"""Token counting + pricing for the pactrun pre-call cost gate.

Uses real tokenizers and live pricing when available (tiktoken for OpenAI
models; litellm for non-OpenAI token counts and for all pricing), and degrades
to a cheap heuristic when those libraries aren't installed or the model is
unknown. Every public function returns a ``(value, tag)`` pair so callers can
tell users how the number was obtained:

- ``"exact"``              — priced from real (post-call) usage via litellm.
- ``"estimated"``          — a real tokenizer + real pricing, but a pre-call bound.
- ``"heuristic-fallback"`` — the crude ``len // 4`` / static-table path
  (a library was missing or the model was unknown).

Honesty note: a pre-call number is a worst-case **bound**, never an exact bill —
you cannot know completion tokens before a call, and Anthropic/Gemini have no
public tokenizer (litellm uses a BPE estimate that can differ from the provider).

Prompt caching: post-call costs price cache reads and writes at their own rates
(``prompt_tokens`` is the total input, cache counts are subsets of it). The
pre-call bound cannot know whether a call will hit the cache, so it never
assumes a discount; it does assume the cache-WRITE premium whenever a write can
happen (Anthropic requests carrying ``cache_control``: 1.25x the input price,
2x for a ``ttl: "1h"`` breakpoint), because a full-price input bound would be
below the real bill for such a call.
"""

from __future__ import annotations

from typing import Any

from pactrun.core.usage import CacheRates, TokenUsage, lookup, price

EXACT = "exact"
ESTIMATED = "estimated"
HEURISTIC = "heuristic-fallback"

# Static fallback pricing (USD per 1M tokens: input, output) — used only when
# litellm can't price a model. Conservative defaults so unknown models err
# toward refusing the call.
_PRICING: dict[str, tuple[float, float]] = {
    "gpt-5.4": (2.50, 15.00),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4o": (2.50, 10.00),
    "o3": (2.00, 8.00),
    "o4-mini": (0.55, 2.20),
    "claude-opus-4": (15.00, 75.00),
    "claude-sonnet-4": (3.00, 15.00),
    "claude-haiku-4": (0.80, 4.00),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.5-flash": (0.30, 2.50),
}
_DEFAULT_PRICE = (2.50, 10.00)

# Cache multipliers for the static fallback, same prefixes as _PRICING and the
# same sources as the provider adapters (checked 2026-10-05): OpenAI pricing
# page cached-input column, no write charge before GPT-5.6; Anthropic pricing
# page read 0.1x / 5-minute write 1.25x / 1-hour write 2x; Gemini pricing page
# 2.5 Pro/Flash 0.1x. Unknown models get no read discount and the highest known
# write premiums, so the fallback errs toward over-counting.
_ANTHROPIC_CACHE = CacheRates(read=0.10, write=1.25, write_1h=2.0)
_CACHE_RATES: dict[str, CacheRates] = {
    "gpt-5.4": CacheRates(read=0.10),
    "gpt-4.1": CacheRates(read=0.25),
    "gpt-4o": CacheRates(read=0.50),
    "o3": CacheRates(read=0.25),
    "o4-mini": CacheRates(read=0.25),
    "claude-opus-4": _ANTHROPIC_CACHE,
    "claude-sonnet-4": _ANTHROPIC_CACHE,
    "claude-haiku-4": _ANTHROPIC_CACHE,
    "gemini-2.5-pro": CacheRates(read=0.10),
    "gemini-2.5-flash": CacheRates(read=0.10),
}
_DEFAULT_CACHE = CacheRates(read=1.0, write=1.25, write_1h=2.0)


def _normalize_messages(messages, system) -> list[dict]:
    msgs: list[dict] = []
    if isinstance(system, str) and system:
        msgs.append({"role": "system", "content": system})
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role", "user")
        content = message.get("content")
        if isinstance(content, str):
            msgs.append({"role": role, "content": content})
        elif isinstance(content, list):
            text = " ".join(
                b["text"] for b in content
                if isinstance(b, dict) and isinstance(b.get("text"), str)
            )
            msgs.append({"role": role, "content": text})
    return msgs


def _text_of(messages, system) -> str:
    return " ".join(m["content"] for m in _normalize_messages(messages, system))


def _is_openai_model(model: str) -> bool:
    m = (model or "").lower()
    return not (m.startswith("claude") or m.startswith("gemini") or m.startswith("anthropic"))


def count_input_tokens(model, messages=None, *, tools=None, system=None) -> tuple[int, str]:
    """Count the input tokens for a request. Returns ``(tokens, tag)``."""
    model = model or ""
    # Non-OpenAI models: tiktoken (o200k_base) undercounts Claude by ~20-33%,
    # which would bias the budget gate toward letting over-budget calls
    # through — the dangerous direction. Use litellm's per-provider estimate.
    if not _is_openai_model(model):
        try:
            import litellm

            n = litellm.token_counter(model=model, messages=_normalize_messages(messages, system))
            return int(n), ESTIMATED
        except Exception:
            pass
    else:
        try:
            import tiktoken

            try:
                enc = tiktoken.encoding_for_model(model)
            except KeyError:
                enc = tiktoken.get_encoding("o200k_base")
            return len(enc.encode(_text_of(messages, system))), ESTIMATED
        except Exception:
            pass
    # Heuristic fallback (~4 chars/token).
    return max(1, len(_text_of(messages, system)) // 4), HEURISTIC


def _static_price(model: str) -> tuple[float, float]:
    return lookup(_PRICING, model) or _DEFAULT_PRICE


def _litellm_info(model: str) -> dict:
    try:
        import litellm

        return dict(litellm.get_model_info(model) or {})
    except Exception:
        return {}


def _price_tokens(
    model: str,
    in_tokens: int,
    out_tokens: int,
    *,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    cache_write_1h_tokens: int = 0,
) -> tuple[float, str]:
    """Return ``(cost_usd, priced_via)`` where ``priced_via`` is 'litellm' or 'static'.

    ``in_tokens`` is the total input; the cache counts are subsets of it.
    """
    try:
        import litellm

        kwargs: dict[str, Any] = {}
        info = _litellm_info(model) if (cache_read_tokens or cache_write_tokens) else {}
        # litellm prices a cache slice at 0 when the model has no rate for it,
        # so only hand it a slice it can price; otherwise that slice stays in
        # prompt_tokens at the full input rate (an over-estimate, never under).
        if cache_read_tokens and info.get("cache_read_input_token_cost"):
            kwargs["cache_read_input_tokens"] = int(cache_read_tokens)
        write_rate = info.get("cache_creation_input_token_cost")
        if cache_write_tokens and write_rate:
            kwargs["cache_creation_input_tokens"] = int(cache_write_tokens)
        in_cost, out_cost = litellm.cost_per_token(
            model=model, prompt_tokens=int(in_tokens), completion_tokens=int(out_tokens), **kwargs
        )
        cost = float(in_cost) + float(out_cost)
        # litellm prices every write at the 5-minute rate; add the 1-hour premium.
        rate_1h = info.get("cache_creation_input_token_cost_above_1hr")
        if cache_write_1h_tokens and write_rate and rate_1h:
            cost += min(int(cache_write_1h_tokens), int(cache_write_tokens)) * (float(rate_1h) - float(write_rate))
        return cost, "litellm"
    except Exception:
        pass
    in_price, out_price = _static_price(model)
    usage = TokenUsage(
        prompt_tokens=int(in_tokens),
        completion_tokens=int(out_tokens),
        cache_read_tokens=int(cache_read_tokens),
        cache_write_tokens=int(cache_write_tokens),
        cache_write_1h_tokens=int(cache_write_1h_tokens),
    )
    rates = lookup(_CACHE_RATES, model) or _DEFAULT_CACHE
    return price(usage, in_price, out_price, rates), "static"


def _cap_output(model: str, max_output_tokens: int) -> int:
    """Cap the assumed worst-case output at the model's real max, when known."""
    try:
        import litellm

        cap = litellm.get_model_info(model).get("max_output_tokens")
        if cap:
            return min(int(max_output_tokens), int(cap))
    except Exception:
        pass
    return int(max_output_tokens)


def worstcase_cost(model, input_tokens, max_output_tokens) -> tuple[float, str]:
    """Worst-case cost of one call: input + the maximum output you allow."""
    model = model or ""
    cost, via = _price_tokens(model, input_tokens, _cap_output(model, max_output_tokens))
    return cost, (ESTIMATED if via == "litellm" else HEURISTIC)


def actual_cost(
    model,
    prompt_tokens,
    completion_tokens,
    *,
    cache_read_tokens=0,
    cache_write_tokens=0,
    cache_write_1h_tokens=0,
) -> tuple[float, str]:
    """Cost from real (post-call) usage. Returns ``(cost_usd, tag)``.

    ``prompt_tokens`` is the total input; ``cache_read_tokens`` /
    ``cache_write_tokens`` (and the 1-hour share of the writes) are subsets of
    it, priced at their own rates. Invalid counts give a ``nan`` cost so the
    session rejects it and the cost predicates fail closed.
    """
    usage = TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        cache_write_1h_tokens=cache_write_1h_tokens,
    )
    if not usage.is_valid():
        return float("nan"), HEURISTIC
    cost, via = _price_tokens(
        model or "",
        prompt_tokens,
        completion_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        cache_write_1h_tokens=cache_write_1h_tokens,
    )
    return cost, (EXACT if via == "litellm" else HEURISTIC)


def _cache_ttls(obj: Any, depth: int = 0) -> set[str]:
    """TTLs of every ``cache_control`` marker in a request fragment ("5m" when unset)."""
    found: set[str] = set()
    if depth > 8:
        return found
    if isinstance(obj, dict):
        marker = obj.get("cache_control")
        if isinstance(marker, dict):
            found.add(str(marker.get("ttl") or "5m"))
        for value in obj.values():
            if isinstance(value, (dict, list)):
                found |= _cache_ttls(value, depth + 1)
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, (dict, list)):
                found |= _cache_ttls(item, depth + 1)
    return found


def cache_write_multiplier(model, messages=None, *, system=None, tools=None, cache_control=None) -> float:
    """Worst-case input-price multiplier from prompt-cache writes for one request.

    Anthropic writes to the cache only when the request carries
    ``cache_control`` (top level or on a block), at 1.25x the input price for
    the default 5-minute TTL and 2x for ``ttl: "1h"`` (Anthropic pricing page,
    checked 2026-10-05). Other providers cache automatically; when litellm
    knows a write rate above the input rate (e.g. OpenAI GPT-5.6+, 1.25x), that
    ratio is used. Otherwise 1.0 (writes cost no more than plain input).
    """
    m = (model or "").lower()
    if m.startswith("claude") or "anthropic" in m:
        top = {"cache_control": cache_control} if isinstance(cache_control, dict) else None
        ttls = _cache_ttls([top, system, messages, tools])
        if not ttls:
            return 1.0
        return 2.0 if "1h" in ttls else 1.25
    info = _litellm_info(model or "")
    write_rate = info.get("cache_creation_input_token_cost")
    in_rate = info.get("input_cost_per_token")
    try:
        if write_rate and in_rate:
            return max(1.0, float(write_rate) / float(in_rate))
    except (TypeError, ValueError, ZeroDivisionError):
        pass
    return 1.0


def precall_worstcase(
    model, messages, max_output_tokens, *, system=None, tools=None, cache_control=None
) -> tuple[float, str]:
    """Pre-call worst-case cost from a request. Returns ``(cost_usd, tag)``.

    The tag is the weakest link: ``"heuristic-fallback"`` if either the token
    count or the pricing fell back, otherwise ``"estimated"``. Cache reads are
    never assumed (a miss is the worst case); cache-write premiums are, when the
    request can write (see :func:`cache_write_multiplier`).
    """
    in_tokens, in_tag = count_input_tokens(model, messages, tools=tools, system=system)
    cost, price_tag = worstcase_cost(model, in_tokens, max_output_tokens)
    multiplier = cache_write_multiplier(
        model, messages, system=system, tools=tools, cache_control=cache_control
    )
    if multiplier > 1.0:
        input_cost, _ = _price_tokens(model or "", in_tokens, 0)
        cost += (multiplier - 1.0) * input_cost
    tag = HEURISTIC if (in_tag == HEURISTIC or price_tag == HEURISTIC) else ESTIMATED
    return cost, tag
