"""Token counting + pricing for the pactrun pre-call cost gate.

Uses real tokenizers when available (tiktoken for OpenAI models; litellm for
non-OpenAI token counts) and degrades to a cheap heuristic when those libraries
aren't installed. Prices come from pactrun's own table
(:mod:`pactrun.core.pricing`); litellm's price map is consulted only for models
that table does not know, and a model neither knows is priced at the
provider's ceiling (never $0). Every public function returns a
``(value, tag)`` pair so callers can tell users how the number was obtained:

- ``"exact"``              — real (post-call) usage at a known model's price.
- ``"estimated"``          — a real tokenizer + a known price, but a pre-call bound.
- ``"heuristic-fallback"`` — the crude ``len // 4`` count, or an unknown model
  priced at the ceiling.

Honesty note: a pre-call number estimates input plus assumed maximum output;
it is not a universal bound on the bill. Tokenization, provider framing, media,
pricing and hidden reasoning can differ from these assumptions. Reported usage
is priced after the call.

Prompt caching: post-call costs price cache reads and writes at their own rates
(``prompt_tokens`` is the total input, cache counts are subsets of it). The
pre-call bound cannot know whether a call will hit the cache, so it never
assumes a discount; it does assume the cache-WRITE premium whenever a write can
happen (Anthropic requests carrying ``cache_control``: 1.25x the input price,
2x for a ``ttl: "1h"`` breakpoint), because a full-price input bound would be
below the real bill for such a call.
"""

from __future__ import annotations

import json
from typing import Any

from pactrun.core import pricing
from pactrun.core.usage import TokenUsage

EXACT = "exact"
ESTIMATED = "estimated"
HEURISTIC = "heuristic-fallback"

def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for block in content if isinstance(content, list) else []:
        if not isinstance(block, dict):
            continue
        if isinstance(block.get("text"), str):
            parts.append(block["text"])
        elif block.get("type") == "tool_use":
            parts.append(json.dumps({"name": block.get("name"), "input": block.get("input")}, ensure_ascii=False, default=str))
        elif block.get("type") == "tool_result":
            parts.append(_content_text(block.get("content")))
    return " ".join(parts)


def _normalize_messages(messages, system) -> list[dict]:
    msgs: list[dict] = []
    system_text = _content_text(system)
    if system_text:
        msgs.append({"role": "system", "content": system_text})
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role", "user")
        text = _content_text(message.get("content"))
        # Assistant tool-call history is billed input even with null content.
        for field in ("tool_calls", "function_call"):
            if message.get(field):
                text += " " + json.dumps(message[field], ensure_ascii=False, default=str)
        msgs.append({"role": role, "content": text})
    return msgs


def _text_of(messages, system, tools=None) -> str:
    text = " ".join(m["content"] for m in _normalize_messages(messages, system))
    if tools:
        # Definitions (descriptions, JSON schemas, etc.) are billed input too.
        # JSON is an estimate of their provider-specific wire representation.
        text += " " + json.dumps(tools, sort_keys=True, ensure_ascii=False, default=str)
    return text


def _is_openai_model(model: str) -> bool:
    m = (model or "").lower()
    return not (m.startswith("claude") or m.startswith("gemini") or m.startswith("anthropic"))


def count_input_tokens(model, messages=None, *, tools=None, system=None) -> tuple[int, str]:
    """Estimate request text and tool-definition tokens. Returns ``(tokens, tag)``.

    Provider framing and non-text media can add billed tokens beyond this count.
    """
    model = model or ""
    # Non-OpenAI models: tiktoken (o200k_base) undercounts Claude by ~20-33%,
    # which would bias the budget gate toward letting over-budget calls
    # through — the dangerous direction. Use litellm's per-provider estimate.
    if not _is_openai_model(model):
        try:
            import litellm

            kwargs = {"tools": tools} if tools else {}
            n = litellm.token_counter(model=model, messages=_normalize_messages(messages, system), **kwargs)
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
            return len(enc.encode(_text_of(messages, system, tools))), ESTIMATED
        except Exception:
            pass
    # Heuristic fallback (~4 chars/token).
    return max(1, len(_text_of(messages, system, tools)) // 4), HEURISTIC


def _litellm_info(model: str) -> dict:
    try:
        import litellm

        return dict(litellm.get_model_info(model) or {})
    except Exception:
        return {}


def _litellm_cost(model: str, usage: TokenUsage) -> float | None:
    """litellm's price for a model pactrun's table lacks, or ``None`` when it has none."""
    try:
        import litellm

        info = _litellm_info(model)
        if not info.get("input_cost_per_token"):
            return None
        kwargs: dict[str, Any] = {}
        # litellm prices a cache slice at 0 when the model has no rate for it,
        # so only hand it a slice it can price; otherwise that slice stays in
        # prompt_tokens at the full input rate (an over-estimate, never under).
        if usage.cache_read_tokens and info.get("cache_read_input_token_cost"):
            kwargs["cache_read_input_tokens"] = int(usage.cache_read_tokens)
        write_rate = info.get("cache_creation_input_token_cost")
        if usage.cache_write_tokens and write_rate:
            kwargs["cache_creation_input_tokens"] = int(usage.cache_write_tokens)
        in_cost, out_cost = litellm.cost_per_token(
            model=model, prompt_tokens=int(usage.prompt_tokens),
            completion_tokens=int(usage.completion_tokens), **kwargs,
        )
        cost = float(in_cost) + float(out_cost)
        # litellm prices every write at the 5-minute rate; add the 1-hour premium.
        rate_1h = info.get("cache_creation_input_token_cost_above_1hr")
        if usage.cache_write_1h_tokens and write_rate and rate_1h:
            cost += min(int(usage.cache_write_1h_tokens), int(usage.cache_write_tokens)) * (
                float(rate_1h) - float(write_rate)
            )
        return cost * pricing.modifier(usage)
    except Exception:
        return None


def is_unknown_model(model: str) -> bool:
    """True when neither pactrun's table nor litellm can price ``model``."""
    return pricing.find(model) is None and not _litellm_info(model).get("input_cost_per_token")


def cost_metadata(model: str) -> dict[str, Any]:
    """Event metadata flagging a cost priced at the unknown-model ceiling."""
    return {pricing.COST_SOURCE: pricing.UNKNOWN_MODEL} if is_unknown_model(model) else {}


def _price_tokens(
    model: str,
    in_tokens: int,
    out_tokens: int,
    *,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    cache_write_1h_tokens: int = 0,
    speed: Any = None,
    inference_geo: Any = None,
) -> tuple[float, str]:
    """Return ``(cost_usd, priced_via)``; ``priced_via`` is 'pactrun', 'litellm' or 'unknown'.

    ``in_tokens`` is the total input; the cache counts are subsets of it.
    """
    usage = TokenUsage(
        prompt_tokens=int(in_tokens),
        completion_tokens=int(out_tokens),
        cache_read_tokens=int(cache_read_tokens),
        cache_write_tokens=int(cache_write_tokens),
        cache_write_1h_tokens=int(cache_write_1h_tokens),
        speed=speed,
        inference_geo=inference_geo,
    )
    if pricing.find(model) is None:
        cost = _litellm_cost(model, usage)
        if cost is not None:
            return cost, "litellm"
    quote = pricing.cost(model, usage)
    return quote.usd, ("pactrun" if quote.known else "unknown")


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


def worstcase_cost(model, input_tokens, max_output_tokens, *, speed=None, inference_geo=None) -> tuple[float, str]:
    """Worst-case cost of one call: input + the maximum output you allow.

    ``speed`` / ``inference_geo`` are the request's Anthropic parameters
    (``"fast"`` / ``"us"``), which raise the price.
    """
    model = model or ""
    cost, via = _price_tokens(
        model, input_tokens, _cap_output(model, max_output_tokens), speed=speed, inference_geo=inference_geo
    )
    return cost, (HEURISTIC if via == "unknown" else ESTIMATED)


def actual_cost(
    model,
    prompt_tokens,
    completion_tokens,
    *,
    cache_read_tokens=0,
    cache_write_tokens=0,
    cache_write_1h_tokens=0,
    speed=None,
    inference_geo=None,
) -> tuple[float, str]:
    """Cost from real (post-call) usage. Returns ``(cost_usd, tag)``.

    ``prompt_tokens`` is the total input; ``cache_read_tokens`` /
    ``cache_write_tokens`` (and the 1-hour share of the writes) are subsets of
    it, priced at their own rates. ``speed`` / ``inference_geo`` are the
    response's Anthropic usage fields. Invalid counts give a ``nan`` cost so
    the session rejects it and the cost predicates fail closed.
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
        speed=speed,
        inference_geo=inference_geo,
    )
    return cost, (HEURISTIC if via == "unknown" else EXACT)


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
    checked 2026-10-19). Other providers cache automatically, so their
    write multiplier from pactrun's table applies (OpenAI GPT-5.6+: 1.25x), or
    litellm's for a model the table lacks, or the provider ceiling's for an
    unknown model.
    """
    m = (model or "").lower()
    if m.startswith("claude") or "anthropic" in m:
        top = {"cache_control": cache_control} if isinstance(cache_control, dict) else None
        ttls = _cache_ttls([top, system, messages, tools])
        if not ttls:
            return 1.0
        return 2.0 if "1h" in ttls else 1.25
    found = pricing.find(model)
    if found is not None:
        return max(1.0, found[1].cache.write)
    info = _litellm_info(model or "")
    write_rate = info.get("cache_creation_input_token_cost")
    in_rate = info.get("input_cost_per_token")
    try:
        if write_rate and in_rate:
            return max(1.0, float(write_rate) / float(in_rate))
    except (TypeError, ValueError, ZeroDivisionError):
        pass
    if in_rate:
        return 1.0
    return max(1.0, pricing.ceiling(pricing.provider_of(model)).cache.write)


def precall_worstcase(
    model, messages, max_output_tokens, *, system=None, tools=None, cache_control=None,
    speed=None, inference_geo=None,
) -> tuple[float, str]:
    """Pre-call worst-case cost from a request. Returns ``(cost_usd, tag)``.

    The tag is the weakest link: ``"heuristic-fallback"`` if either the token
    count or the pricing fell back, otherwise ``"estimated"``. Cache reads are
    never assumed (a miss is the worst case); cache-write premiums are, when the
    request can write (see :func:`cache_write_multiplier`).
    """
    in_tokens, in_tag = count_input_tokens(model, messages, tools=tools, system=system)
    cost, price_tag = worstcase_cost(
        model, in_tokens, max_output_tokens, speed=speed, inference_geo=inference_geo
    )
    multiplier = cache_write_multiplier(
        model, messages, system=system, tools=tools, cache_control=cache_control
    )
    if multiplier > 1.0:
        input_cost, _ = _price_tokens(model or "", in_tokens, 0, speed=speed, inference_geo=inference_geo)
        cost += (multiplier - 1.0) * input_cost
    tag = HEURISTIC if (in_tag == HEURISTIC or price_tag == HEURISTIC) else ESTIMATED
    return cost, tag
