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
"""

from __future__ import annotations

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
    if model in _PRICING:
        return _PRICING[model]
    for key, prices in _PRICING.items():
        if model.startswith(key):
            return prices
    return _DEFAULT_PRICE


def _price_tokens(model: str, in_tokens: int, out_tokens: int) -> tuple[float, str]:
    """Return ``(cost_usd, priced_via)`` where ``priced_via`` is 'litellm' or 'static'."""
    try:
        import litellm

        in_cost, out_cost = litellm.cost_per_token(
            model=model, prompt_tokens=int(in_tokens), completion_tokens=int(out_tokens)
        )
        return float(in_cost) + float(out_cost), "litellm"
    except Exception:
        pass
    in_price, out_price = _static_price(model)
    return (int(in_tokens) * in_price + int(out_tokens) * out_price) / 1_000_000, "static"


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


def actual_cost(model, prompt_tokens, completion_tokens) -> tuple[float, str]:
    """Cost from real (post-call) usage. Returns ``(cost_usd, tag)``."""
    cost, via = _price_tokens(model or "", prompt_tokens, completion_tokens)
    return cost, (EXACT if via == "litellm" else HEURISTIC)


def precall_worstcase(model, messages, max_output_tokens, *, system=None, tools=None) -> tuple[float, str]:
    """Pre-call worst-case cost from a request. Returns ``(cost_usd, tag)``.

    The tag is the weakest link: ``"heuristic-fallback"`` if either the token
    count or the pricing fell back, otherwise ``"estimated"``.
    """
    in_tokens, in_tag = count_input_tokens(model, messages, tools=tools, system=system)
    cost, price_tag = worstcase_cost(model, in_tokens, max_output_tokens)
    tag = HEURISTIC if (in_tag == HEURISTIC or price_tag == HEURISTIC) else ESTIMATED
    return cost, tag
