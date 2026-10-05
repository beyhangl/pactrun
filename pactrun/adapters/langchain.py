"""LangChain / LangGraph adapter — emits events to the active pactrun Session.

Unlike the OpenAI/Anthropic adapters (which patch an SDK method), LangChain and
LangGraph instrument via *callbacks*. ``PactrunCallbackHandler`` is a standard
``BaseCallbackHandler`` you pass through the run config; it forwards every LLM
and tool event to the active pactrun Session, so it works for plain LangChain
chains *and* LangGraph graphs (callbacks propagate through the graph).

Usage::

    from pactrun import Contract, cost_under
    from pactrun.adapters import PactrunCallbackHandler

    handler = PactrunCallbackHandler()
    with Contract("agent").require(cost_under(0.50)).session():
        graph.invoke(state, config={"callbacks": [handler]})

For async or multi-threaded execution where the active-session contextvar may
not propagate to the callback, pass the session explicitly::

    with Contract("agent").session() as session:
        handler = PactrunCallbackHandler(session=session)
        await graph.ainvoke(state, config={"callbacks": [handler]})
"""

from __future__ import annotations

import time
from typing import Any

from pactrun.adapters._base import get_session
from pactrun.core.usage import CacheRates, TokenUsage, count, lookup, price, read_field, total

try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "The 'langchain-core' package is required for the LangChain/LangGraph adapter. "
        "Install it with: pip install 'pactrun[langchain]'"
    ) from exc


class PactrunCallbackHandler(BaseCallbackHandler):
    """A LangChain/LangGraph callback handler that records into a pactrun Session."""

    def __init__(self, session: Any = None) -> None:
        self._session = session
        self._starts: dict[Any, float] = {}

    def _resolve_session(self):
        return self._session if self._session is not None else get_session()

    # -- LLM lifecycle -----------------------------------------------------

    def on_llm_start(self, serialized, prompts, *, run_id=None, **kwargs: Any) -> None:
        self._starts[run_id] = time.monotonic()

    def on_chat_model_start(self, serialized, messages, *, run_id=None, **kwargs: Any) -> None:
        self._starts[run_id] = time.monotonic()

    def on_llm_end(self, response, *, run_id=None, **kwargs: Any) -> None:
        session = self._resolve_session()
        if session is None:
            return

        start = self._starts.pop(run_id, None)
        duration_ms = (time.monotonic() - start) * 1000 if start is not None else 0.0

        model = (getattr(response, "llm_output", None) or {}).get("model_name") or "unknown"
        output = _extract_text(response)
        usage = _extract_usage(response)

        session.emit_llm_response(
            model=model,
            output=output,
            cost=_estimate_cost(model, usage),
            duration_ms=duration_ms,
            **usage.event_kwargs(),
        )

    def on_llm_error(self, error, *, run_id=None, **kwargs: Any) -> None:
        session = self._resolve_session()
        if session is None:
            return
        start = self._starts.pop(run_id, None)
        duration_ms = (time.monotonic() - start) * 1000 if start is not None else 0.0
        session.emit_llm_response(
            model="unknown", output="", duration_ms=duration_ms, metadata={"error": str(error)}
        )

    # -- Tool lifecycle ----------------------------------------------------

    def on_tool_start(self, serialized, input_str, *, run_id=None, **kwargs: Any) -> None:
        session = self._resolve_session()
        if session is None:
            return
        name = (serialized or {}).get("name") or kwargs.get("name") or "tool"
        session.emit_tool_call(name, args={"input": input_str})


# ---------------------------------------------------------------------------
# Extraction helpers (robust across LangChain output shapes)
# ---------------------------------------------------------------------------

def _extract_text(response) -> str:
    try:
        return getattr(response.generations[0][0], "text", "") or ""
    except (AttributeError, IndexError, TypeError):
        return ""


def _extract_usage(response) -> TokenUsage:
    """Token usage of an ``LLMResult`` in pactrun's convention.

    Prefers the message's ``usage_metadata`` (LangChain's provider-neutral
    ``UsageMetadata``): its ``input_tokens`` is documented as the "sum of all
    input token types" (so it includes cache reads/writes) and ``output_tokens``
    likewise includes reasoning; ``input_token_details.cache_read`` /
    ``cache_creation`` and ``output_token_details.reasoning`` are subsets.
    Falls back to the provider-raw ``llm_output`` usage dict.
    """
    try:
        message = getattr(response.generations[0][0], "message", None)
    except (AttributeError, IndexError, TypeError):
        message = None
    meta = getattr(message, "usage_metadata", None)
    if meta:
        return _from_usage_metadata(meta)

    llm_output = getattr(response, "llm_output", None) or {}
    raw = llm_output.get("token_usage") or llm_output.get("usage") or {}
    return _from_raw_usage(raw)


def _from_usage_metadata(meta) -> TokenUsage:
    in_details = read_field(meta, "input_token_details")
    out_details = read_field(meta, "output_token_details")
    write = count(read_field(in_details, "cache_creation"))
    write_1h = count(read_field(in_details, "ephemeral_1h_input_tokens"))
    if not write:
        # langchain-anthropic, when Anthropic splits the write by TTL, zeroes
        # `cache_creation` and reports `ephemeral_5m_input_tokens` /
        # `ephemeral_1h_input_tokens` instead (chat_models._create_usage_metadata).
        write = total(read_field(in_details, "ephemeral_5m_input_tokens"), write_1h)
    return TokenUsage(
        prompt_tokens=count(read_field(meta, "input_tokens")),
        completion_tokens=count(read_field(meta, "output_tokens")),
        cache_read_tokens=count(read_field(in_details, "cache_read")),
        cache_write_tokens=write,
        cache_write_1h_tokens=write_1h,
        reasoning_tokens=count(read_field(out_details, "reasoning")),
    )


def _from_raw_usage(raw) -> TokenUsage:
    if not isinstance(raw, dict):
        return TokenUsage()
    if raw.get("prompt_tokens") is not None or raw.get("completion_tokens") is not None:
        # OpenAI-style: totals already include cached input and reasoning.
        prompt_details = raw.get("prompt_tokens_details") or {}
        completion_details = raw.get("completion_tokens_details") or {}
        return TokenUsage(
            prompt_tokens=count(raw.get("prompt_tokens")),
            completion_tokens=count(raw.get("completion_tokens")),
            cache_read_tokens=count(read_field(prompt_details, "cached_tokens")),
            cache_write_tokens=count(read_field(prompt_details, "cache_write_tokens")),
            reasoning_tokens=count(read_field(completion_details, "reasoning_tokens")),
        )
    # Anthropic-style: input_tokens EXCLUDES cache reads and writes.
    read = count(raw.get("cache_read_input_tokens"))
    write = count(raw.get("cache_creation_input_tokens"))
    return TokenUsage(
        prompt_tokens=total(raw.get("input_tokens"), read, write),
        completion_tokens=count(raw.get("output_tokens")),
        cache_read_tokens=read,
        cache_write_tokens=write,
        cache_write_1h_tokens=count(read_field(raw.get("cache_creation"), "ephemeral_1h_input_tokens")),
    )


# Best-effort pricing per 1M tokens (input, output) for common model families.
# Unknown models fall back to 0.0 — pass a priced model name to get cost.
_PRICING: dict[str, tuple[float, float]] = {
    "gpt-5.4": (2.50, 15.00),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4o": (2.50, 10.00),
    "claude-opus-4": (15.00, 75.00),
    "claude-sonnet-4": (3.00, 15.00),
    "claude-haiku-4": (0.80, 4.00),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-1.5-pro": (1.25, 5.00),
    "gemini-1.5-flash": (0.075, 0.30),
}


# Cache multipliers on the base input price, same prefixes as _PRICING. Sources
# (checked 2026-10-05) are the ones documented in the provider adapters:
# OpenAI pricing page cached-input column (gpt-5.4 0.1x, gpt-4.1 0.25x,
# gpt-4o 0.5x; no write charge before GPT-5.6), Anthropic pricing page (read
# 0.1x, 5-minute write 1.25x, 1-hour write 2x), Gemini pricing page /
# genai-prices 0.1.9 (2.5 Pro/Flash 0.1x, 2.0 Flash and 1.5 Flash 0.25x, 1.5 Pro
# no published cached rate so 1.0x).
_ANTHROPIC_CACHE = CacheRates(read=0.10, write=1.25, write_1h=2.0)
_CACHE_RATES: dict[str, CacheRates] = {
    "gpt-5.4": CacheRates(read=0.10),
    "gpt-4.1": CacheRates(read=0.25),
    "gpt-4o": CacheRates(read=0.50),
    "claude-opus-4": _ANTHROPIC_CACHE,
    "claude-sonnet-4": _ANTHROPIC_CACHE,
    "claude-haiku-4": _ANTHROPIC_CACHE,
    "gemini-2.5-pro": CacheRates(read=0.10),
    "gemini-2.5-flash": CacheRates(read=0.10),
    "gemini-2.0-flash": CacheRates(read=0.25),
    "gemini-1.5-flash": CacheRates(read=0.25),
}


def _estimate_cost(model: str, usage: TokenUsage) -> float:
    pricing = lookup(_PRICING, model)
    if not pricing:
        return 0.0
    rates = lookup(_CACHE_RATES, model) or CacheRates()
    return price(usage, pricing[0], pricing[1], rates)
