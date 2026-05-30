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
        prompt_tokens, completion_tokens = _extract_usage(response)
        cost = _estimate_cost(model, prompt_tokens, completion_tokens)

        session.emit_llm_response(
            model=model,
            output=output,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=cost,
            duration_ms=duration_ms,
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


def _extract_usage(response) -> tuple[int, int]:
    # 1) llm_output.token_usage (OpenAI-style) or .usage
    llm_output = getattr(response, "llm_output", None) or {}
    usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
    prompt = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
    completion = usage.get("completion_tokens") or usage.get("output_tokens") or 0
    if prompt or completion:
        return int(prompt), int(completion)

    # 2) usage_metadata on the message (newer LangChain chat models)
    try:
        message = getattr(response.generations[0][0], "message", None)
        meta = getattr(message, "usage_metadata", None) or {}
        return int(meta.get("input_tokens", 0)), int(meta.get("output_tokens", 0))
    except (AttributeError, IndexError, TypeError):
        return 0, 0


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


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    pricing = _PRICING.get(model)
    if not pricing:
        for key, prices in _PRICING.items():
            if model.startswith(key):
                pricing = prices
                break
    if not pricing:
        return 0.0
    return (prompt_tokens * pricing[0] + completion_tokens * pricing[1]) / 1_000_000
