"""Gemini adapter — auto-emits events to the active pactrun Session.

Patches the google-genai SDK's ``Models.generate_content`` (and the async
``AsyncModels.generate_content``) so every Gemini call is recorded into the
active Session.

Usage::

    from google import genai
    from pactrun import Contract, cost_under
    from pactrun.adapters import GeminiAdapter

    client = genai.Client()
    with Contract("agent").require(cost_under(0.50)).session():
        with GeminiAdapter():
            client.models.generate_content(model="gemini-2.5-flash", contents="Hello")
"""

from __future__ import annotations

import time
from typing import Any

from pactrun.adapters._base import get_session


class GeminiAdapter:
    """Patches the google-genai SDK to auto-emit events to the active pactrun Session."""

    def __init__(self) -> None:
        self._Models: Any = None
        self._AsyncModels: Any = None
        self._original_sync: Any = None
        self._original_async: Any = None
        self._patched = False

    def __enter__(self) -> "GeminiAdapter":
        self._patch()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._unpatch()

    async def __aenter__(self) -> "GeminiAdapter":
        self._patch()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        self._unpatch()

    def _patch(self) -> None:
        if self._patched:
            return
        try:
            from google.genai.models import AsyncModels, Models
        except ImportError as exc:
            raise ImportError(
                "The 'google-genai' package is required for GeminiAdapter. "
                "Install it with: pip install 'pactrun[gemini]'"
            ) from exc

        self._Models = Models
        self._AsyncModels = AsyncModels
        self._original_sync = Models.generate_content
        self._original_async = AsyncModels.generate_content

        adapter = self
        original_sync = self._original_sync
        original_async = self._original_async

        def patched_sync(self_models: Any, *args: Any, **kwargs: Any) -> Any:
            start = time.monotonic()
            try:
                response = original_sync(self_models, *args, **kwargs)
            except Exception as exc:
                adapter._emit_error(kwargs, (time.monotonic() - start) * 1000, str(exc))
                raise
            adapter._emit_response(kwargs, response, (time.monotonic() - start) * 1000)
            return response

        async def patched_async(self_models: Any, *args: Any, **kwargs: Any) -> Any:
            start = time.monotonic()
            try:
                response = await original_async(self_models, *args, **kwargs)
            except Exception as exc:
                adapter._emit_error(kwargs, (time.monotonic() - start) * 1000, str(exc))
                raise
            adapter._emit_response(kwargs, response, (time.monotonic() - start) * 1000)
            return response

        Models.generate_content = patched_sync  # type: ignore[method-assign]
        AsyncModels.generate_content = patched_async  # type: ignore[method-assign]
        self._patched = True

    def _unpatch(self) -> None:
        if not self._patched:
            return
        if self._Models and self._original_sync:
            self._Models.generate_content = self._original_sync
        if self._AsyncModels and self._original_async:
            self._AsyncModels.generate_content = self._original_async
        self._patched = False

    def _emit_response(self, kwargs: dict, response: Any, duration_ms: float) -> None:
        session = get_session()
        if session is None:
            return

        model = kwargs.get("model") or getattr(response, "model_version", None) or "unknown"

        prompt_tokens = 0
        completion_tokens = 0
        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
            completion_tokens = getattr(usage, "candidates_token_count", 0) or 0

        # Emit any function (tool) calls the model requested.
        try:
            for call in getattr(response, "function_calls", None) or []:
                name = getattr(call, "name", None)
                if name:
                    session.emit_tool_call(name, args=getattr(call, "args", None) or {})
        except (AttributeError, TypeError):
            pass

        # ``.text`` is a convenience property that can raise when the response
        # carries no text parts (e.g. a pure function call) — guard it.
        try:
            output = getattr(response, "text", None) or ""
        except Exception:
            output = ""

        cost = _estimate_cost(model, prompt_tokens, completion_tokens)

        session.emit_llm_response(
            model=model,
            output=output,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=cost,
            duration_ms=duration_ms,
        )

    def _emit_error(self, kwargs: dict, duration_ms: float, error: str) -> None:
        session = get_session()
        if session is None:
            return
        session.emit_llm_response(
            model=kwargs.get("model", "unknown"),
            output="",
            duration_ms=duration_ms,
            metadata={"error": error},
        )


# Best-effort pricing per 1M tokens (input, output) for common Gemini models.
_PRICING: dict[str, tuple[float, float]] = {
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
