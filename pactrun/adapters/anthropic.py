"""Anthropic adapter — auto-emits events to active pactrun Session.

Patches ``anthropic.resources.messages.Messages.create`` so every call
is automatically recorded into the active Session.
"""

from __future__ import annotations

import time
from typing import Any

from pactrun.adapters._base import get_session


class AnthropicAdapter:
    """Patches Anthropic SDK to auto-emit events to active pactrun Session."""

    def __init__(self) -> None:
        self._Messages: Any = None
        self._AsyncMessages: Any = None
        self._original_sync: Any = None
        self._original_async: Any = None
        self._patched = False

    def __enter__(self) -> "AnthropicAdapter":
        self._patch()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._unpatch()

    async def __aenter__(self) -> "AnthropicAdapter":
        self._patch()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        self._unpatch()

    def _patch(self) -> None:
        if self._patched:
            return
        try:
            from anthropic.resources.messages import AsyncMessages, Messages
        except ImportError as exc:
            raise ImportError(
                "The 'anthropic' package is required for AnthropicAdapter. "
                "Install it with: pip install 'pactrun[anthropic]'"
            ) from exc

        self._Messages = Messages
        self._AsyncMessages = AsyncMessages
        self._original_sync = Messages.create
        self._original_async = AsyncMessages.create

        adapter = self
        original_sync = self._original_sync
        original_async = self._original_async

        def patched_sync(self_msg: Any, *args: Any, **kwargs: Any) -> Any:
            start = time.monotonic()
            try:
                response = original_sync(self_msg, *args, **kwargs)
            except Exception as exc:
                duration_ms = (time.monotonic() - start) * 1000
                adapter._emit_error(kwargs, duration_ms, str(exc))
                raise
            duration_ms = (time.monotonic() - start) * 1000
            adapter._emit_response(kwargs, response, duration_ms)
            return response

        async def patched_async(self_msg: Any, *args: Any, **kwargs: Any) -> Any:
            start = time.monotonic()
            try:
                response = await original_async(self_msg, *args, **kwargs)
            except Exception as exc:
                duration_ms = (time.monotonic() - start) * 1000
                adapter._emit_error(kwargs, duration_ms, str(exc))
                raise
            duration_ms = (time.monotonic() - start) * 1000
            adapter._emit_response(kwargs, response, duration_ms)
            return response

        Messages.create = patched_sync  # type: ignore[method-assign]
        AsyncMessages.create = patched_async  # type: ignore[method-assign]
        self._patched = True

    def _unpatch(self) -> None:
        if not self._patched:
            return
        if self._Messages and self._original_sync:
            self._Messages.create = self._original_sync
        if self._AsyncMessages and self._original_async:
            self._AsyncMessages.create = self._original_async
        self._patched = False

    def _emit_response(self, kwargs: dict, response: Any, duration_ms: float) -> None:
        session = get_session()
        if session is None:
            return

        model = getattr(response, "model", None) or kwargs.get("model", "unknown")
        prompt_tokens = 0
        completion_tokens = 0
        try:
            usage = response.usage
            prompt_tokens = getattr(usage, "input_tokens", 0) or 0
            completion_tokens = getattr(usage, "output_tokens", 0) or 0
        except AttributeError:
            pass

        output = ""
        try:
            for block in response.content:
                if hasattr(block, "text"):
                    output += block.text
                elif hasattr(block, "name"):
                    # Tool use block
                    session.emit_tool_call(
                        block.name,
                        args=getattr(block, "input", None) or {},
                    )
        except (AttributeError, TypeError):
            pass

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


_PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-4-6": (15.00, 75.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5-20251001": (0.80, 4.00),
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
