"""Anthropic adapter — auto-emits events to active pactrun Session.

Patches ``anthropic.resources.messages.Messages.create`` so every call
is automatically recorded into the active Session.
"""

from __future__ import annotations

import time
from typing import Any

from pactrun.adapters._base import get_session
from pactrun.core import pricing
from pactrun.core.usage import TokenUsage, count, read_field, total


class AnthropicAdapter:
    """Patches Anthropic SDK to auto-emit events to active pactrun Session."""

    def __init__(self) -> None:
        self._Messages: Any = None
        self._AsyncMessages: Any = None
        self._original_sync: Any = None
        self._original_async: Any = None
        self._patched = False

    def __enter__(self) -> AnthropicAdapter:
        self._patch()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._unpatch()

    async def __aenter__(self) -> AnthropicAdapter:
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
        usage = usage_from_anthropic(getattr(response, "usage", None))

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

        quote = pricing.cost(model, usage)
        session.emit_llm_response(
            model=model,
            output=output,
            cost=quote.usd,
            duration_ms=duration_ms,
            metadata=quote.metadata or None,
            **usage.event_kwargs(),
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


def usage_from_anthropic(usage: Any) -> TokenUsage:
    """Normalise an Anthropic ``Usage`` (or streaming usage) to pactrun's convention.

    Anthropic's ``input_tokens`` counts only the input AFTER the last cache
    breakpoint; the total input is
    ``input_tokens + cache_read_input_tokens + cache_creation_input_tokens``
    (Anthropic prompt-caching docs, "Understanding the token breakdown").
    ``cache_creation.ephemeral_1h_input_tokens`` is the part of the cache write
    that went to the one-hour cache (absent on older SDKs: all writes are then
    priced at the five-minute rate). ``output_tokens`` already includes
    thinking; ``output_tokens_details.thinking_tokens`` is a subset of it.
    ``speed`` ("fast") and ``inference_geo`` ("us") change the price
    (anthropic-python ``types/usage.py``; absent on older SDKs).
    """
    if usage is None:
        return TokenUsage()
    read = count(read_field(usage, "cache_read_input_tokens"))
    write = count(read_field(usage, "cache_creation_input_tokens"))
    breakdown = read_field(usage, "cache_creation")
    return TokenUsage(
        prompt_tokens=total(read_field(usage, "input_tokens"), read, write),
        completion_tokens=count(read_field(usage, "output_tokens")),
        cache_read_tokens=read,
        cache_write_tokens=write,
        cache_write_1h_tokens=count(read_field(breakdown, "ephemeral_1h_input_tokens")),
        reasoning_tokens=count(read_field(read_field(usage, "output_tokens_details"), "thinking_tokens")),
        speed=read_field(usage, "speed"),
        inference_geo=read_field(usage, "inference_geo"),
    )
