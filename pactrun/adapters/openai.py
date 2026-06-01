"""OpenAI adapter — auto-emits events to active pactrun Session.

Patches ``openai.resources.chat.completions.Completions.create`` so every
call is automatically recorded into the active Session.

Usage::

    from pactrun.adapters import OpenAIAdapter

    with contract.session() as session:
        with OpenAIAdapter():
            response = client.chat.completions.create(
                model="gpt-5.4-nano",
                messages=[{"role": "user", "content": "Hello"}],
            )
    # session automatically received llm_call + tool_call events
"""

from __future__ import annotations

import json
import time
from typing import Any

from pactrun.adapters._base import get_session


class OpenAIAdapter:
    """Patches OpenAI SDK to auto-emit events to active pactrun Session."""

    def __init__(self) -> None:
        self._Completions: Any = None
        self._AsyncCompletions: Any = None
        self._original_sync: Any = None
        self._original_async: Any = None
        self._patched = False

    def __enter__(self) -> "OpenAIAdapter":
        self._patch()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._unpatch()

    async def __aenter__(self) -> "OpenAIAdapter":
        self._patch()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        self._unpatch()

    def _patch(self) -> None:
        if self._patched:
            return
        try:
            from openai.resources.chat.completions import (
                AsyncCompletions,
                Completions,
            )
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for OpenAIAdapter. "
                "Install it with: pip install 'pactrun[openai]'"
            ) from exc

        self._Completions = Completions
        self._AsyncCompletions = AsyncCompletions
        self._original_sync = Completions.create
        self._original_async = AsyncCompletions.create

        adapter = self
        original_sync = self._original_sync
        original_async = self._original_async

        def patched_sync(self_comp: Any, *args: Any, **kwargs: Any) -> Any:
            start = time.monotonic()
            try:
                response = original_sync(self_comp, *args, **kwargs)
            except Exception as exc:
                duration_ms = (time.monotonic() - start) * 1000
                adapter._emit_error(kwargs, duration_ms, str(exc))
                raise
            duration_ms = (time.monotonic() - start) * 1000
            adapter._emit_response(kwargs, response, duration_ms)
            return response

        async def patched_async(self_comp: Any, *args: Any, **kwargs: Any) -> Any:
            start = time.monotonic()
            try:
                response = await original_async(self_comp, *args, **kwargs)
            except Exception as exc:
                duration_ms = (time.monotonic() - start) * 1000
                adapter._emit_error(kwargs, duration_ms, str(exc))
                raise
            duration_ms = (time.monotonic() - start) * 1000
            adapter._emit_response(kwargs, response, duration_ms)
            return response

        Completions.create = patched_sync  # type: ignore[method-assign]
        AsyncCompletions.create = patched_async  # type: ignore[method-assign]
        self._patched = True

    def _unpatch(self) -> None:
        if not self._patched:
            return
        if self._Completions and self._original_sync:
            self._Completions.create = self._original_sync
        if self._AsyncCompletions and self._original_async:
            self._AsyncCompletions.create = self._original_async
        self._patched = False

    def _emit_response(self, kwargs: dict, response: Any, duration_ms: float) -> None:
        session = get_session()
        if session is None:
            return

        model = getattr(response, "model", None) or kwargs.get("model", "unknown")

        # Extract token usage
        prompt_tokens = 0
        completion_tokens = 0
        try:
            usage = response.usage
            if usage:
                prompt_tokens = usage.prompt_tokens or 0
                completion_tokens = usage.completion_tokens or 0
        except AttributeError:
            pass

        # Extract output text
        output = ""
        try:
            msg = response.choices[0].message
            output = msg.content or ""
        except (AttributeError, IndexError):
            pass

        # Extract cost estimate
        cost = _estimate_cost(model, prompt_tokens, completion_tokens)

        # Emit tool calls if present
        try:
            msg = response.choices[0].message
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                    except (ValueError, AttributeError):
                        args = {}
                    session.emit_tool_call(tc.function.name, args=args)
        except (AttributeError, IndexError):
            pass

        # Emit LLM response
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
        model = kwargs.get("model", "unknown")
        session.emit_llm_response(
            model=model,
            output="",
            duration_ms=duration_ms,
            metadata={"error": error},
        )


# Pricing table (per 1M tokens)
_PRICING: dict[str, tuple[float, float]] = {
    "gpt-5.4": (2.50, 15.00),
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.4-nano": (0.20, 1.25),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.20, 0.80),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "o3": (2.00, 8.00),
    "o4-mini": (0.55, 2.20),
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
