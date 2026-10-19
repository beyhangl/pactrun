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
from pactrun.core import pricing
from pactrun.core.usage import TokenUsage, count, read_field


class OpenAIAdapter:
    """Patches OpenAI SDK to auto-emit events to active pactrun Session."""

    def __init__(self) -> None:
        self._Completions: Any = None
        self._AsyncCompletions: Any = None
        self._original_sync: Any = None
        self._original_async: Any = None
        self._patched = False

    def __enter__(self) -> OpenAIAdapter:
        self._patch()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._unpatch()

    async def __aenter__(self) -> OpenAIAdapter:
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

        usage = usage_from_openai(getattr(response, "usage", None))

        # Extract output text
        output = ""
        try:
            msg = response.choices[0].message
            output = msg.content or ""
        except (AttributeError, IndexError):
            pass

        quote = pricing.cost(model, usage)

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
            cost=quote.usd,
            duration_ms=duration_ms,
            metadata=quote.metadata or None,
            **usage.event_kwargs(),
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


def usage_from_openai(usage: Any) -> TokenUsage:
    """Normalise an OpenAI Chat Completions ``CompletionUsage`` to pactrun's convention.

    OpenAI already reports totals: ``prompt_tokens`` INCLUDES
    ``prompt_tokens_details.cached_tokens`` (cache reads) and
    ``prompt_tokens_details.cache_write_tokens`` (cache writes, reported for
    GPT-5.6 and later; absent on older SDKs), and ``completion_tokens``
    INCLUDES ``completion_tokens_details.reasoning_tokens``. Source:
    openai-python ``types/completion_usage.py`` and the OpenAI prompt-caching
    guide (``ordinaryInputTokens = inputTokens - cachedTokens - cacheWriteTokens``).
    """
    if usage is None:
        return TokenUsage()
    prompt_details = read_field(usage, "prompt_tokens_details")
    completion_details = read_field(usage, "completion_tokens_details")
    return TokenUsage(
        prompt_tokens=count(read_field(usage, "prompt_tokens")),
        completion_tokens=count(read_field(usage, "completion_tokens")),
        cache_read_tokens=count(read_field(prompt_details, "cached_tokens")),
        cache_write_tokens=count(read_field(prompt_details, "cache_write_tokens")),
        reasoning_tokens=count(read_field(completion_details, "reasoning_tokens")),
    )
