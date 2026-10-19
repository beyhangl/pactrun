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
from pactrun.core import pricing
from pactrun.core.usage import TokenUsage, count, read_field, total


class GeminiAdapter:
    """Patches the google-genai SDK to auto-emit events to the active pactrun Session."""

    def __init__(self) -> None:
        self._Models: Any = None
        self._AsyncModels: Any = None
        self._original_sync: Any = None
        self._original_async: Any = None
        self._patched = False

    def __enter__(self) -> GeminiAdapter:
        self._patch()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._unpatch()

    async def __aenter__(self) -> GeminiAdapter:
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

        usage = usage_from_gemini(getattr(response, "usage_metadata", None))

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


def usage_from_gemini(usage: Any) -> TokenUsage:
    """Normalise a google-genai ``GenerateContentResponseUsageMetadata``.

    Per the google-genai SDK field docs, ``total_token_count`` is the sum of
    ``prompt_token_count``, ``candidates_token_count``,
    ``tool_use_prompt_token_count`` and ``thoughts_token_count``:

    - input = ``prompt_token_count`` (which already INCLUDES
      ``cached_content_token_count``) + ``tool_use_prompt_token_count``;
    - output = ``candidates_token_count`` + ``thoughts_token_count`` (thinking
      is billed as output and is NOT inside ``candidates_token_count``).

    Gemini has no per-request cache-write count (explicit caches are created
    separately and billed per hour of storage), so ``cache_write_tokens`` is 0.
    """
    if usage is None:
        return TokenUsage()
    thoughts = count(read_field(usage, "thoughts_token_count"))
    return TokenUsage(
        prompt_tokens=total(
            read_field(usage, "prompt_token_count"), read_field(usage, "tool_use_prompt_token_count")
        ),
        completion_tokens=total(read_field(usage, "candidates_token_count"), thoughts),
        cache_read_tokens=count(read_field(usage, "cached_content_token_count")),
        reasoning_tokens=thoughts,
    )
