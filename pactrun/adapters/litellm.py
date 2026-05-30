"""LiteLLM adapter — auto-emits events to the active pactrun Session.

LiteLLM is a universal gateway that many agent frameworks route their LLM calls
through — notably **CrewAI**, but also any code calling ``litellm.completion``.
This adapter patches ``litellm.completion`` / ``litellm.acompletion`` so every
call is recorded into the active Session, giving you cost/token/tool enforcement
across all of them at once.

Usage (plain LiteLLM)::

    import litellm
    from pactrun import Contract, cost_under
    from pactrun.adapters import LiteLLMAdapter

    with Contract("agent").require(cost_under(0.50)).session():
        with LiteLLMAdapter():
            litellm.completion(model="gpt-4.1", messages=[{"role": "user", "content": "Hi"}])

Usage (CrewAI, which calls LiteLLM under the hood)::

    with Contract("crew").require(cost_under(2.00)).session():
        with LiteLLMAdapter():
            crew.kickoff()
"""

from __future__ import annotations

import json
import time
from typing import Any

from pactrun.adapters._base import get_session


class LiteLLMAdapter:
    """Patches LiteLLM to auto-emit events to the active pactrun Session."""

    def __init__(self) -> None:
        self._litellm: Any = None
        self._original_sync: Any = None
        self._original_async: Any = None
        self._patched = False

    def __enter__(self) -> "LiteLLMAdapter":
        self._patch()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._unpatch()

    async def __aenter__(self) -> "LiteLLMAdapter":
        self._patch()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        self._unpatch()

    def _patch(self) -> None:
        if self._patched:
            return
        try:
            import litellm
        except ImportError as exc:
            raise ImportError(
                "The 'litellm' package is required for LiteLLMAdapter. "
                "Install it with: pip install 'pactrun[litellm]'"
            ) from exc

        self._litellm = litellm
        self._original_sync = litellm.completion
        self._original_async = litellm.acompletion

        adapter = self
        original_sync = self._original_sync
        original_async = self._original_async

        def patched_sync(*args: Any, **kwargs: Any) -> Any:
            start = time.monotonic()
            try:
                response = original_sync(*args, **kwargs)
            except Exception as exc:
                adapter._emit_error(kwargs, (time.monotonic() - start) * 1000, str(exc))
                raise
            adapter._emit_response(kwargs, response, (time.monotonic() - start) * 1000)
            return response

        async def patched_async(*args: Any, **kwargs: Any) -> Any:
            start = time.monotonic()
            try:
                response = await original_async(*args, **kwargs)
            except Exception as exc:
                adapter._emit_error(kwargs, (time.monotonic() - start) * 1000, str(exc))
                raise
            adapter._emit_response(kwargs, response, (time.monotonic() - start) * 1000)
            return response

        litellm.completion = patched_sync
        litellm.acompletion = patched_async
        self._patched = True

    def _unpatch(self) -> None:
        if not self._patched:
            return
        if self._litellm is not None:
            self._litellm.completion = self._original_sync
            self._litellm.acompletion = self._original_async
        self._patched = False

    def _emit_response(self, kwargs: dict, response: Any, duration_ms: float) -> None:
        session = get_session()
        if session is None:
            return

        model = getattr(response, "model", None) or kwargs.get("model", "unknown")

        prompt_tokens = 0
        completion_tokens = 0
        usage = getattr(response, "usage", None)
        if usage is not None:
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0

        output = ""
        try:
            message = response.choices[0].message
            output = getattr(message, "content", None) or ""
            for call in getattr(message, "tool_calls", None) or []:
                fn = getattr(call, "function", None)
                name = getattr(fn, "name", None)
                if name:
                    session.emit_tool_call(name, args=_parse_args(getattr(fn, "arguments", None)))
        except (AttributeError, IndexError, TypeError):
            pass

        session.emit_llm_response(
            model=model,
            output=output,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=_extract_cost(response),
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


def _parse_args(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw:
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            return {}
    return {}


def _extract_cost(response: Any) -> float:
    # LiteLLM stamps the computed cost on the response after a real call.
    hidden = getattr(response, "_hidden_params", None) or {}
    if isinstance(hidden, dict) and hidden.get("response_cost"):
        try:
            return float(hidden["response_cost"])
        except (TypeError, ValueError):
            pass
    # Otherwise ask LiteLLM to price it from its bundled cost map.
    try:
        import litellm

        return float(litellm.completion_cost(completion_response=response) or 0.0)
    except Exception:
        return 0.0
