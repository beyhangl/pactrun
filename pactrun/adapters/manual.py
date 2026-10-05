"""Manual instrumentation — for frameworks without a dedicated adapter.

Usage::

    from pactrun.adapters.manual import emit_llm_call, emit_tool_call

    with contract.session():
        emit_llm_call(model="gpt-5.4-nano", output="Hello", cost=0.001)
        emit_tool_call("search", args={"q": "test"}, result={"found": True})
"""

from __future__ import annotations

from typing import Any

from pactrun.adapters._base import get_session
from pactrun.core.models import Violation


def emit_llm_call(
    model: str,
    output: str,
    *,
    input: Any = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cost: float = 0.0,
    duration_ms: float = 0.0,
    metadata: dict | None = None,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    reasoning_tokens: int = 0,
) -> list[Violation]:
    """Emit an LLM call event to the active session.

    ``prompt_tokens`` is the TOTAL input including cached tokens, and
    ``completion_tokens`` the total output including reasoning; the
    ``cache_*_tokens`` and ``reasoning_tokens`` counts are subsets of those
    (see ``pactrun.core.models.Event``). Anthropic users: add
    ``cache_read_input_tokens`` and ``cache_creation_input_tokens`` to
    ``input_tokens`` to get ``prompt_tokens``.

    Returns list of violations triggered (empty if compliant).
    """
    session = get_session()
    if session is None:
        return []
    return session.emit_llm_response(
        model=model,
        output=output,
        input=input,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost=cost,
        duration_ms=duration_ms,
        metadata=metadata,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        reasoning_tokens=reasoning_tokens,
    )


def emit_tool_call(
    tool_name: str,
    *,
    args: dict | None = None,
    result: Any = None,
    duration_ms: float = 0.0,
    error: str | None = None,
    metadata: dict | None = None,
) -> list[Violation]:
    """Emit a tool call event to the active session.

    Returns list of violations triggered (empty if compliant).
    """
    session = get_session()
    if session is None:
        return []
    return session.emit_tool_call(
        tool_name,
        args=args,
        result=result,
        duration_ms=duration_ms,
        error=error,
        metadata=metadata,
    )
