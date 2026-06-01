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
) -> list[Violation]:
    """Emit an LLM call event to the active session.

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
