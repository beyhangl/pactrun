"""Tool predicates — control which tools agents can/must call."""

from __future__ import annotations

from pactrun.core.enums import EventKind
from pactrun.core.models import Event, PredicateResult, SessionState
from pactrun.predicates.base import predicate


@predicate("must_call")
def must_call(tool: str):
    """Agent must call this tool by session end."""
    def check(event: Event, state: SessionState) -> PredicateResult:
        return PredicateResult(
            passed=tool in state.tool_call_history,
            expected=f"'{tool}' in tool history",
            actual=str(state.tool_call_history),
            message=f"Tool '{tool}' was never called",
        )
    check.predicate_name = "must_call"  # type: ignore[attr-defined]
    check._check_on = "session_end"  # type: ignore[attr-defined]
    return check


@predicate("must_not_call")
def must_not_call(tool: str):
    """Agent must never call this tool."""
    def check(event: Event, state: SessionState) -> PredicateResult:
        if event.kind == EventKind.TOOL_CALL and event.tool_name == tool:
            return PredicateResult(
                passed=False,
                expected=f"'{tool}' never called",
                actual=f"'{tool}' was called",
                message=f"Forbidden tool '{tool}' was called",
            )
        return PredicateResult(passed=True)
    check.predicate_name = "must_not_call"  # type: ignore[attr-defined]
    return check


@predicate("tool_order")
def tool_order(expected: list[str], strict: bool = False):
    """Tools must be called in this order (checked at session end)."""
    def check(event: Event, state: SessionState) -> PredicateResult:
        history = state.tool_call_history
        if strict:
            passed = history == expected
        else:
            it = iter(history)
            passed = all(t in it for t in expected)
        return PredicateResult(
            passed=passed,
            expected=str(expected),
            actual=str(history),
            message=f"Tool order mismatch: expected {expected}, got {history}",
        )
    check.predicate_name = "tool_order"  # type: ignore[attr-defined]
    check._check_on = "session_end"  # type: ignore[attr-defined]
    return check


@predicate("tools_allowed")
def tools_allowed(whitelist: list[str]):
    """Only these tools may be called."""
    def check(event: Event, state: SessionState) -> PredicateResult:
        if event.kind == EventKind.TOOL_CALL and event.tool_name:
            if event.tool_name not in whitelist:
                return PredicateResult(
                    passed=False,
                    expected=f"tool in {whitelist}",
                    actual=event.tool_name,
                    message=f"Tool '{event.tool_name}' not in allowed list: {whitelist}",
                )
        return PredicateResult(passed=True)
    check.predicate_name = "tools_allowed"  # type: ignore[attr-defined]
    return check


@predicate("max_tool_calls")
def max_tool_calls(limit: int):
    """Total tool calls must not exceed limit."""
    def check(event: Event, state: SessionState) -> PredicateResult:
        return PredicateResult(
            passed=state.total_tool_calls <= limit,
            expected=f"<= {limit} tool calls",
            actual=f"{state.total_tool_calls} tool calls",
            message=f"Tool call count {state.total_tool_calls} exceeds limit {limit}",
        )
    check.predicate_name = "max_tool_calls"  # type: ignore[attr-defined]
    return check
