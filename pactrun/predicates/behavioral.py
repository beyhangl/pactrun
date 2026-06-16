"""Behavioral predicates — detect loops, drift, and repetition."""

from __future__ import annotations

from pactrun.core.enums import EventKind
from pactrun.core.models import Event, PredicateResult, SessionState
from pactrun.predicates.base import predicate


@predicate("no_loops")
def no_loops(window: int = 5, threshold: float = 0.8):
    """Detect repetitive tool call patterns (probable infinite loops).

    Checks if the last `window` tool calls have >threshold fraction
    of identical calls.
    """
    def check(event: Event, state: SessionState) -> PredicateResult:
        history = state.tool_call_history
        if len(history) < window:
            return PredicateResult(passed=True)
        recent = history[-window:]
        if not recent:
            return PredicateResult(passed=True)
        most_common_count = max(recent.count(t) for t in set(recent))
        ratio = most_common_count / len(recent)
        return PredicateResult(
            passed=ratio < threshold,
            expected=f"loop ratio < {threshold:.0%}",
            actual=f"{ratio:.0%} repetition in last {window} calls",
            message=f"Possible loop: {ratio:.0%} of last {window} tool calls are identical",
        )
    check.predicate_name = "no_loops"  # type: ignore[attr-defined]
    return check


@predicate("max_retries")
def max_retries(n: int, tool: str | None = None):
    """Max N consecutive calls to the same tool (or a specific tool)."""
    def check(event: Event, state: SessionState) -> PredicateResult:
        history = state.tool_call_history
        if len(history) < 2:
            return PredicateResult(passed=True)

        # Count consecutive identical calls at the end
        target = tool or (history[-1] if history else None)
        if target is None:
            return PredicateResult(passed=True)

        consecutive = 0
        for t in reversed(history):
            if t == target:
                consecutive += 1
            else:
                break

        return PredicateResult(
            passed=consecutive <= n,
            expected=f"<= {n} consecutive '{target}' calls",
            actual=f"{consecutive} consecutive calls",
            message=f"Tool '{target}' called {consecutive} times consecutively (max {n})",
        )
    check.predicate_name = "max_retries"  # type: ignore[attr-defined]
    return check


@predicate("drift_bounds")
def drift_bounds(cost_pct: float | None = None, tokens_pct: float | None = None):
    """Per-turn metrics must stay within N% of session average.

    Detects gradual drift by comparing the latest turn's metrics
    against the running average.
    """
    def check(event: Event, state: SessionState) -> PredicateResult:
        # Need at least 3 turns to detect drift
        if len(state.cost_per_turn) < 3:
            return PredicateResult(passed=True)

        violations: list[str] = []

        if cost_pct is not None and state.cost_per_turn:
            avg = sum(state.cost_per_turn) / len(state.cost_per_turn)
            if avg > 0:
                latest = state.cost_per_turn[-1]
                deviation = (latest - avg) / avg
                if deviation > cost_pct:
                    violations.append(f"cost drift {deviation:+.0%} (limit {cost_pct:+.0%})")

        if tokens_pct is not None and state.tokens_per_turn:
            avg = sum(state.tokens_per_turn) / len(state.tokens_per_turn)
            if avg > 0:
                latest = state.tokens_per_turn[-1]
                deviation = (latest - avg) / avg
                if deviation > tokens_pct:
                    violations.append(f"token drift {deviation:+.0%} (limit {tokens_pct:+.0%})")

        if violations:
            return PredicateResult(
                passed=False,
                message="Drift detected: " + "; ".join(violations),
            )
        return PredicateResult(passed=True)
    check.predicate_name = "drift_bounds"  # type: ignore[attr-defined]
    return check


@predicate("no_repeated_output")
def no_repeated_output(window: int = 3):
    """Agent must not produce identical outputs across recent turns."""
    def check(event: Event, state: SessionState) -> PredicateResult:
        history = state.output_history
        if len(history) < 2:
            return PredicateResult(passed=True)
        recent = history[-window:]
        if len(recent) != len(set(recent)):
            return PredicateResult(
                passed=False,
                message=f"Repeated output detected in last {window} turns",
            )
        return PredicateResult(passed=True)
    check.predicate_name = "no_repeated_output"  # type: ignore[attr-defined]
    return check


def _is_tool_error(e: Event) -> bool:
    """Classify a tool-call event as failed across the shapes adapters populate."""
    if e.error is not None:
        return True
    meta = e.metadata or {}
    if meta.get("is_error") or meta.get("isError"):  # MCP-style result flag
        return True
    if isinstance(e.tool_result, BaseException):
        return True
    return False


@predicate("tool_error_rate_under")
def tool_error_rate_under(max_rate: float = 0.3, window: int = 10, min_calls: int = 3):
    """Rolling tool-failure fraction must stay under a ceiling.

    Looks at the last ``window`` tool calls; if at least ``min_calls`` have
    occurred, fails when the failed fraction exceeds ``max_rate`` (a degraded-
    grounding signal: the agent keeps calling tools that error). Below
    ``min_calls`` it passes (warm-up). A call is "failed" when ``Event.error``
    is set, its ``metadata`` carries an ``is_error`` / ``isError`` flag (MCP
    shape), or its ``tool_result`` is an exception.

    Note: fires only where the adapter populates an error signal (the manual and
    MCP adapters today). It degrades safely — an unflagged call counts as a
    success — so it never false-positives on adapters that don't yet map tool
    errors into ``Event.error``.
    """
    def check(event: Event, state: SessionState) -> PredicateResult:
        tool_events = [e for e in state.events if e.kind == EventKind.TOOL_CALL]
        recent = tool_events[-window:]
        if len(recent) < min_calls:
            return PredicateResult(passed=True)
        errors = sum(1 for e in recent if _is_tool_error(e))
        rate = errors / len(recent)
        return PredicateResult(
            passed=rate <= max_rate,
            expected=f"tool error rate <= {max_rate:.0%}",
            actual=f"{rate:.0%} ({errors}/{len(recent)})",
            message=f"Tool error rate {rate:.0%} ({errors}/{len(recent)}) exceeds {max_rate:.0%}",
        )
    check.predicate_name = "tool_error_rate_under"  # type: ignore[attr-defined]
    return check
