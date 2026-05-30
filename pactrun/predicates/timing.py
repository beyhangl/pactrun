"""Timing predicates — latency and timeout constraints."""

from __future__ import annotations

from pactrun.core.models import Event, PredicateResult, SessionState
from pactrun.predicates.base import predicate


@predicate("max_latency")
def max_latency(max_ms: float):
    """No single event should exceed this latency."""
    def check(event: Event, state: SessionState) -> PredicateResult:
        if event.duration_ms <= 0:
            return PredicateResult(passed=True)
        return PredicateResult(
            passed=event.duration_ms <= max_ms,
            expected=f"<= {max_ms:.0f}ms",
            actual=f"{event.duration_ms:.0f}ms",
            message=f"Latency {event.duration_ms:.0f}ms exceeds limit {max_ms:.0f}ms",
        )
    check.predicate_name = "max_latency"  # type: ignore[attr-defined]
    return check


@predicate("session_timeout")
def session_timeout(max_ms: float):
    """Total session must complete within this time."""
    def check(event: Event, state: SessionState) -> PredicateResult:
        return PredicateResult(
            passed=state.elapsed_ms <= max_ms,
            expected=f"<= {max_ms:.0f}ms",
            actual=f"{state.elapsed_ms:.0f}ms",
            message=f"Session elapsed {state.elapsed_ms:.0f}ms exceeds timeout {max_ms:.0f}ms",
        )
    check.predicate_name = "session_timeout"  # type: ignore[attr-defined]
    return check


@predicate("max_turns")
def max_turns(n: int):
    """Session must not exceed N turns."""
    def check(event: Event, state: SessionState) -> PredicateResult:
        return PredicateResult(
            passed=state.turn_number <= n,
            expected=f"<= {n} turns",
            actual=f"{state.turn_number} turns",
            message=f"Turn count {state.turn_number} exceeds limit {n}",
        )
    check.predicate_name = "max_turns"  # type: ignore[attr-defined]
    return check
