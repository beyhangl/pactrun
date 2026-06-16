"""Rate-limit predicates — windowed (event-time) spend and call-rate caps.

Unlike ``cost_under`` / ``max_tool_calls`` (cumulative whole-session counts),
these enforce a rolling time window over the recorded events, so a self-pacing
agent that steadily burns budget — which never trips a cumulative cap until the
total is reached — is caught. The window is **event-time** (``Event.timestamp``
epoch seconds), so it also works in replay / eval, not only live.
"""

from __future__ import annotations

from pactrun.core.enums import EventKind
from pactrun.core.models import Event, PredicateResult, SessionState
from pactrun.predicates.base import predicate


def _window_events(state: SessionState, event: Event, window_s: float, kind: EventKind):
    cutoff = event.timestamp - window_s
    return [
        e for e in state.events
        if e.kind == kind and cutoff <= e.timestamp <= event.timestamp
    ]


@predicate("spend_rate_under")
def spend_rate_under(max_usd: float, window_s: float):
    """LLM spend within a rolling time window must stay under a cap."""
    def check(event: Event, state: SessionState) -> PredicateResult:
        spent = sum(e.cost_usd for e in _window_events(state, event, window_s, EventKind.LLM_CALL))
        return PredicateResult(
            passed=spent <= max_usd,
            expected=f"<= ${max_usd:.4f} per {window_s:.0f}s",
            actual=f"${spent:.4f} in the last {window_s:.0f}s",
            message=f"Spend rate ${spent:.4f}/{window_s:.0f}s exceeds ${max_usd:.4f}",
        )
    check.predicate_name = "spend_rate_under"  # type: ignore[attr-defined]
    return check


@predicate("call_rate_under")
def call_rate_under(max_calls: int, window_s: float):
    """LLM-call count within a rolling time window must stay under a cap."""
    def check(event: Event, state: SessionState) -> PredicateResult:
        n = len(_window_events(state, event, window_s, EventKind.LLM_CALL))
        return PredicateResult(
            passed=n <= max_calls,
            expected=f"<= {max_calls} calls per {window_s:.0f}s",
            actual=f"{n} calls in the last {window_s:.0f}s",
            message=f"Call rate {n}/{window_s:.0f}s exceeds {max_calls}",
        )
    check.predicate_name = "call_rate_under"  # type: ignore[attr-defined]
    return check


@predicate("tool_rate_limit")
def tool_rate_limit(tool: str, max_calls: int, per_seconds: float):
    """One tool's invocation rate within a rolling window must stay under a cap."""
    def check(event: Event, state: SessionState) -> PredicateResult:
        cutoff = event.timestamp - per_seconds
        n = sum(
            1 for e in state.events
            if e.kind == EventKind.TOOL_CALL and e.tool_name == tool
            and cutoff <= e.timestamp <= event.timestamp
        )
        return PredicateResult(
            passed=n <= max_calls,
            expected=f"<= {max_calls} '{tool}' calls per {per_seconds:.0f}s",
            actual=f"{n} '{tool}' calls in the last {per_seconds:.0f}s",
            message=f"Tool '{tool}' rate {n}/{per_seconds:.0f}s exceeds {max_calls}",
        )
    check.predicate_name = "tool_rate_limit"  # type: ignore[attr-defined]
    return check
