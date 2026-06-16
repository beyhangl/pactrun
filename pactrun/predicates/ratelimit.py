"""Rate-limit predicates — windowed (event-time) spend and call-rate caps.

Unlike ``cost_under`` / ``max_tool_calls`` (cumulative whole-session counts),
these enforce a rolling time window over the recorded events, so a self-pacing
agent that steadily burns budget — which never trips a cumulative cap until the
total is reached — is caught. The window is **event-time** (``Event.timestamp``
epoch seconds), so it also works in replay / eval, not only live.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from pactrun.core.enums import EventKind
from pactrun.core.models import Event, PredicateResult, SessionState
from pactrun.predicates.base import predicate
from pactrun.predicates.tools import _resolve_path


def _window_events(state: SessionState, event: Event, window_s: float, kind: EventKind):
    cutoff = event.timestamp - window_s
    return [
        e for e in state.events
        if e.kind == kind and cutoff <= e.timestamp <= event.timestamp
    ]


def _scalar_key(value) -> str:
    """Stable string form of an extracted bucket key (scalars verbatim, else JSON)."""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(value, sort_keys=True, default=str)


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


@predicate("per_key_rate_limit")
def per_key_rate_limit(
    tool: str,
    key_path: str,
    max_calls: int,
    per_seconds: float,
    on_missing: str = "ignore",
):
    """Rolling-window call cap **bucketed by an extracted argument value**.

    Like :func:`tool_rate_limit`, but maintains an INDEPENDENT window per
    distinct value found at ``key_path`` (a dotted path:
    ``"recipient.phone"``, ``"items.0.id"``) — e.g. at most one message per
    recipient phone per 24h, regardless of how many distinct recipients are
    contacted. The current call counts toward its own bucket (so
    ``max_calls=1`` trips the second call to the *same* key).

    ``on_missing``: ``"ignore"`` (skip the check when the key is absent) or
    ``"block"`` (treat an absent key as a violation).
    """
    if on_missing not in ("ignore", "block"):
        raise ValueError(f"per_key_rate_limit: on_missing must be 'ignore' or 'block', got {on_missing!r}")

    def check(event: Event, state: SessionState) -> PredicateResult:
        if event.kind != EventKind.TOOL_CALL or event.tool_name != tool:
            return PredicateResult(passed=True)
        found, key = _resolve_path(event.tool_args or {}, key_path)
        if not found:
            if on_missing == "block":
                return PredicateResult(
                    passed=False,
                    expected=f"'{key_path}' present on '{tool}' call",
                    actual="key absent",
                    message=f"Tool '{tool}' call missing rate-limit key '{key_path}'",
                )
            return PredicateResult(passed=True)
        bucket = _scalar_key(key)
        cutoff = event.timestamp - per_seconds
        n = 0
        for e in state.events:
            if e.kind != EventKind.TOOL_CALL or e.tool_name != tool:
                continue
            if not (cutoff <= e.timestamp <= event.timestamp):
                continue
            ef, ek = _resolve_path(e.tool_args or {}, key_path)
            if ef and _scalar_key(ek) == bucket:
                n += 1
        return PredicateResult(
            passed=n <= max_calls,
            expected=f"<= {max_calls} '{tool}' calls per {per_seconds:.0f}s for {key_path}={bucket}",
            actual=f"{n} calls for {key_path}={bucket} in the last {per_seconds:.0f}s",
            message=f"Tool '{tool}' rate {n}/{per_seconds:.0f}s for {key_path}={bucket} exceeds {max_calls}",
        )
    check.predicate_name = "per_key_rate_limit"  # type: ignore[attr-defined]
    return check


_WEEKDAY = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}


def _resolve_tz(tz: str):
    if tz is None or tz.upper() == "UTC":
        return timezone.utc
    from zoneinfo import ZoneInfo  # raises ZoneInfoNotFoundError if tzdata is absent

    return ZoneInfo(tz)


@predicate("tool_quota_per_period")
def tool_quota_per_period(
    tool: str,
    limit: int,
    period: str = "month",
    *,
    week_start: str = "mon",
    tz: str = "UTC",
    count_failed: bool = False,
):
    """Calendar-anchored hard cap on a tool, **resetting at the period boundary**.

    Counts invocations of ``tool`` since the start of the current calendar
    period (``"day"`` | ``"week"`` | ``"month"``) in timezone ``tz``. Unlike the
    rolling-window :func:`tool_rate_limit`, this RESETS on the boundary — e.g.
    "160 applies per calendar month, fresh on the 1st", or "N sends per ISO
    week". Failed calls (``Event.error`` set) are excluded unless
    ``count_failed=True``. ``week_start`` is ``"mon"``..``"sun"``.

    ``tz`` defaults to UTC (no external data needed); a named zone like
    ``"America/New_York"`` requires the system tz database or the ``tzdata``
    package and is resolved at construction time.
    """
    if period not in ("day", "week", "month"):
        raise ValueError(f"tool_quota_per_period: period must be day/week/month, got {period!r}")
    week_idx = _WEEKDAY.get(week_start.lower())
    if week_idx is None:
        raise ValueError(f"tool_quota_per_period: bad week_start {week_start!r}")
    tzinfo = _resolve_tz(tz)

    def _period_start_epoch(ts: float) -> float:
        dt = datetime.fromtimestamp(ts, tzinfo)
        midnight = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        if period == "day":
            start = midnight
        elif period == "week":
            start = midnight - timedelta(days=(dt.weekday() - week_idx) % 7)
        else:  # month
            start = midnight.replace(day=1)
        return start.timestamp()

    def check(event: Event, state: SessionState) -> PredicateResult:
        if event.kind != EventKind.TOOL_CALL or event.tool_name != tool:
            return PredicateResult(passed=True)
        start = _period_start_epoch(event.timestamp)
        n = sum(
            1 for e in state.events
            if e.kind == EventKind.TOOL_CALL and e.tool_name == tool
            and start <= e.timestamp <= event.timestamp
            and (count_failed or not e.error)
        )
        return PredicateResult(
            passed=n <= limit,
            expected=f"<= {limit} '{tool}' calls per {period}",
            actual=f"{n} '{tool}' calls this {period}",
            message=f"Tool '{tool}' used {n} times this {period}, over the {limit} quota",
        )
    check.predicate_name = "tool_quota_per_period"  # type: ignore[attr-defined]
    return check
