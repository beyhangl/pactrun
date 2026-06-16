"""Tests for per_key_rate_limit and tool_quota_per_period."""

from datetime import datetime, timezone

import pytest

from pactrun import (
    Contract,
    EventKind,
    per_key_rate_limit,
    tool_quota_per_period,
    tool_rate_limit,
)
from pactrun.core.models import Event


def _tool(name, t, args=None, error=None):
    return Event(kind=EventKind.TOOL_CALL, tool_name=name, tool_args=args or {}, timestamp=t, error=error)


def _ts(y, mo, d, h=12):
    return datetime(y, mo, d, h, 0, 0, tzinfo=timezone.utc).timestamp()


# ---------------------------------------------------------------------------
# per_key_rate_limit
# ---------------------------------------------------------------------------

def test_per_key_isolation():
    c = Contract("t").require(per_key_rate_limit("sms", "to", 1, 86400), on_fail="log")
    with c.session() as s:
        s.record_event(_tool("sms", 1000.0, {"to": "A"}))   # A #1 ok
        s.record_event(_tool("sms", 1001.0, {"to": "B"}))   # B #1 ok
        s.record_event(_tool("sms", 1002.0, {"to": "A"}))   # A #2 within 24h -> trips
    assert len(s.violations) == 1


def test_per_key_independent_windows():
    c = Contract("t").require(per_key_rate_limit("sms", "to", 1, 100), on_fail="log")
    with c.session() as s:
        s.record_event(_tool("sms", 1000.0, {"to": "A"}))
        s.record_event(_tool("sms", 1200.0, {"to": "A"}))  # 200s later, window 100s -> fresh
    assert s.is_compliant


def test_per_key_vs_global_gap():
    """5 distinct recipients pass under per-key max_calls=1 where a global
    tool_rate_limit(...,2,...) would trip."""
    calls = [(1000.0 + i, {"to": f"r{i}"}) for i in range(5)]
    c_key = Contract("t").require(per_key_rate_limit("sms", "to", 1, 3600), on_fail="log")
    with c_key.session() as s_key:
        for t, a in calls:
            s_key.record_event(_tool("sms", t, a))
    assert s_key.is_compliant
    c_glob = Contract("t").require(tool_rate_limit("sms", 2, 3600), on_fail="log")
    with c_glob.session() as s_glob:
        for t, a in calls:
            s_glob.record_event(_tool("sms", t, a))
    assert not s_glob.is_compliant


def test_per_key_nested_path():
    c = Contract("t").require(per_key_rate_limit("sms", "recipient.phone", 1, 86400), on_fail="log")
    with c.session() as s:
        s.record_event(_tool("sms", 1000.0, {"recipient": {"phone": "+15551234"}}))
        s.record_event(_tool("sms", 1001.0, {"recipient": {"phone": "+15551234"}}))
    assert len(s.violations) == 1


def test_per_key_on_missing_ignore():
    c = Contract("t").require(per_key_rate_limit("sms", "to", 1, 86400, on_missing="ignore"), on_fail="log")
    with c.session() as s:
        s.record_event(_tool("sms", 1000.0, {"other": 1}))
        s.record_event(_tool("sms", 1001.0, {"other": 2}))
    assert s.is_compliant


def test_per_key_on_missing_block():
    c = Contract("t").require(per_key_rate_limit("sms", "to", 1, 86400, on_missing="block"), on_fail="log")
    with c.session() as s:
        s.record_event(_tool("sms", 1000.0, {"other": 1}))
    assert not s.is_compliant


def test_per_key_bad_on_missing():
    with pytest.raises(ValueError):
        per_key_rate_limit("sms", "to", 1, 86400, on_missing="explode")


# ---------------------------------------------------------------------------
# tool_quota_per_period
# ---------------------------------------------------------------------------

def test_quota_under_limit_passes():
    c = Contract("t").require(tool_quota_per_period("apply", 3, "month"), on_fail="log")
    with c.session() as s:
        for d in (2, 10, 20):
            s.record_event(_tool("apply", _ts(2026, 6, d)))
    assert s.is_compliant


def test_quota_trips_over_limit():
    c = Contract("t").require(tool_quota_per_period("apply", 2, "month"), on_fail="log")
    with c.session() as s:
        for d in (2, 10, 20):
            s.record_event(_tool("apply", _ts(2026, 6, d)))
    assert not s.is_compliant  # 3rd call this month over limit 2


def test_quota_resets_next_month():
    c = Contract("t").require(tool_quota_per_period("apply", 2, "month"), on_fail="log")
    with c.session() as s:
        s.record_event(_tool("apply", _ts(2026, 6, 20)))  # June
        s.record_event(_tool("apply", _ts(2026, 6, 25)))  # June (2/2)
        s.record_event(_tool("apply", _ts(2026, 7, 1)))   # July — fresh period, ok
    assert s.is_compliant


def test_quota_resets_next_day():
    c = Contract("t").require(tool_quota_per_period("apply", 1, "day"), on_fail="log")
    with c.session() as s:
        s.record_event(_tool("apply", _ts(2026, 6, 16, 9)))   # day 1
        s.record_event(_tool("apply", _ts(2026, 6, 17, 9)))   # day 2 — fresh
    assert s.is_compliant


def test_quota_resets_next_week():
    c = Contract("t").require(tool_quota_per_period("apply", 1, "week"), on_fail="log")
    with c.session() as s:
        # 2026-06-15 is a Monday; 2026-06-22 the next Monday.
        s.record_event(_tool("apply", _ts(2026, 6, 16)))  # week of Jun 15
        s.record_event(_tool("apply", _ts(2026, 6, 23)))  # week of Jun 22 — fresh
    assert s.is_compliant


def test_quota_week_start_boundary():
    # Sunday 2026-06-21 vs Monday 2026-06-22.
    # week_start="mon": these are in DIFFERENT weeks -> both pass under limit 1.
    c_mon = Contract("t").require(tool_quota_per_period("apply", 1, "week", week_start="mon"), on_fail="log")
    with c_mon.session() as s:
        s.record_event(_tool("apply", _ts(2026, 6, 21)))  # Sun
        s.record_event(_tool("apply", _ts(2026, 6, 22)))  # Mon (new week)
    assert s.is_compliant
    # week_start="sun": Sunday starts the week, so Sun+Mon are the SAME week -> trips.
    c_sun = Contract("t").require(tool_quota_per_period("apply", 1, "week", week_start="sun"), on_fail="log")
    with c_sun.session() as s2:
        s2.record_event(_tool("apply", _ts(2026, 6, 21)))  # Sun (week start)
        s2.record_event(_tool("apply", _ts(2026, 6, 22)))  # Mon (same week)
    assert not s2.is_compliant


def test_quota_excludes_failed_by_default():
    c = Contract("t").require(tool_quota_per_period("apply", 1, "month"), on_fail="log")
    with c.session() as s:
        s.record_event(_tool("apply", _ts(2026, 6, 2), error="boom"))  # failed — not counted
        s.record_event(_tool("apply", _ts(2026, 6, 3)))                # 1st success — ok
    assert s.is_compliant


def test_quota_counts_failed_when_requested():
    c = Contract("t").require(tool_quota_per_period("apply", 1, "month", count_failed=True), on_fail="log")
    with c.session() as s:
        s.record_event(_tool("apply", _ts(2026, 6, 2), error="boom"))
        s.record_event(_tool("apply", _ts(2026, 6, 3)))
    assert not s.is_compliant


def test_quota_only_named_tool():
    c = Contract("t").require(tool_quota_per_period("apply", 1, "month"), on_fail="log")
    with c.session() as s:
        s.record_event(_tool("search", _ts(2026, 6, 2)))
        s.record_event(_tool("search", _ts(2026, 6, 3)))
        s.record_event(_tool("apply", _ts(2026, 6, 4)))  # only this counts
    assert s.is_compliant


def test_quota_bad_period():
    with pytest.raises(ValueError):
        tool_quota_per_period("apply", 1, "fortnight")


def test_quota_bad_week_start():
    with pytest.raises(ValueError):
        tool_quota_per_period("apply", 1, "week", week_start="moonday")


def test_quota_named_tz_resolves_or_skips():
    # UTC default needs nothing; a named zone needs tzdata — skip if absent.
    try:
        pred = tool_quota_per_period("apply", 1, "day", tz="America/New_York")
    except Exception:
        pytest.skip("tzdata not available for named timezone")
    # 2026-03-08 02:30 UTC is 2026-03-07 21:30 in New_York (prev day).
    c = Contract("t").require(pred, on_fail="log")
    with c.session() as s:
        s.record_event(_tool("apply", datetime(2026, 3, 8, 2, 30, tzinfo=timezone.utc).timestamp()))
        s.record_event(_tool("apply", datetime(2026, 3, 8, 12, 0, tzinfo=timezone.utc).timestamp()))
    # First is NY Mar 7, second is NY Mar 8 — different days, both pass limit 1.
    assert s.is_compliant


def test_registered():
    import pactrun
    names = pactrun.list_predicates()
    assert "per_key_rate_limit" in names
    assert "tool_quota_per_period" in names
