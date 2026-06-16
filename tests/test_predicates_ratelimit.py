"""Tests for windowed rate-limit predicates.

These enforce a rolling *time window* over recorded events, so a self-pacing
agent that steadily burns budget — never tripping a cumulative cap until the
total is hit — is still caught. Windows are event-time (``Event.timestamp``),
so the tests synthesize events with hand-set timestamps.
"""

from types import SimpleNamespace as NS

from pactrun import (
    Contract,
    EventKind,
    call_rate_under,
    cost_under,
    spend_rate_under,
    tool_rate_limit,
    wrap,
)
from pactrun.core.models import Event


def _llm(cost, t):
    return Event(kind=EventKind.LLM_CALL, cost_usd=cost, timestamp=t)


def _tool(name, t):
    return Event(kind=EventKind.TOOL_CALL, tool_name=name, timestamp=t)


# ---------------------------------------------------------------------------
# spend_rate_under
# ---------------------------------------------------------------------------

def test_spend_rate_trips_on_burst():
    """A burst of spend inside the window trips even when the cumulative
    total would clear a generous whole-session cap."""
    c = Contract("t").require(spend_rate_under(5.0, 60), on_fail="log")
    with c.session() as s:
        # four $2 calls within 40s == $8 in the 60s window
        for i, t in enumerate([1000.0, 1010.0, 1020.0, 1030.0]):
            s.record_event(_llm(2.0, t))
    assert not s.is_compliant
    assert any("Spend rate" in (v.message or "") for v in s.violations)


def test_spend_rate_paces_ok():
    """The same total spend spread beyond the window never trips."""
    c = Contract("t").require(spend_rate_under(5.0, 60), on_fail="log")
    with c.session() as s:
        # $2 every 90s — at most one prior call ever sits inside a 60s window
        for t in [1000.0, 1090.0, 1180.0, 1270.0]:
            s.record_event(_llm(2.0, t))
    assert s.is_compliant


def test_spend_rate_vs_cumulative_cap():
    """A pace that trips the windowed rate would *pass* a cumulative cost cap —
    this is the gap the windowed predicate closes."""
    pace = [(2.0, 1000.0), (2.0, 1005.0), (2.0, 1010.0)]  # $6 in 10s
    # Cumulative $10 cap: passes (total is only $6).
    c_cum = Contract("t").require(cost_under(10.0), on_fail="log")
    with c_cum.session() as s_cum:
        for cost, t in pace:
            s_cum.record_event(_llm(cost, t))
    assert s_cum.is_compliant
    # Windowed $5/60s cap: trips.
    c_win = Contract("t").require(spend_rate_under(5.0, 60), on_fail="log")
    with c_win.session() as s_win:
        for cost, t in pace:
            s_win.record_event(_llm(cost, t))
    assert not s_win.is_compliant


def test_spend_rate_window_slides():
    """Old spend that has aged out of the window no longer counts."""
    c = Contract("t").require(spend_rate_under(5.0, 60), on_fail="log")
    with c.session() as s:
        s.record_event(_llm(4.0, 1000.0))   # ages out by t=1100
        s.record_event(_llm(4.0, 1100.0))   # window now only holds this one ($4)
    assert s.is_compliant


# ---------------------------------------------------------------------------
# call_rate_under
# ---------------------------------------------------------------------------

def test_call_rate_trips():
    c = Contract("t").require(call_rate_under(3, 60), on_fail="log")
    with c.session() as s:
        for t in [1000.0, 1005.0, 1010.0, 1015.0]:  # 4 calls in 15s
            s.record_event(_llm(0.0, t))
    assert not s.is_compliant


def test_call_rate_ok_when_spread():
    c = Contract("t").require(call_rate_under(3, 60), on_fail="log")
    with c.session() as s:
        for t in [1000.0, 1100.0, 1200.0, 1300.0]:  # 1 per window
            s.record_event(_llm(0.0, t))
    assert s.is_compliant


# ---------------------------------------------------------------------------
# tool_rate_limit
# ---------------------------------------------------------------------------

def test_tool_rate_limit_trips_for_named_tool():
    c = Contract("t").require(tool_rate_limit("search", 2, 30), on_fail="log")
    with c.session() as s:
        for t in [1000.0, 1005.0, 1010.0]:  # 3 'search' calls in 10s
            s.record_event(_tool("search", t))
    assert not s.is_compliant


def test_tool_rate_limit_ignores_other_tools():
    c = Contract("t").require(tool_rate_limit("search", 2, 30), on_fail="log")
    with c.session() as s:
        s.record_event(_tool("search", 1000.0))
        s.record_event(_tool("write", 1001.0))
        s.record_event(_tool("write", 1002.0))   # other tool — doesn't count
        s.record_event(_tool("search", 1003.0))  # 2 'search' total — at cap, ok
    assert s.is_compliant


# ---------------------------------------------------------------------------
# wrap() wiring
# ---------------------------------------------------------------------------

def test_wrap_accepts_rate_kwargs():
    client = NS(chat=NS(completions=NS(create=lambda **kw: None)))
    guarded = wrap(
        client,
        max_cost_per_min=1.0,
        tool_rate_limits={"search": (5, 60)},
        on_violation="log",
    )
    names = {c.predicate_name for c in guarded._contract.clauses}
    assert "spend_rate_under" in names
    assert "tool_rate_limit" in names
