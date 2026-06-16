"""Tests for flow_progression (diagnostic + gate modes)."""

import pytest

from pactrun import Contract, EventKind, flow_progression
from pactrun.core.models import Event


def _run(pred, calls, on_fail="log"):
    """calls: list of ('tool', name) or ('out', text)."""
    c = Contract("t").require(pred, on_fail=on_fail)
    with c.session() as s:
        for kind, val in calls:
            if kind == "tool":
                s.emit_tool_call(val)
            else:
                s.emit_llm_response(model="m", output=val)
    return s


STAGES = ["search", "draft", "review", "send"]


# ---------------------------------------------------------------------------
# diagnostic mode
# ---------------------------------------------------------------------------

def test_diagnostic_happy_path():
    s = _run(flow_progression(STAGES),
             [("tool", "search"), ("tool", "draft"), ("tool", "review"), ("tool", "send")])
    assert s.is_compliant


def test_diagnostic_dropoff_reports_stage():
    s = _run(flow_progression(STAGES),
             [("tool", "search"), ("tool", "draft")])
    assert not s.is_compliant
    v = s.violations[0]
    assert "reached_stage=draft" in v.actual
    assert "(2/4)" in v.actual


def test_diagnostic_zero_progress():
    s = _run(flow_progression(STAGES), [("tool", "unrelated")])
    assert not s.is_compliant
    assert "reached_stage=none (0/4)" in s.violations[0].actual


def test_diagnostic_requires_order():
    # send before the earlier stages -> only 'send' can't count out of order
    s = _run(flow_progression(STAGES),
             [("tool", "send"), ("tool", "search"), ("tool", "draft"), ("tool", "review")])
    # forward scan: search(1) draft(2) review(3) -> never reaches send terminal in order
    assert not s.is_compliant


def test_diagnostic_output_marker_stage():
    s = _run(flow_progression(["start", "DONE"]),
             [("tool", "start"), ("out", "the task is DONE now")])
    assert s.is_compliant


def test_diagnostic_callable_stage():
    is_big = lambda e: e.kind == EventKind.TOOL_CALL and e.tool_name == "big"
    s = _run(flow_progression(["small", is_big]),
             [("tool", "small"), ("tool", "big")])
    assert s.is_compliant


def test_diagnostic_custom_terminal():
    # terminal='review' means we only need to reach review, not send
    s = _run(flow_progression(STAGES, terminal="review"),
             [("tool", "search"), ("tool", "draft"), ("tool", "review")])
    assert s.is_compliant


def test_diagnostic_terminal_not_in_stages():
    with pytest.raises(ValueError):
        flow_progression(STAGES, terminal="ship")


def test_empty_stages_rejected():
    with pytest.raises(ValueError):
        flow_progression([])


def test_bad_mode_rejected():
    with pytest.raises(ValueError):
        flow_progression(STAGES, mode="walk")


# ---------------------------------------------------------------------------
# gate mode
# ---------------------------------------------------------------------------

def test_gate_in_order_passes():
    s = _run(flow_progression(STAGES, mode="gate"),
             [("tool", "search"), ("tool", "draft"), ("tool", "review"), ("tool", "send")])
    assert s.is_compliant


def test_gate_blocks_premature_stage():
    # 'send' fires at phase 0 before search/draft/review -> blocked
    s = _run(flow_progression(STAGES, mode="gate"),
             [("tool", "search"), ("tool", "send")])
    assert not s.is_compliant
    assert "Out-of-order" in s.violations[0].message


def test_gate_enter_condition_blocks():
    # 'send' may only be entered if a prior 'approved' flag is on state.metadata
    enter = {"send": lambda e, st: st.metadata.get("approved", False)}
    pred = flow_progression(["draft", "send"], mode="gate", enter=enter)
    c = Contract("t").require(pred, on_fail="log")
    with c.session() as s:
        s.emit_tool_call("draft")
        s.emit_tool_call("send")  # approved not set -> blocked
    assert not s.is_compliant


def test_gate_enter_condition_allows_when_met():
    enter = {"send": lambda e, st: st.metadata.get("approved", False)}
    pred = flow_progression(["draft", "send"], mode="gate", enter=enter)
    c = Contract("t").require(pred, on_fail="log")
    with c.session() as s:
        s.emit_tool_call("draft")
        s.state.metadata["approved"] = True
        s.emit_tool_call("send")
    assert s.is_compliant


def test_gate_repeat_allowed_by_default():
    s = _run(flow_progression(["a", "b"], mode="gate"),
             [("tool", "a"), ("tool", "a"), ("tool", "b")])
    assert s.is_compliant


def test_gate_repeat_blocked_when_disallowed():
    s = _run(flow_progression(["a", "b"], mode="gate", allow_repeats=False),
             [("tool", "a"), ("tool", "b"), ("tool", "a")])  # 'a' repeats after advancing
    assert not s.is_compliant


def test_gate_unrelated_events_passthrough():
    s = _run(flow_progression(["a", "b"], mode="gate"),
             [("tool", "noise"), ("tool", "a"), ("tool", "noise"), ("tool", "b")])
    assert s.is_compliant


def test_registered():
    import pactrun
    assert "flow_progression" in pactrun.list_predicates()


def test_check_on_resolution():
    diag = flow_progression(STAGES, mode="diagnostic")
    gate = flow_progression(STAGES, mode="gate")
    assert diag._check_on == "session_end"
    assert gate._check_on == "every_event"
