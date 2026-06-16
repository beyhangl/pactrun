"""Tests for tool_error_rate_under."""

from pactrun import Contract, EventKind, tool_error_rate_under
from pactrun.core.models import Event, SessionState


def _tool(ok, t, *, meta=None, result=None):
    return Event(
        kind=EventKind.TOOL_CALL,
        tool_name="search",
        timestamp=t,
        error=None if ok else "boom",
        metadata=meta or {},
        tool_result=result,
    )


def _run(pred, events):
    c = Contract("t").require(pred, on_fail="log")
    with c.session() as s:
        for e in events:
            s.record_event(e)
    return s


def test_below_min_calls_passes_even_all_errors():
    s = _run(tool_error_rate_under(0.3, window=10, min_calls=3),
             [_tool(False, 1.0), _tool(False, 2.0)])  # only 2 calls
    assert s.is_compliant


def test_boundary_equal_rate_passes():
    # Evaluation only happens once min_calls is reached: with window=min_calls=4
    # and the single error LAST, the only checked point is 1/4 = 25% == max_rate.
    events = [_tool(True, 1.0), _tool(True, 2.0), _tool(True, 3.0), _tool(False, 4.0)]
    s = _run(tool_error_rate_under(0.25, window=4, min_calls=4), events)
    assert s.is_compliant  # 25% == max_rate, <= passes


def test_above_rate_fails():
    events = [_tool(False, 1.0), _tool(False, 2.0), _tool(True, 3.0)]  # 2/3 = 67%
    s = _run(tool_error_rate_under(0.3, window=10, min_calls=3), events)
    assert not s.is_compliant


def test_window_only_sees_recent_calls():
    # Per-event eval would trip on the early burst, so test the predicate's
    # verdict at a late point directly: the last 5 calls are all successes.
    pred = tool_error_rate_under(0.2, window=5, min_calls=3)
    state = SessionState()
    state.events = [_tool(False, float(i)) for i in range(3)] + [_tool(True, float(i)) for i in range(3, 10)]
    result = pred(state.events[-1], state)
    assert result.passed  # last 5 are all successes -> 0%


def test_early_burst_trips_when_it_happens():
    # The burst itself is a real failure signal and should be caught live.
    events = [_tool(False, 1.0), _tool(False, 2.0), _tool(False, 3.0)]
    s = _run(tool_error_rate_under(0.2, window=5, min_calls=3), events)
    assert not s.is_compliant


def test_metadata_is_error_flag_counts():
    events = [_tool(True, 1.0, meta={"is_error": True}),
              _tool(True, 2.0, meta={"is_error": True}),
              _tool(True, 3.0)]
    s = _run(tool_error_rate_under(0.3, window=10, min_calls=3), events)
    assert not s.is_compliant  # 2/3 flagged errors


def test_metadata_isError_camel_counts():
    events = [_tool(True, 1.0, meta={"isError": True}),
              _tool(True, 2.0, meta={"isError": True}),
              _tool(True, 3.0)]
    s = _run(tool_error_rate_under(0.3, window=10, min_calls=3), events)
    assert not s.is_compliant


def test_exception_result_counts():
    events = [_tool(True, 1.0, result=RuntimeError("x")),
              _tool(True, 2.0, result=ValueError("y")),
              _tool(True, 3.0)]
    s = _run(tool_error_rate_under(0.3, window=10, min_calls=3), events)
    assert not s.is_compliant


def test_clean_events_not_counted():
    events = [_tool(True, float(i)) for i in range(5)]
    s = _run(tool_error_rate_under(0.3, window=10, min_calls=3), events)
    assert s.is_compliant


def test_llm_calls_not_counted():
    c = Contract("t").require(tool_error_rate_under(0.3, window=10, min_calls=3), on_fail="log")
    with c.session() as s:
        s.emit_llm_response(model="m", output="a")  # not a tool call
        s.emit_llm_response(model="m", output="b")
        s.record_event(_tool(False, 5.0))  # 1 tool call, error, but < min_calls
    assert s.is_compliant


def test_registered():
    import pactrun
    assert "tool_error_rate_under" in pactrun.list_predicates()
