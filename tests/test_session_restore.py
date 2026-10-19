"""Session restore from a recorded trace, and pausing the wall clock."""

import asyncio
import json
import time

import pytest

from pactrun import Contract, Event, EventKind, ViolationError
from pactrun.approval_store import SQLiteApprovalStore
from pactrun.core.amounts import INVALID_AMOUNT_EVENTS_KEY, INVALID_AMOUNTS_KEY
from pactrun.observability.trace import TraceRecorder
from pactrun.predicates.consent import (
    consent_token_required,
    mint_approval_token,
    mint_consent_token,
    multi_party_approval_required,
)
from pactrun.predicates.cost import cost_under
from pactrun.predicates.flow import flow_progression
from pactrun.predicates.ratelimit import tool_quota_per_period, tool_rate_limit
from pactrun.predicates.timing import max_turns, session_timeout
from pactrun.predicates.tools import max_tool_calls, must_not_call
from pactrun.replay import TraceLoadError, load_trace
from pactrun.session import get_active_session

SECRET = "restore-test-secret"


class Spy:
    def __init__(self):
        self.events, self.violations, self.ends = [], [], 0

    def on_event(self, event, state):
        self.events.append(event.id)

    def on_violation(self, violation, event=None):
        self.violations.append(violation)

    def on_session_end(self, state):
        self.ends += 1


class Clock:
    def __init__(self, monkeypatch, start=1_000_000.0):
        self.now = start
        monkeypatch.setattr(time, "time", lambda: self.now)

    def tick(self, seconds):
        self.now += seconds


def _llm(cost, ts=None, **kw):
    return Event(kind=EventKind.LLM_CALL, model="m", output="o", cost_usd=cost,
                 timestamp=time.time() if ts is None else ts, **kw)


def _tool(name, ts=None, **kw):
    return Event(kind=EventKind.TOOL_CALL, tool_name=name,
                 timestamp=time.time() if ts is None else ts, **kw)


def _record(contract, path, emit, **kwargs):
    with contract.session(observers=[TraceRecorder(path)], **kwargs) as s:
        emit(s)
    return s


def _write_jsonl(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


# ---------------------------------------------------------------------------
# budgets and histories
# ---------------------------------------------------------------------------

def test_cost_budget_continues_across_restore(tmp_path):
    trace = tmp_path / "run.jsonl"
    contract = Contract("c").require(cost_under(1.0))

    def first(s):
        s.emit_llm_response("m", "a", cost=0.4)
        s.emit_llm_response("m", "b", cost=0.4)

    _record(contract, trace, first)
    with contract.session(restore_from=trace) as s:
        assert s.state.total_cost_usd == pytest.approx(0.8)
        assert s.state.total_llm_calls == 2
        s.emit_llm_response("m", "c", cost=0.15)
        assert s.is_compliant
        with pytest.raises(ViolationError):
            s.emit_llm_response("m", "d", cost=0.1)
    assert s.violations[0].predicate_name == "cost_under"
    assert s.violations[0].restored is False
    assert s.summary().total_cost_usd == pytest.approx(1.05)


def test_restore_from_event_list_leaves_callers_events_untouched():
    events = [_llm(0.5), _llm(float("nan"))]
    with Contract("c").require(cost_under(0.6)).session(restore_from=events) as s:
        assert s.state.total_cost_usd == pytest.approx(0.5)
    assert events[1].cost_usd != events[1].cost_usd
    assert INVALID_AMOUNTS_KEY not in events[1].metadata


def test_turn_limit_and_tool_history_survive(tmp_path):
    trace = tmp_path / "run.jsonl"
    contract = Contract("c").require(max_turns(3)).require(max_tool_calls(3))

    def first(s):
        for name in ("search", "read", "search"):
            s.emit_tool_call(name)
            s.advance_turn()

    _record(contract, trace, first)
    with contract.session(restore_from=trace) as s:
        assert s.state.turn_number == 3
        assert s.state.tool_call_history == ["search", "read", "search"]
        assert s.state.total_tool_calls == 3
        with pytest.raises(ViolationError) as exc:
            s.advance_turn()
    assert exc.value.violation.predicate_name == "max_turns"
    assert s.summary().tool_call_history == ["search", "read", "search"]


def test_restore_does_not_raise_on_block_violation_in_the_trace(tmp_path):
    trace = tmp_path / "run.jsonl"
    contract = Contract("c").forbid(must_not_call("delete"))

    def first(s):
        with pytest.raises(ViolationError):
            s.emit_tool_call("delete")

    first_run = _record(contract, trace, first)
    assert len(first_run.violations) == 1

    with contract.session(restore_from=trace) as s:
        assert s.state.tool_call_history == ["delete"]
        assert s.violations == []
        s.emit_tool_call("search")
        with pytest.raises(ViolationError):
            s.emit_tool_call("delete")
    assert len(s.violations) == 1


def test_restored_violations_are_marked_and_never_reenforced(tmp_path):
    trace = tmp_path / "run.jsonl"
    contract = Contract("c").forbid(must_not_call("delete"))

    def first(s):
        with pytest.raises(ViolationError):
            s.emit_tool_call("delete")

    earlier = _record(contract, trace, first).violations
    spy = Spy()
    with contract.session(
        restore_from=trace,
        restore_violations=[earlier[0].to_dict()],
        observers=[spy],
    ) as s:
        s.emit_tool_call("search")
    assert [v.id for v in s.violations] == [earlier[0].id]
    assert s.violations[0].restored is True
    assert earlier[0].restored is False
    assert not s.is_compliant
    assert spy.violations == []
    assert s.summary().violations[0].to_dict()["restored"] is True


# ---------------------------------------------------------------------------
# observers
# ---------------------------------------------------------------------------

def test_observers_not_notified_for_restored_events(tmp_path):
    trace = tmp_path / "run.jsonl"
    contract = Contract("c").require(cost_under(10))

    def first(s):
        s.emit_llm_response("m", "a", cost=0.1)
        s.emit_tool_call("search")

    _record(contract, trace, first)
    old = [e.id for e in load_trace(trace)]
    spy = Spy()
    recorder = TraceRecorder(trace, append=True)
    with contract.session(restore_from=trace, observers=[recorder, spy]) as s:
        s.emit_tool_call("read")
        s.emit_llm_response("m", "b", cost=0.1)

    new = [e.id for e in s.state.events[len(old):]]
    assert spy.events == new
    assert spy.ends == 1
    assert [e.id for e in load_trace(trace)] == old + new

    with contract.session(restore_from=trace) as again:
        assert again.state.total_cost_usd == pytest.approx(0.2)
        assert again.state.tool_call_history == ["search", "read"]


def test_truncating_recorder_on_the_restore_path_is_refused(tmp_path):
    trace = tmp_path / "run.jsonl"
    _write_jsonl(trace, [_llm(0.5).to_dict()])
    with pytest.raises(ValueError, match="append=True"):
        Contract("c").session(restore_from=trace, observers=[TraceRecorder(trace)])
    assert get_active_session() is None


# ---------------------------------------------------------------------------
# amounts and corrupt traces
# ---------------------------------------------------------------------------

def test_invalid_amounts_in_a_trace_are_sanitised(tmp_path):
    trace = tmp_path / "run.jsonl"
    good = _llm(0.3).to_dict()
    negative = _llm(0.0).to_dict() | {"cost_usd": -50.0}
    nan_tokens = _llm(0.1).to_dict() | {"prompt_tokens": float("nan")}
    text = _llm(0.0).to_dict() | {"cost_usd": "lots"}
    trace.write_text("".join(json.dumps(r) + "\n" for r in (good, negative, nan_tokens, text)))

    with Contract("c").require(cost_under(0.5)).session(restore_from=trace) as s:
        assert s.state.total_cost_usd == pytest.approx(0.4)
        assert s.state.total_tokens == 0
        assert s.state.metadata[INVALID_AMOUNT_EVENTS_KEY] == 3
        restored = s.state.events
        assert restored[1].metadata[INVALID_AMOUNTS_KEY] == {"cost_usd": "-50.0"}
        assert restored[2].metadata[INVALID_AMOUNTS_KEY] == {"prompt_tokens": "nan"}
        assert restored[3].cost_usd == 0
        s.emit_llm_response("m", "x", cost=0.05)
        assert s.is_compliant
        with pytest.raises(ViolationError):
            s.emit_llm_response("m", "y", cost=0.1)


def test_rejections_recorded_by_the_earlier_run_still_count(tmp_path):
    trace = tmp_path / "run.jsonl"
    contract = Contract("c").require(cost_under(1.0), on_fail="log")
    _record(contract, trace, lambda s: s.emit_llm_response("m", "a", cost=-3.0))

    [stored] = load_trace(trace)
    assert stored.cost_usd == 0 and INVALID_AMOUNTS_KEY in stored.metadata
    with contract.session(restore_from=trace) as s:
        assert s.state.metadata[INVALID_AMOUNT_EVENTS_KEY] == 1
        assert s.state.total_cost_usd == 0


@pytest.mark.parametrize("line", [
    "{not json",
    json.dumps({"kind": "teleport"}),
    json.dumps(["a", "list"]),
    json.dumps({"kind": "llm_call", "timestamp": "yesterday"}),
    json.dumps({"kind": "llm_call", "timestamp": 1.0, "metadata": ["x"]}),
])
def test_corrupt_trace_raises_and_does_not_open_the_session(tmp_path, line):
    trace = tmp_path / "run.jsonl"
    trace.write_text(json.dumps(_llm(0.1).to_dict()) + "\n" + line + "\n")
    spy = Spy()
    with pytest.raises(TraceLoadError):
        Contract("c").session(restore_from=trace, observers=[spy])
    assert get_active_session() is None
    assert spy.ends == 0


def test_corrupt_event_list_raises():
    with pytest.raises(TraceLoadError):
        Contract("c").session(restore_from=[_llm(0.1), "garbage"])


# ---------------------------------------------------------------------------
# rate limits, quotas and per-run predicate state
# ---------------------------------------------------------------------------

def test_rolling_rate_limit_counts_restored_calls():
    now = time.time()
    contract = Contract("c").require(tool_rate_limit("search", max_calls=2, per_seconds=60))
    recent = [_tool("search", now - 10), _tool("search", now - 5)]
    with contract.session(restore_from=recent) as s:
        with pytest.raises(ViolationError):
            s.emit_tool_call("search")

    aged = [_tool("search", now - 300), _tool("search", now - 200)]
    with contract.session(restore_from=aged) as s:
        s.emit_tool_call("search")
        s.emit_tool_call("search")
        assert s.is_compliant


def test_period_quota_counts_restored_calls(tmp_path):
    trace = tmp_path / "run.jsonl"
    contract = Contract("c").require(tool_quota_per_period("send", limit=3, period="month"))

    def first(s):
        s.emit_tool_call("send")
        s.emit_tool_call("send")

    _record(contract, trace, first)
    with contract.session(restore_from=trace) as s:
        s.emit_tool_call("send")
        with pytest.raises(ViolationError):
            s.emit_tool_call("send")


def test_flow_gate_phase_survives_restore(tmp_path):
    trace = tmp_path / "run.jsonl"
    contract = Contract("c").require(flow_progression(["verify", "quote", "refund"], mode="gate"))

    def first(s):
        s.emit_tool_call("verify")
        s.emit_tool_call("quote")

    _record(contract, trace, first)
    with contract.session(restore_from=trace) as s:
        s.emit_tool_call("refund")
    assert s.is_compliant

    with contract.session(restore_from=[_tool("verify")]) as s:
        with pytest.raises(ViolationError):
            s.emit_tool_call("refund")


# ---------------------------------------------------------------------------
# approvals
# ---------------------------------------------------------------------------

class CountingStore(SQLiteApprovalStore):
    def __init__(self, path):
        super().__init__(path)
        self.calls = 0

    def consume_many(self, nonces):
        self.calls += 1
        return super().consume_many(nonces)


def test_consumed_consent_token_cannot_be_replayed_after_restart(tmp_path):
    trace = tmp_path / "run.jsonl"
    db = tmp_path / "approvals.sqlite"
    pred = consent_token_required("refund", secret=SECRET, replay_store=SQLiteApprovalStore(db))
    token = mint_consent_token("refund", secret=SECRET)
    contract = Contract("c").require(pred)
    _record(contract, trace, lambda s: s.emit_tool_call("refund", metadata={"user_consent": token}))

    store = CountingStore(db)
    pred = consent_token_required("refund", secret=SECRET, replay_store=store)
    with Contract("c").require(pred).session(restore_from=trace) as s:
        assert store.calls == 0
        with pytest.raises(ViolationError):
            s.emit_tool_call("refund", metadata={"user_consent": token})
    assert store.calls == 1


def test_restore_never_consumes_an_unused_grant(tmp_path):
    db = tmp_path / "approvals.sqlite"
    store = CountingStore(db)
    token = mint_consent_token("refund", secret=SECRET)
    pred = consent_token_required("refund", secret=SECRET, replay_store=store)
    restored = [_tool("refund", metadata={"user_consent": token})]
    with Contract("c").require(pred).session(restore_from=restored) as s:
        assert store.calls == 0
        s.emit_tool_call("refund", metadata={"user_consent": token})
        assert s.is_compliant
        with pytest.raises(ViolationError):
            s.emit_tool_call("refund", metadata={"user_consent": token})


def test_monitor_restore_rebuilds_shadow_nonces_without_touching_the_store(tmp_path):
    store = CountingStore(tmp_path / "approvals.sqlite")
    token = mint_consent_token("refund", secret=SECRET)
    quorum = [mint_approval_token(a, tool="wire", secret=SECRET) for a in ("ann", "bob")]
    contract = (
        Contract("c")
        .require(consent_token_required("refund", secret=SECRET, replay_store=store))
        .require(multi_party_approval_required("wire", secret=SECRET, replay_store=store))
    )
    restored = [
        _tool("refund", metadata={"user_consent": token}),
        _tool("wire", metadata={"approvals": quorum}),
    ]
    with contract.session(mode="monitor", restore_from=restored) as s:
        assert s.state.approval_nonces == {token["nonce"], *(t["nonce"] for t in quorum)}
        s.emit_tool_call("refund", metadata={"user_consent": token})
        s.emit_tool_call("wire", metadata={"approvals": quorum})
    assert store.calls == 0
    assert sorted(v.predicate_name for v in s.violations) == [
        "consent_token_required", "multi_party_approval_required",
    ]
    assert all(v.enforced is False for v in s.violations)


# ---------------------------------------------------------------------------
# clocks
# ---------------------------------------------------------------------------

def test_pause_excludes_suspended_time_from_session_timeout(monkeypatch):
    clock = Clock(monkeypatch)
    contract = Contract("c").require(session_timeout(1000))
    with contract.session() as s:
        clock.tick(0.5)
        s.emit_tool_call("a")
        s.pause()
        assert s.is_paused
        clock.tick(3600)
        s.resume()
        clock.tick(0.2)
        s.emit_tool_call("b")
        assert s.state.elapsed_ms == pytest.approx(700)
        s.pause()
        clock.tick(7200)
    assert s.state.elapsed_ms == pytest.approx(700)
    assert s.is_compliant

    with contract.session() as s:
        clock.tick(3600)
        with pytest.raises(ViolationError):
            s.emit_tool_call("a")


def test_recording_an_event_resumes_the_clock(monkeypatch):
    clock = Clock(monkeypatch)
    with Contract("c").session() as s:
        s.pause()
        clock.tick(10)
        s.emit_tool_call("a")
        assert not s.is_paused
        clock.tick(2)
        s.emit_tool_call("b")
        assert s.state.elapsed_ms == pytest.approx(2000)


def test_pause_requires_an_active_session():
    s = Contract("c").session()
    with pytest.raises(RuntimeError):
        s.pause()


def test_restored_wall_elapsed_continues_from_the_trace(monkeypatch):
    clock = Clock(monkeypatch)
    hours_ago = clock.now - 7200
    restored = [_tool("a", hours_ago), _tool("b", hours_ago + 0.4)]
    contract = Contract("c").require(session_timeout(1000))
    with contract.session(restore_from=restored) as s:
        assert s.state.elapsed_ms == pytest.approx(400)
        clock.tick(0.3)
        s.emit_tool_call("c")
        assert s.state.elapsed_ms == pytest.approx(700)
        assert s.is_compliant

    with contract.session(restore_from=restored, restore_elapsed_ms=950) as s:
        clock.tick(0.1)
        with pytest.raises(ViolationError):
            s.emit_tool_call("c")


def test_restore_elapsed_ms_is_validated():
    with pytest.raises(ValueError):
        Contract("c").session(restore_elapsed_ms=-1)
    with pytest.raises(ValueError):
        Contract("c").session(clock="event", restore_elapsed_ms=10)


def test_event_clock_is_unaffected_by_pause_and_continues_after_restore():
    t0 = 1_700_000_000.0
    contract = Contract("c").require(session_timeout(1000))
    restored = [_tool("a", t0), _tool("b", t0 + 0.4)]
    with contract.session(clock="event", restore_from=restored) as s:
        assert s.state.elapsed_ms == pytest.approx(400)
        s.pause()
        s.record_event(_tool("c", t0 + 0.9))
        assert s.state.elapsed_ms == pytest.approx(900)
        assert s.is_compliant
        s.pause()
        with pytest.raises(ViolationError):
            s.record_event(_tool("d", t0 + 1.5))
    assert s.state.elapsed_ms == pytest.approx(1500)


# ---------------------------------------------------------------------------
# async
# ---------------------------------------------------------------------------

def test_async_session_restores(tmp_path):
    trace = tmp_path / "run.jsonl"
    contract = Contract("c").require(cost_under(1.0))
    _write_jsonl(trace, [_llm(0.9).to_dict()])

    async def run():
        async with contract.session(restore_from=trace) as s:
            assert get_active_session() is s
            assert s.state.total_cost_usd == pytest.approx(0.9)
            with pytest.raises(ViolationError):
                s.emit_llm_response("m", "x", cost=0.2)
        return s

    s = asyncio.run(run())
    assert not s.is_active
    assert len(s.violations) == 1
