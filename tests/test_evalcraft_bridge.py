"""Regression tests for the optional cassette bridge (no API calls)."""

from __future__ import annotations

import subprocess
import sys
from copy import deepcopy

import pytest

pytest.importorskip("evalcraft")
from evalcraft.core.models import Cassette, Span, SpanKind, TokenUsage

from pactrun import Contract, Event, EventKind, cost_under, max_tool_calls, max_turns, token_budget
from pactrun.integrations.evalcraft import EvalcraftRecorder, cassette_to_events, replay_cassette


def test_record_save_load_replay_preserves_cost_cache_identity_and_result(tmp_path):
    recorder = EvalcraftRecorder(name="support")
    contract = Contract("support").require(cost_under(1)).require(token_budget(120))
    with contract.session(observers=[recorder]) as session:
        session.emit_llm_response(
            "synthetic", "refund", cost=0.1, prompt_tokens=100, completion_tokens=10,
            cache_read_tokens=60, cache_write_tokens=10, reasoning_tokens=3,
        )
        session.emit_tool_call(
            "refund", args={"amount_minor": 2500},
            result={"status": "confirmed"}, metadata={"host_tenant": "synthetic"},
        )
    span = recorder.cassette.get_llm_calls()[0]
    assert span.token_usage.prompt_tokens == 30
    assert span.token_usage.total_tokens == 110
    path = recorder.cassette.save(tmp_path / "run.json")
    result = replay_cassette(contract, path)
    assert result.compliant
    assert result.summary.total_tokens == 110
    assert result.summary.total_cost_usd == 0.1
    assert result.events[0].cache_read_tokens == 60
    assert result.events[0].cache_write_tokens == 10
    assert result.events[0].reasoning_tokens == 3
    assert [e.id for e in result.events] == [e.id for e in session.state.events]
    tool = result.events[1]
    assert tool.tool_result == {"status": "confirmed"}
    assert tool.metadata == {"host_tenant": "synthetic"}


def test_stricter_policy_catches_recorded_tool_attempt_without_execution():
    recorder = EvalcraftRecorder()
    with Contract("observe").session(observers=[recorder]) as session:
        session.emit_tool_call("refund", args={"amount_minor": 2500})
    result = replay_cassette(Contract("strict").require(max_tool_calls(0)), recorder.cassette)
    assert not result.compliant
    assert result.events[0].tool_result is None
    assert all(not v.enforced for v in result.violations)
    assert result.event_index(result.violations[0]) == 1


def test_conversion_is_pure_and_never_creates_approval_or_trust_labels():
    cassette = Cassette(spans=[Span(
        kind=SpanKind.TOOL_CALL, tool_name="refund", tool_args={"amount_minor": 2500},
        metadata={"custom": {"values": [1]}},
    )])
    original = deepcopy(cassette.to_dict())
    events = cassette_to_events(cassette)
    assert "user_consent" not in events[0].metadata
    assert "trusted" not in events[0].metadata
    events[0].metadata["custom"]["values"].append(2)
    events[0].tool_args["amount_minor"] = 9999
    assert cassette.to_dict() == original


def test_request_response_and_linked_result_are_counted_once():
    cassette = Cassette(spans=[
        Span(kind=SpanKind.LLM_REQUEST),
        Span(kind=SpanKind.LLM_RESPONSE, cost_usd=0.0, token_usage=TokenUsage(prompt_tokens=5)),
        Span(id="tool-1", kind=SpanKind.TOOL_CALL, tool_name="refund"),
        Span(kind=SpanKind.TOOL_RESULT, parent_id="tool-1", output={"status": "confirmed"}),
    ])
    result = replay_cassette(Contract("counts"), cassette)
    assert result.summary.total_llm_calls == 1
    assert result.summary.total_tool_calls == 1
    assert result.events[1].tool_result == {"status": "confirmed"}


def test_unlinked_result_is_refused_instead_of_inventing_execution():
    cassette = Cassette(spans=[Span(kind=SpanKind.TOOL_RESULT, output="ok")])
    with pytest.raises(ValueError, match="identified call"):
        cassette_to_events(cassette)


def test_linked_result_keeps_untrusted_provenance_for_exfiltration_policy():
    from pactrun import untrusted_taint_to_sink

    text = "untrusted-secret-content"
    cassette = Cassette(spans=[
        Span(id="lookup-1", kind=SpanKind.TOOL_CALL, tool_name="lookup"),
        Span(kind=SpanKind.TOOL_RESULT, parent_id="lookup-1", output=text,
             metadata={"untrusted": True}),
        Span(kind=SpanKind.TOOL_CALL, tool_name="send_email", tool_args={"body": text}),
    ])
    contract = Contract("exfiltration").require(untrusted_taint_to_sink(["send_email"], min_overlap=8))
    result = replay_cassette(contract, cassette)
    assert result.events[0].metadata["untrusted"] is True
    assert not result.compliant


def test_conflicting_or_interleaved_result_provenance_is_refused():
    call = Span(id="lookup-1", kind=SpanKind.TOOL_CALL, tool_name="lookup",
                metadata={"untrusted": False})
    result = Span(kind=SpanKind.TOOL_RESULT, parent_id="lookup-1", output="data",
                  metadata={"untrusted": True})
    with pytest.raises(ValueError, match="Conflicting"):
        cassette_to_events(Cassette(spans=[call, result]))
    with pytest.raises(ValueError, match="Interleaved"):
        cassette_to_events(Cassette(spans=[call, Span(kind=SpanKind.AGENT_OUTPUT), result]))


def test_duplicate_ids_and_results_are_refused_instead_of_overwritten():
    call = Span(id="call-1", kind=SpanKind.TOOL_CALL, tool_name="lookup")
    first = Span(kind=SpanKind.TOOL_RESULT, parent_id=call.id, output="first")
    second = Span(kind=SpanKind.TOOL_RESULT, parent_id=call.id, output="second")
    with pytest.raises(ValueError, match="Duplicate tool result"):
        cassette_to_events(Cassette(spans=[call, first, second]))
    with pytest.raises(ValueError, match="Duplicate cassette span"):
        cassette_to_events(Cassette(spans=[call, call]))
    call.tool_result = "already observed"
    with pytest.raises(ValueError, match="Duplicate tool result"):
        cassette_to_events(Cassette(spans=[call, first]))


def test_unknown_cost_is_refused_unless_caller_explicitly_accepts_zero():
    cassette = Cassette(spans=[Span(kind=SpanKind.LLM_RESPONSE)])
    with pytest.raises(ValueError, match="no recorded cost"):
        replay_cassette(Contract("budget").require(cost_under(0.01)), cassette)
    events = cassette_to_events(cassette, missing_cost="zero")
    assert events[0].cost_usd == 0.0
    with pytest.raises(ValueError, match="missing_cost"):
        cassette_to_events(cassette, missing_cost="ignore")


def test_recorder_keeps_all_supported_event_kinds_and_does_not_mutate_metadata():
    recorder = EvalcraftRecorder()
    events = [
        Event(kind=kind, metadata={"nested": {"value": 1}})
        for kind in (EventKind.INPUT, EventKind.OUTPUT, EventKind.ERROR, EventKind.TURN_END)
    ]
    with Contract("events").session(observers=[recorder]) as session:
        for event in events:
            session.record_event(event)
    restored = cassette_to_events(recorder.cassette)
    assert [e.kind for e in restored] == [e.kind for e in events]
    assert all(e.metadata == {"nested": {"value": 1}} for e in events)
    assert all(e.metadata == {"nested": {"value": 1}} for e in restored)


def test_recorder_refuses_impossible_cache_subset():
    recorder = EvalcraftRecorder()
    with pytest.raises(ValueError, match="cache subsets"):
        recorder.on_event(
            Event(prompt_tokens=3, cache_read_tokens=4),
            Contract("invalid").session().state,
        )


def test_base_import_keeps_evalcraft_optional_and_use_has_install_guidance():
    script = '''
import importlib.abc
import sys
class NoEvalcraft(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "evalcraft" or fullname.startswith("evalcraft."):
            raise ImportError("Evalcraft deliberately unavailable")
sys.meta_path.insert(0, NoEvalcraft())
import pactrun
try:
    pactrun.EvalcraftRecorder()
except ImportError as exc:
    assert "pactrun[evalcraft]" in str(exc)
else:
    raise AssertionError("Optional integration should require Evalcraft when used")
'''
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr


def test_monitor_replay_does_not_modify_original_cassette():
    recorder = EvalcraftRecorder()
    with Contract("run").session(observers=[recorder]) as session:
        session.emit_llm_response("synthetic", "ok", prompt_tokens=5, cost=0.0)
    before = deepcopy(recorder.cassette.to_dict())
    replay_cassette(Contract("tight").require(token_budget(0)), recorder.cassette)
    assert recorder.cassette.to_dict() == before


@pytest.mark.parametrize("turns, compliant", [(1, True), (2, False)])
def test_live_and_replayed_turn_limits_agree(turns, compliant):
    recorder = EvalcraftRecorder()
    contract = Contract("turn-limit").require(max_turns(1)).monitor()
    with contract.session(observers=[recorder]) as session:
        for _ in range(turns):
            session.advance_turn()
    replayed = replay_cassette(contract, recorder.cassette)
    assert session.state.turn_number == turns
    assert replayed.summary.turn_count == turns
    assert session.is_compliant == compliant
    assert replayed.compliant == compliant


def _scenario_cassette():
    from evalcraft.scenarios import run_scenario

    from pactrun.scenarios import refund_environment, refund_scenarios

    return run_scenario(refund_scenarios()[0], refund_environment).cassette


def test_stateful_annotations_preserve_actual_policy_events_and_tool_counts():
    from evalcraft.scenarios import run_scenarios

    from pactrun.scenarios import refund_environment, refund_scenarios

    suite = run_scenarios(refund_scenarios(), refund_environment)
    assert suite.passed
    for case in suite.cases:
        before = deepcopy(case.cassette.to_dict())
        events = cassette_to_events(case.cassette)
        attempts = case.cassette.get_tool_calls()
        assert sum(event.kind == EventKind.TOOL_CALL for event in events) == len(attempts)
        assert {span.id for span in attempts}.issubset({event.id for event in events})
        assert all("evalcraft_scenario" not in event.metadata for event in events)
        replayed = replay_cassette(Contract("tool-count").require(max_tool_calls(0)), case.cassette)
        assert replayed.compliant == (not attempts)
        assert case.cassette.to_dict() == before


@pytest.mark.parametrize("mutation", [
    "missing_owner", "future_owner", "boolean_owner", "future_marker", "boolean_step",
    "unknown_event", "tool_evidence", "model_evidence", "unknown_metadata", "wrong_name",
    "missing_parent", "duplicate_outcome", "missing_outcome",
])
def test_invalid_scenario_annotations_cannot_silently_hide_evidence(mutation):
    cassette = _scenario_cassette()
    span = cassette.spans[0]
    if mutation == "missing_owner":
        cassette.metadata.pop("evalcraft_scenario")
    elif mutation == "future_owner":
        cassette.metadata["evalcraft_scenario"]["version"] = 2
    elif mutation == "boolean_owner":
        cassette.metadata["evalcraft_scenario"]["version"] = True
    elif mutation == "future_marker":
        span.metadata["evalcraft_scenario"]["version"] = 2
    elif mutation == "boolean_step":
        span.metadata["evalcraft_scenario"]["step"] = True
    elif mutation == "unknown_event":
        span.metadata["evalcraft_scenario"]["event"] = "execute_tool"
    elif mutation == "tool_evidence":
        span.tool_name, span.tool_args = "refund", {"amount_minor": 2500}
    elif mutation == "model_evidence":
        span.cost_usd = 0.5
    elif mutation == "unknown_metadata":
        span.metadata["pactrun_bridge"] = {"event_kind": "tool_call"}
    elif mutation == "wrong_name":
        span.name = "unrecognized"
    elif mutation == "missing_parent":
        cassette.spans[1].parent_id = "missing"
    elif mutation == "duplicate_outcome":
        duplicate = deepcopy(cassette.spans[1])
        duplicate.id = "duplicate-outcome"
        cassette.spans.insert(2, duplicate)
    elif mutation == "missing_outcome":
        cassette.spans.pop(1)
    with pytest.raises(ValueError, match="scenario"):
        cassette_to_events(cassette)


def test_scenario_markers_never_omit_real_tool_or_model_spans():
    cassette = _scenario_cassette()
    cassette.spans.append(Span(
        kind=SpanKind.TOOL_CALL, tool_name="external_action",
        metadata={"evalcraft_scenario": {"version": 1}},
    ))
    cassette.spans.append(Span(
        kind=SpanKind.LLM_RESPONSE, cost_usd=0.25,
        metadata={"evalcraft_scenario": {"version": 1}},
    ))
    events = cassette_to_events(cassette)
    assert events[-2].tool_name == "external_action"
    assert events[-1].cost_usd == 0.25
    with pytest.raises(ValueError, match="Unsupported cassette span kind"):
        cassette_to_events(Cassette(spans=[Span(kind=SpanKind.AGENT_STEP)]))
