"""Tests for trace replay, TraceRecorder, and YAML policy tests."""

import json

import pytest
from click.testing import CliRunner

from pactrun import Contract, EventKind, cost_under, must_not_call, session_timeout
from pactrun.cli.main import cli
from pactrun.core.errors import ContractLoadError
from pactrun.core.models import Event
from pactrun.observability import TraceRecorder
from pactrun.replay import (
    TraceLoadError,
    load_trace,
    replay_trace,
    run_contract_tests,
)


def _write_trace(path, events):
    path.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# load_trace
# ---------------------------------------------------------------------------

class TestLoadTrace:
    def test_reads_events_and_skips_blank_lines(self, tmp_path):
        p = tmp_path / "t.jsonl"
        p.write_text(
            json.dumps({"kind": "llm_call", "cost_usd": 0.1}) + "\n\n"
            + json.dumps({"kind": "tool_call", "tool_name": "search"}) + "\n",
            encoding="utf-8",
        )
        events = load_trace(p)
        assert [e.kind for e in events] == [EventKind.LLM_CALL, EventKind.TOOL_CALL]
        assert events[1].tool_name == "search"

    def test_bad_json_reports_the_line(self, tmp_path):
        p = tmp_path / "t.jsonl"
        p.write_text('{"kind": "llm_call"}\n{not json\n', encoding="utf-8")
        with pytest.raises(TraceLoadError, match=r"t\.jsonl:2"):
            load_trace(p)

    def test_unknown_event_kind_is_rejected(self, tmp_path):
        p = _write_trace(tmp_path / "t.jsonl", [{"kind": "teleport"}])
        with pytest.raises(TraceLoadError):
            load_trace(p)

    def test_non_mapping_line_is_rejected(self, tmp_path):
        p = tmp_path / "t.jsonl"
        p.write_text("[1, 2]\n", encoding="utf-8")
        with pytest.raises(TraceLoadError, match="mapping"):
            load_trace(p)


# ---------------------------------------------------------------------------
# replay_trace / Contract.replay
# ---------------------------------------------------------------------------

class TestReplay:
    def test_flags_violations_without_raising_even_on_block_clauses(self):
        c = Contract("t").require(cost_under(0.5)).forbid(must_not_call("delete"))  # default block
        events = [Event(kind=EventKind.LLM_CALL, cost_usd=0.9),
                  Event(kind=EventKind.TOOL_CALL, tool_name="delete")]
        result = c.replay(events)  # must not raise
        assert not result.compliant
        assert result.violated == ["cost_under", "must_not_call"]
        assert all(v.enforced is False for v in result.violations)

    def test_compliant_trace(self):
        result = Contract("t").require(cost_under(1.0)).replay(
            [Event(kind=EventKind.LLM_CALL, cost_usd=0.1)])
        assert result.compliant and result.violated == []

    def test_does_not_mutate_the_callers_events(self):
        original = Event(kind=EventKind.LLM_CALL, cost_usd=-5.0)  # would be sanitized
        Contract("t").require(cost_under(1.0)).replay([original])
        assert original.cost_usd == -5.0
        assert "pactrun.invalid_amounts" not in original.metadata

    def test_event_index_points_at_the_triggering_event(self):
        events = [Event(kind=EventKind.TOOL_CALL, tool_name="ok"),
                  Event(kind=EventKind.TOOL_CALL, tool_name="delete")]
        result = Contract("t").forbid(must_not_call("delete")).replay(events)
        assert result.event_index(result.violations[0]) == 2

    def test_session_end_clause_has_no_event_index(self):
        from pactrun import must_call

        result = Contract("t").require(must_call("finish")).replay(
            [Event(kind=EventKind.TOOL_CALL, tool_name="start")])
        assert result.violated == ["must_call"]
        assert result.event_index(result.violations[0]) is None

    def test_runs_on_the_event_clock(self):
        # A run that took 10 minutes must breach a 60s timeout even though the
        # replay itself takes milliseconds.
        events = [Event(kind=EventKind.LLM_CALL, timestamp=1000.0),
                  Event(kind=EventKind.LLM_CALL, timestamp=1600.0)]
        result = Contract("t").require(session_timeout(60_000)).replay(events)
        assert result.violated == ["session_timeout"]

    def test_short_run_passes_timeout_on_event_clock(self):
        events = [Event(kind=EventKind.LLM_CALL, timestamp=1000.0),
                  Event(kind=EventKind.LLM_CALL, timestamp=1001.0)]
        assert Contract("t").require(session_timeout(60_000)).replay(events).compliant

    def test_replay_trace_function_matches_method(self):
        c = Contract("t").require(cost_under(0.5))
        events = [Event(kind=EventKind.LLM_CALL, cost_usd=0.9)]
        assert replay_trace(c, events).violated == c.replay(events).violated


# ---------------------------------------------------------------------------
# TraceRecorder round-trip
# ---------------------------------------------------------------------------

class TestTraceRecorder:
    def test_live_verdict_equals_replayed_verdict(self, tmp_path):
        p = tmp_path / "run.jsonl"
        contract = Contract("t").require(cost_under(0.5), on_fail="log").forbid(
            must_not_call("delete"), on_fail="log")
        with contract.session(observers=[TraceRecorder(p)]) as live:
            live.emit_llm_response(model="m", output="x", cost=0.9)
            live.emit_tool_call("delete", args={})
        replayed = contract.replay(load_trace(p))
        live_names = sorted({v.predicate_name for v in live.violations})
        assert replayed.violated == live_names == ["cost_under", "must_not_call"]

    def test_redacts_credential_arguments(self, tmp_path):
        p = tmp_path / "run.jsonl"
        with Contract("t").session(observers=[TraceRecorder(p)]) as s:
            s.emit_tool_call("api", args={"api_key": "sk-SECRET", "nested": {"password": "hunter2"}})
        text = p.read_text()
        assert "sk-SECRET" not in text and "hunter2" not in text

    def test_starts_fresh_unless_append(self, tmp_path):
        p = tmp_path / "run.jsonl"
        for _ in range(2):
            with Contract("t").session(observers=[TraceRecorder(p)]) as s:
                s.emit_tool_call("a")
        assert len(load_trace(p)) == 1
        with Contract("t").session(observers=[TraceRecorder(p, append=True)]) as s:
            s.emit_tool_call("a")
        assert len(load_trace(p)) == 2


# ---------------------------------------------------------------------------
# YAML policy tests
# ---------------------------------------------------------------------------

CONTRACT = """\
name: agent
clauses:
  - require: cost_under
    args: {max_usd: 0.5}
  - forbid: must_not_call
    args: {tool: delete}
"""


def _contract_file(tmp_path, tests_yaml):
    p = tmp_path / "c.yaml"
    p.write_text(CONTRACT + tests_yaml, encoding="utf-8")
    return p


class TestPolicyTests:
    def test_passing_suite(self, tmp_path):
        _write_trace(tmp_path / "ok.jsonl", [{"kind": "llm_call", "cost_usd": 0.1}])
        p = _contract_file(tmp_path, """\
tests:
  - name: benign
    trace: ok.jsonl
    expect: pass
  - name: runaway bill
    events:
      - {kind: llm_call, cost_usd: 0.9}
    expect: {violated: [cost_under]}
""")
        report = run_contract_tests(p)
        assert report.passed and len(report.results) == 2

    def test_missed_detection_fails_the_test(self, tmp_path):
        p = _contract_file(tmp_path, """\
tests:
  - name: should be caught but is not
    events:
      - {kind: llm_call, cost_usd: 0.1}
    expect: {violated: [cost_under]}
""")
        result = run_contract_tests(p).results[0]
        assert not result.passed
        assert "did not fire: cost_under" in result.detail

    def test_over_blocking_fails_the_test(self, tmp_path):
        # Exact-set matching: an unexpected extra violation is a failure too.
        p = _contract_file(tmp_path, """\
tests:
  - name: expected only the budget
    events:
      - {kind: llm_call, cost_usd: 0.9}
      - {kind: tool_call, tool_name: delete}
    expect: {violated: [cost_under]}
""")
        result = run_contract_tests(p).results[0]
        assert not result.passed
        assert "fired unexpectedly: must_not_call" in result.detail

    def test_no_tests_block_is_an_error(self, tmp_path):
        p = tmp_path / "c.yaml"
        p.write_text(CONTRACT, encoding="utf-8")
        with pytest.raises(ContractLoadError, match="no 'tests:' block"):
            run_contract_tests(p)

    @pytest.mark.parametrize("expect", ["'fail'", "{violated: []}", "{blocked: [x]}"])
    def test_bad_expect_is_an_error(self, tmp_path, expect):
        p = _contract_file(tmp_path, f"tests:\n  - events: [{{kind: llm_call}}]\n    expect: {expect}\n")
        with pytest.raises(ContractLoadError):
            run_contract_tests(p)

    def test_test_without_events_is_an_error(self, tmp_path):
        p = _contract_file(tmp_path, "tests:\n  - name: empty\n    expect: pass\n")
        with pytest.raises(ContractLoadError, match="needs a 'trace:'"):
            run_contract_tests(p)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestCli:
    def _setup(self, tmp_path):
        c = tmp_path / "c.yaml"
        c.write_text(CONTRACT, encoding="utf-8")
        ok = _write_trace(tmp_path / "ok.jsonl", [{"kind": "llm_call", "cost_usd": 0.1}])
        bad = _write_trace(tmp_path / "bad.jsonl", [{"kind": "tool_call", "tool_name": "delete"}])
        return c, ok, bad

    def test_replay_exit_codes(self, tmp_path):
        c, ok, bad = self._setup(tmp_path)
        runner = CliRunner()
        assert runner.invoke(cli, ["replay", str(c), str(ok)]).exit_code == 0
        r = runner.invoke(cli, ["replay", str(c), str(bad)])
        assert r.exit_code == 1 and "must_not_call" in r.output

    def test_replay_json_output(self, tmp_path):
        c, _, bad = self._setup(tmp_path)
        r = CliRunner().invoke(cli, ["replay", str(c), str(bad), "--json"])
        data = json.loads(r.output)
        assert data["compliant"] is False
        assert data["violated"] == ["must_not_call"]
        assert data["violations"][0]["event_index"] == 1

    def test_replay_bad_trace_exits_2(self, tmp_path):
        c, _, _ = self._setup(tmp_path)
        broken = tmp_path / "broken.jsonl"
        broken.write_text("{nope\n", encoding="utf-8")
        assert CliRunner().invoke(cli, ["replay", str(c), str(broken)]).exit_code == 2

    def test_test_command_exit_codes(self, tmp_path):
        good = _contract_file(tmp_path, "tests:\n  - events: [{kind: llm_call, cost_usd: 0.9}]\n"
                                        "    expect: {violated: [cost_under]}\n")
        assert CliRunner().invoke(cli, ["test", str(good)]).exit_code == 0

        failing = tmp_path / "f.yaml"
        failing.write_text(CONTRACT + "tests:\n  - events: [{kind: llm_call, cost_usd: 0.9}]\n"
                                      "    expect: pass\n", encoding="utf-8")
        assert CliRunner().invoke(cli, ["test", str(failing)]).exit_code == 1

        no_tests = tmp_path / "n.yaml"
        no_tests.write_text(CONTRACT, encoding="utf-8")
        assert CliRunner().invoke(cli, ["test", str(no_tests)]).exit_code == 2


# ---------------------------------------------------------------------------
# Loader fixes shipped alongside
# ---------------------------------------------------------------------------

class TestLoader:
    def test_invalid_limit_is_a_contract_error_not_a_crash(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("name: x\nclauses:\n  - require: cost_under\n    args: {max_usd: -1}\n",
                     encoding="utf-8")
        with pytest.raises(ContractLoadError, match="cost_under"):
            Contract.from_yaml(p)
        r = CliRunner().invoke(cli, ["validate", str(p)])
        assert r.exit_code == 1
        assert "Traceback" not in r.output

    def test_unknown_argument_is_a_contract_error(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("name: x\nclauses:\n  - require: cost_under\n    args: {budget: 1}\n",
                     encoding="utf-8")
        with pytest.raises(ContractLoadError):
            Contract.from_yaml(p)

    def test_yaml_monitor_mode(self, tmp_path):
        p = tmp_path / "m.yaml"
        p.write_text("name: x\nmode: monitor\nclauses:\n  - require: cost_under\n    args: {max_usd: 0.1}\n",
                     encoding="utf-8")
        c = Contract.from_yaml(p)
        assert c.mode == "monitor"
        with c.session() as s:  # default on_fail is block - must not raise in monitor mode
            s.emit_llm_response(model="m", output="x", cost=5.0)
        assert s.violations[0].enforced is False

    def test_bad_yaml_mode_rejected(self, tmp_path):
        p = tmp_path / "m.yaml"
        p.write_text("name: x\nmode: yolo\nclauses: []\n", encoding="utf-8")
        with pytest.raises(ContractLoadError):
            Contract.from_yaml(p)


def test_violation_carries_predicate_name_and_round_trips():
    from pactrun.core.models import Violation

    with Contract("t").require(cost_under(0.1), on_fail="log").session() as s:
        s.emit_llm_response(model="m", output="x", cost=1.0)
    v = s.violations[0]
    assert v.predicate_name == "cost_under"
    assert Violation.from_dict(v.to_dict()).predicate_name == "cost_under"
