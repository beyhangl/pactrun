"""Cross-project scenario, artifact and current-code playback acceptance tests.

Every model response uses a real SDK over HTTP MockTransport; the payment
provider is SQLite-only. Existing recordings and golden fixtures are untouched.
"""

from __future__ import annotations

import copy
import json

import pytest

pytest.importorskip("evalcraft")

from evalcraft import Cassette, playback
from evalcraft.scenarios import (
    Scenario,
    ScenarioAction,
    ScenarioStep,
    run_scenario,
    run_scenarios,
)

from pactrun.scenarios import RefundScenarioEnvironment, refund_environment, refund_scenarios


def test_entire_refund_pack_roundtrips_new_artifacts_and_repeats_in_isolation(tmp_path):
    cases = refund_scenarios()
    original_initial_states = copy.deepcopy([case.initial_state for case in cases])
    first = run_scenarios(cases, refund_environment, output_dir=tmp_path / "first")
    second = run_scenarios(cases, refund_environment, output_dir=tmp_path / "second")
    assert len(first.cases) == len(second.cases) == 26
    assert first.passed, first.failures
    assert second.passed, second.failures
    assert [case.initial_state for case in cases] == original_initial_states

    for directory, report in [(tmp_path / "first", first), (tmp_path / "second", second)]:
        stored_suite = json.loads((directory / "report.json").read_text())
        assert stored_suite == report.to_dict()
        case_directories = sorted(path for path in directory.iterdir() if path.is_dir())
        assert len(case_directories) == 26
        for case_directory, case_report in zip(case_directories, report.cases, strict=True):
            cassette = Cassette.load(case_directory / "cassette.json")
            stored_case = json.loads((case_directory / "report.json").read_text())
            assert stored_case == case_report.to_dict()
            assert cassette.to_dict() == case_report.cassette.to_dict()
            assert cassette.metadata["final_state"] == case_report.final_state
            assert cassette.metadata["passed"] is True
            assert cassette.metadata["evalcraft_scenario"] == {"version": 1}

    for left, right in zip(first.cases, second.cases, strict=True):
        assert left.cassette.id != right.cassette.id
        assert left.final_state["orders"] == right.final_state["orders"]
        assert left.final_state["provider_refunds"] == right.final_state["provider_refunds"]


@pytest.mark.parametrize("suite", [False, True])
def test_saving_never_overwrites_an_existing_cassette_or_report(tmp_path, suite):
    directory = tmp_path / "existing"
    directory.mkdir()
    cassette = directory / "cassette.json"
    report = directory / "report.json"
    cassette.write_text('{"baseline":"do not replace"}\n')
    report.write_text('{"reviewed":"do not replace"}\n')
    calls = []

    def factory(initial_state):
        calls.append(initial_state)
        return refund_environment(initial_state)

    with pytest.raises(FileExistsError):
        if suite:
            run_scenarios(refund_scenarios(), factory, output_dir=directory)
        else:
            run_scenario(refund_scenarios()[0], factory, output_dir=directory)
    assert not calls
    assert cassette.read_text() == '{"baseline":"do not replace"}\n'
    assert report.read_text() == '{"reviewed":"do not replace"}\n'


def test_sdk_cassette_playback_runs_current_scenario_code_and_detects_changed_refund(tmp_path):
    httpx = pytest.importorskip("httpx")
    openai = pytest.importorskip("openai")
    from evalcraft.adapters import OpenAIAdapter

    base = refund_scenarios()[0]
    proposal = copy.deepcopy(base.steps[0].args["proposal"])
    messages = [{"role": "user", "content": "Return the approved refund proposal as JSON."}]
    reply = {
        "id": "scenario-synthetic-reply", "object": "chat.completion", "created": 1,
        "model": "gpt-4.1-mini", "choices": [{"index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": json.dumps(proposal)}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    requests = []

    def recorded_transport(request):
        requests.append(request)
        return httpx.Response(200, json=reply)

    class AgentEnvironment:
        def __init__(self, initial_state, client, *, changed=False):
            self.inner = RefundScenarioEnvironment(initial_state)
            self.client = client
            self.changed = changed

        def execute(self, action):
            if action.action != "agent_refund":
                return self.inner.execute(action)
            response = self.client.chat.completions.create(model="gpt-4.1-mini", messages=messages)
            current_proposal = json.loads(response.choices[0].message.content)
            if self.changed:
                current_proposal["amount_minor"] += 1
            return self.inner.execute(ScenarioAction(
                "refund", {"proposal": current_proposal, "approval": "grant"},
            ))

        def snapshot(self):
            return self.inner.snapshot()

        def close(self):
            self.inner.close()

    case = Scenario("sdk-refund-scenario", base.initial_state, [
        copy.deepcopy(base.steps[0]),
        ScenarioStep("agent_refund", {}, "succeeded", expected_result={"status": "verified"}),
    ], base.expected_state)
    transport = httpx.MockTransport(recorded_transport)
    with openai.OpenAI(api_key="synthetic-not-a-real-key", http_client=httpx.Client(transport=transport)) as client:
        with OpenAIAdapter():
            baseline = run_scenario(
                case, lambda initial: AgentEnvironment(initial, client), output_dir=tmp_path / "baseline",
            )
    assert baseline.passed, baseline.failures
    assert len(requests) == 1
    assert baseline.cassette.llm_call_count == baseline.cassette.tool_call_count == 1
    assert baseline.final_state["provider_refunds"] == 1
    saved = Cassette.load(tmp_path / "baseline" / "cassette.json")

    def deny_transport(request):
        pytest.fail("Playback reached the SDK transport instead of answering from the cassette")

    # Playback must execute each new environment and current agent code. A deny
    # transport makes any SDK HTTP request fail even if the network guard changes.
    for changed in (False, True):
        with openai.OpenAI(
            api_key="synthetic-not-a-real-key", http_client=httpx.Client(transport=httpx.MockTransport(deny_transport)),
        ) as client:
            with playback(saved):
                current = run_scenario(
                    case, lambda initial, changed=changed: AgentEnvironment(initial, client, changed=changed),
                    output_dir=tmp_path / ("changed" if changed else "unchanged"),
                )
        assert current.cassette.llm_call_count == 1
        if changed:
            assert not current.passed
            assert current.steps[-1].outcome.status == "blocked"
            assert current.final_state["provider_refunds"] == 0
            assert current.final_state["orders"][0]["remaining"] == 10000
            assert any("final_state.provider_refunds" in failure for failure in current.failures)
        else:
            assert current.passed, current.failures
            assert current.final_state["provider_refunds"] == 1
    assert len(requests) == 1
