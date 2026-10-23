"""Reusable refund cases exercise real controls and independently persisted state."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from dataclasses import replace

import pytest

pytest.importorskip("evalcraft")

from evalcraft.scenarios import (
    Scenario,
    ScenarioAction,
    ScenarioStep,
    StepOutcome,
    run_scenario,
    run_scenarios,
)

from pactrun.scenarios import (
    FakePaymentProvider,
    Identity,
    ProviderRejectedError,
    Refund,
    RefundScenarioEnvironment,
    refund_environment,
    refund_scenarios,
)


@pytest.mark.parametrize("case", refund_scenarios(), ids=lambda case: case.name)
def test_reviewed_refund_scenario(case):
    report = run_scenario(case, refund_environment)
    assert report.passed, report.failures


def test_pack_is_fresh_and_repeated_runs_have_the_same_business_outcomes():
    cases = refund_scenarios()
    cases[0].initial_state["orders"][0]["remaining"] = 1
    assert refund_scenarios()[0].initial_state["orders"][0]["remaining"] == 10000
    first = run_scenarios(refund_scenarios(), refund_environment)
    second = run_scenarios(refund_scenarios(), refund_environment)
    assert first.passed and second.passed
    assert [case.final_state["orders"] for case in first.cases] == [
        case.final_state["orders"] for case in second.cases
    ]


def test_environment_owns_isolated_state_and_cleanup():
    initial = refund_scenarios()[0].initial_state
    before = copy.deepcopy(initial)
    first = refund_environment(initial)
    second = refund_environment(initial)
    try:
        assert first.directory != second.directory
        first.execute(ScenarioAction("host_update_order", {"order_id": "order-1", "remaining": 1}))
        assert first.snapshot()["orders"][0]["remaining"] == 1
        assert second.snapshot()["orders"][0]["remaining"] == 10000
        assert initial == before
        directory = first.directory
        first.close()
        first.close()
        assert not directory.exists()
        with pytest.raises(RuntimeError, match="closed"):
            first.snapshot()
    finally:
        first.close()
        second.close()


def test_scenario_persists_cassette_and_report_without_snapshot_secrets(tmp_path):
    case = refund_scenarios()[0]
    report = run_scenario(case, refund_environment, output_dir=tmp_path / "case")
    assert report.passed
    assert report.cassette.tool_call_count == 1
    encoded = json.dumps(report.final_state)
    assert "synthetic-refund-scenario-key" not in encoded
    assert '"sig"' not in encoded and '"nonce"' not in encoded
    assert list((tmp_path / "case").glob("*.json"))
    # Structured observations retain before/after business state, while tool
    # spans describe attempts and cannot establish that a payment occurred.
    assert report.steps[-1].before["provider_refunds"] == 0
    assert report.steps[-1].after["provider_refunds"] == 1


def test_expectations_detect_an_unsafe_approval_bypass():
    class UnsafeEnvironment(RefundScenarioEnvironment):
        def _refund(self, args, *, workflow=None):
            flow = workflow or self.workflow
            proposal = Refund(**args["proposal"])
            receipt = flow.provider.refund(proposal, self.identity)
            return StepOutcome("succeeded", result={"status": "verified", "receipt": receipt})

    case = next(case for case in refund_scenarios() if case.name == "refund-changed-approved-amount")
    report = run_scenario(case, UnsafeEnvironment)
    assert not report.passed
    assert report.final_state["provider_refunds"] == 1
    assert report.final_state["orders"][0]["remaining"] == 7499
    assert any("status" in failure for failure in report.failures)
    assert any("provider_refunds" in failure for failure in report.failures)


@pytest.mark.parametrize("fault", ["typo", "", "pretend_live_api"])
def test_unknown_fault_is_a_harness_failure_even_when_error_is_expected(fault):
    base = refund_scenarios()[0]
    case = Scenario("bad-fault", base.initial_state, [
        ScenarioStep("refund", {"proposal": base.steps[0].args["proposal"]}, "error", fault=fault),
    ], {"provider_refunds": 0})
    report = run_scenario(case, refund_environment)
    assert not report.passed
    assert "Unknown synthetic provider fault" in " ".join(report.failures)


def test_fault_configuration_cannot_be_injected_through_proposed_tool_arguments():
    base = refund_scenarios()[0]
    proposal = {**base.steps[0].args["proposal"], "timeout_after_commit": True}
    case = Scenario("fault-in-tool-args", base.initial_state, [
        ScenarioStep("refund", {"proposal": proposal}, "blocked"),
    ], {"provider_refunds": 0})
    report = run_scenario(case, refund_environment)
    assert not report.passed  # malformed harness input is not a policy verdict
    assert report.final_state["provider_refunds"] == 0


def test_fake_provider_enforces_idempotency_and_changed_key_arguments(tmp_path):
    provider = FakePaymentProvider(tmp_path / "provider.sqlite")
    provider.seed_order()
    identity = Identity("tenant-a", "customer-a", "beyhangl")
    proposal = Refund("refund-1", "order-1", 2500, "USD")
    first = provider.refund(proposal, identity)
    assert provider.refund(proposal, identity) == first
    with pytest.raises(ProviderRejectedError, match="different action"):
        provider.refund(replace(proposal, amount_minor=2501), identity)
    assert provider.committed_refunds == 1
    assert provider.order("order-1")["remaining"] == 7500


def test_pactrun_scenario_namespace_and_plain_environment_do_not_require_evalcraft():
    source = """
import builtins
original_import = builtins.__import__
def without_evalcraft(name, *args, **kwargs):
    if name == 'evalcraft' or name.startswith('evalcraft.'):
        raise ImportError('optional dependency deliberately absent')
    return original_import(name, *args, **kwargs)
builtins.__import__ = without_evalcraft
from pactrun.scenarios import RefundScenarioEnvironment, RefundWorkflow
environment = RefundScenarioEnvironment({'identity': {
    'tenant_id': 't', 'customer_id': 'c', 'principal_id': 'p'}, 'orders': []})
assert environment.snapshot()['provider_refunds'] == 0
environment.close()
"""
    result = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
