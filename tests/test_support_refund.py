"""End-to-end synthetic support actions, cassettes and changed-code playback."""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("evalcraft")

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "support_refund.py"
SPEC = importlib.util.spec_from_file_location("pactrun_support_refund_example", EXAMPLE)
demo = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = demo
SPEC.loader.exec_module(demo)

from pactrun import ViolationError


@pytest.fixture
def workflow(tmp_path):
    flow = demo.RefundWorkflow(tmp_path, secret="synthetic-test-key")
    flow.provider.seed_order()
    return flow


@pytest.fixture
def identity():
    return demo.Identity("tenant-a", "customer-a", "beyhangl")


@pytest.fixture
def proposal():
    return demo.Refund("refund-1", "order-1", 2500, "USD")


def test_timeout_restart_reconciliation_never_blindly_retries(workflow, identity, proposal):
    grant = workflow.approve(proposal, identity)
    assert workflow.execute(proposal, identity, grant, timeout_after_commit=True)["status"] == "unknown"
    assert workflow.provider.committed_refunds == 1
    restarted = demo.RefundWorkflow(workflow.directory, secret="synthetic-test-key")
    assert restarted.reconcile(proposal.action_id)["status"] == "verified"
    with pytest.raises(ViolationError):
        restarted.execute(proposal, identity, grant)
    assert restarted.provider.committed_refunds == 1
    assert restarted.provider.order(proposal.order_id)["remaining"] == 7500


@pytest.mark.parametrize("foreign", [
    demo.Identity("tenant-b", "customer-a", "beyhangl"),
    demo.Identity("tenant-a", "customer-b", "beyhangl"),
])
def test_host_identity_must_own_order(workflow, foreign, proposal):
    grant = workflow.approve(proposal, foreign)
    with pytest.raises(ViolationError, match="business"):
        workflow.execute(proposal, foreign, grant)
    assert workflow.provider.committed_refunds == 0


@pytest.mark.parametrize("changes", [
    {"amount_minor": 10001}, {"amount_minor": 0}, {"amount_minor": -1},
    {"amount_minor": True}, {"amount_minor": 25.5}, {"currency": "EUR"},
])
def test_invalid_refund_never_reaches_provider(workflow, identity, proposal, changes):
    invalid = replace(proposal, **changes)
    grant = workflow.approve(invalid, identity)
    with pytest.raises(ViolationError):
        workflow.execute(invalid, identity, grant)
    assert workflow.provider.committed_refunds == 0


def test_changed_action_after_approval_is_blocked(workflow, identity, proposal):
    grant = workflow.approve(proposal, identity)
    with pytest.raises(ViolationError):
        workflow.execute(replace(proposal, amount_minor=2501), identity, grant)
    assert workflow.provider.committed_refunds == 0


def test_timestamp_edit_does_not_refresh_a_grant(workflow, identity, proposal):
    grant = workflow.approve(proposal, identity, issued_at=time.time() - 1000)
    grant["issued_at"] = time.time()
    with pytest.raises(ViolationError):
        workflow.execute(proposal, identity, grant)
    assert workflow.provider.committed_refunds == 0


def test_changed_business_state_invalidates_old_approval(workflow, identity, proposal):
    grant = workflow.approve(proposal, identity)
    workflow.provider.refund(replace(proposal, action_id="external-action"), identity)
    with pytest.raises(ViolationError):
        workflow.execute(proposal, identity, grant)
    assert workflow.provider.committed_refunds == 1


def test_same_grant_concurrently_admits_at_most_one_action(workflow, identity, proposal):
    grant = workflow.approve(proposal, identity)
    barrier = threading.Barrier(2)

    def attempt():
        flow = demo.RefundWorkflow(workflow.directory, secret="synthetic-test-key")
        barrier.wait(timeout=5)
        try:
            return flow.execute(proposal, identity, grant)["status"]
        except ViolationError:
            return "blocked"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: attempt(), range(2)))
    assert sorted(results) == ["blocked", "verified"]
    assert workflow.provider.committed_refunds == 1


def test_missing_or_mismatched_provider_evidence_is_not_verified(workflow, identity, proposal, monkeypatch):
    grant = workflow.approve(proposal, identity)
    monkeypatch.setattr(workflow.provider, "lookup", lambda action_id: None)
    assert workflow.execute(proposal, identity, grant)["status"] == "unknown"
    assert workflow.provider.committed_refunds == 1
    monkeypatch.setattr(workflow.provider, "lookup", lambda action_id: {
        "refund_id": action_id, "status": "confirmed", "amount_minor": 9999,
    })
    assert workflow.reconcile(proposal.action_id)["status"] == "mismatch"
    assert workflow.provider.committed_refunds == 1


def test_reserved_action_survives_crash_before_result_is_recorded(workflow, identity, proposal):
    grant = workflow.approve(proposal, identity)

    def crash_after_commit():
        with workflow.contract(proposal, identity).session() as session:
            session.emit_tool_call("refund", args=demo.asdict(proposal), metadata={"user_consent": grant})
            workflow._reserve(proposal, identity)
            workflow.provider.refund(proposal, identity)
            raise RuntimeError("synthetic process crash")

    with pytest.raises(RuntimeError):
        crash_after_commit()
    restarted = demo.RefundWorkflow(workflow.directory, secret="synthetic-test-key")
    assert restarted.reconcile(proposal.action_id)["status"] == "verified"
    assert restarted.provider.committed_refunds == 1


def test_unknown_outcome_before_provider_commit_does_not_automatically_resend(workflow, identity, proposal):
    workflow._reserve(proposal, identity)
    assert workflow.reconcile(proposal.action_id)["status"] == "unknown"
    assert workflow.provider.committed_refunds == 0


def test_cli_demo_creates_inspectable_cassette_and_report(tmp_path):
    report = demo.run_demo(tmp_path)
    assert report["timeout_outcome"] == "unknown"
    assert report["after_restart"] == "verified"
    assert report["repeat_blocked"]
    assert report["provider_refunds"] == 1
    assert report["recorded_policy_compliant"]
    assert Path(report["cassette"]).is_file()
    assert json.loads((tmp_path / "report.json").read_text()) == report


def test_changed_agent_code_runs_against_recorded_reply_and_guard_blocks_changed_amount(tmp_path):
    import httpx
    import openai
    from evalcraft import CaptureContext, assert_tool_called, playback
    from evalcraft.adapters import OpenAIAdapter

    from pactrun.integrations.evalcraft import EvalcraftRecorder

    identity = demo.Identity("tenant-a", "customer-a", "beyhangl")
    proposal = demo.Refund("refund-1", "order-1", 2500, "USD")
    messages = [{"role": "user", "content": "Propose the approved USD refund for order-1 as JSON."}]
    reply = {
        "id": "synthetic-model-reply", "object": "chat.completion", "created": 1,
        "model": "gpt-4.1-mini", "choices": [{"index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": json.dumps(demo.asdict(proposal))}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=reply))

    def agent(client, flow, grant, *, changed=False):
        response = client.chat.completions.create(model="gpt-4.1-mini", messages=messages)
        action = demo.Refund(**json.loads(response.choices[0].message.content))
        if changed:
            action = replace(action, amount_minor=action.amount_minor + 1)
        return flow.execute(action, identity, grant)

    baseline = demo.RefundWorkflow(tmp_path / "baseline", secret="synthetic-test-key")
    baseline.provider.seed_order()
    with CaptureContext(name="refund-agent") as captured, OpenAIAdapter():
        baseline.recorder = EvalcraftRecorder(captured.cassette)
        with openai.OpenAI(api_key="synthetic", http_client=httpx.Client(transport=transport)) as client:
            assert agent(client, baseline, baseline.approve(proposal, identity))["status"] == "verified"
    cassette = captured.cassette
    assert cassette.llm_call_count == 1
    assert cassette.tool_call_count == 1
    recorded_action = assert_tool_called(cassette, "refund", with_args=demo.asdict(proposal))
    assert recorded_action.passed, recorded_action.message

    current = demo.RefundWorkflow(tmp_path / "current", secret="synthetic-test-key")
    current.provider.seed_order()
    with openai.OpenAI(api_key="synthetic") as client, playback(cassette):
        assert agent(client, current, current.approve(proposal, identity))["status"] == "verified"

    broken = demo.RefundWorkflow(tmp_path / "changed", secret="synthetic-test-key")
    broken.provider.seed_order()
    with openai.OpenAI(api_key="synthetic") as client, playback(cassette):
        with pytest.raises(ViolationError):
            agent(client, broken, broken.approve(proposal, identity), changed=True)
    assert broken.provider.committed_refunds == 0
