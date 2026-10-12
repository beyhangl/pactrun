"""Approval security regressions use host-issued synthetic data, no real tools."""

import time
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from pactrun import (
    Contract,
    ViolationError,
    consent_token_required,
    mint_approval_token,
    mint_consent_token,
    multi_party_approval_required,
    must_not_call,
)
from pactrun.approval_store import SQLiteApprovalStore
from pactrun.core.enums import EventKind
from pactrun.core.models import Event, SessionState

SECRET = "host-owned-test-key"
CONTEXT = {
    "tenant": "shop-a", "principal": "customer-17",
    "policy": "refund-v3", "state": {"order_version": 8},
}


def _mint(kind, **kwargs):
    if kind == "consent":
        return mint_consent_token("refund", secret=SECRET, **kwargs)
    return mint_approval_token("support-user-1", tool="refund", secret=SECRET, **kwargs)


def _predicate(kind, **kwargs):
    if kind == "consent":
        return consent_token_required("refund", secret=SECRET, **kwargs)
    return multi_party_approval_required("refund", n_required=1, secret=SECRET, **kwargs)


def _event(kind, token, *, args=None, timestamp=None, metadata=None):
    token_key = "user_consent" if kind == "consent" else "approvals"
    values = {token_key: token}
    values.update(metadata or {})
    kwargs = {} if timestamp is None else {"timestamp": timestamp}
    return Event(kind=EventKind.TOOL_CALL, tool_name="refund", tool_args=args or {}, metadata=values, **kwargs)


@pytest.mark.parametrize("kind", ["consent", "approval"])
def test_expired_timestamp_cannot_be_rewritten(kind):
    token = _mint(kind, issued_at=time.time() - 1000)
    assert not _predicate(kind)(_event(kind, token), SessionState()).passed
    token["issued_at"] = time.time()
    assert not _predicate(kind)(_event(kind, token), SessionState()).passed


@pytest.mark.parametrize("kind", ["consent", "approval"])
@pytest.mark.parametrize("issued", [True, None, "2026-10-12", float("nan"), float("inf"), -float("inf")])
def test_invalid_timestamps_rejected_even_without_expiration(kind, issued):
    token = _mint(kind)
    token["issued_at"] = issued
    assert not _predicate(kind, max_age_s=None)(_event(kind, token), SessionState()).passed


@pytest.mark.parametrize("kind", ["consent", "approval"])
def test_signed_future_timestamp_rejected(kind):
    token = _mint(kind, issued_at=time.time() + 60)
    assert not _predicate(kind, max_age_s=None)(_event(kind, token), SessionState()).passed


@pytest.mark.parametrize("kind", ["consent", "approval"])
@pytest.mark.parametrize("field,value", [("action", "other_tool"), ("nonce", "other-nonce"), ("version", 1)])
def test_signed_envelope_cannot_be_repointed(kind, field, value):
    token = _mint(kind)
    token[field] = value
    assert not _predicate(kind)(_event(kind, token), SessionState()).passed


@pytest.mark.parametrize("kind", ["consent", "approval"])
def test_exact_host_context_binding_rejects_model_assertions(kind):
    token = _mint(kind, context=CONTEXT)
    current = {**CONTEXT, "tenant": "shop-b"}
    # The model repeats the approved tenant, but the host's live tenant differs.
    event = _event(kind, token, metadata={"tenant": "shop-a", "principal": "customer-17"})
    assert not _predicate(kind, context=lambda event, state: current)(event, SessionState()).passed
    assert _predicate(kind, context=lambda event, state: CONTEXT)(event, SessionState()).passed


@pytest.mark.parametrize("kind", ["consent", "approval"])
@pytest.mark.parametrize("field", ["tenant", "principal", "policy", "state"])
def test_tampered_context_rejected_even_if_expected_context_matches(kind, field):
    token = _mint(kind, context=CONTEXT)
    modified = {**CONTEXT, field: "attacker-controlled"}
    token["context"] = modified
    assert not _predicate(kind, context=modified)(_event(kind, token), SessionState()).passed


def test_minter_detaches_mutable_business_state():
    state = {"remaining_cents": 500, "nested": {"version": 1}}
    token = mint_consent_token("refund", secret=SECRET, context=state)
    state["nested"]["version"] = 2
    assert token["context"]["nested"]["version"] == 1
    assert not consent_token_required("refund", secret=SECRET, context=state)(
        _event("consent", token), SessionState(),
    ).passed


@pytest.mark.parametrize("kind", ["consent", "approval"])
def test_missing_bound_argument_cannot_equal_explicit_null(kind):
    token = _mint(kind, args={"amount": None}, bind_args=["amount"])
    pred = _predicate(kind, bind_args=["amount"])
    assert not pred(_event(kind, token), SessionState()).passed
    assert pred(_event(kind, token, args={"amount": None}), SessionState()).passed


@pytest.mark.parametrize("kind", ["consent", "approval"])
def test_legacy_envelope_and_non_ascii_signature_fail_closed(kind):
    token = _mint(kind)
    del token["version"]
    assert not _predicate(kind)(_event(kind, token), SessionState()).passed
    token = _mint(kind)
    token["sig"] = "\N{SNOWMAN}"
    assert not _predicate(kind)(_event(kind, token), SessionState()).passed


@pytest.mark.parametrize("kind", ["consent", "approval"])
def test_reuse_rejected_across_sessions_and_store_instances(kind, tmp_path):
    path = tmp_path / "approvals.sqlite"
    token = _mint(kind)
    assert _predicate(kind, replay_store=SQLiteApprovalStore(path))(
        _event(kind, token), SessionState(),
    ).passed
    assert not _predicate(kind, replay_store=SQLiteApprovalStore(path))(
        _event(kind, token), SessionState(),
    ).passed


@pytest.mark.parametrize("kind", ["consent", "approval"])
def test_concurrent_authorization_uses_one_grant(kind, tmp_path):
    path = tmp_path / "approvals.sqlite"
    SQLiteApprovalStore(path)
    token = _mint(kind)
    ready = Barrier(2)

    def authorize():
        pred = _predicate(kind, replay_store=SQLiteApprovalStore(path))
        ready.wait(timeout=5)
        return pred(_event(kind, token), SessionState()).passed

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: authorize(), range(2)))
    assert sorted(results) == [False, True]


@pytest.mark.parametrize("kind", ["consent", "approval"])
def test_monitor_simulates_reuse_without_consuming_durable_grant(kind, tmp_path):
    store = SQLiteApprovalStore(tmp_path / "approvals.sqlite")
    token = _mint(kind)
    contract = Contract("shadow").require(_predicate(kind, replay_store=store)).monitor()
    with contract.session() as session:
        session.record_event(_event(kind, token))
        assert session.is_compliant
        session.record_event(_event(kind, token))
    assert not session.is_compliant
    assert not session.violations[0].enforced
    assert _predicate(kind, replay_store=store)(_event(kind, token), SessionState()).passed


@pytest.mark.parametrize("kind", ["consent", "approval"])
def test_replay_uses_event_clock_and_does_not_touch_live_store(kind, tmp_path):
    token = _mint(kind, issued_at=100)
    event = _event(kind, token, timestamp=110)
    store = SQLiteApprovalStore(tmp_path / "approvals.sqlite")
    contract = Contract("recorded").require(_predicate(kind, max_age_s=20, replay_store=store))
    result = contract.replay([event])
    assert result.summary.is_compliant
    assert store.consume_many([token["nonce"]])
    # Live evaluation cannot revive an expired grant by supplying an old event timestamp.
    assert not _predicate(kind, max_age_s=20)(event, SessionState()).passed


def test_later_clause_failure_consumes_grant_and_requires_reapproval(tmp_path):
    store = SQLiteApprovalStore(tmp_path / "approvals.sqlite")
    token = _mint("consent")
    pred = _predicate("consent", replay_store=store)
    contract = Contract("later-denial").require(pred).require(must_not_call("refund"))
    with pytest.raises(ViolationError):
        with contract.session() as session:
            session.record_event(_event("consent", token))
    assert not pred(_event("consent", token), SessionState()).passed


@pytest.mark.parametrize("kind", ["consent", "approval"])
def test_store_and_host_lookup_errors_do_not_grant_authority(kind):
    class UnavailableStore:
        def consume_many(self, nonces):
            raise OSError("host-owned store unavailable")

    token = _mint(kind)
    assert not _predicate(kind, replay_store=UnavailableStore())(_event(kind, token), SessionState()).passed

    def unavailable_context(event, state):
        raise OSError("authoritative backend unavailable")

    assert not _predicate(kind, context=unavailable_context)(_event(kind, token), SessionState()).passed


def test_quorum_shortfall_and_reused_quorum_never_partially_consume(tmp_path):
    store = SQLiteApprovalStore(tmp_path / "approvals.sqlite")
    first = mint_approval_token("alice", tool="refund", secret=SECRET)
    second = mint_approval_token("bob", tool="refund", secret=SECRET)
    pred = multi_party_approval_required("refund", secret=SECRET, replay_store=store)
    assert not pred(_event("approval", first), SessionState()).passed
    assert pred(_event("approval", [first, second]), SessionState()).passed
    third = mint_approval_token("carol", tool="refund", secret=SECRET)
    assert not pred(_event("approval", [third, second]), SessionState()).passed
    # Failed quorum rolled back the first insertion; carol's grant remains usable.
    assert store.consume_many([third["nonce"]])


def test_unkeyed_compatibility_mode_is_reusable_but_cannot_consume(tmp_path):
    token = mint_consent_token("refund")
    pred = consent_token_required("refund")
    assert pred(_event("consent", token), SessionState()).passed
    assert pred(_event("consent", token), SessionState()).passed
    with pytest.raises(ValueError, match="HMAC secret"):
        consent_token_required("refund", replay_store=SQLiteApprovalStore(tmp_path / "unused.sqlite"))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, True])
def test_invalid_expiration_configuration_rejected(value):
    with pytest.raises(ValueError):
        consent_token_required("refund", max_age_s=value)


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_quorum_configuration_rejected(value):
    with pytest.raises(ValueError):
        multi_party_approval_required("refund", n_required=value)


def test_empty_approver_allowlist_denies_all():
    token = _mint("approval")
    assert not multi_party_approval_required("refund", n_required=1, approvers=[], secret=SECRET)(
        _event("approval", token), SessionState(),
    ).passed


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True])
def test_minter_rejects_nonfinite_or_boolean_timestamp(value):
    with pytest.raises(ValueError):
        mint_consent_token("refund", issued_at=value)


def test_minter_rejects_missing_bound_values_and_lossy_context():
    with pytest.raises(ValueError, match="missing"):
        mint_consent_token("refund", args={}, bind_args=["amount"])
    with pytest.raises(ValueError, match="finite JSON"):
        mint_consent_token("refund", context={"authority": object()})


@pytest.mark.parametrize("kind", ["consent", "approval"])
def test_missing_host_context_does_not_become_empty_context(kind):
    token = _mint(kind)
    assert not _predicate(kind, context=lambda event, state: None)(
        _event(kind, token), SessionState(),
    ).passed


@pytest.mark.parametrize("kind", ["consent", "approval"])
def test_invalid_replay_clock_fails_closed(kind):
    token = _mint(kind)
    event = _event(kind, token, timestamp=float("nan"))
    assert not _predicate(kind)(event, SessionState(
        evaluation_clock="event", evaluation_mode="monitor",
    )).passed


@pytest.mark.parametrize("kind", ["consent", "approval"])
def test_live_event_clock_cannot_revive_expired_approval(kind):
    issued = time.time() - 1000
    token = _mint(kind, issued_at=issued)
    event = _event(kind, token, timestamp=issued + 1)
    assert not _predicate(kind)(event, SessionState(evaluation_clock="event")).passed
    assert _predicate(kind)(event, SessionState(
        evaluation_clock="event", evaluation_mode="monitor",
    )).passed


def test_empty_secret_and_missing_approver_rejected():
    with pytest.raises(ValueError, match="nonempty"):
        mint_consent_token("refund", secret="")
    with pytest.raises(ValueError, match="Approver"):
        mint_approval_token(None, tool="refund")
