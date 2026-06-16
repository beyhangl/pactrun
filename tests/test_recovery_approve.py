"""Tests for the `approve` recovery action (human/policy-in-the-loop)."""

import pytest

from pactrun import (
    Contract,
    OnFail,
    ViolationError,
    auto_approver,
    cost_under,
)
from pactrun.core.models import Violation
from pactrun.recovery import apply_recovery


def _v(on_fail=OnFail.APPROVE):
    return Violation(clause_description="cost under $1", on_fail=on_fail, message="over budget")


# ---------------------------------------------------------------------------
# apply_recovery branch
# ---------------------------------------------------------------------------

def test_approver_allows_proceed():
    # Returns None (does not raise) when the approver says yes.
    assert apply_recovery(_v(), approval_handler=lambda v: True) is None


def test_approver_denies_blocks():
    with pytest.raises(ViolationError):
        apply_recovery(_v(), approval_handler=lambda v: False)


def test_missing_approver_fails_closed():
    with pytest.raises(ViolationError):
        apply_recovery(_v(), approval_handler=None)


def test_erroring_approver_fails_closed():
    def boom(v):
        raise RuntimeError("approver down")

    with pytest.raises(ViolationError):
        apply_recovery(_v(), approval_handler=boom)


# ---------------------------------------------------------------------------
# Contract.on_approve wiring
# ---------------------------------------------------------------------------

def test_contract_on_approve_allows():
    calls = []
    c = Contract("t").require(cost_under(1.0), on_fail="approve").on_approve(
        lambda v: calls.append(v) or True
    )
    # Approved → no raise, run continues.
    with c.session() as s:
        s.emit_llm_response(model="m", output="x", cost=5.0)
    assert len(calls) == 1
    # Caveat: the violation is still recorded, so the session is not "compliant".
    assert not s.is_compliant


def test_contract_on_approve_denies_raises():
    c = Contract("t").require(cost_under(1.0), on_fail="approve").on_approve(auto_approver(False))
    with pytest.raises(ViolationError):
        with c.session() as s:
            s.emit_llm_response(model="m", output="x", cost=5.0)


def test_contract_approve_without_handler_blocks():
    # on_fail="approve" but no on_approve() registered → fail closed.
    c = Contract("t").require(cost_under(1.0), on_fail="approve")
    with pytest.raises(ViolationError):
        with c.session() as s:
            s.emit_llm_response(model="m", output="x", cost=5.0)


def test_session_kwarg_overrides_contract():
    c = Contract("t").require(cost_under(1.0), on_fail="approve")
    # Approval handler supplied at session construction time.
    with c.session(approval_handler=auto_approver(True)) as s:
        s.emit_llm_response(model="m", output="x", cost=5.0)
    assert not s.is_compliant  # recorded, but not blocked


def test_auto_approver_decision():
    assert auto_approver(True)(_v()) is True
    assert auto_approver(False)(_v()) is False
