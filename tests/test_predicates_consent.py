"""Tests for consent_token_required + mint_consent_token."""

import time

import pytest

from pactrun import (
    Contract,
    ViolationError,
    auto_approver,
    consent_token_required,
    mint_consent_token,
)


def _run(pred, tool, args, *, metadata=None, on_fail="log", **session_kwargs):
    c = Contract("t").require(pred, on_fail=on_fail)
    with c.session(**session_kwargs) as s:
        s.emit_tool_call(tool, args=args, metadata=metadata)
    return s


# ---------------------------------------------------------------------------
# happy / missing
# ---------------------------------------------------------------------------

def test_ungated_tool_passes_without_token():
    s = _run(consent_token_required(["place_call"]), "search", {"q": "x"})
    assert s.is_compliant


def test_gated_tool_without_token_blocks():
    s = _run(consent_token_required(["place_call"]), "place_call", {"to": "x"})
    assert not s.is_compliant


def test_matching_fresh_token_passes():
    tok = mint_consent_token("place_call")
    s = _run(consent_token_required(["place_call"]), "place_call", {"to": "x"},
             metadata={"user_consent": tok})
    assert s.is_compliant


def test_session_level_token_fallback():
    tok = mint_consent_token("place_call")
    c = Contract("t").require(consent_token_required(["place_call"]), on_fail="log")
    with c.session() as s:
        s.state.metadata["user_consent"] = tok  # host set at session scope
        s.emit_tool_call("place_call", args={"to": "x"})
    assert s.is_compliant


# ---------------------------------------------------------------------------
# action / arg binding
# ---------------------------------------------------------------------------

def test_token_for_other_action_rejected():
    tok = mint_consent_token("send_email")  # token for a different tool
    s = _run(consent_token_required(["place_call"]), "place_call", {"to": "x"},
             metadata={"user_consent": tok})
    assert not s.is_compliant


def test_bind_args_mismatch_rejected():
    # token issued binding recipient A, call places to B
    tok = mint_consent_token("place_call", args={"to": "A"}, bind_args=["to"])
    s = _run(consent_token_required(["place_call"], bind_args=["to"]),
             "place_call", {"to": "B"}, metadata={"user_consent": tok})
    assert not s.is_compliant


def test_bind_args_match_passes():
    tok = mint_consent_token("place_call", args={"to": "A"}, bind_args=["to"])
    s = _run(consent_token_required(["place_call"], bind_args=["to"]),
             "place_call", {"to": "A"}, metadata={"user_consent": tok})
    assert s.is_compliant


# ---------------------------------------------------------------------------
# freshness
# ---------------------------------------------------------------------------

def test_expired_token_rejected():
    tok = mint_consent_token("place_call", issued_at=time.time() - 1000)
    s = _run(consent_token_required(["place_call"], max_age_s=300),
             "place_call", {"to": "x"}, metadata={"user_consent": tok})
    assert not s.is_compliant


def test_fresh_token_within_age_passes():
    tok = mint_consent_token("place_call", issued_at=time.time() - 10)
    s = _run(consent_token_required(["place_call"], max_age_s=300),
             "place_call", {"to": "x"}, metadata={"user_consent": tok})
    assert s.is_compliant


def test_max_age_none_disables_freshness():
    tok = mint_consent_token("place_call", issued_at=time.time() - 100000)
    s = _run(consent_token_required(["place_call"], max_age_s=None),
             "place_call", {"to": "x"}, metadata={"user_consent": tok})
    assert s.is_compliant


def test_undated_token_rejected():
    s = _run(consent_token_required(["place_call"]), "place_call", {"to": "x"},
             metadata={"user_consent": {"action": "place_call", "sig": "deadbeef"}})
    assert not s.is_compliant


# ---------------------------------------------------------------------------
# HMAC secret
# ---------------------------------------------------------------------------

def test_hmac_token_passes_with_right_secret():
    tok = mint_consent_token("place_call", secret="s3cr3t")
    s = _run(consent_token_required(["place_call"], secret="s3cr3t"),
             "place_call", {"to": "x"}, metadata={"user_consent": tok})
    assert s.is_compliant


def test_hmac_tamper_or_wrong_secret_rejected():
    tok = mint_consent_token("place_call", secret="attacker-guess")
    s = _run(consent_token_required(["place_call"], secret="real-secret"),
             "place_call", {"to": "x"}, metadata={"user_consent": tok})
    assert not s.is_compliant


# ---------------------------------------------------------------------------
# recovery integration
# ---------------------------------------------------------------------------

def test_blocks_run_when_on_fail_block():
    with pytest.raises(ViolationError):
        _run(consent_token_required(["place_call"]), "place_call", {"to": "x"}, on_fail="block")


def test_approve_recovery_allows():
    c = Contract("t").require(consent_token_required(["place_call"]), on_fail="approve").on_approve(
        auto_approver(True)
    )
    with c.session() as s:
        s.emit_tool_call("place_call", args={"to": "x"})  # no token, but approver allows
    # approved → no raise; violation still recorded (approve caveat)
    assert not s.is_compliant


def test_approve_recovery_denies_raises():
    c = Contract("t").require(consent_token_required(["place_call"]), on_fail="approve").on_approve(
        auto_approver(False)
    )
    with pytest.raises(ViolationError):
        with c.session() as s:
            s.emit_tool_call("place_call", args={"to": "x"})


def test_non_tool_event_never_trips():
    c = Contract("t").require(consent_token_required(["place_call"]), on_fail="log")
    with c.session() as s:
        s.emit_llm_response(model="m", output="place_call")
    assert s.is_compliant


def test_registered():
    import pactrun
    assert "consent_token_required" in pactrun.list_predicates()
    assert callable(pactrun.mint_consent_token)
