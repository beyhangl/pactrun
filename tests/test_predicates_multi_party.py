"""Tests for multi_party_approval_required + mint_approval_token."""

import time

import pytest

from pactrun import (
    Contract,
    ViolationError,
    mint_approval_token,
    multi_party_approval_required,
)


def _run(pred, tool, args, tokens, *, on_fail="log", key="approvals", session_meta=False):
    c = Contract("t").require(pred, on_fail=on_fail)
    if session_meta:
        with c.session(metadata={key: tokens}) as s:
            s.emit_tool_call(tool, args=args)
    else:
        with c.session() as s:
            s.emit_tool_call(tool, args=args, metadata={key: tokens})
    return s


def test_two_distinct_approvals_pass():
    toks = [mint_approval_token("alice", tool="wire"), mint_approval_token("bob", tool="wire")]
    s = _run(multi_party_approval_required(["wire"], n_required=2, approvers={"alice", "bob"}),
             "wire", {}, toks)
    assert s.is_compliant


def test_same_approver_twice_counts_once():
    toks = [mint_approval_token("alice", tool="wire"), mint_approval_token("alice", tool="wire")]
    s = _run(multi_party_approval_required(["wire"], n_required=2, approvers={"alice", "bob"}),
             "wire", {}, toks)
    assert not s.is_compliant  # only 1 distinct approver


def test_shortfall_blocks():
    toks = [mint_approval_token("alice", tool="wire")]
    s = _run(multi_party_approval_required(["wire"], n_required=2), "wire", {}, toks)
    assert not s.is_compliant


def test_unlisted_approver_ignored():
    toks = [mint_approval_token("alice", tool="wire"), mint_approval_token("mallory", tool="wire")]
    s = _run(multi_party_approval_required(["wire"], n_required=2, approvers={"alice", "bob"}),
             "wire", {}, toks)
    assert not s.is_compliant  # mallory not allowed -> only alice counts


def test_approver_id_swap_without_resign_fails():
    tok = mint_approval_token("alice", tool="wire")
    tok2 = dict(tok)
    tok2["approver"] = "bob"  # tamper: claim it's bob's, but sig covers 'alice'
    s = _run(multi_party_approval_required(["wire"], n_required=2),
             "wire", {}, [mint_approval_token("carol", tool="wire"), tok2])
    assert not s.is_compliant  # tok2 sig invalid for bob -> only carol counts


def test_bind_args_match_required():
    toks = [
        mint_approval_token("alice", tool="wire", args={"amount": 100}, bind_args=["amount"]),
        mint_approval_token("bob", tool="wire", args={"amount": 100}, bind_args=["amount"]),
    ]
    pred = multi_party_approval_required(["wire"], n_required=2, bind_args=["amount"])
    assert _run(pred, "wire", {"amount": 100}, toks).is_compliant
    # different amount at call time -> tokens don't bind -> block
    assert not _run(pred, "wire", {"amount": 999}, toks).is_compliant


def test_expired_token_ignored():
    toks = [
        mint_approval_token("alice", tool="wire", issued_at=time.time() - 10000),
        mint_approval_token("bob", tool="wire"),
    ]
    s = _run(multi_party_approval_required(["wire"], n_required=2, max_age_s=600),
             "wire", {}, toks)
    assert not s.is_compliant  # alice expired -> only bob


def test_hmac_secret_enforced():
    good = [mint_approval_token("alice", tool="wire", secret="k"),
            mint_approval_token("bob", tool="wire", secret="k")]
    assert _run(multi_party_approval_required(["wire"], n_required=2, secret="k"),
                "wire", {}, good).is_compliant
    wrong = [mint_approval_token("alice", tool="wire", secret="guess"),
             mint_approval_token("bob", tool="wire", secret="guess")]
    assert not _run(multi_party_approval_required(["wire"], n_required=2, secret="real"),
                    "wire", {}, wrong).is_compliant


def test_session_metadata_fallback():
    toks = [mint_approval_token("alice", tool="wire"), mint_approval_token("bob", tool="wire")]
    s = _run(multi_party_approval_required(["wire"], n_required=2),
             "wire", {}, toks, session_meta=True)
    assert s.is_compliant


def test_ungated_tool_passes():
    s = _run(multi_party_approval_required(["wire"], n_required=2), "search", {}, [])
    assert s.is_compliant


def test_block_raises():
    with pytest.raises(ViolationError):
        _run(multi_party_approval_required(["wire"], n_required=2), "wire", {}, [], on_fail="block")


def test_registered():
    import pactrun
    assert "multi_party_approval_required" in pactrun.list_predicates()
    assert callable(pactrun.mint_approval_token)
