"""Tests for the no_secrets credential-leak predicate."""

import pytest

from pactrun import Contract, no_secrets

SAMPLES = {
    "AWS access key id": "AKIAIOSFODNN7EXAMPLE",
    "GitHub token": "ghp_" + "a" * 36,
    "Google API key": "AIza" + "B" * 35,
    "Slack token": "xox" + "b-" + "1" * 12 + "-" + "a" * 16,
    "payment live key": "sk_live_" + "x" * 24,
    "JWT": "eyJhbGc.eyJzdWIi.SflKxwRJ",
    "private key": "-----BEGIN RSA PRIVATE KEY-----",
}


@pytest.mark.parametrize("secret", list(SAMPLES.values()), ids=list(SAMPLES))
def test_detects_secret(secret):
    c = Contract("t").require(no_secrets(), on_fail="log")
    with c.session() as s:
        s.emit_llm_response(model="m", output=f"here is the key {secret}")
    assert not s.is_compliant


def test_clean_prose_passes():
    c = Contract("t").require(no_secrets(), on_fail="log")
    with c.session() as s:
        s.emit_llm_response(model="m", output="Contact me at alice@example.com about the report.")
    assert s.is_compliant


def test_message_redacts_secret():
    secret = "ghp_" + "z" * 36
    c = Contract("t").require(no_secrets(), on_fail="log")
    with c.session() as s:
        s.emit_llm_response(model="m", output=f"token: {secret}")
    assert not s.is_compliant
    surfaced = " ".join((v.actual or "") + (v.message or "") for v in s.violations)
    assert "redacted" in surfaced
    assert secret not in surfaced  # the full secret must never be echoed


def test_scan_tool_args_when_enabled():
    c = Contract("t").require(no_secrets(scan_tool_args=True), on_fail="log")
    with c.session() as s:
        s.emit_tool_call("post", args={"body": "AKIAIOSFODNN7EXAMPLE"})
    assert not s.is_compliant


def test_tool_args_not_scanned_by_default():
    c = Contract("t").require(no_secrets(), on_fail="log")
    with c.session() as s:
        s.emit_tool_call("post", args={"body": "AKIAIOSFODNN7EXAMPLE"})
    assert s.is_compliant
