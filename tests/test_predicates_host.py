"""Tests for tool_host_within — network-egress / SSRF guard."""

import pytest

from pactrun import Contract, tool_host_within


def _run(pred, url, *, tool="fetch", key="url"):
    c = Contract("t").require(pred, on_fail="log")
    with c.session() as s:
        s.emit_tool_call(tool, args={key: url})
    return s


# ---------------------------------------------------------------------------
# allow / deny
# ---------------------------------------------------------------------------

def test_allow_exact_passes():
    assert _run(tool_host_within(allow=["api.corp.com"]), "https://api.corp.com/v1").is_compliant


def test_allow_implicit_deny():
    assert not _run(tool_host_within(allow=["api.corp.com"]), "https://evil.com/x").is_compliant


def test_allow_wildcard_subdomain():
    assert _run(tool_host_within(allow=["*.corp.com"]), "https://api.corp.com/v1").is_compliant


def test_wildcard_does_not_match_suffix_spoof():
    # *.corp.com must NOT match corp.com.evil.com
    assert not _run(tool_host_within(allow=["*.corp.com"]), "https://corp.com.evil.com/x").is_compliant


def test_deny_wins():
    s = _run(tool_host_within(allow=["*.corp.com"], deny=["secret.corp.com"]),
             "https://secret.corp.com/x")
    assert not s.is_compliant


def test_deny_only_passes_other():
    assert _run(tool_host_within(deny=["evil.com"]), "https://good.com/x").is_compliant


# ---------------------------------------------------------------------------
# block_private — SSRF
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url", [
    "http://169.254.169.254/latest/meta-data/",  # cloud metadata
    "http://127.0.0.1:8080/",
    "http://10.1.2.3/x",
    "http://192.168.0.1/x",
    "http://[::1]:9000/",
    "http://localhost:3000/admin",
])
def test_block_private_blocks(url):
    assert not _run(tool_host_within(block_private=True), url).is_compliant


def test_block_private_passes_public():
    assert _run(tool_host_within(block_private=True), "https://example.com/x").is_compliant


def test_block_private_passes_public_ip():
    assert _run(tool_host_within(block_private=True), "http://8.8.8.8/x").is_compliant


# ---------------------------------------------------------------------------
# CIDR
# ---------------------------------------------------------------------------

def test_cidr_deny():
    s = _run(tool_host_within(deny=["10.0.0.0/8"]), "http://10.5.6.7/x")
    assert not s.is_compliant


def test_cidr_allow_outside_blocked():
    s = _run(tool_host_within(allow=["10.0.0.0/8"]), "http://11.0.0.1/x")
    assert not s.is_compliant


# ---------------------------------------------------------------------------
# arg targeting / parsing robustness
# ---------------------------------------------------------------------------

def test_arg_scoping():
    pred = tool_host_within(allow=["api.corp.com"], arg="endpoint")
    c = Contract("t").require(pred, on_fail="log")
    with c.session() as s:
        # 'note' is not the targeted arg → ignored even though it's a bad host
        s.emit_tool_call("fetch", args={"endpoint": "https://api.corp.com", "note": "https://evil.com"})
    assert s.is_compliant


def test_default_scan_finds_url_in_any_key():
    pred = tool_host_within(allow=["api.corp.com"])
    c = Contract("t").require(pred, on_fail="log")
    with c.session() as s:
        s.emit_tool_call("fetch", args={"callback": "https://evil.com/hook"})
    assert not s.is_compliant


def test_bare_host_value():
    assert not _run(tool_host_within(allow=["api.corp.com"]), "evil.com:443").is_compliant


def test_non_url_string_ignored():
    # A plain sentence shouldn't be parsed as a host.
    assert _run(tool_host_within(allow=["api.corp.com"]), "just a normal note").is_compliant


def test_malformed_url_does_not_crash():
    assert _run(tool_host_within(block_private=True), "ht!tp://::::").is_compliant


def test_non_tool_event_passes():
    c = Contract("t").require(tool_host_within(block_private=True), on_fail="log")
    with c.session() as s:
        s.emit_llm_response(model="m", output="http://127.0.0.1/")
    assert s.is_compliant


def test_requires_a_policy():
    with pytest.raises(ValueError):
        tool_host_within()


def test_registered():
    import pactrun
    assert "tool_host_within" in pactrun.list_predicates()


def test_integration_blocks_run():
    from pactrun import ViolationError
    c = Contract("t").require(tool_host_within(block_private=True), on_fail="block")
    with pytest.raises(ViolationError):
        with c.session() as s:
            s.emit_tool_call("fetch", args={"url": "http://169.254.169.254/"})
