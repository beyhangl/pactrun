"""Tests for no_exfiltration_after_untrusted and lethal_trifecta_guard."""

import pytest

from pactrun import (
    Contract,
    ViolationError,
    auto_approver,
    lethal_trifecta_guard,
    no_exfiltration_after_untrusted,
)


# ---------------------------------------------------------------------------
# no_exfiltration_after_untrusted
# ---------------------------------------------------------------------------

def _run(pred, steps, on_fail="log", **skw):
    """steps: list of ('tool', name, args, metadata)."""
    c = Contract("t").require(pred, on_fail=on_fail)
    with c.session(**skw) as s:
        for name, args, meta in steps:
            s.emit_tool_call(name, args=args, metadata=meta)
    return s


def test_ingest_then_exfil_fails():
    s = _run(no_exfiltration_after_untrusted(),
             [("web_fetch", {"url": "http://x"}, None), ("send_email", {"to": "a@b.com"}, None)])
    assert not s.is_compliant


def test_exfil_alone_passes():
    s = _run(no_exfiltration_after_untrusted(),
             [("send_email", {"to": "a@b.com"}, None)])
    assert s.is_compliant


def test_ingest_alone_passes():
    s = _run(no_exfiltration_after_untrusted(),
             [("web_fetch", {"url": "http://x"}, None)])
    assert s.is_compliant


def test_order_matters_exfil_before_ingest_passes():
    s = _run(no_exfiltration_after_untrusted(),
             [("send_email", {"to": "a@b.com"}, None), ("web_fetch", {"url": "http://x"}, None)])
    assert s.is_compliant


def test_encoded_arg_channel_fails_off_exfil_list():
    blob = "A" * 80
    s = _run(no_exfiltration_after_untrusted(),
             [("web_fetch", {"url": "http://x"}, None),
              ("note", {"data": f"http://collect.me/?d={blob}"}, None)])
    assert not s.is_compliant  # 'note' isn't an exfil tool, but the encoded blob is the channel


def test_metadata_tagged_ingest():
    s = _run(no_exfiltration_after_untrusted(),
             [("custom_read", {}, {"source": "external"}),
              ("send_email", {"to": "a@b.com"}, None)])
    assert not s.is_compliant


def test_tag_values_filter():
    pred = no_exfiltration_after_untrusted(untrusted_tag_values=["external"])
    # tagged 'internal' -> not untrusted -> exfil passes
    s = _run(pred, [("custom_read", {}, {"source": "internal"}),
                    ("send_email", {"to": "a@b.com"}, None)])
    assert s.is_compliant


def test_approve_routes():
    c = Contract("t").require(no_exfiltration_after_untrusted(), on_fail="approve").on_approve(
        auto_approver(False)
    )
    with pytest.raises(ViolationError):
        with c.session() as s:
            s.emit_tool_call("web_fetch", args={"url": "http://x"})
            s.emit_tool_call("send_email", args={"to": "a@b.com"})


# ---------------------------------------------------------------------------
# lethal_trifecta_guard
# ---------------------------------------------------------------------------

TRIO = dict(
    untrusted_sources=["fetch_*", "read_email"],
    private_data_tools=["db_query", "read_file"],
    egress_tools=["send_email", "http_post"],
)


def test_all_three_fails():
    s = _run(lethal_trifecta_guard(**TRIO),
             [("fetch_page", {}, None), ("db_query", {}, None), ("send_email", {}, None)])
    assert not s.is_compliant


@pytest.mark.parametrize("steps", [
    [("fetch_page", {}, None), ("db_query", {}, None)],            # no egress
    [("fetch_page", {}, None), ("send_email", {}, None)],          # no private
    [("db_query", {}, None), ("send_email", {}, None)],            # no untrusted
])
def test_two_of_three_passes(steps):
    s = _run(lethal_trifecta_guard(**TRIO), steps)
    assert s.is_compliant


def test_taint_metadata_supplies_untrusted_leg():
    s = _run(lethal_trifecta_guard(**TRIO),
             [("read_file", {}, {"untrusted": True}), ("db_query", {}, None), ("http_post", {}, None)])
    assert not s.is_compliant


def test_glob_matching():
    s = _run(lethal_trifecta_guard(**TRIO),
             [("fetch_news", {}, None), ("read_file", {}, None), ("http_post", {}, None)])
    assert not s.is_compliant


def test_custom_taint_key():
    pred = lethal_trifecta_guard(**TRIO, taint_key="tainted")
    s = _run(pred, [("noop", {}, {"tainted": 1}), ("db_query", {}, None), ("send_email", {}, None)])
    assert not s.is_compliant


def test_empty_class_rejected():
    with pytest.raises(ValueError):
        lethal_trifecta_guard(untrusted_sources=[], private_data_tools=["x"], egress_tools=["y"])


def test_bad_mode_rejected():
    with pytest.raises(ValueError):
        lethal_trifecta_guard(**TRIO, mode="strict")


def test_check_on_modes():
    diag = lethal_trifecta_guard(**TRIO, mode="diagnostic")
    inc = lethal_trifecta_guard(**TRIO, mode="incremental")
    assert diag._check_on == "session_end"
    assert inc._check_on == "every_event"


def test_incremental_blocks_at_third_leg():
    c = Contract("t").require(lethal_trifecta_guard(**TRIO, mode="incremental"), on_fail="block")
    with pytest.raises(ViolationError):
        with c.session() as s:
            s.emit_tool_call("fetch_page")
            s.emit_tool_call("db_query")
            s.emit_tool_call("send_email")  # third leg -> blocks here


def test_registered():
    import pactrun
    names = pactrun.list_predicates()
    assert "no_exfiltration_after_untrusted" in names
    assert "lethal_trifecta_guard" in names
