"""Tests for tool_arg_value_guard and required_disclosure."""

import pytest

from pactrun import Contract, required_disclosure, tool_arg_value_guard


def _run(pred, calls):
    """Run a list of (tool, args) tool calls under a log-mode contract; return session."""
    c = Contract("t").require(pred, on_fail="log")
    with c.session() as s:
        for tool, args in calls:
            s.emit_tool_call(tool, args=args)
    return s


# ---------------------------------------------------------------------------
# tool_arg_value_guard — deny
# ---------------------------------------------------------------------------

def test_deny_exact_blocks():
    s = _run(tool_arg_value_guard("apply", "company", deny=["BadCo"]),
             [("apply", {"company": "BadCo"})])
    assert not s.is_compliant


def test_deny_passes_when_not_listed():
    s = _run(tool_arg_value_guard("apply", "company", deny=["BadCo"]),
             [("apply", {"company": "GoodCo"})])
    assert s.is_compliant


def test_deny_passes_when_field_absent():
    s = _run(tool_arg_value_guard("apply", "company", deny=["BadCo"]),
             [("apply", {"role": "eng"})])
    assert s.is_compliant


def test_deny_ci_match():
    s = _run(tool_arg_value_guard("apply", "company", deny=["badco"], match="ci"),
             [("apply", {"company": "BADCO"})])
    assert not s.is_compliant


def test_deny_glob_match():
    s = _run(tool_arg_value_guard("send", "to", deny=["*@spam.com"], match="glob"),
             [("send", {"to": "x@spam.com"})])
    assert not s.is_compliant


def test_deny_regex_match():
    s = _run(tool_arg_value_guard("send", "to", deny=[r"^admin@"], match="regex"),
             [("send", {"to": "admin@corp.com"})])
    assert not s.is_compliant


def test_normalize_applied_to_both_sides():
    norm = lambda v: v.strip().lower()
    s = _run(tool_arg_value_guard("send", "to", deny=["  ALICE@x.com "], normalize=norm),
             [("send", {"to": "alice@x.com"})])
    assert not s.is_compliant


def test_dotted_path():
    s = _run(tool_arg_value_guard("send", "recipient.email", deny=["a@b.com"]),
             [("send", {"recipient": {"email": "a@b.com"}})])
    assert not s.is_compliant


def test_list_index_path():
    s = _run(tool_arg_value_guard("batch", "items.0.name", deny=["X"]),
             [("batch", {"items": [{"name": "X"}]})])
    assert not s.is_compliant


def test_callable_denylist_reevaluated_each_event():
    box = {"deny": set()}
    pred = tool_arg_value_guard("send", "to", deny=lambda: box["deny"])
    c = Contract("t").require(pred, on_fail="log")
    with c.session() as s:
        s.emit_tool_call("send", args={"to": "a@b.com"})  # denylist empty → ok
        box["deny"] = {"a@b.com"}                          # file "updated"
        s.emit_tool_call("send", args={"to": "a@b.com"})  # now denied
    assert len([v for v in s.violations]) == 1


# ---------------------------------------------------------------------------
# tool_arg_value_guard — allow (fail-closed)
# ---------------------------------------------------------------------------

def test_allow_passes_when_listed():
    s = _run(tool_arg_value_guard("call", "region", allow=["us", "eu"]),
             [("call", {"region": "eu"})])
    assert s.is_compliant


def test_allow_blocks_when_not_listed():
    s = _run(tool_arg_value_guard("call", "region", allow=["us", "eu"]),
             [("call", {"region": "cn"})])
    assert not s.is_compliant


def test_allow_fails_closed_on_missing_field():
    s = _run(tool_arg_value_guard("call", "region", allow=["us"]),
             [("call", {"other": 1})])
    assert not s.is_compliant


# ---------------------------------------------------------------------------
# tool_arg_value_guard — dedupe
# ---------------------------------------------------------------------------

def test_dedupe_blocks_second_same_key():
    s = _run(tool_arg_value_guard("email", "to", dedupe_within_session=True),
             [("email", {"to": "a@b.com"}), ("email", {"to": "a@b.com"})])
    assert len(s.violations) == 1  # only the second trips


def test_dedupe_distinct_keys_both_pass():
    s = _run(tool_arg_value_guard("email", "to", dedupe_within_session=True),
             [("email", {"to": "a@b.com"}), ("email", {"to": "b@b.com"})])
    assert s.is_compliant


def test_dedupe_is_replay_deterministic():
    # Same recorded events → same verdict, regardless of how many times evaluated.
    pred = tool_arg_value_guard("email", "to", dedupe_within_session=True)
    calls = [("email", {"to": "a@b.com"}), ("email", {"to": "a@b.com"})]
    s1 = _run(pred, calls)
    s2 = _run(pred, calls)
    assert len(s1.violations) == len(s2.violations) == 1


# ---------------------------------------------------------------------------
# tool_arg_value_guard — scoping & validation
# ---------------------------------------------------------------------------

def test_tool_scoping():
    s = _run(tool_arg_value_guard("apply", "company", deny=["BadCo"]),
             [("other", {"company": "BadCo"})])  # different tool — ignored
    assert s.is_compliant


def test_none_tool_matches_any():
    s = _run(tool_arg_value_guard(None, "company", deny=["BadCo"]),
             [("whatever", {"company": "BadCo"})])
    assert not s.is_compliant


def test_deny_and_allow_rejected():
    with pytest.raises(ValueError):
        tool_arg_value_guard("t", "f", deny=["a"], allow=["b"])


def test_neither_rejected():
    with pytest.raises(ValueError):
        tool_arg_value_guard("t", "f")


def test_bad_match_rejected():
    with pytest.raises(ValueError):
        tool_arg_value_guard("t", "f", deny=["a"], match="fuzzy")


# ---------------------------------------------------------------------------
# required_disclosure
# ---------------------------------------------------------------------------

def test_disclosure_all_present_passes():
    pred = required_disclosure("send", "body", ["automated assistant", "on behalf of"])
    s = _run(pred, [("send", {"body": "I am an automated assistant writing on behalf of Acme."})])
    assert s.is_compliant


def test_disclosure_one_missing_fails_all():
    pred = required_disclosure("send", "body", ["automated assistant", "on behalf of"])
    s = _run(pred, [("send", {"body": "I am an automated assistant."})])
    assert not s.is_compliant


def test_disclosure_any_mode():
    pred = required_disclosure("send", "body", ["automated", "AI assistant"], match="any")
    s = _run(pred, [("send", {"body": "This is an automated note."})])
    assert s.is_compliant


def test_disclosure_fail_closed_on_missing_arg():
    pred = required_disclosure("send", "body", ["automated"])
    s = _run(pred, [("send", {"subject": "hi"})])
    assert not s.is_compliant


def test_disclosure_fail_closed_on_non_string():
    pred = required_disclosure("send", "body", ["automated"])
    s = _run(pred, [("send", {"body": 123})])
    assert not s.is_compliant


def test_disclosure_case_insensitive_default():
    pred = required_disclosure("send", "body", ["Automated"])
    s = _run(pred, [("send", {"body": "this is automated"})])
    assert s.is_compliant


def test_disclosure_case_sensitive():
    pred = required_disclosure("send", "body", ["Automated"], case_sensitive=True)
    s = _run(pred, [("send", {"body": "this is automated"})])
    assert not s.is_compliant


def test_disclosure_regex_pattern():
    pred = required_disclosure("send", "body", [r"on behalf of \w+"], pattern=True)
    s = _run(pred, [("send", {"body": "writing on behalf of Acme"})])
    assert s.is_compliant


def test_disclosure_wrong_tool_noop():
    pred = required_disclosure("send", "body", ["automated"])
    s = _run(pred, [("other", {"body": "no disclosure here"})])
    assert s.is_compliant


def test_disclosure_bad_match_rejected():
    with pytest.raises(ValueError):
        required_disclosure("send", "body", ["x"], match="some")


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------

def test_registered():
    import pactrun

    names = pactrun.list_predicates()
    assert "tool_arg_value_guard" in names
    assert "required_disclosure" in names
