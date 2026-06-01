"""Regression tests for session-end predicate resolution.

Predicates like ``must_call`` / ``tool_order`` / ``output_contains`` can only
be satisfied once the whole session has run, so they tag their checker with
``_check_on = "session_end"``. Previously ``Contract.require()`` / ``forbid()``
and the YAML loader ignored that hint and evaluated such clauses on *every*
event, so they failed on the very first event — before the required tool could
ever be called. These tests exercise the real ``.require()`` and loader paths
(not the ``.postcondition()`` workaround the older tests used) to guard the fix.
"""

import pytest

from pactrun import (
    Contract,
    ViolationError,
    must_call,
    tool_order,
    output_contains,
    tools_allowed,
    must_not_call,
)


class TestRequireHonorsCheckOn:
    def test_require_must_call_resolves_to_session_end(self):
        c = Contract("t").require(must_call("search"), on_fail="log")
        assert c.clauses[0].check_on == "session_end"

    def test_must_call_via_require_passes_when_eventually_called(self):
        c = Contract("t").require(must_call("search"), on_fail="log")
        with c.session() as s:
            s.emit_llm_response(model="m", output="thinking...")  # first event, no tool yet
            s.emit_tool_call("search")
        assert s.is_compliant

    def test_must_call_via_require_fails_when_never_called(self):
        c = Contract("t").require(must_call("search"), on_fail="log")
        with c.session() as s:
            s.emit_tool_call("other")
        assert not s.is_compliant

    def test_require_does_not_block_on_first_event(self):
        # Regression: the old code raised ViolationError on the first event
        # (before 'search' was ever called) because check_on was every_event.
        c = Contract("t").require(must_call("search"), on_fail="block")
        with c.session() as s:
            s.emit_llm_response(model="m", output="working on it")
            s.emit_tool_call("search")
        assert s.is_compliant

    def test_tool_order_via_require_passes_in_order(self):
        c = Contract("t").require(tool_order(["search", "format"]), on_fail="log")
        with c.session() as s:
            s.emit_tool_call("search")
            s.emit_tool_call("format")
        assert s.is_compliant

    def test_output_contains_via_require_passes(self):
        c = Contract("t").require(output_contains("done"), on_fail="log")
        with c.session() as s:
            s.emit_output("all done")
        assert s.is_compliant


class TestEveryEventPredicatesUnchanged:
    def test_tools_allowed_stays_every_event(self):
        c = Contract("t").require(tools_allowed(["search"]), on_fail="log")
        assert c.clauses[0].check_on == "every_event"

    def test_must_not_call_stays_every_event(self):
        c = Contract("t").forbid(must_not_call("delete"), on_fail="log")
        assert c.clauses[0].check_on == "every_event"


class TestExplicitCheckOnOverride:
    def test_explicit_check_on_wins_over_hint(self):
        c = Contract("t").require(must_call("x"), check_on="every_event", on_fail="log")
        assert c.clauses[0].check_on == "every_event"


class TestLoaderHonorsCheckOn:
    def test_yaml_require_must_call_resolves_session_end(self):
        data = {
            "name": "t",
            "clauses": [
                {"require": "must_call", "args": {"tool": "search"}, "on_fail": "log"},
            ],
        }
        c = Contract.from_dict(data)
        assert c.clauses[0].check_on == "session_end"

    def test_yaml_require_must_call_passes_when_called(self):
        data = {
            "name": "t",
            "clauses": [
                {"require": "must_call", "args": {"tool": "search"}, "on_fail": "log"},
            ],
        }
        c = Contract.from_dict(data)
        with c.session() as s:
            s.emit_llm_response(model="m", output="x")
            s.emit_tool_call("search")
        assert s.is_compliant

    def test_yaml_explicit_check_on_wins(self):
        data = {
            "name": "t",
            "clauses": [
                {
                    "require": "must_call",
                    "args": {"tool": "search"},
                    "check_on": "every_event",
                    "on_fail": "log",
                },
            ],
        }
        c = Contract.from_dict(data)
        assert c.clauses[0].check_on == "every_event"

    def test_yaml_precondition_resolves_session_start(self):
        data = {
            "name": "t",
            "clauses": [
                {"precondition": "tools_allowed", "args": {"whitelist": ["search"]}},
            ],
        }
        c = Contract.from_dict(data)
        assert c.clauses[0].check_on == "session_start"
