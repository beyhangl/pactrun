"""Tests for pactrun Contract builder."""

import json
import pytest

from pactrun import Contract, ClauseKind, OnFail, Severity, PredicateResult


def _always_pass(event, state):
    return PredicateResult(passed=True)


def _always_fail(event, state):
    return PredicateResult(passed=False, message="always fails")


class TestContractBuilder:
    def test_create_empty(self):
        c = Contract("test")
        assert c.name == "test"
        assert c.version == "1.0"
        assert len(c.clauses) == 0

    def test_require(self):
        c = Contract("test").require(_always_pass, description="always pass")
        assert len(c.clauses) == 1
        assert c.clauses[0].kind == ClauseKind.REQUIRE

    def test_forbid(self):
        c = Contract("test").forbid(_always_fail, description="never do this")
        assert len(c.clauses) == 1
        assert c.clauses[0].kind == ClauseKind.FORBID
        assert c.clauses[0].severity == Severity.CRITICAL

    def test_fluent_chaining(self):
        c = (
            Contract("test")
            .require(_always_pass, description="a")
            .require(_always_pass, description="b")
            .forbid(_always_fail, description="c")
        )
        assert len(c.clauses) == 3

    def test_on_violation_sets_default(self):
        c = Contract("test").on_violation("warn")
        assert c.default_on_fail == OnFail.WARN

    def test_precondition(self):
        c = Contract("test").precondition(_always_pass, description="pre")
        assert c.clauses[0].kind == ClauseKind.PRECONDITION
        assert c.clauses[0].check_on == "session_start"

    def test_postcondition(self):
        c = Contract("test").postcondition(_always_pass, description="post")
        assert c.clauses[0].kind == ClauseKind.POSTCONDITION
        assert c.clauses[0].check_on == "session_end"

    def test_custom_severity(self):
        c = Contract("test").require(_always_pass, severity=Severity.WARNING)
        assert c.clauses[0].severity == Severity.WARNING

    def test_custom_on_fail(self):
        c = Contract("test").require(_always_pass, on_fail="log")
        assert c.clauses[0].on_fail == OnFail.LOG

    def test_get_clauses_by_kind(self):
        c = (
            Contract("test")
            .require(_always_pass)
            .forbid(_always_fail)
            .require(_always_pass)
        )
        requires = c.get_clauses(kind=ClauseKind.REQUIRE)
        assert len(requires) == 2

    def test_get_clauses_by_check_on(self):
        c = (
            Contract("test")
            .precondition(_always_pass)
            .require(_always_pass)
            .postcondition(_always_pass)
        )
        session_end = c.get_clauses(check_on="session_end")
        assert len(session_end) == 1


class TestContractSerialization:
    def test_to_dict(self):
        c = Contract("test", version="2.0").require(_always_pass, description="check")
        d = c.to_dict()
        assert d["name"] == "test"
        assert d["version"] == "2.0"
        assert len(d["clauses"]) == 1

    def test_save_and_load_json(self, tmp_path):
        c = Contract("test").require(_always_pass, description="check")
        path = c.save(tmp_path / "contract.json")
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["name"] == "test"


class TestContractEnforce:
    def test_enforce_decorator_sync(self):
        c = Contract("test")  # No clauses = always compliant

        @c.enforce
        def my_agent():
            return "hello"

        result = my_agent()
        assert result == "hello"

    def test_enforce_decorator_async(self):
        import asyncio
        c = Contract("test")

        @c.enforce
        async def my_agent():
            return "async hello"

        result = asyncio.run(my_agent())
        assert result == "async hello"
