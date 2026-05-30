"""Tests for pactrun core data models."""

import pytest

from pactrun.core.enums import ClauseKind, EventKind, OnFail, Severity
from pactrun.core.models import (
    Clause, Event, PredicateResult, SessionState, SessionSummary, Violation,
)


class TestEvent:
    def test_default_values(self):
        e = Event()
        assert e.kind == EventKind.LLM_CALL
        assert e.cost_usd == 0.0
        assert e.id  # UUID generated

    def test_llm_event(self, sample_llm_event):
        assert sample_llm_event.model == "gpt-4.1-mini"
        assert sample_llm_event.cost_usd == 0.003

    def test_tool_event(self, sample_tool_event):
        assert sample_tool_event.tool_name == "search"
        assert sample_tool_event.tool_args == {"q": "weather"}

    def test_to_dict_roundtrip(self, sample_llm_event):
        d = sample_llm_event.to_dict()
        restored = Event.from_dict(d)
        assert restored.model == sample_llm_event.model
        assert restored.cost_usd == sample_llm_event.cost_usd

    def test_unique_ids(self):
        e1 = Event()
        e2 = Event()
        assert e1.id != e2.id


class TestPredicateResult:
    def test_default_passes(self):
        r = PredicateResult()
        assert r.passed is True

    def test_failure_with_message(self):
        r = PredicateResult(passed=False, message="too expensive")
        assert not r.passed
        assert "expensive" in r.message

    def test_to_dict(self):
        r = PredicateResult(passed=True, expected="< $1", actual="$0.50")
        d = r.to_dict()
        assert d["passed"] is True
        assert d["expected"] == "< $1"


class TestSessionState:
    def test_default_values(self):
        s = SessionState()
        assert s.total_cost_usd == 0.0
        assert s.turn_number == 0
        assert s.tool_call_history == []

    def test_to_dict(self):
        s = SessionState(total_cost_usd=0.05, turn_number=3)
        d = s.to_dict()
        assert d["total_cost_usd"] == 0.05
        assert d["turn_number"] == 3


class TestViolation:
    def test_creation(self):
        v = Violation(
            clause_id="c1",
            clause_description="cost limit",
            severity=Severity.ERROR,
            message="Cost exceeded",
        )
        assert v.severity == Severity.ERROR
        assert v.message == "Cost exceeded"

    def test_to_dict_roundtrip(self):
        v = Violation(clause_id="c1", severity=Severity.CRITICAL, message="Bad")
        d = v.to_dict()
        restored = Violation.from_dict(d)
        assert restored.clause_id == "c1"
        assert restored.severity == Severity.CRITICAL


class TestSessionSummary:
    def test_is_compliant_with_no_violations(self):
        s = SessionSummary()
        assert s.is_compliant
        assert s.violation_count == 0

    def test_has_critical(self):
        s = SessionSummary(violations=[
            Violation(severity=Severity.WARNING),
            Violation(severity=Severity.CRITICAL),
        ])
        assert s.has_critical
        assert s.has_errors

    def test_violations_by_severity(self):
        s = SessionSummary(violations=[
            Violation(severity=Severity.WARNING),
            Violation(severity=Severity.ERROR),
            Violation(severity=Severity.WARNING),
        ])
        warnings = s.violations_by_severity(Severity.WARNING)
        assert len(warnings) == 2

    def test_to_dict_roundtrip(self):
        s = SessionSummary(
            session_id="s1",
            contract_name="test",
            total_cost_usd=0.05,
            violations=[Violation(severity=Severity.ERROR, message="bad")],
        )
        d = s.to_dict()
        restored = SessionSummary.from_dict(d)
        assert restored.session_id == "s1"
        assert restored.violation_count == 1
