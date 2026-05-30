"""Output predicates — validate agent outputs."""

from __future__ import annotations

import re

from pactrun.core.enums import EventKind
from pactrun.core.models import Event, PredicateResult, SessionState
from pactrun.predicates.base import predicate


_PII_PATTERNS = [
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "email"),
    (r"\b\d{3}[-.]?\d{2}[-.]?\d{4}\b", "SSN"),
    (r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "phone"),
    (r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", "credit card"),
]


@predicate("no_pii")
def no_pii():
    """Output must not contain PII (email, SSN, phone, credit card)."""
    def check(event: Event, state: SessionState) -> PredicateResult:
        output = str(event.output or "")
        if not output:
            return PredicateResult(passed=True)
        for pattern, pii_type in _PII_PATTERNS:
            match = re.search(pattern, output)
            if match:
                return PredicateResult(
                    passed=False,
                    expected="no PII in output",
                    actual=f"Found {pii_type}: {match.group()[:20]}...",
                    message=f"Output contains {pii_type}",
                )
        return PredicateResult(passed=True)
    check.predicate_name = "no_pii"  # type: ignore[attr-defined]
    return check


@predicate("output_contains")
def output_contains(substring: str, case_sensitive: bool = True):
    """Output must contain this substring."""
    def check(event: Event, state: SessionState) -> PredicateResult:
        if not state.output_history:
            return PredicateResult(passed=False, message="No output to check")
        last_output = state.output_history[-1]
        if case_sensitive:
            passed = substring in last_output
        else:
            passed = substring.lower() in last_output.lower()
        return PredicateResult(
            passed=passed,
            expected=f"contains '{substring}'",
            actual=last_output[:100],
            message=f"Output does not contain '{substring}'",
        )
    check.predicate_name = "output_contains"  # type: ignore[attr-defined]
    check._check_on = "session_end"  # type: ignore[attr-defined]
    return check


@predicate("output_matches")
def output_matches(pattern: str):
    """Output must match regex pattern."""
    def check(event: Event, state: SessionState) -> PredicateResult:
        if not state.output_history:
            return PredicateResult(passed=False, message="No output to check")
        last_output = state.output_history[-1]
        passed = bool(re.search(pattern, last_output))
        return PredicateResult(
            passed=passed,
            expected=f"matches '{pattern}'",
            actual=last_output[:100],
            message=f"Output does not match pattern '{pattern}'",
        )
    check.predicate_name = "output_matches"  # type: ignore[attr-defined]
    check._check_on = "session_end"  # type: ignore[attr-defined]
    return check


@predicate("max_output_length")
def max_output_length(max_chars: int):
    """Output must not exceed character limit."""
    def check(event: Event, state: SessionState) -> PredicateResult:
        output = str(event.output or "")
        return PredicateResult(
            passed=len(output) <= max_chars,
            expected=f"<= {max_chars} chars",
            actual=f"{len(output)} chars",
            message=f"Output length {len(output)} exceeds limit {max_chars}",
        )
    check.predicate_name = "max_output_length"  # type: ignore[attr-defined]
    return check


@predicate("output_must_not_contain")
def output_must_not_contain(pattern: str):
    """Output must not match this regex pattern."""
    def check(event: Event, state: SessionState) -> PredicateResult:
        output = str(event.output or "")
        if not output:
            return PredicateResult(passed=True)
        match = re.search(pattern, output)
        if match:
            return PredicateResult(
                passed=False,
                expected=f"does not match '{pattern}'",
                actual=f"matched: {match.group()[:50]}",
                message=f"Output contains forbidden pattern '{pattern}'",
            )
        return PredicateResult(passed=True)
    check.predicate_name = "output_must_not_contain"  # type: ignore[attr-defined]
    return check
