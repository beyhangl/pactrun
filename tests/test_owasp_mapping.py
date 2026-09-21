"""Tests for the OWASP Agentic Top-10 (2026) predicate mapping."""

import pytest

import pactrun
from pactrun.predicates.base import (
    OWASP_AGENTIC_2026,
    owasp_coverage,
    predicate,
    predicate_owasp,
)


def test_catalog_has_ten_risks_with_expected_ids():
    assert len(OWASP_AGENTIC_2026) == 10
    assert list(OWASP_AGENTIC_2026) == [f"ASI{i:02d}" for i in range(1, 11)]


def test_every_risk_id_present_in_coverage():
    coverage = owasp_coverage()
    assert set(coverage) == set(OWASP_AGENTIC_2026)


def test_tagged_predicates_use_valid_ids():
    valid = set(OWASP_AGENTIC_2026)
    for name in pactrun.list_predicates():
        for rid in predicate_owasp(name):
            assert rid in valid, f"{name} tagged with unknown id {rid}"


def test_unknown_id_is_rejected_at_registration():
    with pytest.raises(ValueError, match="unknown OWASP ids"):
        @predicate("bogus_predicate_for_test", owasp=("ASI99",))
        def _bogus():  # pragma: no cover - registration raises first
            ...


def test_coverage_is_consistent_with_tags():
    coverage = owasp_coverage()
    for name in pactrun.list_predicates():
        for rid in predicate_owasp(name):
            assert name in coverage[rid]


def test_untagged_predicate_returns_empty():
    # A predicate with no security mapping (e.g. output shape) reports nothing.
    assert predicate_owasp("valid_json") == ()


def test_substantial_coverage_exists():
    coverage = owasp_coverage()
    covered = [rid for rid, preds in coverage.items() if preds]
    assert len(covered) >= 8, f"expected broad coverage, got {covered}"


def test_key_security_predicates_are_tagged():
    expected = {
        "no_injection_phrases": "ASI01",
        "tools_allowed": "ASI02",
        "consent_token_required": "ASI03",
        "no_destructive_args": "ASI05",
        "untrusted_taint_to_sink": "ASI06",
        "tool_host_within": "ASI07",
        "no_loops": "ASI08",
        "ai_disclosure_in_output": "ASI09",
    }
    for name, rid in expected.items():
        assert rid in predicate_owasp(name), f"{name} should be tagged {rid}"


def test_predicate_exposes_owasp_attribute():
    from pactrun import tools_allowed

    assert "ASI02" in tools_allowed.owasp


def test_uncovered_risks_are_honestly_empty():
    # ASI10 (rogue agents) is an identity/orchestration concern a runtime
    # contract library cannot observe — it must not be fake-tagged. ASI04 is
    # covered only for the part we can actually see at runtime: a tool server
    # mutating what it advertises mid-run. Package/registry compromise itself
    # still happens outside the process.
    coverage = owasp_coverage()
    assert coverage["ASI10"] == []
    assert coverage["ASI04"] == ["tool_definitions_stable"]
