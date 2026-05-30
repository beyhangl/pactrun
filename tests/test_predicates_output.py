"""Tests for output predicates."""

import pytest
from pactrun import Contract, no_pii, output_contains, output_matches, max_output_length, output_must_not_contain


class TestNoPii:
    def test_clean_output_passes(self):
        c = Contract("test").require(no_pii(), on_fail="log")
        with c.session() as s:
            s.emit_llm_response(model="gpt-5.4-nano", output="The weather is sunny.")
        assert s.is_compliant

    def test_detects_email(self):
        c = Contract("test").require(no_pii(), on_fail="log")
        with c.session() as s:
            s.emit_llm_response(model="gpt-5.4-nano", output="Contact user@example.com for help")
        assert not s.is_compliant

    def test_detects_phone(self):
        c = Contract("test").require(no_pii(), on_fail="log")
        with c.session() as s:
            s.emit_llm_response(model="gpt-5.4-nano", output="Call 555-123-4567")
        assert not s.is_compliant

    def test_detects_ssn(self):
        c = Contract("test").require(no_pii(), on_fail="log")
        with c.session() as s:
            s.emit_llm_response(model="gpt-5.4-nano", output="SSN: 123-45-6789")
        assert not s.is_compliant

    def test_detects_credit_card(self):
        c = Contract("test").require(no_pii(), on_fail="log")
        with c.session() as s:
            s.emit_llm_response(model="gpt-5.4-nano", output="Card: 4111 1111 1111 1111")
        assert not s.is_compliant

    def test_empty_output_passes(self):
        c = Contract("test").require(no_pii(), on_fail="log")
        with c.session() as s:
            s.emit_llm_response(model="gpt-5.4-nano", output="")
        assert s.is_compliant


class TestOutputContains:
    def test_passes_when_found(self):
        c = Contract("test").postcondition(output_contains("Paris"), on_fail="log")
        with c.session() as s:
            s.emit_llm_response(model="gpt-5.4-nano", output="Paris is beautiful")
        assert s.is_compliant

    def test_fails_when_not_found(self):
        c = Contract("test").postcondition(output_contains("Tokyo"), on_fail="log")
        with c.session() as s:
            s.emit_llm_response(model="gpt-5.4-nano", output="Paris is beautiful")
        assert not s.is_compliant

    def test_case_insensitive(self):
        c = Contract("test").postcondition(output_contains("paris", case_sensitive=False), on_fail="log")
        with c.session() as s:
            s.emit_llm_response(model="gpt-5.4-nano", output="PARIS is great")
        assert s.is_compliant


class TestOutputMatches:
    def test_passes_when_matched(self):
        c = Contract("test").postcondition(output_matches(r"\d+°[CF]"), on_fail="log")
        with c.session() as s:
            s.emit_llm_response(model="gpt-5.4-nano", output="It's 18°C in Paris")
        assert s.is_compliant

    def test_fails_when_not_matched(self):
        c = Contract("test").postcondition(output_matches(r"\d+°[CF]"), on_fail="log")
        with c.session() as s:
            s.emit_llm_response(model="gpt-5.4-nano", output="It's warm in Paris")
        assert not s.is_compliant


class TestMaxOutputLength:
    def test_passes_under_limit(self):
        c = Contract("test").require(max_output_length(100), on_fail="log")
        with c.session() as s:
            s.emit_llm_response(model="gpt-5.4-nano", output="Short")
        assert s.is_compliant

    def test_fails_over_limit(self):
        c = Contract("test").require(max_output_length(5), on_fail="log")
        with c.session() as s:
            s.emit_llm_response(model="gpt-5.4-nano", output="This is way too long")
        assert not s.is_compliant


class TestOutputMustNotContain:
    def test_passes_when_no_match(self):
        c = Contract("test").require(output_must_not_contain(r"password"), on_fail="log")
        with c.session() as s:
            s.emit_llm_response(model="gpt-5.4-nano", output="Your account is active")
        assert s.is_compliant

    def test_fails_when_matched(self):
        c = Contract("test").require(output_must_not_contain(r"password"), on_fail="log")
        with c.session() as s:
            s.emit_llm_response(model="gpt-5.4-nano", output="Your password is abc123")
        assert not s.is_compliant
