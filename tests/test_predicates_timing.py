"""Tests for timing predicates."""

import pytest
from pactrun import Contract, max_latency, session_timeout, max_turns


class TestMaxLatency:
    def test_passes_under_limit(self):
        c = Contract("test").require(max_latency(1000), on_fail="log")
        with c.session() as s:
            s.emit_llm_response(model="gpt-5.4-nano", output="hi", duration_ms=200)
        assert s.is_compliant

    def test_fails_over_limit(self):
        c = Contract("test").require(max_latency(100), on_fail="log")
        with c.session() as s:
            s.emit_llm_response(model="gpt-5.4-nano", output="hi", duration_ms=500)
        assert not s.is_compliant

    def test_zero_duration_passes(self):
        c = Contract("test").require(max_latency(100), on_fail="log")
        with c.session() as s:
            s.emit_llm_response(model="gpt-5.4-nano", output="hi", duration_ms=0)
        assert s.is_compliant


class TestMaxTurns:
    def test_under_limit_passes(self):
        c = Contract("test").require(max_turns(5), on_fail="log")
        with c.session() as s:
            s.advance_turn()
            s.advance_turn()
        assert s.is_compliant

    def test_over_limit_fails(self):
        c = Contract("test").require(max_turns(2), on_fail="log")
        with c.session() as s:
            s.advance_turn()
            s.advance_turn()
            s.advance_turn()
        assert not s.is_compliant
