"""Tests for cost predicates."""

import pytest
from pactrun import cost_under, cost_per_turn_under, token_budget, Contract


class TestCostUnder:
    def test_passes_under_budget(self):
        c = Contract("test").require(cost_under(1.0), on_fail="log")
        with c.session() as s:
            s.emit_llm_response(model="gpt-5.4-nano", output="hi", cost=0.05)
        assert s.is_compliant

    def test_fails_over_budget(self):
        c = Contract("test").require(cost_under(0.01), on_fail="log")
        with c.session() as s:
            s.emit_llm_response(model="gpt-5.4-nano", output="hi", cost=0.05)
        assert not s.is_compliant

    def test_cumulative_cost(self):
        c = Contract("test").require(cost_under(0.10), on_fail="log")
        with c.session() as s:
            s.emit_llm_response(model="gpt-5.4-nano", output="a", cost=0.04)
            s.emit_llm_response(model="gpt-5.4-nano", output="b", cost=0.04)
            s.emit_llm_response(model="gpt-5.4-nano", output="c", cost=0.04)
        assert not s.is_compliant  # 0.12 > 0.10

    def test_zero_cost_passes(self):
        c = Contract("test").require(cost_under(0.01), on_fail="log")
        with c.session() as s:
            s.emit_llm_response(model="gpt-5.4-nano", output="hi", cost=0.0)
        assert s.is_compliant

    def test_exact_limit_passes(self):
        c = Contract("test").require(cost_under(0.05), on_fail="log")
        with c.session() as s:
            s.emit_llm_response(model="gpt-5.4-nano", output="hi", cost=0.05)
        assert s.is_compliant


class TestCostPerTurnUnder:
    def test_passes(self):
        c = Contract("test").require(cost_per_turn_under(0.05), on_fail="log")
        with c.session() as s:
            s.emit_llm_response(model="gpt-5.4-nano", output="hi", cost=0.01)
        assert s.is_compliant

    def test_fails_on_expensive_turn(self):
        c = Contract("test").require(cost_per_turn_under(0.01), on_fail="log")
        with c.session() as s:
            s.emit_llm_response(model="gpt-5.4-nano", output="hi", cost=0.05)
        assert not s.is_compliant


class TestTokenBudget:
    def test_passes_under_budget(self):
        c = Contract("test").require(token_budget(1000), on_fail="log")
        with c.session() as s:
            s.emit_llm_response(model="gpt-5.4-nano", output="hi", prompt_tokens=50, completion_tokens=20)
        assert s.is_compliant

    def test_fails_over_budget(self):
        c = Contract("test").require(token_budget(100), on_fail="log")
        with c.session() as s:
            s.emit_llm_response(model="gpt-5.4-nano", output="hi", prompt_tokens=80, completion_tokens=50)
        assert not s.is_compliant

    def test_cumulative_tokens(self):
        c = Contract("test").require(token_budget(100), on_fail="log")
        with c.session() as s:
            s.emit_llm_response(model="gpt-5.4-nano", output="a", prompt_tokens=40, completion_tokens=10)
            s.emit_llm_response(model="gpt-5.4-nano", output="b", prompt_tokens=40, completion_tokens=20)
        assert not s.is_compliant  # 110 > 100
