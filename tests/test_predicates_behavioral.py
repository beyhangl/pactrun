"""Tests for behavioral predicates."""

import pytest
from pactrun import Contract, no_loops, max_retries, drift_bounds, no_repeated_output


class TestNoLoops:
    def test_no_loop_passes(self):
        c = Contract("test").require(no_loops(window=5, threshold=0.8), on_fail="log")
        with c.session() as s:
            for tool in ["a", "b", "c", "d", "e"]:
                s.emit_tool_call(tool)
        assert s.is_compliant

    def test_loop_detected(self):
        c = Contract("test").require(no_loops(window=5, threshold=0.8), on_fail="log")
        with c.session() as s:
            for _ in range(6):
                s.emit_tool_call("search")
        assert not s.is_compliant

    def test_short_history_passes(self):
        c = Contract("test").require(no_loops(window=5, threshold=0.8), on_fail="log")
        with c.session() as s:
            s.emit_tool_call("search")
        assert s.is_compliant


class TestMaxRetries:
    def test_under_limit_passes(self):
        c = Contract("test").require(max_retries(3), on_fail="log")
        with c.session() as s:
            s.emit_tool_call("search")
            s.emit_tool_call("search")
        assert s.is_compliant

    def test_over_limit_fails(self):
        c = Contract("test").require(max_retries(2), on_fail="log")
        with c.session() as s:
            s.emit_tool_call("search")
            s.emit_tool_call("search")
            s.emit_tool_call("search")
        assert not s.is_compliant

    def test_different_tools_reset(self):
        c = Contract("test").require(max_retries(2), on_fail="log")
        with c.session() as s:
            s.emit_tool_call("search")
            s.emit_tool_call("search")
            s.emit_tool_call("format")  # Breaks the streak
            s.emit_tool_call("search")
        assert s.is_compliant  # Only 1 consecutive search at the end

    def test_specific_tool(self):
        c = Contract("test").require(max_retries(2, tool="search"), on_fail="log")
        with c.session() as s:
            s.emit_tool_call("search")
            s.emit_tool_call("search")
            s.emit_tool_call("search")
        assert not s.is_compliant


class TestDriftBounds:
    def test_stable_cost_passes(self):
        c = Contract("test").require(drift_bounds(cost_pct=0.50), on_fail="log")
        with c.session() as s:
            for _ in range(5):
                s.emit_llm_response(model="gpt-5.4-nano", output="hi", cost=0.01)
        assert s.is_compliant

    def test_cost_spike_fails(self):
        c = Contract("test").require(drift_bounds(cost_pct=0.50), on_fail="log")
        with c.session() as s:
            s.emit_llm_response(model="gpt-5.4-nano", output="a", cost=0.01)
            s.emit_llm_response(model="gpt-5.4-nano", output="b", cost=0.01)
            s.emit_llm_response(model="gpt-5.4-nano", output="c", cost=0.01)
            s.emit_llm_response(model="gpt-5.4-nano", output="d", cost=0.10)  # 10x spike
        assert not s.is_compliant

    def test_few_turns_passes(self):
        c = Contract("test").require(drift_bounds(cost_pct=0.10), on_fail="log")
        with c.session() as s:
            s.emit_llm_response(model="gpt-5.4-nano", output="a", cost=0.01)
        assert s.is_compliant  # Too few turns to detect drift


class TestNoRepeatedOutput:
    def test_unique_outputs_pass(self):
        c = Contract("test").require(no_repeated_output(window=3), on_fail="log")
        with c.session() as s:
            s.emit_llm_response(model="gpt-5.4-nano", output="Hello")
            s.emit_llm_response(model="gpt-5.4-nano", output="World")
            s.emit_llm_response(model="gpt-5.4-nano", output="Foo")
        assert s.is_compliant

    def test_repeated_output_fails(self):
        c = Contract("test").require(no_repeated_output(window=3), on_fail="log")
        with c.session() as s:
            s.emit_llm_response(model="gpt-5.4-nano", output="Hello")
            s.emit_llm_response(model="gpt-5.4-nano", output="Hello")
        assert not s.is_compliant

    def test_single_output_passes(self):
        c = Contract("test").require(no_repeated_output(), on_fail="log")
        with c.session() as s:
            s.emit_llm_response(model="gpt-5.4-nano", output="Hello")
        assert s.is_compliant
