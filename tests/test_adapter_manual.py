"""Tests for manual instrumentation adapter."""

import pytest

from pactrun import Contract, cost_under, must_not_call, ViolationError
from pactrun.adapters.manual import emit_llm_call, emit_tool_call


class TestManualEmit:
    def test_emit_llm_call_updates_session(self):
        c = Contract("test")
        with c.session() as session:
            emit_llm_call(model="gpt-5.4-nano", output="Hello", cost=0.01)

        assert session.state.total_llm_calls == 1
        assert session.state.total_cost_usd == 0.01

    def test_emit_tool_call_updates_session(self):
        c = Contract("test")
        with c.session() as session:
            emit_tool_call("search", args={"q": "test"})

        assert session.state.total_tool_calls == 1
        assert "search" in session.state.tool_call_history

    def test_emit_without_session_returns_empty(self):
        # No session active — should return empty list, not error
        violations = emit_llm_call(model="gpt-5.4-nano", output="Hello")
        assert violations == []

        violations = emit_tool_call("search")
        assert violations == []

    def test_violations_returned(self):
        c = Contract("test").require(cost_under(0.001), on_fail="log")
        with c.session() as session:
            violations = emit_llm_call(model="gpt-5.4-nano", output="Hello", cost=0.01)

        assert len(violations) > 0
        assert not session.is_compliant

    def test_forbidden_tool_via_manual(self):
        c = Contract("test").forbid(must_not_call("delete"), on_fail="log")
        with c.session() as session:
            violations = emit_tool_call("delete")

        assert len(violations) > 0
        assert not session.is_compliant

    def test_combined_usage(self):
        c = (
            Contract("test")
            .require(cost_under(0.10), on_fail="log")
            .forbid(must_not_call("drop_table"), on_fail="log")
        )
        with c.session() as session:
            emit_llm_call(model="gpt-5.4-nano", output="Planning...", cost=0.005)
            emit_tool_call("search", args={"q": "data"}, result={"found": True})
            emit_llm_call(model="gpt-5.4-nano", output="Done.", cost=0.003)

        assert session.is_compliant
        assert session.state.total_llm_calls == 2
        assert session.state.total_tool_calls == 1
