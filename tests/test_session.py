"""Tests for pactrun Session runtime enforcement."""

import pytest

from pactrun import (
    Contract, Event, EventKind, OnFail, PredicateResult, Severity, ViolationError,
)
from pactrun.session import Session, get_active_session


def _cost_check(max_usd):
    def check(event, state):
        return PredicateResult(
            passed=state.total_cost_usd <= max_usd,
            expected=f"<= ${max_usd}",
            actual=f"${state.total_cost_usd:.4f}",
            message=f"Cost exceeds ${max_usd}",
        )
    return check


def _tool_forbidden(tool_name):
    def check(event, state):
        if event.kind == EventKind.TOOL_CALL and event.tool_name == tool_name:
            return PredicateResult(passed=False, message=f"{tool_name} is forbidden")
        return PredicateResult(passed=True)
    return check


def _must_call_tool(tool_name):
    def check(event, state):
        return PredicateResult(
            passed=tool_name in state.tool_call_history,
            message=f"{tool_name} was not called",
        )
    return check


class TestSessionLifecycle:
    def test_context_manager(self, empty_contract):
        with empty_contract.session() as s:
            assert s.is_active
        assert not s.is_active

    def test_async_context_manager(self, empty_contract):
        import asyncio

        async def run():
            async with empty_contract.session() as s:
                assert s.is_active
            return s

        s = asyncio.run(run())
        assert not s.is_active

    def test_sets_active_session(self, empty_contract):
        assert get_active_session() is None
        with empty_contract.session() as s:
            assert get_active_session() is s
        assert get_active_session() is None

    def test_session_id_unique(self, empty_contract):
        with empty_contract.session() as s1:
            pass
        with empty_contract.session() as s2:
            pass
        assert s1.session_id != s2.session_id


class TestSessionEventTracking:
    def test_llm_response_updates_state(self, empty_contract):
        with empty_contract.session() as s:
            s.emit_llm_response(model="gpt-5.4-nano", output="hi", cost=0.005, prompt_tokens=50, completion_tokens=10)

        assert s.state.total_cost_usd == 0.005
        assert s.state.total_tokens == 60
        assert s.state.total_llm_calls == 1

    def test_tool_call_updates_state(self, empty_contract):
        with empty_contract.session() as s:
            s.emit_tool_call("search", args={"q": "test"})
            s.emit_tool_call("format")

        assert s.state.total_tool_calls == 2
        assert s.state.tool_call_history == ["search", "format"]

    def test_advance_turn(self, empty_contract):
        with empty_contract.session() as s:
            s.advance_turn()
            s.advance_turn()

        assert s.state.turn_number == 2

    def test_output_recorded(self, empty_contract):
        with empty_contract.session() as s:
            s.emit_output("Final answer")

        assert "Final answer" in s.state.output_history


class TestSessionEnforcement:
    def test_require_clause_passes(self):
        c = Contract("test").require(_cost_check(1.0), description="budget")
        with c.session() as s:
            s.emit_llm_response(model="gpt-5.4-nano", output="hi", cost=0.01)
        assert s.is_compliant
        assert s.violation_count == 0

    def test_require_clause_fails_log(self):
        c = Contract("test").require(_cost_check(0.001), description="tiny budget", on_fail="log")
        with c.session() as s:
            s.emit_llm_response(model="gpt-5.4-nano", output="hi", cost=0.01)
        assert not s.is_compliant
        assert s.violation_count == 1
        assert "exceeds" in s.violations[0].message

    def test_forbid_clause_blocks(self):
        c = Contract("test").forbid(_tool_forbidden("delete"), description="no delete", on_fail="block")
        with pytest.raises(ViolationError, match="forbidden"):
            with c.session() as s:
                s.emit_tool_call("delete")

    def test_forbid_clause_log_mode(self):
        c = Contract("test").forbid(_tool_forbidden("delete"), description="no delete", on_fail="log")
        with c.session() as s:
            s.emit_tool_call("delete")
        assert s.violation_count == 1

    def test_postcondition_checked_at_end(self):
        c = Contract("test").postcondition(_must_call_tool("search"), description="must search", on_fail="log")
        with c.session() as s:
            pass  # Never called search
        assert not s.is_compliant
        assert any("search" in v.message for v in s.violations)

    def test_postcondition_passes(self):
        c = Contract("test").postcondition(
            _must_call_tool("search"), description="must search", on_fail="log"
        )
        with c.session() as s:
            s.emit_tool_call("search")
        assert s.is_compliant

    def test_multiple_violations(self):
        c = (
            Contract("test")
            .require(_cost_check(0.001), description="budget", on_fail="log")
            .forbid(_tool_forbidden("delete"), description="no delete", on_fail="log")
        )
        with c.session() as s:
            s.emit_llm_response(model="gpt-5.4-nano", output="hi", cost=0.01)
            s.emit_tool_call("delete")
        assert s.violation_count >= 2  # cost check fires on llm + tool events


class TestSessionSummary:
    def test_summary_generation(self, simple_contract):
        with simple_contract.session() as s:
            s.emit_llm_response(model="gpt-5.4-nano", output="hi", cost=0.003)
            s.emit_tool_call("search")
        summary = s.summary()
        assert summary.contract_name == "test_agent"
        assert summary.total_llm_calls == 1
        assert summary.total_tool_calls == 1
        assert summary.is_compliant

    def test_summary_serialization(self, simple_contract):
        with simple_contract.session() as s:
            s.emit_llm_response(model="gpt-5.4-nano", output="hi", cost=0.003)
        summary = s.summary()
        d = summary.to_dict()
        assert d["contract_name"] == "test_agent"
        assert isinstance(d["violations"], list)
