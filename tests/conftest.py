"""Shared fixtures for pactrun tests."""

import pytest

from pactrun import Contract, Event, EventKind, PredicateResult, SessionState


@pytest.fixture
def simple_contract():
    """A contract with one require and one forbid clause."""
    def cost_check(event, state):
        return PredicateResult(
            passed=state.total_cost_usd <= 0.10,
            expected="<= $0.10",
            actual=f"${state.total_cost_usd:.4f}",
            message=f"Cost ${state.total_cost_usd:.4f} exceeds $0.10",
        )

    def no_delete(event, state):
        if event.kind == EventKind.TOOL_CALL and event.tool_name == "delete":
            return PredicateResult(passed=False, message="delete tool is forbidden")
        return PredicateResult(passed=True)

    return (
        Contract("test_agent", version="1.0")
        .require(cost_check, description="cost_under_0.10")
        .forbid(no_delete, description="no_delete", on_fail="log")
    )


@pytest.fixture
def empty_contract():
    return Contract("empty")


@pytest.fixture
def sample_llm_event():
    return Event(
        kind=EventKind.LLM_CALL,
        model="gpt-4.1-mini",
        output="Hello world",
        prompt_tokens=50,
        completion_tokens=10,
        cost_usd=0.003,
        duration_ms=200.0,
    )


@pytest.fixture
def sample_tool_event():
    return Event(
        kind=EventKind.TOOL_CALL,
        tool_name="search",
        tool_args={"q": "weather"},
        tool_result={"temp": 18},
        duration_ms=45.0,
    )
