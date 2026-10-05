"""Tests for the Pydantic AI adapter (PactrunCapability) with offline models only.

``TestModel`` / ``FunctionModel`` never touch the network, so these runs cost
nothing and are deterministic.
"""

import dataclasses
import os

import pytest

pytest.importorskip("pydantic_ai")

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

from decimal import Decimal

from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RequestUsage

from pactrun import (
    Contract,
    EventKind,
    PredicateResult,
    ViolationError,
    cost_under,
    must_not_call,
    no_destructive_args,
    token_budget,
    tools_allowed,
)
from pactrun.adapters import PactrunCapability
from pactrun.core.errors import SessionError
from pactrun.session import get_active_session


def _scripted(calls, *, usage=None, seen=None, model_name=None):
    """FunctionModel that makes ``calls`` [(tool, args), ...] once, then answers text."""

    def fn(messages, info):
        if seen is not None:
            seen.append(messages)
        kwargs = {"usage": usage} if usage is not None else {}
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart(name, args) for name, args in calls], **kwargs)
        return ModelResponse(parts=[TextPart("done")], **kwargs)

    return FunctionModel(fn, model_name=model_name)


def _kinds(session):
    return [e.kind for e in session.state.events]


def test_normal_run_records_llm_and_tool_events():
    cap = PactrunCapability(Contract("t"))
    agent = Agent(TestModel(), capabilities=[cap])

    @agent.tool_plain
    def lookup(city: str) -> str:
        return f"sunny in {city}"

    agent.run_sync("weather?")

    session = cap.last_session
    assert session is not None and not session.is_active  # per-run session was closed
    assert _kinds(session) == [EventKind.LLM_CALL, EventKind.TOOL_CALL, EventKind.LLM_CALL]
    first_llm, tool, last_llm = session.state.events
    assert first_llm.model == "test"
    assert first_llm.prompt_tokens > 0 and first_llm.completion_tokens > 0
    assert first_llm.metadata["tool_calls"] == ["lookup"]
    assert first_llm.metadata["cost_source"] == "unavailable"  # the test model has no price
    assert tool.tool_name == "lookup"
    assert tool.tool_args == {"city": "a"}
    assert tool.tool_result == "sunny in a"
    assert tool.error is None
    assert session.state.total_tokens == sum(
        e.prompt_tokens + e.completion_tokens for e in (first_llm, last_llm)
    )
    assert session.is_compliant
    assert get_active_session() is None


def test_must_not_call_blocks_before_tool_runs():
    ran = []
    cap = PactrunCapability(Contract("t").forbid(must_not_call("delete_file")))
    agent = Agent(_scripted([("delete_file", {"path": "/etc/hosts"})]), capabilities=[cap])

    @agent.tool_plain
    def delete_file(path: str) -> str:
        ran.append(path)
        return "deleted"

    with pytest.raises(ViolationError, match="delete_file"):
        agent.run_sync("clean up")

    assert ran == []
    assert [v.predicate_name for v in cap.last_session.violations] == ["must_not_call"]
    assert cap.last_session.violations[0].enforced is True


def test_tools_allowed_blocks_unlisted_tool():
    ran = []
    cap = PactrunCapability(Contract("t").require(tools_allowed(["search"])))
    agent = Agent(_scripted([("send_email", {"to": "x@example.com"})]), capabilities=[cap])

    @agent.tool_plain
    def search(q: str) -> str:
        return q

    @agent.tool_plain
    def send_email(to: str) -> str:
        ran.append(to)
        return "sent"

    with pytest.raises(ViolationError, match="send_email"):
        agent.run_sync("go")
    assert ran == []


def test_no_destructive_args_sees_the_real_args():
    ran = []
    cap = PactrunCapability(Contract("t").forbid(no_destructive_args()))
    agent = Agent(_scripted([("shell", {"cmd": "rm -rf /"})]), capabilities=[cap])

    @agent.tool_plain
    def shell(cmd: str) -> str:
        ran.append(cmd)
        return "ok"

    with pytest.raises(ViolationError, match="destructive"):
        agent.run_sync("go")
    assert ran == []
    event = cap.last_session.state.events[-1]
    assert event.tool_args == {"cmd": "rm -rf /"}


def test_tool_args_match_validates_real_args():
    pytest.importorskip("jsonschema")
    from pactrun import tool_args_match

    schema = {"type": "object", "properties": {"amount": {"type": "number", "maximum": 100}}}
    ran = []
    cap = PactrunCapability(Contract("t").require(tool_args_match("refund", schema)))
    agent = Agent(_scripted([("refund", {"amount": 5000})]), capabilities=[cap])

    @agent.tool_plain
    def refund(amount: float) -> str:
        ran.append(amount)
        return "refunded"

    with pytest.raises(ViolationError, match="refund"):
        agent.run_sync("refund me")
    assert ran == []


def test_args_are_the_validated_args_the_tool_receives():
    seen_args = []

    def spy(event, state):
        if event.kind == EventKind.TOOL_CALL:
            seen_args.append(dict(event.tool_args))
        return PredicateResult(passed=True)

    cap = PactrunCapability(Contract("t").require(spy, description="spy"))
    received = []
    agent = Agent(_scripted([("add", '{"a": "2", "b": 3}')]), capabilities=[cap])

    @agent.tool_plain
    def add(a: int, b: int) -> int:
        received.append((a, b))
        return a + b

    agent.run_sync("add")
    assert received == [(2, 3)]
    assert seen_args == [{"a": 2, "b": 3}]  # validated + coerced, not the raw JSON string


def test_cost_under_fires_from_reported_usage_cost():
    if "cost" not in {f.name for f in dataclasses.fields(RequestUsage)}:
        pytest.skip("RequestUsage.cost does not exist in this pydantic-ai version")
    usage = RequestUsage(input_tokens=10, output_tokens=5, cost=Decimal("0.25"))
    cap = PactrunCapability(Contract("t").require(cost_under(0.10)))
    agent = Agent(_scripted([], usage=usage), capabilities=[cap])

    with pytest.raises(ViolationError, match="exceeds"):
        agent.run_sync("hi")
    event = cap.last_session.state.events[0]
    assert event.cost_usd == pytest.approx(0.25)
    assert event.metadata["cost_source"] == "pydantic_ai"


def test_cost_falls_back_to_pydantic_ai_pricing():
    usage = RequestUsage(input_tokens=1_000, output_tokens=500)
    cap = PactrunCapability(Contract("t"))

    def fn(messages, info):
        return ModelResponse(parts=[TextPart("hi")], usage=usage, provider_name="openai")

    Agent(FunctionModel(fn, model_name="gpt-4o"), capabilities=[cap]).run_sync("hi")
    event = cap.last_session.state.events[0]
    # Newer Pydantic AI fills usage.cost before the hook runs; older versions
    # leave it to ModelResponse.cost(). Either way the model gets priced.
    assert event.metadata["cost_source"] in ("pydantic_ai", "genai_prices")
    assert event.cost_usd > 0


def test_token_budget_fires_before_the_requested_tool_runs():
    ran = []
    usage = RequestUsage(input_tokens=900, output_tokens=200)
    cap = PactrunCapability(Contract("t").require(token_budget(1_000)))
    agent = Agent(_scripted([("search", {"q": "x"})], usage=usage), capabilities=[cap])

    @agent.tool_plain
    def search(q: str) -> str:
        ran.append(q)
        return "r"

    with pytest.raises(ViolationError, match="1,?100|token"):
        agent.run_sync("hi")
    assert ran == []
    assert cap.last_session.state.total_tokens == 1_100


def test_monitor_mode_records_but_does_not_block():
    ran = []
    cap = PactrunCapability(Contract("t").forbid(must_not_call("delete_file")).monitor())
    agent = Agent(_scripted([("delete_file", {"path": "a"})]), capabilities=[cap])

    @agent.tool_plain
    def delete_file(path: str) -> str:
        ran.append(path)
        return "deleted"

    result = agent.run_sync("go")
    assert result.output == "done"
    assert ran == ["a"]
    [violation] = cap.last_session.violations
    assert violation.enforced is False
    assert not cap.last_session.is_compliant


def test_monitor_mode_via_session_kwargs():
    ran = []
    cap = PactrunCapability(
        Contract("t").forbid(must_not_call("delete_file")), session_kwargs={"mode": "monitor"}
    )
    agent = Agent(_scripted([("delete_file", {"path": "a"})]), capabilities=[cap])

    @agent.tool_plain
    def delete_file(path: str) -> str:
        ran.append(path)
        return "deleted"

    agent.run_sync("go")
    assert ran == ["a"]
    assert cap.last_session.violations[0].enforced is False


def test_return_to_model_skips_tool_and_tells_the_model():
    ran = []
    seen = []
    cap = PactrunCapability(
        Contract("t").forbid(must_not_call("delete_file")), on_tool_block="return_to_model"
    )
    agent = Agent(_scripted([("delete_file", {"path": "a"})], seen=seen), capabilities=[cap])

    @agent.tool_plain
    def delete_file(path: str) -> str:
        ran.append(path)
        return "deleted"

    result = agent.run_sync("go")
    assert result.output == "done"
    assert ran == []
    returns = [p for m in seen[-1] for p in getattr(m, "parts", []) if isinstance(p, ToolReturnPart)]
    assert returns and "pactrun blocked tool 'delete_file'" in str(returns[0].content)
    assert len(cap.last_session.violations) == 1


def test_active_session_fallback_sync():
    ran = []
    contract = Contract("t").forbid(must_not_call("delete_file"))
    agent = Agent(_scripted([("delete_file", {"path": "a"})]), capabilities=[PactrunCapability()])

    @agent.tool_plain
    def delete_file(path: str) -> str:
        ran.append(path)
        return "deleted"

    with contract.session() as session:
        with pytest.raises(ViolationError):
            agent.run_sync("go")
    assert ran == []
    assert [e.kind for e in session.state.events] == [EventKind.LLM_CALL, EventKind.TOOL_CALL]
    assert len(session.violations) == 1


async def test_active_session_fallback_async_accumulates_across_runs():
    agent = Agent(TestModel(), capabilities=[PactrunCapability()])

    @agent.tool_plain
    def lookup(city: str) -> str:
        assert get_active_session() is not None  # the session is ambient inside tools too
        return city

    async with Contract("t").session() as session:
        await agent.run("one")
        await agent.run("two")
    assert session.state.total_tool_calls == 2
    assert session.state.total_llm_calls == 4


def test_no_contract_and_no_active_session_refuses_the_run():
    ran = []
    agent = Agent(TestModel(), capabilities=[PactrunCapability()])

    @agent.tool_plain
    def lookup(city: str) -> str:
        ran.append(city)
        return city

    with pytest.raises(SessionError, match="no pactrun session is active"):
        agent.run_sync("go")
    assert ran == []


async def test_async_run_blocks_and_closes_session():
    ran = []
    cap = PactrunCapability(Contract("t").forbid(must_not_call("delete_file")))
    agent = Agent(_scripted([("delete_file", {"path": "a"})]), capabilities=[cap])

    @agent.tool_plain
    async def delete_file(path: str) -> str:
        ran.append(path)
        return "deleted"

    with pytest.raises(ViolationError):
        await agent.run("go")
    assert ran == []
    assert not cap.last_session.is_active
    assert get_active_session() is None


async def test_run_stream_records_and_blocks():
    cap = PactrunCapability(Contract("t"))
    agent = Agent(TestModel(), capabilities=[cap])

    @agent.tool_plain
    def lookup(city: str) -> str:
        return city

    async with agent.run_stream("x") as result:
        await result.get_output()
    assert _kinds(cap.last_session) == [EventKind.LLM_CALL, EventKind.TOOL_CALL, EventKind.LLM_CALL]

    ran = []
    cap = PactrunCapability(Contract("t").forbid(must_not_call("lookup")))
    agent = Agent(TestModel(), capabilities=[cap])

    @agent.tool_plain
    def lookup(city: str) -> str:  # noqa: F811
        ran.append(city)
        return city

    with pytest.raises(ViolationError):
        async with agent.run_stream("x") as result:
            await result.get_output()
    assert ran == []


def test_each_run_gets_a_fresh_session():
    cap = PactrunCapability(Contract("t"))
    agent = Agent(TestModel(), capabilities=[cap])
    agent.run_sync("one")
    first = cap.last_session
    agent.run_sync("two")
    assert cap.last_session is not first
    assert cap.last_session.state.total_llm_calls == 1


def test_explicit_session_is_used_as_is():
    contract = Contract("t")
    session = contract.session()
    agent = Agent(TestModel(), capabilities=[PactrunCapability(session)])
    agent.run_sync("one")
    agent.run_sync("two")
    assert session.state.total_llm_calls == 2


def test_tool_error_is_back_filled_on_the_event():
    cap = PactrunCapability(Contract("t"))
    agent = Agent(_scripted([("flaky", {"x": 1})]), capabilities=[cap])

    @agent.tool_plain
    def flaky(x: int) -> str:
        raise RuntimeError("backend down")

    with pytest.raises(RuntimeError, match="backend down"):
        agent.run_sync("go")
    tool = next(e for e in cap.last_session.state.events if e.kind == EventKind.TOOL_CALL)
    assert tool.error == "RuntimeError: backend down"


def test_unexpected_hook_failure_fails_closed():
    class BrokenObserver:
        def on_event(self, event, state):
            if event.kind == EventKind.TOOL_CALL:
                raise OSError("audit sink unavailable")

    ran = []
    cap = PactrunCapability(Contract("t"), session_kwargs={"observers": [BrokenObserver()]})
    agent = Agent(_scripted([("search", {"q": "x"})]), capabilities=[cap])

    @agent.tool_plain
    def search(q: str) -> str:
        ran.append(q)
        return "r"

    with pytest.raises(RuntimeError, match="fail-closed"):
        agent.run_sync("go")
    assert ran == []


def test_invalid_options_rejected():
    with pytest.raises(ValueError, match="on_tool_block"):
        PactrunCapability(Contract("t"), on_tool_block="ignore")
    with pytest.raises(ValueError, match="session_kwargs"):
        PactrunCapability(session_kwargs={"mode": "monitor"})
