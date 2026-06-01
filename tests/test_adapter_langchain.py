"""Tests for the LangChain / LangGraph adapter (PactrunCallbackHandler)."""

import pytest

pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, Generation, LLMResult

from pactrun import Contract, cost_under
from pactrun.adapters import PactrunCallbackHandler


def _result_with_token_usage():
    return LLMResult(
        generations=[[Generation(text="Hello from the graph")]],
        llm_output={
            "model_name": "gpt-4.1",
            "token_usage": {"prompt_tokens": 30, "completion_tokens": 12},
        },
    )


class TestLangChainAdapter:
    def test_emits_llm_event_from_token_usage(self):
        handler = PactrunCallbackHandler()
        with Contract("t").session() as s:
            handler.on_llm_start({"name": "m"}, ["prompt"], run_id="r1")
            handler.on_llm_end(_result_with_token_usage(), run_id="r1")
        assert s.state.total_llm_calls == 1
        assert s.state.total_tokens == 42
        assert "Hello from the graph" in s.state.output_history
        assert s.state.total_cost_usd > 0  # gpt-4.1 is priced

    def test_emits_llm_event_from_usage_metadata(self):
        handler = PactrunCallbackHandler()
        message = AIMessage(
            content="hi there",
            usage_metadata={"input_tokens": 5, "output_tokens": 7, "total_tokens": 12},
        )
        result = LLMResult(generations=[[ChatGeneration(message=message)]])
        with Contract("t").session() as s:
            handler.on_llm_end(result, run_id="r2")
        assert s.state.total_llm_calls == 1
        assert s.state.total_tokens == 12

    def test_emits_tool_call(self):
        handler = PactrunCallbackHandler()
        with Contract("t").session() as s:
            handler.on_tool_start({"name": "search"}, "weather in Paris", run_id="t1")
        assert "search" in s.state.tool_call_history

    def test_explicit_session_argument(self):
        session = Contract("t").session()  # not the active contextvar session
        handler = PactrunCallbackHandler(session=session)
        handler.on_llm_end(_result_with_token_usage(), run_id="r1")
        assert session.state.total_llm_calls == 1

    def test_no_active_session_is_silent(self):
        handler = PactrunCallbackHandler()
        handler.on_llm_end(_result_with_token_usage(), run_id="r1")  # must not raise
        handler.on_tool_start({"name": "x"}, "in", run_id="t1")

    def test_contract_enforced_through_callbacks(self):
        # gpt-4.1 usage above costs ~$0.000156 — blows a tiny cap.
        contract = Contract("t").require(cost_under(0.0000001), on_fail="log")
        handler = PactrunCallbackHandler()
        with contract.session() as s:
            handler.on_llm_end(_result_with_token_usage(), run_id="r1")
        assert not s.is_compliant

    def test_integration_with_real_langchain_invoke(self):
        # Drive a real LangChain chat model so callbacks fire through actual
        # LangChain dispatch (on_chat_model_start / on_llm_end), as they would
        # inside a LangGraph node.
        fake = pytest.importorskip("langchain_core.language_models.fake_chat_models")
        model = fake.GenericFakeChatModel(messages=iter([AIMessage(content="hi from fake")]))
        handler = PactrunCallbackHandler()
        with Contract("t").session() as s:
            model.invoke("hello", config={"callbacks": [handler]})
        assert s.state.total_llm_calls >= 1
        assert any("hi from fake" in out for out in s.state.output_history)
