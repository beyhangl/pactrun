"""Tests for the LiteLLM adapter (also the practical CrewAI integration)."""

from __future__ import annotations

import pytest

pytest.importorskip("litellm")

import litellm
from litellm import ModelResponse
from litellm.types.utils import Usage

from pactrun import Contract, cost_under
from pactrun.adapters.litellm import LiteLLMAdapter


def _model_response(content="hello from litellm", tool_calls=None, cost=0.0002):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    response = ModelResponse(
        model="gpt-4.1",
        choices=[{"index": 0, "message": message, "finish_reason": "stop"}],
        usage=Usage(prompt_tokens=30, completion_tokens=12, total_tokens=42),
    )
    object.__setattr__(response, "_hidden_params", {"response_cost": cost})
    return response


class TestLiteLLMAdapter:
    def test_patch_and_unpatch(self):
        original = litellm.completion
        adapter = LiteLLMAdapter()
        adapter._patch()
        try:
            assert litellm.completion is not original
        finally:
            adapter._unpatch()
        assert litellm.completion is original

    def test_emits_llm_event(self):
        adapter = LiteLLMAdapter()
        with Contract("t").session() as s:
            adapter._emit_response({"model": "gpt-4.1"}, _model_response(), 12.0)
        assert s.state.total_llm_calls == 1
        assert s.state.total_tokens == 42
        assert "hello from litellm" in s.state.output_history
        assert s.state.total_cost_usd > 0

    def test_detects_tool_call(self):
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
            }
        ]
        adapter = LiteLLMAdapter()
        with Contract("t").session() as s:
            adapter._emit_response(
                {"model": "gpt-4.1"}, _model_response(content=None, tool_calls=tool_calls), 5.0
            )
        assert "get_weather" in s.state.tool_call_history

    def test_no_session_silent(self):
        adapter = LiteLLMAdapter()
        adapter._emit_response({"model": "gpt-4.1"}, _model_response(), 5.0)  # must not raise

    def test_through_patched_completion(self, monkeypatch):
        response = _model_response()
        monkeypatch.setattr(litellm, "completion", lambda *a, **k: response)
        with Contract("t").session() as s:
            with LiteLLMAdapter():
                out = litellm.completion(model="gpt-4.1", messages=[{"role": "user", "content": "hi"}])
        assert out is response
        assert s.state.total_llm_calls == 1
        assert s.state.total_tokens == 42

    def test_enforces_cost_contract(self):
        adapter = LiteLLMAdapter()
        contract = Contract("t").require(cost_under(0.0001), on_fail="log")  # 0.0002 > cap
        with contract.session() as s:
            adapter._emit_response({"model": "gpt-4.1"}, _model_response(cost=0.0002), 5.0)
        assert not s.is_compliant
