"""Tests for Anthropic adapter."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from pactrun import Contract, cost_under
from pactrun.adapters.anthropic import AnthropicAdapter


class TestAnthropicAdapter:
    def _make_fake_anthropic(self):
        class FakeTextBlock:
            type = "text"
            text = "Hello from Claude"

        class FakeUsage:
            input_tokens = 40
            output_tokens = 15

        class FakeResponse:
            model = "claude-sonnet-4-6"
            content = [FakeTextBlock()]
            usage = FakeUsage()
            stop_reason = "end_turn"

        class FakeMessages:
            @staticmethod
            def create(*args, **kwargs):
                return FakeResponse()

        class FakeAsyncMessages:
            @staticmethod
            async def create(*args, **kwargs):
                return FakeResponse()

        module = ModuleType("anthropic")
        sub = ModuleType("anthropic.resources.messages")
        sub.Messages = FakeMessages
        sub.AsyncMessages = FakeAsyncMessages
        module.resources = ModuleType("anthropic.resources")
        module.resources.messages = sub

        return module, FakeMessages, FakeAsyncMessages

    def test_import_error_when_missing(self):
        adapter = AnthropicAdapter()
        with patch.dict(sys.modules, {"anthropic": None, "anthropic.resources": None, "anthropic.resources.messages": None}):
            with pytest.raises(ImportError, match="anthropic"):
                adapter._patch()

    def test_patch_and_unpatch(self):
        module, FakeMsg, _ = self._make_fake_anthropic()
        with patch.dict(sys.modules, {
            "anthropic": module,
            "anthropic.resources": module.resources,
            "anthropic.resources.messages": module.resources.messages,
        }):
            original = FakeMsg.create
            adapter = AnthropicAdapter()
            adapter._patch()
            assert FakeMsg.create is not original
            adapter._unpatch()
            assert FakeMsg.create is original

    def test_emits_llm_event(self):
        module, FakeMsg, _ = self._make_fake_anthropic()
        with patch.dict(sys.modules, {
            "anthropic": module,
            "anthropic.resources": module.resources,
            "anthropic.resources.messages": module.resources.messages,
        }):
            c = Contract("test")
            with c.session() as session:
                with AnthropicAdapter():
                    FakeMsg.create(FakeMsg(), model="claude-sonnet-4-6", messages=[], max_tokens=100)

            assert session.state.total_llm_calls >= 1
            assert session.state.total_tokens > 0

    def test_no_session_silent(self):
        module, FakeMsg, _ = self._make_fake_anthropic()
        with patch.dict(sys.modules, {
            "anthropic": module,
            "anthropic.resources": module.resources,
            "anthropic.resources.messages": module.resources.messages,
        }):
            with AnthropicAdapter():
                response = FakeMsg.create(FakeMsg(), model="claude-sonnet-4-6", messages=[], max_tokens=100)
                assert response.content[0].text == "Hello from Claude"

    def test_detects_tool_use(self):
        module, FakeMsg, _ = self._make_fake_anthropic()

        class FakeToolUseBlock:
            type = "tool_use"
            name = "get_weather"
            input = {"city": "Paris"}
            id = "tool_123"

        original_create = FakeMsg.create
        def create_with_tools(*args, **kwargs):
            resp = original_create(*args, **kwargs)
            resp.content.append(FakeToolUseBlock())
            return resp
        FakeMsg.create = staticmethod(create_with_tools)

        with patch.dict(sys.modules, {
            "anthropic": module,
            "anthropic.resources": module.resources,
            "anthropic.resources.messages": module.resources.messages,
        }):
            c = Contract("test")
            with c.session() as session:
                with AnthropicAdapter():
                    FakeMsg.create(FakeMsg(), model="claude-sonnet-4-6", messages=[], max_tokens=100)

            assert "get_weather" in session.state.tool_call_history
