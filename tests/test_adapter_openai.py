"""Tests for OpenAI adapter."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from pactrun import Contract, cost_under, must_not_call
from pactrun.adapters.openai import OpenAIAdapter


class TestOpenAIAdapter:
    def _make_fake_openai(self):
        class FakeUsage:
            prompt_tokens = 50
            completion_tokens = 20
            total_tokens = 70

        class FakeMessage:
            content = "Hello from GPT"
            tool_calls = None
            role = "assistant"

        class FakeChoice:
            message = FakeMessage()
            finish_reason = "stop"

        class FakeResponse:
            model = "gpt-5.4-nano"
            choices = [FakeChoice()]
            usage = FakeUsage()

        class FakeCompletions:
            @staticmethod
            def create(*args, **kwargs):
                return FakeResponse()

        class FakeAsyncCompletions:
            @staticmethod
            async def create(*args, **kwargs):
                return FakeResponse()

        class FakeChat:
            completions = FakeCompletions()

        module = ModuleType("openai")
        sub = ModuleType("openai.resources.chat.completions")
        sub.Completions = FakeCompletions
        sub.AsyncCompletions = FakeAsyncCompletions
        module.resources = ModuleType("openai.resources")
        module.resources.chat = ModuleType("openai.resources.chat")
        module.resources.chat.completions = sub

        return module, FakeCompletions, FakeAsyncCompletions

    def test_import_error_when_missing(self):
        adapter = OpenAIAdapter()
        with patch.dict(sys.modules, {"openai": None, "openai.resources": None, "openai.resources.chat": None, "openai.resources.chat.completions": None}):
            with pytest.raises(ImportError, match="openai"):
                adapter._patch()

    def test_patch_and_unpatch(self):
        module, FakeComp, FakeAsync = self._make_fake_openai()
        with patch.dict(sys.modules, {
            "openai": module,
            "openai.resources": module.resources,
            "openai.resources.chat": module.resources.chat,
            "openai.resources.chat.completions": module.resources.chat.completions,
        }):
            original = FakeComp.create
            adapter = OpenAIAdapter()
            adapter._patch()
            assert FakeComp.create is not original
            adapter._unpatch()
            assert FakeComp.create is original

    def test_emits_llm_event(self):
        module, FakeComp, _ = self._make_fake_openai()
        with patch.dict(sys.modules, {
            "openai": module,
            "openai.resources": module.resources,
            "openai.resources.chat": module.resources.chat,
            "openai.resources.chat.completions": module.resources.chat.completions,
        }):
            c = Contract("test").require(cost_under(1.0), on_fail="log")
            with c.session() as session:
                with OpenAIAdapter():
                    FakeComp.create(FakeComp(), model="gpt-5.4-nano", messages=[])

            assert session.state.total_llm_calls >= 1
            assert session.state.total_tokens > 0

    def test_no_session_silent(self):
        module, FakeComp, _ = self._make_fake_openai()
        with patch.dict(sys.modules, {
            "openai": module,
            "openai.resources": module.resources,
            "openai.resources.chat": module.resources.chat,
            "openai.resources.chat.completions": module.resources.chat.completions,
        }):
            with OpenAIAdapter():
                # No session active — should not raise
                response = FakeComp.create(FakeComp(), model="gpt-5.4-nano", messages=[])
                assert response.model == "gpt-5.4-nano"

    def test_detects_tool_calls(self):
        module, FakeComp, _ = self._make_fake_openai()

        class FakeToolCall:
            class function:
                name = "search"
                arguments = '{"q": "test"}'
            id = "call_123"

        # Patch the response to include tool calls
        original_create = FakeComp.create
        def create_with_tools(*args, **kwargs):
            resp = original_create(*args, **kwargs)
            resp.choices[0].message.tool_calls = [FakeToolCall()]
            return resp
        FakeComp.create = staticmethod(create_with_tools)

        with patch.dict(sys.modules, {
            "openai": module,
            "openai.resources": module.resources,
            "openai.resources.chat": module.resources.chat,
            "openai.resources.chat.completions": module.resources.chat.completions,
        }):
            c = Contract("test")
            with c.session() as session:
                with OpenAIAdapter():
                    FakeComp.create(FakeComp(), model="gpt-5.4-nano", messages=[])

            assert "search" in session.state.tool_call_history

    def test_violation_detected_through_adapter(self):
        module, FakeComp, _ = self._make_fake_openai()

        # Make response expensive
        class ExpensiveUsage:
            prompt_tokens = 5000
            completion_tokens = 2000
            total_tokens = 7000

        original_create = FakeComp.create
        def expensive_create(*args, **kwargs):
            resp = original_create(*args, **kwargs)
            resp.usage = ExpensiveUsage()
            return resp
        FakeComp.create = staticmethod(expensive_create)

        with patch.dict(sys.modules, {
            "openai": module,
            "openai.resources": module.resources,
            "openai.resources.chat": module.resources.chat,
            "openai.resources.chat.completions": module.resources.chat.completions,
        }):
            c = Contract("test").require(cost_under(0.0001), on_fail="log")
            with c.session() as session:
                with OpenAIAdapter():
                    FakeComp.create(FakeComp(), model="gpt-5.4-nano", messages=[])

            assert not session.is_compliant
