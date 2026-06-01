"""Tests for the Gemini adapter."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import patch

import pytest

from pactrun import Contract
from pactrun.adapters.gemini import GeminiAdapter


def _make_fake_genai():
    class FakeUsage:
        prompt_token_count = 40
        candidates_token_count = 15
        total_token_count = 55

    class FakeResponse:
        model_version = "gemini-2.5-flash"
        usage_metadata = FakeUsage()
        function_calls: list = []
        text = "Hello from Gemini"

    class FakeModels:
        @staticmethod
        def generate_content(*args, **kwargs):
            return FakeResponse()

    class FakeAsyncModels:
        @staticmethod
        async def generate_content(*args, **kwargs):
            return FakeResponse()

    google_mod = ModuleType("google")
    genai_mod = ModuleType("google.genai")
    models_mod = ModuleType("google.genai.models")
    models_mod.Models = FakeModels
    models_mod.AsyncModels = FakeAsyncModels
    genai_mod.models = models_mod
    google_mod.genai = genai_mod
    return google_mod, genai_mod, models_mod, FakeModels, FakeResponse


def _sysmods(google_mod, genai_mod, models_mod):
    return {
        "google": google_mod,
        "google.genai": genai_mod,
        "google.genai.models": models_mod,
    }


class TestGeminiAdapter:
    def test_import_error_when_missing(self):
        adapter = GeminiAdapter()
        with patch.dict(sys.modules, {"google.genai.models": None}):
            with pytest.raises(ImportError, match="google-genai"):
                adapter._patch()

    def test_patch_and_unpatch(self):
        g, gn, m, FakeModels, _ = _make_fake_genai()
        with patch.dict(sys.modules, _sysmods(g, gn, m)):
            original = FakeModels.generate_content
            adapter = GeminiAdapter()
            adapter._patch()
            assert FakeModels.generate_content is not original
            adapter._unpatch()
            assert FakeModels.generate_content is original

    def test_emits_llm_event(self):
        g, gn, m, FakeModels, _ = _make_fake_genai()
        with patch.dict(sys.modules, _sysmods(g, gn, m)):
            with Contract("t").session() as s:
                with GeminiAdapter():
                    FakeModels.generate_content(FakeModels(), model="gemini-2.5-flash", contents="hi")
            assert s.state.total_llm_calls == 1
            assert s.state.total_tokens == 55
            assert "Hello from Gemini" in s.state.output_history
            assert s.state.total_cost_usd > 0  # gemini-2.5-flash is priced

    def test_detects_function_call(self):
        g, gn, m, FakeModels, _ = _make_fake_genai()

        class FakeFunctionCall:
            name = "get_weather"
            args = {"city": "Paris"}

        original = FakeModels.generate_content

        def with_tools(*args, **kwargs):
            resp = original(*args, **kwargs)
            resp.function_calls = [FakeFunctionCall()]
            return resp

        FakeModels.generate_content = staticmethod(with_tools)

        with patch.dict(sys.modules, _sysmods(g, gn, m)):
            with Contract("t").session() as s:
                with GeminiAdapter():
                    FakeModels.generate_content(FakeModels(), model="gemini-2.5-flash", contents="weather?")
            assert "get_weather" in s.state.tool_call_history

    def test_no_session_silent(self):
        g, gn, m, FakeModels, _ = _make_fake_genai()
        with patch.dict(sys.modules, _sysmods(g, gn, m)):
            with GeminiAdapter():
                resp = FakeModels.generate_content(FakeModels(), model="gemini-2.5-flash", contents="hi")
                assert resp.text == "Hello from Gemini"
