"""Tests for pactrun.wrap() — the one-line pre-call enforcement gate."""

from __future__ import annotations

import pytest

import pactrun
from pactrun import ViolationError


# --- fake OpenAI-style client ---------------------------------------------
class _Usage:
    def __init__(self, p, c):
        self.prompt_tokens = p
        self.completion_tokens = c


class _Fn:
    def __init__(self, name):
        self.name = name
        self.arguments = "{}"


class _ToolCall:
    def __init__(self, name):
        self.function = _Fn(name)


class _Msg:
    def __init__(self, content="ok", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class _Choice:
    def __init__(self, msg):
        self.message = msg


class _OAIResp:
    def __init__(self, content="ok", tool_calls=None, p=10, c=5, model="gpt-4.1"):
        self.model = model
        self.choices = [_Choice(_Msg(content, tool_calls))]
        self.usage = _Usage(p, c)


class _Completions:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        return self._responses[min(self.calls - 1, len(self._responses) - 1)]


class _FakeOpenAI:
    def __init__(self, responses):
        self.chat = type("_Chat", (), {"completions": _Completions(responses)})()
        self.api_key = "sk-test"


class TestWrapOpenAI:
    def test_delegates_unknown_attrs(self):
        g = pactrun.wrap(_FakeOpenAI([_OAIResp()]), max_cost="$0.50")
        assert g.api_key == "sk-test"

    def test_records_call(self):
        g = pactrun.wrap(_FakeOpenAI([_OAIResp(p=100, c=50)]), max_cost="$1.00")
        g.chat.completions.create(model="gpt-4.1", messages=[{"role": "user", "content": "hi"}])
        assert g.session.state.total_llm_calls == 1
        assert g.session.state.total_tokens == 150

    def test_precall_refuses_before_billing(self):
        client = _FakeOpenAI([_OAIResp()])
        g = pactrun.wrap(client, max_cost="$0.01")
        with pytest.raises(ViolationError, match="pre-call"):
            g.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=100_000,  # 100k * $8/1M = $0.80 worst case > $0.01
            )
        assert client.chat.completions.calls == 0  # the real call was NOT made

    def test_precall_allows_within_budget(self):
        client = _FakeOpenAI([_OAIResp()])
        g = pactrun.wrap(client, max_cost="$1.00")
        g.chat.completions.create(
            model="gpt-4.1", messages=[{"role": "user", "content": "hi"}], max_tokens=100
        )
        assert client.chat.completions.calls == 1

    def test_forbidden_tool_blocks_response(self):
        resp = _OAIResp(content=None, tool_calls=[_ToolCall("delete_account")])
        g = pactrun.wrap(
            _FakeOpenAI([resp]), max_cost="$1.00", forbid_tools=["delete_account"], default_max_tokens=10
        )
        with pytest.raises(ViolationError, match="delete_account"):
            g.chat.completions.create(
                model="gpt-4.1", messages=[{"role": "user", "content": "hi"}], max_tokens=10
            )

    def test_log_mode_does_not_raise(self):
        client = _FakeOpenAI([_OAIResp()])
        g = pactrun.wrap(client, max_cost="$0.0000001", on_violation="log")
        g.chat.completions.create(
            model="gpt-4.1", messages=[{"role": "user", "content": "hi"}], max_tokens=1000
        )
        assert not g.session.is_compliant  # recorded, not raised
        assert client.chat.completions.calls == 1  # log mode still made the call


# --- fake Anthropic-style client ------------------------------------------
class _ABlockText:
    def __init__(self, text):
        self.text = text


class _ABlockTool:
    def __init__(self, name):
        self.name = name
        self.input = {}


class _AUsage:
    def __init__(self, i, o):
        self.input_tokens = i
        self.output_tokens = o


class _AResp:
    def __init__(self, content=None, i=10, o=5, model="claude-sonnet-4-6"):
        self.model = model
        self.content = content or [_ABlockText("hi")]
        self.usage = _AUsage(i, o)


class _Messages:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        return self._responses[min(self.calls - 1, len(self._responses) - 1)]


class _FakeAnthropic:
    def __init__(self, responses):
        self.messages = _Messages(responses)


class TestWrapAnthropic:
    def test_records_call(self):
        g = pactrun.wrap(_FakeAnthropic([_AResp(i=40, o=15)]), max_cost="$1.00")
        g.messages.create(
            model="claude-sonnet-4-6", messages=[{"role": "user", "content": "hi"}], max_tokens=100
        )
        assert g.session.state.total_llm_calls == 1
        assert g.session.state.total_tokens == 55

    def test_forbidden_tool_blocks(self):
        g = pactrun.wrap(
            _FakeAnthropic([_AResp(content=[_ABlockTool("delete_account")])]),
            max_cost="$1.00",
            forbid_tools=["delete_account"],
            default_max_tokens=10,
        )
        with pytest.raises(ViolationError, match="delete_account"):
            g.messages.create(
                model="claude-sonnet-4-6", messages=[{"role": "user", "content": "hi"}], max_tokens=10
            )


class TestWrapErrors:
    def test_unknown_client_raises(self):
        with pytest.raises(ValueError, match="OpenAI and Anthropic"):
            pactrun.wrap(object(), max_cost="$1.00")
