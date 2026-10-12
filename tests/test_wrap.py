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


def test_responses_api_is_explicitly_refused():
    from types import SimpleNamespace

    calls = []
    client = _FakeOpenAI([_OAIResp()])
    client.responses = SimpleNamespace(create=lambda **kw: calls.append(kw))
    g = pactrun.wrap(client, max_cost=0)
    with pytest.raises(NotImplementedError, match="Responses API"):
        g.responses.create(model="gpt-4.1", input="hello")
    assert calls == []
    assert g.session.state.total_llm_calls == 0


def test_sync_inflight_reservations_prevent_concurrent_overspend(monkeypatch):
    import importlib
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace

    module = importlib.import_module("pactrun.wrap")
    monkeypatch.setattr(module, "_estimate_worstcase_cost", lambda *a: (0.75, "test-estimate"))
    monkeypatch.setattr(module, "_actual_cost", lambda *a: 0.25)
    started, release = threading.Event(), threading.Event()
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        started.set()
        assert release.wait(5)
        return _OAIResp()

    g = pactrun.wrap(SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), max_cost=1)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(g.chat.completions.create, model="gpt-4.1")
        try:
            assert started.wait(5)
            second = pool.submit(g.chat.completions.create, model="gpt-4.1")
            with pytest.raises(ViolationError, match="pre-call"):
                second.result(timeout=5)
            assert len(calls) == 1
        finally:
            release.set()
        first.result(timeout=5)
    assert not g._reservations
    assert g.session.state.total_cost_usd == 0.25
    # Settlement replaces the pending estimate; it does not add both charges.
    g.chat.completions.create(model="gpt-4.1")
    assert g.session.state.total_cost_usd == 0.5
    assert len(calls) == 2


def test_provider_error_is_accounted_conservatively_and_preserved(monkeypatch):
    import importlib
    from types import SimpleNamespace

    module = importlib.import_module("pactrun.wrap")
    monkeypatch.setattr(module, "_estimate_worstcase_cost", lambda *a: (0.75, "test-estimate"))
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("provider outcome unknown")

    g = pactrun.wrap(SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), max_cost=1)
    with pytest.raises(RuntimeError, match="provider outcome unknown"):
        g.chat.completions.create(model="gpt-4.1")
    assert not g._reservations
    assert g.session.state.total_cost_usd == 0.75
    assert g.session.state.total_llm_calls == 1
    assert g.session.state.events[-1].metadata["cost_basis"] == "pre-call-estimate"
    with pytest.raises(ViolationError, match="pre-call"):
        g.chat.completions.create(model="gpt-4.1")
    assert len(calls) == 1


@pytest.mark.parametrize("estimate", [-1.0, float("nan"), float("inf")])
def test_invalid_estimate_refuses_before_provider_call(monkeypatch, estimate):
    import importlib

    module = importlib.import_module("pactrun.wrap")
    monkeypatch.setattr(module, "_estimate_worstcase_cost", lambda *a: (estimate, "test-estimate"))
    client = _FakeOpenAI([_OAIResp()])
    g = pactrun.wrap(client, max_cost=1)
    with pytest.raises(ValueError, match="finite and non-negative"):
        g.chat.completions.create(model="gpt-4.1")
    assert client.chat.completions.calls == 0


def test_missing_response_usage_keeps_reserved_estimate(monkeypatch):
    import importlib

    module = importlib.import_module("pactrun.wrap")
    monkeypatch.setattr(module, "_estimate_worstcase_cost", lambda *a: (0.75, "test-estimate"))
    response = _OAIResp()
    response.usage = None
    g = pactrun.wrap(_FakeOpenAI([response]), max_cost=1)
    g.chat.completions.create(model="gpt-4.1")
    assert g.session.state.total_cost_usd == 0.75
    assert not g._reservations
    assert g.session.state.events[-1].metadata["usage_unavailable_reason"] == "response-usage-unavailable"
    with pytest.raises(ViolationError, match="pre-call"):
        g.chat.completions.create(model="gpt-4.1")


def test_postcall_approval_uses_configured_handler(monkeypatch):
    import importlib

    module = importlib.import_module("pactrun.wrap")
    monkeypatch.setattr(module, "_estimate_worstcase_cost", lambda *a: (0.75, "test-estimate"))
    monkeypatch.setattr(module, "_actual_cost", lambda *a: 1.5)
    approvals = []
    g = pactrun.wrap(
        _FakeOpenAI([_OAIResp()]), max_cost=1, on_violation="approve",
        approval_handler=lambda violation: approvals.append(violation) or True,
    )
    response = g.chat.completions.create(model="gpt-4.1")
    assert response.choices[0].message.content == "ok"
    assert len(approvals) == 1
    assert not g._reservations
    assert g.session.state.total_cost_usd == 1.5


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("shape", ["empty", "partial", "negative", "nan"])
def test_incomplete_or_invalid_usage_never_refunds_reservation(monkeypatch, provider, shape):
    import importlib
    from types import SimpleNamespace as NS

    module = importlib.import_module("pactrun.wrap")
    monkeypatch.setattr(module, "_estimate_worstcase_cost", lambda *a: (0.75, "test-estimate"))
    input_field, output_field = (
        ("prompt_tokens", "completion_tokens") if provider == "openai" else ("input_tokens", "output_tokens")
    )
    fields = {} if shape == "empty" else {input_field: 0}
    if shape in ("negative", "nan"):
        fields.update({input_field: -1 if shape == "negative" else float("nan"), output_field: 0})
    response = _OAIResp() if provider == "openai" else _AResp()
    response.usage = NS(**fields)
    client = _FakeOpenAI([response]) if provider == "openai" else _FakeAnthropic([response])
    g = pactrun.wrap(client, max_cost=1)
    create = g.chat.completions.create if provider == "openai" else g.messages.create
    if shape in ("negative", "nan"):
        with pytest.raises(ViolationError, match="Invalid provider accounting"):
            create(model="gpt-4.1" if provider == "openai" else "claude-sonnet-4")
        assert g.session.state.events[-1].metadata["pactrun.invalid_reported_usage"] == "reported-usage-invalid"
    else:
        create(model="gpt-4.1" if provider == "openai" else "claude-sonnet-4")
    assert g.session.state.total_cost_usd == 0.75
    assert not g._reservations
    with pytest.raises(ViolationError, match="pre-call"):
        create(model="gpt-4.1" if provider == "openai" else "claude-sonnet-4")


@pytest.mark.parametrize("bad_cost", [-1.0, float("nan"), float("inf")])
def test_invalid_priced_cost_retains_estimate_and_diagnostic(monkeypatch, bad_cost):
    import importlib

    module = importlib.import_module("pactrun.wrap")
    monkeypatch.setattr(module, "_estimate_worstcase_cost", lambda *a: (0.75, "test-estimate"))
    monkeypatch.setattr(module, "_actual_cost", lambda *a: bad_cost)
    client = _FakeOpenAI([_OAIResp()])
    g = pactrun.wrap(client, max_cost=1)
    with pytest.raises(ViolationError, match="Invalid provider accounting"):
        g.chat.completions.create(model="gpt-4.1")
    event = g.session.state.events[-1]
    assert event.metadata["reported_computed_cost"] == repr(bad_cost)
    assert g.session.state.total_cost_usd == 0.75
    assert not g._reservations
    with pytest.raises(ViolationError, match="pre-call"):
        g.chat.completions.create(model="gpt-4.1")
    assert client.chat.completions.calls == 1


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_explicit_zero_reported_usage_is_valid_not_missing(monkeypatch, provider):
    import importlib

    module = importlib.import_module("pactrun.wrap")
    monkeypatch.setattr(module, "_estimate_worstcase_cost", lambda *a: (0.75, "test-estimate"))
    client = _FakeOpenAI([_OAIResp(p=0, c=0)]) if provider == "openai" else _FakeAnthropic([_AResp(i=0, o=0)])
    g = pactrun.wrap(client, max_cost=1)
    create = g.chat.completions.create if provider == "openai" else g.messages.create
    model = "gpt-4.1" if provider == "openai" else "claude-sonnet-4"
    create(model=model)
    create(model=model)
    assert g.session.state.total_cost_usd == 0
    assert g.session.state.total_llm_calls == 2
    assert g.session.is_compliant
