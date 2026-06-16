"""Tests for async + streaming support in pactrun.wrap()."""

from __future__ import annotations

import inspect
from types import SimpleNamespace as NS

import pytest

import pactrun
from pactrun import ViolationError


# --- response / chunk builders (match the real SDK shapes) -----------------
def _oai_resp(content="ok", tool=None, p=10, c=5, model="gpt-4.1"):
    tool_calls = [NS(function=NS(name=tool, arguments="{}"))] if tool else []
    msg = NS(content=content, tool_calls=tool_calls)
    return NS(model=model, choices=[NS(message=msg)], usage=NS(prompt_tokens=p, completion_tokens=c))


def _chunk(content=None, tool=None, usage=None):
    choices = []
    if content is not None or tool:
        delta = NS(content=content, tool_calls=([NS(function=NS(name=tool))] if tool else []))
        choices = [NS(delta=delta)]
    return NS(choices=choices, usage=usage)


def _ousage(p, c):
    return NS(prompt_tokens=p, completion_tokens=c)


# Anthropic streaming events
def _a_start(input_tokens):
    return NS(type="message_start", message=NS(usage=NS(input_tokens=input_tokens)))


def _a_text(text):
    return NS(type="content_block_delta", delta=NS(text=text))


def _a_delta(output_tokens):
    return NS(type="message_delta", usage=NS(output_tokens=output_tokens))


def _a_stop():
    return NS(type="message_stop")


# --- fake clients (class names starting with "Async" mark async) -----------
def _async_create(responses):
    state = {"i": 0}

    async def create(**kwargs):
        r = responses[min(state["i"], len(responses) - 1)]
        state["i"] += 1
        return r

    return create


class _AsyncStream:
    def __init__(self, chunks):
        self._it = iter(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration


class AsyncFakeOpenAI:
    def __init__(self, responses):
        self.chat = NS(completions=NS(create=_async_create(responses)))


class AsyncStreamOpenAI:
    def __init__(self, chunks):
        async def create(**kwargs):
            return _AsyncStream(chunks)

        self.chat = NS(completions=NS(create=create))


class AsyncStreamAnthropic:
    def __init__(self, chunks):
        async def create(**kwargs):
            return _AsyncStream(chunks)

        self.messages = NS(create=create)


class SyncStreamOpenAI:
    def __init__(self, chunks):
        self._chunks = chunks

        def create(**kwargs):
            return iter(self._chunks)

        self.chat = NS(completions=NS(create=create))


# --- async non-streaming ---------------------------------------------------
async def test_async_create_returns_awaitable_and_records():
    g = pactrun.wrap(AsyncFakeOpenAI([_oai_resp(p=100, c=50)]), max_cost="$1.00")
    coro = g.chat.completions.create(model="gpt-4.1", messages=[{"role": "user", "content": "hi"}])
    assert inspect.isawaitable(coro)  # regression: not a silently-dropped sync call
    await coro
    assert g.session.state.total_llm_calls == 1
    assert g.session.state.total_tokens == 150


async def test_async_forbidden_tool_blocks():
    g = pactrun.wrap(
        AsyncFakeOpenAI([_oai_resp(content=None, tool="delete_account")]),
        max_cost="$1.00", forbid_tools=["delete_account"], default_max_tokens=10,
    )
    with pytest.raises(ViolationError, match="delete_account"):
        await g.chat.completions.create(
            model="gpt-4.1", messages=[{"role": "user", "content": "hi"}], max_tokens=10
        )


# --- async streaming -------------------------------------------------------
async def test_async_streaming_records_usage():
    chunks = [_chunk(content="hel"), _chunk(content="lo"), _chunk(usage=_ousage(40, 12))]
    g = pactrun.wrap(AsyncStreamOpenAI(chunks), max_cost="$1.00")
    stream = await g.chat.completions.create(
        model="gpt-4.1", messages=[{"role": "user", "content": "hi"}], stream=True
    )
    got = [chunk async for chunk in stream]
    assert len(got) == 3
    assert g.session.state.total_llm_calls == 1
    assert g.session.state.total_tokens == 52


async def test_async_streaming_blocks_forbidden_tool_mid_stream():
    chunks = [_chunk(content="ok"), _chunk(tool="delete_account"), _chunk(content="after"), _chunk(usage=_ousage(10, 5))]
    g = pactrun.wrap(
        AsyncStreamOpenAI(chunks), max_cost="$1.00", forbid_tools=["delete_account"], default_max_tokens=10
    )
    stream = await g.chat.completions.create(
        model="gpt-4.1", messages=[{"role": "user", "content": "hi"}], stream=True, max_tokens=10
    )
    seen = []
    with pytest.raises(ViolationError, match="delete_account"):
        async for chunk in stream:
            seen.append(chunk)
    assert len(seen) == 1  # the forbidden-tool chunk + "after" were never yielded


async def test_async_anthropic_streaming_records():
    chunks = [_a_start(40), _a_text("hel"), _a_text("lo"), _a_delta(12), _a_stop()]
    g = pactrun.wrap(AsyncStreamAnthropic(chunks), max_cost="$1.00")
    stream = await g.messages.create(
        model="claude-sonnet-4-6", messages=[{"role": "user", "content": "hi"}], max_tokens=100, stream=True
    )
    async for _ in stream:
        pass
    assert g.session.state.total_llm_calls == 1
    assert g.session.state.total_tokens == 52


# --- sync streaming + cancellation -----------------------------------------
def test_sync_streaming_records_usage():
    chunks = [_chunk(content="hi"), _chunk(usage=_ousage(30, 10))]
    g = pactrun.wrap(SyncStreamOpenAI(chunks), max_cost="$1.00")
    stream = g.chat.completions.create(
        model="gpt-4.1", messages=[{"role": "user", "content": "hi"}], stream=True
    )
    list(stream)
    assert g.session.state.total_llm_calls == 1
    assert g.session.state.total_tokens == 40


def test_cancelled_stream_records_worstcase():
    chunks = [_chunk(content="a"), _chunk(content="b"), _chunk(usage=_ousage(30, 10))]
    g = pactrun.wrap(SyncStreamOpenAI(chunks), max_cost="$100.00", default_max_tokens=50)
    stream = g.chat.completions.create(
        model="gpt-4.1", messages=[{"role": "user", "content": "hi"}], stream=True, max_tokens=50
    )
    for _ in stream:
        break  # cancel after the first chunk — no usage chunk reached
    stream.close()
    assert g.session.state.total_llm_calls == 1
    assert g.session.state.total_cost_usd > 0  # worst-case fallback, not silently dropped
