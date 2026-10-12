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
        except StopIteration as exc:
            raise StopAsyncIteration from exc


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


async def test_async_inflight_reservation_blocks_second_request(monkeypatch):
    import asyncio
    import importlib

    module = importlib.import_module("pactrun.wrap")
    monkeypatch.setattr(module, "_estimate_worstcase_cost", lambda *a: (0.75, "test-estimate"))
    monkeypatch.setattr(module, "_actual_cost", lambda *a: 0.25)
    started, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        started.set()
        await release.wait()
        return _oai_resp()

    client = AsyncFakeOpenAI([])
    client.chat.completions.create = create
    g = pactrun.wrap(client, max_cost=1)
    first = asyncio.create_task(g.chat.completions.create(model="gpt-4.1"))
    try:
        await asyncio.wait_for(started.wait(), 5)
        with pytest.raises(ViolationError, match="pre-call"):
            await g.chat.completions.create(model="gpt-4.1")
        assert len(calls) == 1
    finally:
        release.set()
        await first
    assert g.session.state.total_cost_usd == 0.25
    assert not g._reservations
    await g.chat.completions.create(model="gpt-4.1")
    assert g.session.state.total_cost_usd == 0.5


@pytest.mark.parametrize("mode", ["log", "warn", "approve"])
async def test_nonblocking_recovery_still_reserves_inflight_cost(monkeypatch, mode):
    import asyncio
    import importlib
    import warnings

    module = importlib.import_module("pactrun.wrap")
    monkeypatch.setattr(module, "_estimate_worstcase_cost", lambda *a: (0.75, "test-estimate"))
    monkeypatch.setattr(module, "_actual_cost", lambda *a: 0.25)
    first_started, both_started, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    calls, approvals = [], []

    async def create(**kwargs):
        calls.append(kwargs)
        first_started.set()
        if len(calls) == 2:
            both_started.set()
        await release.wait()
        return _oai_resp()

    client = AsyncFakeOpenAI([])
    client.chat.completions.create = create
    g = pactrun.wrap(
        client, max_cost=1, on_violation=mode,
        approval_handler=lambda violation: approvals.append(violation) or True,
    )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        first = asyncio.create_task(g.chat.completions.create(model="gpt-4.1"))
        await asyncio.wait_for(first_started.wait(), 5)
        second = asyncio.create_task(g.chat.completions.create(model="gpt-4.1"))
        try:
            await asyncio.wait_for(both_started.wait(), 5)
            assert sum(g._reservations.values()) == 1.5
        finally:
            release.set()
            await asyncio.gather(first, second)
    assert len(g.session.violations) == 1
    assert bool(captured) == (mode == "warn")
    assert len(approvals) == (1 if mode == "approve" else 0)
    assert not g._reservations
    assert g.session.state.total_cost_usd == 0.5


async def test_cancelled_request_settles_pending_estimate(monkeypatch):
    import asyncio
    import importlib

    module = importlib.import_module("pactrun.wrap")
    monkeypatch.setattr(module, "_estimate_worstcase_cost", lambda *a: (0.75, "test-estimate"))
    started = asyncio.Event()

    async def create(**kwargs):
        started.set()
        await asyncio.Event().wait()

    client = AsyncFakeOpenAI([])
    client.chat.completions.create = create
    g = pactrun.wrap(client, max_cost=1)
    task = asyncio.create_task(g.chat.completions.create(model="gpt-4.1"))
    await asyncio.wait_for(started.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not g._reservations
    assert g.session.state.total_cost_usd == 0.75
    assert g.session.state.total_llm_calls == 1


def test_open_stream_keeps_reservation_and_close_settles_once(monkeypatch):
    import importlib

    module = importlib.import_module("pactrun.wrap")
    monkeypatch.setattr(module, "_estimate_worstcase_cost", lambda *a: (0.75, "test-estimate"))
    g = pactrun.wrap(SyncStreamOpenAI([_chunk(content="hi")]), max_cost=1)
    stream = g.chat.completions.create(model="gpt-4.1", stream=True)
    with pytest.raises(ViolationError, match="pre-call"):
        g.chat.completions.create(model="gpt-4.1", stream=True)
    assert g.session.state.total_llm_calls == 0
    stream.close()
    stream.close()
    assert not g._reservations
    assert g.session.state.total_cost_usd == 0.75
    assert g.session.state.total_llm_calls == 1


def test_final_usage_replaces_stream_estimate(monkeypatch):
    import importlib

    module = importlib.import_module("pactrun.wrap")
    monkeypatch.setattr(module, "_estimate_worstcase_cost", lambda *a: (0.75, "test-estimate"))
    monkeypatch.setattr(module, "_actual_cost", lambda *a: 0.25)
    g = pactrun.wrap(SyncStreamOpenAI([_chunk(usage=_ousage(10, 5))]), max_cost=1)
    stream = g.chat.completions.create(model="gpt-4.1", stream=True)
    list(stream)
    stream.close()
    assert not g._reservations
    assert g.session.state.total_cost_usd == 0.25
    assert g.session.state.total_llm_calls == 1
    list(g.chat.completions.create(model="gpt-4.1", stream=True))
    assert g.session.state.total_cost_usd == 0.5


def test_stream_cleanup_on_postcall_violation(monkeypatch):
    import importlib

    module = importlib.import_module("pactrun.wrap")
    monkeypatch.setattr(module, "_estimate_worstcase_cost", lambda *a: (0.75, "test-estimate"))
    monkeypatch.setattr(module, "_actual_cost", lambda *a: 1.5)

    class Stream:
        closed = False

        def __next__(self):
            return _chunk(usage=_ousage(10, 5))

        def close(self):
            self.closed = True

    upstream = Stream()
    client = NS(chat=NS(completions=NS(create=lambda **kw: upstream)))
    g = pactrun.wrap(client, max_cost=1)
    stream = g.chat.completions.create(model="gpt-4.1", stream=True)
    with pytest.raises(ViolationError):
        next(stream)
    assert upstream.closed
    assert not g._reservations
    assert g.session.state.total_cost_usd == 1.5
    assert g.session.state.total_llm_calls == 1


async def test_cancelled_stream_closes_sdk_async_close_and_settles_once(monkeypatch):
    import asyncio
    import importlib

    module = importlib.import_module("pactrun.wrap")
    monkeypatch.setattr(module, "_estimate_worstcase_cost", lambda *a: (0.75, "test-estimate"))
    started = asyncio.Event()

    class Stream:
        closed = False

        async def __anext__(self):
            started.set()
            await asyncio.Event().wait()

        async def close(self):
            self.closed = True

    upstream = Stream()
    client = AsyncFakeOpenAI([])

    async def create(**kwargs):
        return upstream

    client.chat.completions.create = create
    g = pactrun.wrap(client, max_cost=1)
    stream = await g.chat.completions.create(model="gpt-4.1", stream=True)
    task = asyncio.create_task(stream.__anext__())
    await asyncio.wait_for(started.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await stream.aclose()
    assert upstream.closed
    assert not g._reservations
    assert g.session.state.total_cost_usd == 0.75
    assert g.session.state.total_llm_calls == 1


def test_upstream_stream_error_is_preserved_and_accounted(monkeypatch):
    import importlib

    module = importlib.import_module("pactrun.wrap")
    monkeypatch.setattr(module, "_estimate_worstcase_cost", lambda *a: (0.75, "test-estimate"))

    class Stream:
        closed = False

        def __next__(self):
            raise RuntimeError("stream transport failed")

        def close(self):
            self.closed = True

    upstream = Stream()
    g = pactrun.wrap(NS(chat=NS(completions=NS(create=lambda **kw: upstream))), max_cost=1)
    stream = g.chat.completions.create(model="gpt-4.1", stream=True)
    with pytest.raises(RuntimeError, match="stream transport failed"):
        next(stream)
    assert upstream.closed
    assert not g._reservations
    assert g.session.state.total_cost_usd == 0.75
    assert g.session.state.total_llm_calls == 1


def test_context_exit_closes_upstream_even_when_cost_check_raises(monkeypatch):
    import importlib

    module = importlib.import_module("pactrun.wrap")
    monkeypatch.setattr(module, "_estimate_worstcase_cost", lambda *a: (0.75, "test-estimate"))

    class Stream:
        exited = False

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.exited = True

    upstream = Stream()
    g = pactrun.wrap(NS(chat=NS(completions=NS(create=lambda **kw: upstream))), max_cost=1)
    stream = g.chat.completions.create(model="gpt-4.1", stream=True)
    # A previously admitted call can report more than its estimate. Simulate
    # that settlement while this stream remains open.
    g.session.emit_llm_response(model="gpt-4.1", output="other", cost=0.5)
    with pytest.raises(ViolationError):
        with stream:
            pass
    assert upstream.exited
    assert not g._reservations
    assert g.session.state.total_cost_usd == 1.25
    stream.close()
    assert g.session.state.total_llm_calls == 2


async def test_async_context_exit_closes_upstream_on_cost_violation(monkeypatch):
    import importlib

    module = importlib.import_module("pactrun.wrap")
    monkeypatch.setattr(module, "_estimate_worstcase_cost", lambda *a: (0.75, "test-estimate"))

    class Stream:
        exited = False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            self.exited = True

    upstream = Stream()
    client = AsyncFakeOpenAI([])

    async def create(**kwargs):
        return upstream

    client.chat.completions.create = create
    g = pactrun.wrap(client, max_cost=1)
    stream = await g.chat.completions.create(model="gpt-4.1", stream=True)
    g.session.emit_llm_response(model="gpt-4.1", output="other", cost=0.5)
    with pytest.raises(ViolationError):
        async with stream:
            pass
    assert upstream.exited
    assert not g._reservations
    assert g.session.state.total_cost_usd == 1.25
    await stream.aclose()
    assert g.session.state.total_llm_calls == 2


@pytest.mark.parametrize("shape", ["empty", "partial", "negative", "nan"])
async def test_async_invalid_or_partial_usage_keeps_estimate(monkeypatch, shape):
    import importlib

    module = importlib.import_module("pactrun.wrap")
    monkeypatch.setattr(module, "_estimate_worstcase_cost", lambda *a: (0.75, "test-estimate"))
    fields = {} if shape == "empty" else {"prompt_tokens": 0}
    if shape in ("negative", "nan"):
        fields.update(prompt_tokens=-1 if shape == "negative" else float("nan"), completion_tokens=0)
    response = _oai_resp()
    response.usage = NS(**fields)
    g = pactrun.wrap(AsyncFakeOpenAI([response]), max_cost=1)
    if shape in ("negative", "nan"):
        with pytest.raises(ViolationError):
            await g.chat.completions.create(model="gpt-4.1")
    else:
        await g.chat.completions.create(model="gpt-4.1")
    assert g.session.state.total_cost_usd == 0.75
    assert not g._reservations
    with pytest.raises(ViolationError, match="pre-call"):
        await g.chat.completions.create(model="gpt-4.1")


@pytest.mark.parametrize("shape", ["empty", "partial", "negative", "nan", "invalid_cost", "zero"])
async def test_stream_usage_accounting_requires_complete_valid_totals(monkeypatch, shape):
    import importlib

    module = importlib.import_module("pactrun.wrap")
    monkeypatch.setattr(module, "_estimate_worstcase_cost", lambda *a: (0.75, "test-estimate"))
    fields = {} if shape == "empty" else {"prompt_tokens": 0}
    if shape in ("negative", "nan", "zero", "invalid_cost"):
        fields.update(prompt_tokens=-1 if shape == "negative" else float("nan") if shape == "nan" else 0, completion_tokens=0)
    if shape == "invalid_cost":
        monkeypatch.setattr(module, "_actual_cost", lambda *a: float("nan"))
    g = pactrun.wrap(AsyncStreamOpenAI([_chunk(usage=NS(**fields))]), max_cost=1)
    stream = await g.chat.completions.create(model="gpt-4.1", stream=True)
    if shape in ("negative", "nan", "invalid_cost"):
        with pytest.raises(ViolationError):
            async for _ in stream:
                pass
    else:
        async for _ in stream:
            pass
    await stream.aclose()
    assert not g._reservations
    assert g.session.state.total_llm_calls == 1
    assert g.session.state.total_cost_usd == (0 if shape == "zero" else 0.75)
    if shape != "zero":
        with pytest.raises(ViolationError, match="pre-call"):
            await g.chat.completions.create(model="gpt-4.1", stream=True)


async def test_neutral_named_async_subclass_records_and_enforces():
    class CustomClient(AsyncFakeOpenAI):
        pass

    g = pactrun.wrap(CustomClient([_oai_resp(tool="delete_account")]), forbid_tools=["delete_account"])
    with pytest.raises(ViolationError, match="delete_account"):
        await g.chat.completions.create(model="gpt-4.1")
    assert g.session.state.total_llm_calls == 1
    assert g.session.state.total_tool_calls == 1


async def test_neutral_named_client_async_method_is_detected():
    class CustomClient:
        def __init__(self):
            self.chat = NS(completions=NS(create=_async_create([_oai_resp()])))

    g = pactrun.wrap(CustomClient())
    await g.chat.completions.create(model="gpt-4.1")
    assert g.session.state.total_llm_calls == 1


async def test_real_async_sdk_neutral_subclass_records_without_network():
    openai = pytest.importorskip("openai")

    class CustomClient(openai.AsyncOpenAI):
        pass

    client = CustomClient(api_key="test-local-only")

    async def response():
        return _oai_resp()

    # Real AsyncOpenAI create wrappers need not be coroutine functions.
    def create(**kwargs):
        return response()

    client.chat.completions.create = create
    try:
        g = pactrun.wrap(client)
        await g.chat.completions.create(model="gpt-4.1")
        assert g.session.state.total_llm_calls == 1
    finally:
        await client.close()
