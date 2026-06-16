"""pactrun.wrap() — a one-line pre-call enforcement gate.

Wrap an LLM client and every call is checked **before** it bills: pactrun
estimates the worst-case cost of the next call (your prompt + the maximum
output you allow) and refuses it if that would push your run over budget — and
it enforces tool / loop / turn limits on the response before your code proceeds.

    import openai, pactrun

    client = pactrun.wrap(
        openai.OpenAI(),
        max_cost="$0.50",          # whole-run budget — refused before the call that would cross it
        no_loops=True,             # stop repeating tool loops
        forbid_tools=["delete_account"],
    )
    client.chat.completions.create(model="gpt-4.1", messages=[...])

Works with sync **and** async clients (``AsyncOpenAI`` / ``AsyncAnthropic``,
including ``await`` + ``async for``) and with streaming calls (``stream=True``):
the pre-call cost gate runs before the request, forbidden-tool/loop rules fire on
the first tool-name delta (before your code can dispatch the tool), and real token
usage is recorded from the final usage chunk.

Honesty note: the pre-call cost check is a **worst-case bound** — you cannot know
the real completion-token count before a call, and reasoning models (o3/o4) can
spend hidden tokens beyond the visible estimate. The post-call check (on the real
usage reported by the provider) is the exact backstop. Content already streamed to
your code cannot be un-sent; a cancelled stream falls back to a worst-case cost
record. ``wrap()`` supports the OpenAI and Anthropic client SDKs (sync + async)
today; for other providers use the framework adapters with a ``Contract`` directly.
"""

from __future__ import annotations

from typing import Any

from pactrun.contract import Contract
from pactrun.core.enums import ClauseKind, OnFail, Severity
from pactrun.core.errors import ViolationError
from pactrun.core.models import Violation
from pactrun.predicates import (
    cost_under,
    drift_bounds,
    max_turns as max_turns_predicate,
    must_not_call,
    no_destructive_args,
    no_loops as no_loops_predicate,
    tool_args_match,
    tools_allowed as tools_allowed_predicate,
)
from pactrun.recovery.engine import apply_recovery


def _parse_cost(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value).replace("$", "").strip())


def wrap(
    client: Any,
    *,
    max_cost: Any = None,
    max_turns: int | None = None,
    no_loops: bool = False,
    max_drift: float | None = None,
    forbid_tools: list[str] | None = None,
    tools_allowed: list[str] | None = None,
    forbid_args: bool | list[str] | None = None,
    args_schema: dict | None = None,
    on_violation: str = "block",
    default_max_tokens: int = 4096,
    escalation_handler: Any = None,
) -> "GuardedClient":
    """Wrap an OpenAI/Anthropic client with a pre-call enforcement gate.

    Returns a drop-in client proxy: call it exactly as you would the original
    (``client.chat.completions.create`` / ``client.messages.create``). On each
    call the worst-case cost is checked *before* the request, and tool/loop/turn
    limits are checked on the response before it is handed back.
    """
    budget = _parse_cost(max_cost)

    contract = Contract("pactrun.wrap").on_violation(on_violation)
    if budget is not None:
        contract.require(cost_under(budget))
    if isinstance(max_turns, int):
        contract.require(max_turns_predicate(max_turns))
    if no_loops:
        contract.require(no_loops_predicate())
    if max_drift is not None:
        contract.require(drift_bounds(cost_pct=max_drift))
    for tool in forbid_tools or []:
        contract.forbid(must_not_call(tool))
    if tools_allowed:
        contract.require(tools_allowed_predicate(list(tools_allowed)))
    if forbid_args:
        extra = forbid_args if isinstance(forbid_args, list) else None
        contract.forbid(no_destructive_args(extra=extra))
    for tool_name, schema in (args_schema or {}).items():
        contract.require(tool_args_match(tool_name, schema))

    return GuardedClient(
        client,
        contract,
        budget=budget,
        on_fail=OnFail(on_violation),
        default_max_tokens=default_max_tokens,
        escalation_handler=escalation_handler,
    )


class _Namespace:
    """A tiny attribute holder used to mirror nested SDK call paths."""

    def __init__(self, **attrs: Any) -> None:
        self.__dict__.update(attrs)


class GuardedClient:
    """A client proxy that enforces a contract around every LLM call."""

    def __init__(
        self,
        client: Any,
        contract: Contract,
        *,
        budget: float | None,
        on_fail: OnFail,
        default_max_tokens: int,
        escalation_handler: Any = None,
    ) -> None:
        self._client = client
        self._contract = contract
        self._session = contract.session()
        self._budget = budget
        self._on_fail = on_fail
        self._max_tokens = default_max_tokens
        self._escalation_handler = escalation_handler
        self._kind, self._is_async = _detect_kind(client)
        guard = self._guard_async if self._is_async else self._guard

        if self._kind == "openai":
            real = client.chat.completions.create
            self.chat = _Namespace(completions=_Namespace(create=guard(real)))
        elif self._kind == "anthropic":
            real = client.messages.create
            self.messages = _Namespace(create=guard(real))
        else:
            raise ValueError(
                "pactrun.wrap() supports OpenAI and Anthropic client SDKs today. "
                "For other providers, use the framework adapters with a Contract directly."
            )

    def __getattr__(self, name: str) -> Any:
        client = self.__dict__.get("_client")
        if client is None:
            raise AttributeError(name)
        return getattr(client, name)

    @property
    def session(self):
        """The underlying enforcement Session (for summaries / violations)."""
        return self._session

    def __enter__(self) -> "GuardedClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    async def __aenter__(self) -> "GuardedClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    # -- the gate ----------------------------------------------------------

    def _guard(self, real_create):
        def guarded(*args: Any, **kwargs: Any):
            self._precall_cost_gate(kwargs)
            if self._kind == "openai" and kwargs.get("stream"):
                kwargs.setdefault("stream_options", {"include_usage": True})
            response = real_create(*args, **kwargs)
            return self._maybe_wrap_stream(kwargs, response)

        return guarded

    def _guard_async(self, real_create):
        async def guarded(*args: Any, **kwargs: Any):
            self._precall_cost_gate(kwargs)  # pure sync, no I/O
            if self._kind == "openai" and kwargs.get("stream"):
                kwargs.setdefault("stream_options", {"include_usage": True})
            response = await real_create(*args, **kwargs)
            return self._maybe_wrap_stream(kwargs, response)

        return guarded

    def _maybe_wrap_stream(self, kwargs: dict, response: Any) -> Any:
        if kwargs.get("stream"):
            return GuardedStream(response, self, kwargs)
        self._record(kwargs, response)  # post-call: tokens, tools, loops, turns
        return response

    def _precall_cost_gate(self, kwargs: dict) -> None:
        if self._budget is None:
            return
        worst, tag = _estimate_worstcase_cost(kwargs, self._max_tokens)
        projected = self._session.state.total_cost_usd + worst
        if projected <= self._budget:
            return
        violation = Violation(
            clause_description="pre-call cost gate",
            kind=ClauseKind.REQUIRE,
            severity=Severity.ERROR,
            on_fail=self._on_fail,
            message=(
                f"pre-call refusal: worst-case cost ~${worst:.4f} would push the run to "
                f"~${projected:.4f}, over the ${self._budget:.4f} budget (cost basis: {tag})"
            ),
            expected=f"<= ${self._budget:.4f}",
            actual=f"~${projected:.4f} projected",
        )
        self._session._violations.append(violation)
        apply_recovery(violation, escalation_handler=self._escalation_handler)

    def _record(self, kwargs: dict, response: Any) -> None:
        if self._kind == "openai":
            _record_openai(self._session, kwargs, response)
        elif self._kind == "anthropic":
            _record_anthropic(self._session, kwargs, response)


class GuardedStream:
    """Wraps a streaming response and enforces the contract as chunks arrive.

    Honesty: content already streamed to your code cannot be un-sent. The hard
    guarantees are the pre-call cost/bill gate and a tool block on the FIRST
    delta of a forbidden tool's name (before your code can dispatch it). A
    cancelled stream may never yield a final usage chunk, so its cost is then
    recorded from the worst-case estimate rather than real usage.
    """

    def __init__(self, stream: Any, owner: "GuardedClient", kwargs: dict) -> None:
        self._stream = stream
        self._owner = owner
        self._session = owner._session
        self._kind = owner._kind
        self._kwargs = kwargs
        self._model = kwargs.get("model", "unknown")
        self._seen_tools: set[str] = set()
        self._content: list[str] = []
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._recorded = False

    # -- sync iteration / context manager ----------------------------------

    def __iter__(self) -> "GuardedStream":
        return self

    def __next__(self) -> Any:
        try:
            chunk = next(self._stream)
        except StopIteration:
            self._finalize(usage_seen=False)
            raise
        self._on_chunk(chunk)
        return chunk

    def __enter__(self) -> "GuardedStream":
        enter = getattr(self._stream, "__enter__", None)
        if enter:
            enter()
        return self

    def __exit__(self, *exc: Any) -> Any:
        self._finalize(usage_seen=False)
        exit_ = getattr(self._stream, "__exit__", None)
        return exit_(*exc) if exit_ else None

    def close(self) -> None:
        self._finalize(usage_seen=False)
        close = getattr(self._stream, "close", None)
        if close:
            close()

    # -- async iteration / context manager ---------------------------------

    def __aiter__(self) -> "GuardedStream":
        return self

    async def __anext__(self) -> Any:
        try:
            chunk = await self._stream.__anext__()
        except StopAsyncIteration:
            self._finalize(usage_seen=False)
            raise
        self._on_chunk(chunk)
        return chunk

    async def __aenter__(self) -> "GuardedStream":
        enter = getattr(self._stream, "__aenter__", None)
        if enter:
            await enter()
        return self

    async def __aexit__(self, *exc: Any) -> Any:
        self._finalize(usage_seen=False)
        exit_ = getattr(self._stream, "__aexit__", None)
        return await exit_(*exc) if exit_ else None

    async def aclose(self) -> None:
        self._finalize(usage_seen=False)
        aclose = getattr(self._stream, "aclose", None)
        if aclose:
            await aclose()

    # -- chunk handling ----------------------------------------------------

    def _on_chunk(self, chunk: Any) -> None:
        if self._kind == "openai":
            self._on_openai_chunk(chunk)
        else:
            self._on_anthropic_chunk(chunk)

    def _on_openai_chunk(self, chunk: Any) -> None:
        for choice in getattr(chunk, "choices", None) or []:
            delta = getattr(choice, "delta", None)
            if delta is None:
                continue
            content = getattr(delta, "content", None)
            if content:
                self._content.append(content)
            for call in getattr(delta, "tool_calls", None) or []:
                self._note_tool(getattr(getattr(call, "function", None), "name", None))
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            self._prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            self._completion_tokens = getattr(usage, "completion_tokens", 0) or 0
            self._finalize(usage_seen=True)

    def _on_anthropic_chunk(self, chunk: Any) -> None:
        ctype = getattr(chunk, "type", None)
        if ctype == "content_block_start":
            self._note_tool(getattr(getattr(chunk, "content_block", None), "name", None))
        elif ctype == "content_block_delta":
            text = getattr(getattr(chunk, "delta", None), "text", None)
            if text:
                self._content.append(text)
        elif ctype == "message_start":
            usage = getattr(getattr(chunk, "message", None), "usage", None)
            if usage is not None:
                self._prompt_tokens = getattr(usage, "input_tokens", 0) or 0
        elif ctype == "message_delta":
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                self._completion_tokens = getattr(usage, "output_tokens", 0) or 0
        elif ctype == "message_stop":
            self._finalize(usage_seen=True)

    def _note_tool(self, name: Any) -> None:
        if name and name not in self._seen_tools:
            self._seen_tools.add(name)
            self._session.emit_tool_call(name)  # raises on forbidden -> earliest block

    def _finalize(self, usage_seen: bool) -> None:
        if self._recorded:
            return
        self._recorded = True
        output = "".join(self._content)
        if usage_seen and (self._prompt_tokens or self._completion_tokens):
            cost = _actual_cost(self._model, self._prompt_tokens, self._completion_tokens)
            self._session.emit_llm_response(
                model=self._model, output=output,
                prompt_tokens=self._prompt_tokens, completion_tokens=self._completion_tokens, cost=cost,
            )
        else:
            # No usage chunk (e.g. a cancelled stream) — don't silently drop the
            # cost; record the worst-case estimate instead.
            worst, _ = _estimate_worstcase_cost(self._kwargs, self._owner._max_tokens)
            self._session.emit_llm_response(model=self._model, output=output, cost=worst)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def _detect_kind(client: Any) -> tuple[str, bool]:
    # AsyncOpenAI/AsyncAnthropic .create is NOT a coroutine function, so detect
    # async by the client class name (verified) — otherwise async clients run
    # the sync path and silently never await/record.
    is_async = type(client).__name__.startswith("Async")
    chat = getattr(client, "chat", None)
    if chat is not None and getattr(getattr(chat, "completions", None), "create", None):
        return "openai", is_async
    messages = getattr(client, "messages", None)
    if messages is not None and getattr(messages, "create", None):
        return "anthropic", is_async
    return "unknown", is_async


# ---------------------------------------------------------------------------
# Cost estimation — delegates to pactrun.cost_model (real tokenizer + live
# pricing via tiktoken/litellm when available, heuristic fallback otherwise).
# ---------------------------------------------------------------------------

def _estimate_worstcase_cost(kwargs: dict, default_max_tokens: int) -> tuple[float, str]:
    from pactrun import cost_model

    model = kwargs.get("model", "") or ""
    max_output = (
        kwargs.get("max_tokens")
        or kwargs.get("max_completion_tokens")
        or default_max_tokens
    )
    return cost_model.precall_worstcase(
        model, kwargs.get("messages"), max_output,
        system=kwargs.get("system"), tools=kwargs.get("tools"),
    )


# ---------------------------------------------------------------------------
# Post-call recording (mirrors the OpenAI / Anthropic adapters)
# ---------------------------------------------------------------------------

def _record_openai(session: Any, kwargs: dict, response: Any) -> None:
    model = getattr(response, "model", None) or kwargs.get("model", "unknown")
    prompt_tokens = 0
    completion_tokens = 0
    usage = getattr(response, "usage", None)
    if usage is not None:
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0

    output = ""
    tool_names: list[tuple[str, dict]] = []
    try:
        message = response.choices[0].message
        output = getattr(message, "content", None) or ""
        for call in getattr(message, "tool_calls", None) or []:
            fn = getattr(call, "function", None)
            name = getattr(fn, "name", None)
            if name:
                tool_names.append((name, _parse_tool_args(getattr(fn, "arguments", None))))
    except (AttributeError, IndexError, TypeError):
        pass

    cost = _actual_cost(model, prompt_tokens, completion_tokens)
    session.emit_llm_response(
        model=model, output=output, prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens, cost=cost,
    )
    for name, args in tool_names:
        session.emit_tool_call(name, args=args)


def _record_anthropic(session: Any, kwargs: dict, response: Any) -> None:
    model = getattr(response, "model", None) or kwargs.get("model", "unknown")
    prompt_tokens = 0
    completion_tokens = 0
    usage = getattr(response, "usage", None)
    if usage is not None:
        prompt_tokens = getattr(usage, "input_tokens", 0) or 0
        completion_tokens = getattr(usage, "output_tokens", 0) or 0

    output = ""
    tool_calls: list[tuple[str, dict]] = []
    try:
        for block in response.content or []:
            if hasattr(block, "text"):
                output += block.text
            elif getattr(block, "name", None):
                tool_calls.append((block.name, getattr(block, "input", None) or {}))
    except (AttributeError, TypeError):
        pass

    cost = _actual_cost(model, prompt_tokens, completion_tokens)
    session.emit_llm_response(
        model=model, output=output, prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens, cost=cost,
    )
    for name, args in tool_calls:
        session.emit_tool_call(name, args=args)


def _parse_tool_args(raw: Any) -> dict:
    import json

    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw:
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except (ValueError, TypeError):
            return {}
    return {}


def _actual_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    from pactrun import cost_model

    return cost_model.actual_cost(model, prompt_tokens, completion_tokens)[0]
