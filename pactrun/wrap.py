"""pactrun.wrap() — a one-line pre-call enforcement gate.

Wrap an LLM client and supported calls are checked before dispatch: pactrun
estimates input/tool-definition cost plus the assumed maximum output and
reserves that amount against other in-flight calls on the same wrapper.
Blocking mode refuses estimates over budget. Tool / loop / turn limits are
checked on the response before your code proceeds.

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

Honesty note: this is an estimated budget, not a universal hard billing cap.
Provider framing, media, pricing changes and hidden reasoning can exceed it.
Callers must set a provider output limit; ``default_max_tokens`` supplies an
estimate when no limit is sent. The post-call check records reported usage
after billing. Content already streamed cannot be un-sent; missing usage,
request errors and cancellation retain the pre-call estimate. Only OpenAI Chat
Completions and Anthropic Messages create calls are wrapped (sync + async).
OpenAI Responses is explicitly refused. See docs/wrapper-coverage.md.
"""

from __future__ import annotations

import inspect
import logging
import math
import threading
from typing import Any

from pactrun.contract import Contract
from pactrun.core.amounts import is_valid_amount
from pactrun.core.enums import ClauseKind, OnFail, Severity
from pactrun.core.models import Event, PredicateResult, SessionState, Violation
from pactrun.core.usage import TokenUsage, read_field
from pactrun.predicates import (
    cost_under,
    drift_bounds,
    must_not_call,
    no_destructive_args,
    spend_rate_under,
    tool_args_match,
    tool_rate_limit,
)
from pactrun.predicates import (
    max_turns as max_turns_predicate,
)
from pactrun.predicates import (
    no_duplicate_side_effect as no_duplicate_side_effect_predicate,
)
from pactrun.predicates import (
    no_loops as no_loops_predicate,
)
from pactrun.predicates import (
    tools_allowed as tools_allowed_predicate,
)
from pactrun.recovery.engine import apply_recovery

logger = logging.getLogger("pactrun")
_INVALID_REPORTED_USAGE = "pactrun.invalid_reported_usage"


def _reported_usage_valid(event: Event, state: SessionState) -> PredicateResult:
    diagnostic = event.metadata.get(_INVALID_REPORTED_USAGE)
    return PredicateResult(
        passed=not diagnostic, expected="valid reported usage and computed cost",
        actual=diagnostic or "valid", message=f"Invalid provider accounting: {diagnostic}",
    )


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
    max_cost_per_min: float | None = None,
    tool_rate_limits: dict | None = None,
    no_duplicate_side_effect: dict | None = None,
    on_violation: str = "block",
    default_max_tokens: int = 4096,
    escalation_handler: Any = None,
    approval_handler: Any = None,
) -> GuardedClient:
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
    contract.require(_reported_usage_valid, description="reported provider usage must be valid")
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
    if max_cost_per_min is not None:
        contract.require(spend_rate_under(max_cost_per_min, 60))
    for tool_name, (n_calls, per_seconds) in (tool_rate_limits or {}).items():
        contract.require(tool_rate_limit(tool_name, n_calls, per_seconds))
    for tool_name, key_fields in (no_duplicate_side_effect or {}).items():
        contract.require(no_duplicate_side_effect_predicate(tool_name, key_fields=list(key_fields)))

    return GuardedClient(
        client,
        contract,
        budget=budget,
        on_fail=OnFail(on_violation),
        default_max_tokens=default_max_tokens,
        escalation_handler=escalation_handler,
        approval_handler=approval_handler,
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
        approval_handler: Any = None,
    ) -> None:
        self._client = client
        self._contract = contract
        self._session = contract.session(
            escalation_handler=escalation_handler, approval_handler=approval_handler,
        )
        self._budget = budget
        self._on_fail = on_fail
        self._max_tokens = default_max_tokens
        self._escalation_handler = escalation_handler
        self._approval_handler = approval_handler
        # The provider call never holds this lock. Admission and settlement do,
        # so another thread/task cannot spend a call's still-pending estimate.
        self._budget_lock = threading.RLock()
        self._reservations: dict[object, float] = {}
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
        if name == "responses" and self._kind == "openai":
            raise NotImplementedError(
                "pactrun.wrap() does not cover OpenAI Responses API calls. "
                "Use chat.completions.create or an explicitly instrumented adapter; "
                "the underlying Responses client is not passed through."
            )
        client = self.__dict__.get("_client")
        if client is None:
            raise AttributeError(name)
        return getattr(client, name)

    @property
    def session(self):
        """The underlying enforcement Session (for summaries / violations)."""
        return self._session

    def __enter__(self) -> GuardedClient:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    async def __aenter__(self) -> GuardedClient:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    # -- the gate ----------------------------------------------------------

    def _guard(self, real_create):
        def guarded(*args: Any, **kwargs: Any):
            reservation = self._precall_cost_gate(kwargs)
            if self._kind == "openai" and kwargs.get("stream"):
                kwargs.setdefault("stream_options", {"include_usage": True})
            try:
                response = real_create(*args, **kwargs)
            except BaseException:
                self._settle_failed_call(reservation, kwargs)
                raise
            return self._maybe_wrap_stream(kwargs, response, reservation)

        return guarded

    def _guard_async(self, real_create):
        async def guarded(*args: Any, **kwargs: Any):
            reservation = self._precall_cost_gate(kwargs)
            if self._kind == "openai" and kwargs.get("stream"):
                kwargs.setdefault("stream_options", {"include_usage": True})
            try:
                response = await real_create(*args, **kwargs)
            except BaseException:
                self._settle_failed_call(reservation, kwargs)
                raise
            return self._maybe_wrap_stream(kwargs, response, reservation)

        return guarded

    def _maybe_wrap_stream(self, kwargs: dict, response: Any, reservation: object) -> Any:
        if kwargs.get("stream"):
            return GuardedStream(response, self, kwargs, reservation)
        with self._budget_lock:
            estimate = self._reservations.pop(reservation)
            self._record(kwargs, response, missing_usage_cost=estimate)
        return response

    def _precall_cost_gate(self, kwargs: dict) -> object:
        worst, tag = _estimate_worstcase_cost(kwargs, self._max_tokens)
        if not math.isfinite(worst) or worst < 0:
            raise ValueError("The pre-call cost estimate must be finite and non-negative")
        with self._budget_lock:
            projected = self._session.state.total_cost_usd + sum(self._reservations.values()) + worst
            if self._budget is not None and projected > self._budget:
                self._handle_precall_violation(worst, tag, projected)
            reservation = object()
            self._reservations[reservation] = worst
            return reservation

    def _handle_precall_violation(self, worst: float, tag: str, projected: float) -> None:
        assert self._budget is not None
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
        apply_recovery(
            violation,
            escalation_handler=self._escalation_handler,
            approval_handler=self._approval_handler,
        )

    def _settle_estimate(self, reservation: object, kwargs: dict, *, output: str = "", reason: str) -> None:
        with self._budget_lock:
            estimate = self._reservations.pop(reservation, None)
            if estimate is None:
                return
            self._session.emit_llm_response(
                model=kwargs.get("model", "unknown"), output=output, cost=estimate,
                metadata={"cost_basis": "pre-call-estimate", "usage_unavailable_reason": reason},
            )

    def _settle_failed_call(self, reservation: object, kwargs: dict) -> None:
        # A request failure/cancellation does not establish that the provider
        # charged nothing. Keep its estimate and preserve the original error.
        try:
            self._settle_estimate(reservation, kwargs, reason="request-error-or-cancellation")
        except Exception:
            logger.exception("Policy evaluation failed while accounting for a failed request")

    def _record(self, kwargs: dict, response: Any, *, missing_usage_cost: float = 0.0) -> None:
        if self._kind == "openai":
            _record_openai(self._session, kwargs, response, missing_usage_cost=missing_usage_cost)
        elif self._kind == "anthropic":
            _record_anthropic(self._session, kwargs, response, missing_usage_cost=missing_usage_cost)


class GuardedStream:
    """Wraps a streaming response and enforces the contract as chunks arrive.

    Content already streamed cannot be un-sent. Admission reserves an estimate;
    a complete forbidden tool name is blocked before its chunk is delivered.
    A cancelled stream may lack final usage, so it retains the pre-call estimate.
    """

    def __init__(self, stream: Any, owner: GuardedClient, kwargs: dict, reservation: object) -> None:
        self._stream = stream
        self._owner = owner
        self._session = owner._session
        self._kind = owner._kind
        self._kwargs = kwargs
        self._reservation = reservation
        self._model = kwargs.get("model", "unknown")
        self._seen_tools: set[str] = set()
        self._content: list[str] = []
        self._usage: TokenUsage | None = None
        self._usage_complete = False
        # Anthropic streams usage in two events: message_start (input + cache
        # counts, incl. the 1-hour split) and message_delta (cumulative output,
        # optionally cumulative input/cache counts).
        self._a_start_usage: Any = None
        self._a_delta_usage: Any = None
        self._recorded = False
        self._closed = False

    # -- sync iteration / context manager ----------------------------------

    def __iter__(self) -> GuardedStream:
        return self

    def __next__(self) -> Any:
        try:
            chunk = next(self._stream)
        except StopIteration:
            self.close()
            raise
        except BaseException:
            self._close_after_error()
            raise
        try:
            self._on_chunk(chunk)
        except BaseException:
            self._close_after_error()
            raise
        return chunk

    def __enter__(self) -> GuardedStream:
        enter = getattr(self._stream, "__enter__", None)
        if enter:
            try:
                enter()
            except BaseException:
                self._close_after_error()
                raise
        return self

    def __exit__(self, *exc: Any) -> Any:
        result = None
        try:
            self._finalize(usage_seen=False)
        finally:
            self._closed = True
            exit_ = getattr(self._stream, "__exit__", None)
            if exit_:
                result = exit_(*exc)
            else:
                close = getattr(self._stream, "close", None)
                if close:
                    close()
        return result

    def close(self) -> None:
        try:
            self._finalize(usage_seen=False)
        finally:
            if not self._closed:
                self._closed = True
                close = getattr(self._stream, "close", None)
                if close:
                    close()

    def _close_after_error(self) -> None:
        try:
            self.close()
        except Exception:
            logger.exception("Stream cleanup failed while preserving the original error")

    # -- async iteration / context manager ---------------------------------

    def __aiter__(self) -> GuardedStream:
        return self

    async def __anext__(self) -> Any:
        try:
            chunk = await self._stream.__anext__()
        except StopAsyncIteration:
            await self.aclose()
            raise
        except BaseException:
            await self._aclose_after_error()
            raise
        try:
            self._on_chunk(chunk)
        except BaseException:
            await self._aclose_after_error()
            raise
        return chunk

    async def __aenter__(self) -> GuardedStream:
        enter = getattr(self._stream, "__aenter__", None)
        if enter:
            try:
                await enter()
            except BaseException:
                await self._aclose_after_error()
                raise
        return self

    async def __aexit__(self, *exc: Any) -> Any:
        result = None
        try:
            self._finalize(usage_seen=False)
        finally:
            self._closed = True
            exit_ = getattr(self._stream, "__aexit__", None)
            if exit_:
                result = await exit_(*exc)
            else:
                await self._close_upstream_async()
        return result

    async def aclose(self) -> None:
        try:
            self._finalize(usage_seen=False)
        finally:
            if not self._closed:
                self._closed = True
                await self._close_upstream_async()

    async def _close_upstream_async(self) -> None:
        # OpenAI's async Stream exposes async close(), while other streams
        # expose aclose(). Support both without delegating around accounting.
        import inspect

        close = getattr(self._stream, "aclose", None) or getattr(self._stream, "close", None)
        if close:
            result = close()
            if inspect.isawaitable(result):
                await result

    async def _aclose_after_error(self) -> None:
        try:
            await self.aclose()
        except Exception:
            logger.exception("Stream cleanup failed while preserving the original error")

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
            from pactrun.adapters.openai import usage_from_openai

            self._usage = usage_from_openai(usage)
            self._usage_complete = _usage_has_totals(usage, "openai")
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
                self._a_start_usage = usage
                self._usage = _merge_anthropic_stream_usage(self._a_start_usage, self._a_delta_usage)
        elif ctype == "message_delta":
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                self._a_delta_usage = usage
                self._usage = _merge_anthropic_stream_usage(self._a_start_usage, self._a_delta_usage)
        elif ctype == "message_stop":
            self._usage_complete = (
                _anthropic_stream_totals(self._a_start_usage, self._a_delta_usage)
            )
            self._finalize(usage_seen=True)

    def _note_tool(self, name: Any) -> None:
        if name and name not in self._seen_tools:
            self._seen_tools.add(name)
            with self._owner._budget_lock:
                self._session.emit_tool_call(name)  # raises on forbidden -> earliest block

    def _finalize(self, usage_seen: bool) -> None:
        with self._owner._budget_lock:
            if self._recorded:
                return
            self._recorded = True
            output = "".join(self._content)
            usage = self._usage
            if usage_seen and usage is not None:
                estimate = self._owner._reservations.pop(self._reservation)
                cost, metadata = _accounted_usage(
                    self._model, usage, available=self._usage_complete, estimate=estimate,
                    unavailable_reason="stream-usage-incomplete",
                )
                self._session.emit_llm_response(
                    model=self._model, output=output, cost=cost, metadata=metadata,
                    **usage.event_kwargs(),
                )
            else:
                self._owner._settle_estimate(
                    self._reservation, self._kwargs, output=output, reason="stream-usage-unavailable",
                )


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def _detect_kind(client: Any) -> tuple[str, bool]:
    # AsyncOpenAI/AsyncAnthropic .create is NOT a coroutine function, so detect
    # async by the client class name (verified) — otherwise async clients run
    # the sync path and silently never await/record.
    is_async = any(cls.__name__.startswith("Async") for cls in type(client).__mro__)
    chat = getattr(client, "chat", None)
    create = getattr(getattr(chat, "completions", None), "create", None)
    if chat is not None and create:
        return "openai", is_async or inspect.iscoroutinefunction(create)
    messages = getattr(client, "messages", None)
    create = getattr(messages, "create", None)
    if messages is not None and create:
        return "anthropic", is_async or inspect.iscoroutinefunction(create)
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
        cache_control=kwargs.get("cache_control"),
    )


# ---------------------------------------------------------------------------
# Post-call recording (mirrors the OpenAI / Anthropic adapters)
# ---------------------------------------------------------------------------

def _usage_has_totals(raw: Any, kind: str) -> bool:
    fields = ("prompt_tokens", "completion_tokens") if kind == "openai" else ("input_tokens", "output_tokens")
    return all(read_field(raw, field) is not None for field in fields)


def _anthropic_stream_totals(start: Any, delta: Any) -> bool:
    def present(field: str) -> bool:
        return read_field(delta, field) is not None or read_field(start, field) is not None

    return present("input_tokens") and present("output_tokens")


def _accounted_usage(
    model: str, usage: TokenUsage, *, available: bool, estimate: float, unavailable_reason: str,
) -> tuple[float, dict[str, Any]]:
    """Trust only complete, valid usage with a valid priced amount.

    Unknown/invalid accounting never frees its reserved budget. Invalid fields
    remain on the event for ingestion diagnostics, and a wrapper clause routes
    the invalid-report diagnostic through the configured recovery action.
    """
    invalid = None
    computed = None
    if not usage.is_valid():
        reason = "reported-usage-invalid"
        invalid = reason
    elif not available:
        reason = unavailable_reason
    else:
        computed = _actual_cost(model, usage)
        if is_valid_amount(computed):
            return computed, {}
        reason = "computed-cost-invalid"
        invalid = reason
    metadata: dict[str, Any] = {
        "cost_basis": "pre-call-estimate", "usage_unavailable_reason": reason,
    }
    if invalid:
        metadata[_INVALID_REPORTED_USAGE] = invalid
        metadata["reported_usage"] = {field: repr(value) for field, value in vars(usage).items()}
        if computed is not None:
            metadata["reported_computed_cost"] = repr(computed)
    return estimate, metadata


def _record_openai(session: Any, kwargs: dict, response: Any, *, missing_usage_cost: float = 0.0) -> None:
    from pactrun.adapters.openai import usage_from_openai

    model = getattr(response, "model", None) or kwargs.get("model", "unknown")
    raw_usage = getattr(response, "usage", None)
    usage = usage_from_openai(raw_usage)
    cost, metadata = _accounted_usage(
        model, usage, available=_usage_has_totals(raw_usage, "openai"), estimate=missing_usage_cost,
        unavailable_reason="response-usage-unavailable" if raw_usage is None else "response-usage-incomplete",
    )

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

    session.emit_llm_response(
        model=model, output=output,
        cost=cost, metadata=metadata,
        **usage.event_kwargs(),
    )
    for name, args in tool_names:
        session.emit_tool_call(name, args=args)


def _record_anthropic(session: Any, kwargs: dict, response: Any, *, missing_usage_cost: float = 0.0) -> None:
    from pactrun.adapters.anthropic import usage_from_anthropic

    model = getattr(response, "model", None) or kwargs.get("model", "unknown")
    raw_usage = getattr(response, "usage", None)
    usage = usage_from_anthropic(raw_usage)
    cost, metadata = _accounted_usage(
        model, usage, available=_usage_has_totals(raw_usage, "anthropic"), estimate=missing_usage_cost,
        unavailable_reason="response-usage-unavailable" if raw_usage is None else "response-usage-incomplete",
    )

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

    session.emit_llm_response(
        model=model, output=output,
        cost=cost, metadata=metadata,
        **usage.event_kwargs(),
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


def _merge_anthropic_stream_usage(start: Any, delta: Any) -> TokenUsage:
    """Combine Anthropic's message_start and message_delta usage into one total.

    ``message_delta`` usage is cumulative (anthropic-python
    ``MessageDeltaUsage``): its output count is final, and its input/cache
    counts, when present, supersede message_start's. The 1-hour write split
    (``cache_creation``) only arrives on message_start.
    """
    from types import SimpleNamespace

    from pactrun.adapters.anthropic import usage_from_anthropic
    from pactrun.core.usage import read_field

    def pick(name: str) -> Any:
        value = read_field(delta, name)
        return read_field(start, name) if value is None else value

    merged = SimpleNamespace(
        input_tokens=pick("input_tokens"),
        cache_read_input_tokens=pick("cache_read_input_tokens"),
        cache_creation_input_tokens=pick("cache_creation_input_tokens"),
        cache_creation=read_field(start, "cache_creation"),
        output_tokens=pick("output_tokens"),
        output_tokens_details=pick("output_tokens_details"),
    )
    return usage_from_anthropic(merged)


def _actual_cost(model: str, usage: TokenUsage) -> float:
    from pactrun import cost_model

    return cost_model.actual_cost(
        model,
        usage.prompt_tokens,
        usage.completion_tokens,
        cache_read_tokens=usage.cache_read_tokens,
        cache_write_tokens=usage.cache_write_tokens,
        cache_write_1h_tokens=usage.cache_write_1h_tokens,
    )[0]
