"""Pydantic AI adapter — enforce a pactrun Contract inside a Pydantic AI agent run.

``PactrunCapability`` is a Pydantic AI *capability* (the framework's own hook
API, ``pydantic_ai.capabilities.AbstractCapability``). Add it to an agent and
every model response and every tool call of every run flows through a pactrun
Session::

    from pydantic_ai import Agent
    from pactrun import Contract, must_not_call, token_budget
    from pactrun.adapters import PactrunCapability

    contract = Contract("support").forbid(must_not_call("delete_user")).require(token_budget(50_000))
    agent = Agent("openai:gpt-5.2", tools=[...], capabilities=[PactrunCapability(contract)])
    result = agent.run_sync("Close ticket 42")  # raises ViolationError before delete_user runs

Where the session comes from:

- ``PactrunCapability(contract)`` opens a **fresh session per agent run** (so
  budgets are per run) and closes it when the run ends, which also evaluates
  ``session_end`` clauses. The most recent one is ``capability.last_session``
  (with concurrent runs of one agent that is whichever started last; pass a
  Session or use the active-session form if you need a specific one).
  ``session_kwargs`` are passed to ``contract.session(...)`` (e.g.
  ``{"mode": "monitor"}``).
- ``PactrunCapability(session)`` records into that existing Session for every
  run; you own its lifecycle.
- ``PactrunCapability()`` uses the session active when the run starts
  (``with contract.session():``). If none is active the run is refused with
  ``SessionError`` — a guard the caller explicitly added must not silently
  enforce nothing.

What is recorded:

- one ``LLM_CALL`` event per model response, from ``after_model_request``:
  model name, ``input_tokens``/``output_tokens`` as prompt/completion tokens
  (Pydantic AI already normalises ``input_tokens`` to INCLUDE cache reads and
  writes, matching pactrun's convention), ``cache_read_tokens`` /
  ``cache_write_tokens``, reasoning tokens from ``usage.details``, the response
  text, and cost. Cost is ``usage.cost`` when Pydantic AI has
  already set it, otherwise Pydantic AI's own ``ModelResponse.cost()``
  (genai-prices). When that cannot price the model (unknown model, the offline
  ``test``/``function`` models) cost is recorded as ``0`` and
  ``metadata["cost_source"]`` is ``"unavailable"`` — pactrun does not guess.
  Use ``token_budget`` rather than ``cost_under`` for models genai-prices does
  not know.
- one ``TOOL_CALL`` event per tool execution, from ``wrap_tool_execute``,
  recorded **before** the tool runs with the validated arguments the tool will
  receive. After the tool returns, the same event is back-filled with
  ``tool_result`` (or ``error``) and ``duration_ms`` so later cross-event
  checks (``tool_error_rate_under``, ``untrusted_taint_to_sink``) see it.

What blocks:

- A ``block``/``escalate`` clause failing on a ``TOOL_CALL`` raises
  ``ViolationError`` before the tool body runs; the tool never executes. By
  default the error propagates out of ``agent.run()`` like every other pactrun
  adapter. With ``on_tool_block="return_to_model"`` the tool is still skipped,
  but the refusal text is returned to the model as the tool result and the run
  continues (the violation stays recorded on the session).
- A failing budget clause on an ``LLM_CALL`` raises out of the run after that
  response arrived (the tokens were already spent) and before any tool it asked
  for runs. ``on_tool_block`` does not apply to model events.
- In monitor mode (``contract.monitor()`` or ``session_kwargs={"mode":
  "monitor"}``) nothing raises: violations are recorded with
  ``enforced=False`` and tools run.
- Fail closed: if recording the tool event fails for any reason other than a
  contract violation (an observer bug, say), the tool is not run and a
  ``RuntimeError`` propagates.

Not covered / caveats (honest list):

- Output tools (structured ``output_type``) and provider-native tools (web
  search run by the provider) do not pass through ``wrap_tool_execute``, so
  they produce no ``TOOL_CALL`` event and cannot be blocked here.
- Checks that scan a tool result on the *same* event (e.g.
  ``no_injection_phrases`` with ``scan=("tool_result",)``) are evaluated before
  the result exists, so they do not see it. pactrun has no separate
  tool-result event kind to re-evaluate on.
- Streaming runs (``run_stream``) record each model response only once it has
  finished streaming, so a budget violation cannot cut a stream short (tool
  blocking works the same as in ``run``). Token counts are whatever the
  provider reports for a stream.
- Other capabilities that rewrite tool arguments should be listed *before*
  this one in ``capabilities=[...]`` so pactrun judges the final arguments
  (capabilities are middleware; later entries are inner layers). Another
  capability's ``on_tool_execute_error`` could swallow the ``ViolationError``;
  the tool still does not run.
- Durable execution (Temporal/DBOS/Prefect) is untested: the per-run session
  is process-local state.
- Do not also use an SDK patch adapter (``OpenAIAdapter`` etc.) inside the same
  run: both would record the same model call.
"""

from __future__ import annotations

import time
from dataclasses import KW_ONLY, dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

from pactrun.core.enums import EventKind
from pactrun.core.errors import SessionError, ViolationError
from pactrun.core.models import Event
from pactrun.session import Session, get_active_session

try:
    from pydantic_ai.capabilities import AbstractCapability
except ImportError as exc:  # pragma: no cover - only without the extra
    raise ImportError(
        "The 'pydantic-ai-slim' package (>= 1.71) is required for the Pydantic AI adapter. "
        "Install it with: pip install 'pactrun[pydantic-ai]'"
    ) from exc

if TYPE_CHECKING:
    from pydantic_ai.capabilities import WrapRunHandler, WrapToolExecuteHandler
    from pydantic_ai.messages import ModelResponse, ToolCallPart
    from pydantic_ai.models import ModelRequestContext
    from pydantic_ai.run import AgentRunResult
    from pydantic_ai.tools import RunContext, ToolDefinition

_FRAMEWORK = "pydantic_ai"


@dataclass
class PactrunCapability(AbstractCapability[Any]):
    """Pydantic AI capability that enforces a pactrun Contract on each agent run."""

    contract: Any = None
    """A ``Contract`` (fresh session per run), a ``Session`` (used as-is), or
    ``None`` (use the session active when the run starts)."""

    _: KW_ONLY

    on_tool_block: Literal["raise", "return_to_model"] = "raise"
    """What to do when a clause blocks a tool call. The tool never runs either way."""

    session_kwargs: dict[str, Any] = field(default_factory=dict)
    """Keyword arguments for ``contract.session(...)`` when a Contract is given."""

    def __post_init__(self) -> None:
        if self.on_tool_block not in ("raise", "return_to_model"):
            raise ValueError(
                f"on_tool_block must be 'raise' or 'return_to_model', got {self.on_tool_block!r}"
            )
        if self.session_kwargs and not _is_contract(self.contract):
            raise ValueError("session_kwargs only applies when a Contract is given")
        self.last_session: Session | None = None
        self._session: Session | None = None
        self._owns_session = False
        self._model_started: float | None = None

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None

    # -- run lifecycle -------------------------------------------------------

    async def for_run(self, ctx: RunContext[Any]) -> PactrunCapability:
        # replace() copies every init field (including the base class's, which
        # vary across Pydantic AI versions) and re-runs __post_init__, so the
        # copy starts with clean per-run state.
        run_cap = replace(self)
        if isinstance(self.contract, Session):
            run_cap._session = self.contract
        elif self.contract is not None:
            run_cap._session = self.contract.session(**self.session_kwargs)
            run_cap._owns_session = True
        else:
            active = get_active_session()
            if active is None:
                raise SessionError(
                    "PactrunCapability() has no contract and no pactrun session is active. "
                    "Pass a Contract (PactrunCapability(contract)) or run the agent inside "
                    "`with contract.session():`."
                )
            run_cap._session = active
        self.last_session = run_cap._session
        return run_cap

    async def wrap_run(self, ctx: RunContext[Any], *, handler: WrapRunHandler) -> AgentRunResult[Any]:
        session = self._session
        if session is None or not self._owns_session:
            return await handler()
        async with session:
            return await handler()

    # -- model requests --------------------------------------------------------

    async def before_model_request(
        self, ctx: RunContext[Any], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        self._model_started = time.monotonic()
        return request_context

    async def after_model_request(
        self,
        ctx: RunContext[Any],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        session = self._require_session()
        duration_ms = 0.0
        if self._model_started is not None:
            duration_ms = (time.monotonic() - self._model_started) * 1000
            self._model_started = None
        usage = response.usage
        cost, cost_source = _response_cost(response)
        model = response.model_name or getattr(request_context.model, "model_name", None) or "unknown"
        session.emit_llm_response(
            model=model,
            output=response.text or "",
            prompt_tokens=int(usage.input_tokens or 0),
            completion_tokens=int(usage.output_tokens or 0),
            cache_read_tokens=int(getattr(usage, "cache_read_tokens", 0) or 0),
            cache_write_tokens=int(getattr(usage, "cache_write_tokens", 0) or 0),
            reasoning_tokens=_reasoning_tokens(usage),
            cost=cost,
            duration_ms=duration_ms,
            metadata={
                "framework": _FRAMEWORK,
                "provider": response.provider_name,
                "cost_source": cost_source,
                "tool_calls": [part.tool_name for part in response.tool_calls],
            },
        )
        return response

    # -- tool execution ------------------------------------------------------

    async def wrap_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        handler: WrapToolExecuteHandler,
    ) -> Any:
        session = self._require_session()
        event = Event(
            kind=EventKind.TOOL_CALL,
            tool_name=call.tool_name,
            tool_args=dict(args),
            metadata={"framework": _FRAMEWORK, "tool_call_id": call.tool_call_id},
        )
        try:
            session.record_event(event)
        except ViolationError as exc:
            if self.on_tool_block == "raise":
                raise
            return f"pactrun blocked tool '{call.tool_name}': {exc.violation.message}"
        except Exception as exc:
            raise RuntimeError(
                f"pactrun could not evaluate tool call '{call.tool_name}'; not running it (fail-closed)"
            ) from exc

        start = time.monotonic()
        try:
            result = await handler(args)
        except Exception as exc:
            event.error = f"{type(exc).__name__}: {exc}"
            event.duration_ms = (time.monotonic() - start) * 1000
            raise
        event.tool_result = result
        event.duration_ms = (time.monotonic() - start) * 1000
        return result

    # -- internal ------------------------------------------------------------

    def _require_session(self) -> Session:
        if self._session is None:
            # Hooks only fire on the per-run copy from for_run(); reaching this
            # means the framework skipped it. Refuse rather than record nowhere.
            raise SessionError("PactrunCapability hook called without a run session (for_run was not called)")
        return self._session


def _is_contract(obj: Any) -> bool:
    return obj is not None and not isinstance(obj, Session)


# `RequestUsage.details` keys that hold reasoning tokens, per provider model
# in pydantic-ai 2.54: OpenAI `reasoning_tokens`, Anthropic `thinking_tokens`,
# Google `thoughts_tokens`. Each is already inside `output_tokens`.
_REASONING_DETAIL_KEYS = ("reasoning_tokens", "thinking_tokens", "thoughts_tokens")


def _reasoning_tokens(usage: Any) -> int:
    details = getattr(usage, "details", None) or {}
    for key in _REASONING_DETAIL_KEYS:
        value = details.get(key)
        if value:
            return int(value)
    return 0


def _response_cost(response: Any) -> tuple[float, str]:
    """USD cost of one response as Pydantic AI reports it, or ``(0.0, "unavailable")``."""
    reported = getattr(response.usage, "cost", None)
    if reported is not None:
        return float(reported), "pydantic_ai"
    if not response.model_name:
        return 0.0, "unavailable"
    try:
        return float(response.cost().total_price), "genai_prices"
    except Exception:  # noqa: BLE001 - unknown model/provider: record no cost, never guess
        return 0.0, "unavailable"
