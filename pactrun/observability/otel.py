"""OpenTelemetry GenAI emitter (experimental).

Reports a pactrun Session as OpenTelemetry GenAI telemetry, so your runtime
contracts also show up in Langfuse / Arize Phoenix / Datadog / any OTLP
backend. Attach it to a session:

    from pactrun import Contract, cost_under
    from pactrun.observability import OTelObserver

    contract = Contract("agent").require(cost_under(0.50))
    with contract.session(observers=[OTelObserver()]):
        ...

Status: experimental. The GenAI semantic conventions live in their own
repository (open-telemetry/semantic-conventions-genai), which has **not cut a
release yet**. This module follows its main branch at ``547934a``
(2026-10-18; schema ``gen-ai-dev/1.42.0-dev``, built on core semconv
v1.44.0). Every ``gen_ai.*`` name is still Development status and the Python
constants for them are marked deprecated pending that release, so names are
written as literals on purpose. Not "stable / standard-compliant".

What it emits:

- One ``invoke_agent {gen_ai.agent.name}`` INTERNAL span per session, parenting
  the rest. The agent name is ``agent_name=`` or else the contract name.
  ``gen_ai.conversation.id`` is set only from ``conversation_id=`` or the
  session metadata key ``conversation_id`` (the spec forbids made-up fallbacks).
- ``chat {gen_ai.request.model}`` CLIENT spans (``chat`` when the model is
  unknown) and ``execute_tool {gen_ai.tool.name}`` INTERNAL spans, with
  ``gen_ai.tool.call.id`` from ``event.metadata["tool_call_id"]`` when the
  provider supplied one. ``emit_llm_spans=False`` drops the chat spans, for
  when another instrumentation (e.g. a provider SDK's own tracing) already
  emits them; everything else is kept.
- A ``gen_ai.evaluation.result`` span event per violation, on the span of the
  event that broke the clause (else the ``invoke_agent`` span). Name is the
  predicate name, label ``fail`` (enforce mode) or ``would_fail`` (monitor
  mode: detected, no action taken), score 0.0, explanation the message. The
  span is also still set to ERROR with an exception event, as before.
- Metrics, through ``meter_provider=`` or the global one (no-op until one is
  configured): ``gen_ai.client.inference.usage.*`` token counters,
  ``gen_ai.client.inference.operation.{input,output}_tokens`` and
  ``gen_ai.client.inference.duration`` histograms, ``gen_ai.execute_tool.duration``,
  and ``gen_ai.invoke_agent.{duration,inference_calls,tool_calls}``. Durations
  are recorded only when the event reports ``duration_ms``. pactrun has no
  per-modality breakdown, so ``gen_ai.token.modality`` is always ``unknown``.

Token usage on ``chat`` spans: ``gen_ai.usage.input_tokens`` is the TOTAL input
including cached tokens, ``gen_ai.usage.cache_read.input_tokens`` /
``gen_ai.usage.cache_write.input_tokens`` are subsets of it, and
``gen_ai.usage.reasoning.output_tokens`` is a subset of
``gen_ai.usage.output_tokens``. The breakdown attributes are emitted only when
non-zero. The ``invoke_agent`` span carries only the input/output totals.

``gen_ai.provider.name`` is required on chat spans and metrics but is an enum,
so it is omitted when it cannot be derived (pass ``provider=``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from pactrun.core.enums import EventKind
from pactrun.core.errors import ViolationError

try:
    from opentelemetry import trace
    from opentelemetry.trace import SpanKind, Status, StatusCode
except ImportError as exc:  # pragma: no cover - only without the extra
    raise ImportError(
        "OpenTelemetry is required for OTelObserver. "
        "Install it with: pip install 'pactrun[otel]'"
    ) from exc

_DURATION_BUCKETS = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, 40.96, 81.92]
_AGENT_DURATION_BUCKETS = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6, 51.2, 102.4, 204.8, 409.6]
_TOKEN_BUCKETS = [1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864]
_CALL_BUCKETS = [1, 2, 4, 8, 16, 32, 64, 128]


def _provider_of(model: str) -> str | None:
    """Best-effort map a model name to a ``gen_ai.provider.name`` enum member.

    Returns ``None`` when the provider cannot be determined. The attribute is
    an enum in the GenAI conventions, so emitting a placeholder like "unknown"
    would be invalid — omitting it is the conformant choice.
    """
    m = (model or "").lower()
    if m.startswith(("gpt", "o1", "o3", "o4")):
        return "openai"
    if m.startswith("claude"):
        return "anthropic"
    if m.startswith("gemini"):
        return "gcp.gemini"
    return None


def _model_of(event: Any) -> str | None:
    # Adapters record "unknown" when a failed call had no model.
    model = event.model
    return model if model and model != "unknown" else None


def _set_system(span: Any, provider: str) -> None:
    # `gen_ai.system` was renamed to `gen_ai.provider.name` and has since been
    # removed from the GenAI registry entirely. Still emitted by default
    # because deployed backends continue to read it; disable with
    # OTelObserver(emit_legacy_system=False).
    try:
        from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as ga

        span.set_attribute(ga.GEN_AI_SYSTEM, provider)
    except Exception:
        span.set_attribute("gen_ai.system", provider)


@dataclass
class _Run:
    """One session seen as one agent invocation."""
    state: Any  # held so id(state) stays unique while the run is open
    span: Any
    start_ns: int
    agent_name: str | None
    conversation_id: str | None
    llm_calls: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    error_type: str | None = None


@dataclass
class _Op:
    """An LLM or tool event between on_event and on_event_end."""
    run: _Run | None
    span: Any
    start_ns: int
    end_ns: int | None
    error_type: str | None = None


class _Instruments:
    def __init__(self, meter: Any) -> None:
        def counter(name: str, brief: str) -> Any:
            return meter.create_counter(name, unit="{token}", description=brief)

        self.usage = {
            "input": counter("gen_ai.client.inference.usage.input_tokens", "Input tokens, including cached."),
            "output": counter("gen_ai.client.inference.usage.output_tokens", "Output tokens, including reasoning."),
            "cache_read": counter(
                "gen_ai.client.inference.usage.cache_read.input_tokens", "Input tokens served from a provider cache."
            ),
            "cache_write": counter(
                "gen_ai.client.inference.usage.cache_write.input_tokens", "Input tokens written to a provider cache."
            ),
            "reasoning": counter(
                "gen_ai.client.inference.usage.reasoning.output_tokens", "Output tokens used for reasoning."
            ),
        }
        h = self._histogram
        self.op_input = h(meter, "gen_ai.client.inference.operation.input_tokens", "{token}",
                          "Input tokens per inference operation.", _TOKEN_BUCKETS)
        self.op_output = h(meter, "gen_ai.client.inference.operation.output_tokens", "{token}",
                           "Output tokens per inference operation.", _TOKEN_BUCKETS)
        self.inference_duration = h(meter, "gen_ai.client.inference.duration", "s",
                                    "GenAI client inference operation duration.", _DURATION_BUCKETS)
        self.tool_duration = h(meter, "gen_ai.execute_tool.duration", "s",
                               "The duration of a single tool execution.", _DURATION_BUCKETS)
        self.agent_duration = h(meter, "gen_ai.invoke_agent.duration", "s",
                                "The duration of a single agent invocation.", _AGENT_DURATION_BUCKETS)
        self.agent_inference_calls = h(meter, "gen_ai.invoke_agent.inference_calls", "{inference_call}",
                                       "Inference calls per agent invocation.", _CALL_BUCKETS)
        self.agent_tool_calls = h(meter, "gen_ai.invoke_agent.tool_calls", "{tool_call}",
                                  "Tool calls per agent invocation.", _CALL_BUCKETS)

    @staticmethod
    def _histogram(meter: Any, name: str, unit: str, brief: str, buckets: list) -> Any:
        try:
            return meter.create_histogram(
                name, unit=unit, description=brief, explicit_bucket_boundaries_advisory=buckets
            )
        except TypeError:  # API older than the advisory parameter
            return meter.create_histogram(name, unit=unit, description=brief)


def _instruments(meter_provider: Any) -> _Instruments | None:
    try:
        from opentelemetry import metrics
    except ImportError:  # pragma: no cover - the API has shipped metrics since 1.12
        return None
    mp = meter_provider or metrics.get_meter_provider()
    if isinstance(mp, metrics.NoOpMeterProvider):
        return None
    return _Instruments(mp.get_meter("pactrun"))


class OTelObserver:
    """Emits GenAI spans, evaluation events and metrics for a pactrun Session."""

    def __init__(
        self,
        tracer_provider: Any = None,
        semconv_version: str = "unreleased",
        *,
        provider: str | None = None,
        emit_legacy_system: bool = True,
        meter_provider: Any = None,
        agent_name: str | None = None,
        agent_id: str | None = None,
        agent_description: str | None = None,
        agent_version: str | None = None,
        conversation_id: str | None = None,
        emit_agent_span: bool = True,
        emit_llm_spans: bool = True,
    ) -> None:
        tp = tracer_provider or trace.get_tracer_provider()
        self._tracer = tp.get_tracer("pactrun")
        self._semconv_version = semconv_version
        self._provider = provider
        self._emit_legacy_system = emit_legacy_system
        self._agent_name = agent_name
        self._agent_id = agent_id
        self._agent_description = agent_description
        self._agent_version = agent_version
        self._conversation_id = conversation_id
        self._emit_agent_span = emit_agent_span
        self._emit_llm_spans = emit_llm_spans
        self._metrics = _instruments(meter_provider)
        # Keyed by id(state) so one observer can serve several sessions.
        self._runs: dict[int, _Run] = {}
        self._open: dict[str, _Op] = {}

    # -- Observer hooks ----------------------------------------------------

    def on_event(self, event: Any, state: Any) -> None:
        run = self._run_for(state)
        if event.kind == EventKind.LLM_CALL:
            model = _model_of(event)
            name = f"chat {model}" if model else "chat"
            kind = SpanKind.CLIENT
            run.llm_calls += 1
            run.input_tokens += int(event.prompt_tokens or 0)
            run.output_tokens += int(event.completion_tokens or 0)
        elif event.kind == EventKind.TOOL_CALL:
            name = f"execute_tool {event.tool_name or 'unknown'}"
            # The GenAI conventions say execute_tool spans SHOULD be INTERNAL.
            kind = SpanKind.INTERNAL
            run.tool_calls += 1
        else:
            return
        now = time.time_ns()
        start, end = now, None
        if event.duration_ms and event.duration_ms > 0:
            # The event arrives when the call has finished.
            start, end = now - int(event.duration_ms * 1e6), now
        span = None
        if event.kind == EventKind.TOOL_CALL or self._emit_llm_spans:
            context = trace.set_span_in_context(run.span) if run.span is not None else None
            span = self._tracer.start_span(name, context=context, kind=kind, start_time=start)
            self._set_attributes(span, event, run)
        op = _Op(run=run, span=span, start_ns=start, end_ns=end)
        if (event.metadata or {}).get("error") or event.error:
            op.error_type = "_OTHER"
            if span is not None:
                span.set_status(Status(StatusCode.ERROR, str(event.error or event.metadata.get("error"))))
                span.set_attribute("error.type", "_OTHER")
        self._open[event.id] = op

    def on_violation(self, violation: Any, event: Any) -> None:
        op = self._open.get(event.id)
        run = op.run if op is not None else self._active_run()
        span = op.span if op is not None else None
        if span is None and run is not None:
            span = run.span
        standalone = span is None
        if standalone:
            span = self._tracer.start_span("pactrun.violation", kind=SpanKind.CLIENT)
        error_type = violation.clause_description or violation.kind.value
        span.set_status(Status(StatusCode.ERROR, violation.message))
        span.set_attribute("pactrun.violation", True)
        span.set_attribute("error.type", error_type)
        try:
            span.record_exception(ViolationError(violation))
        except Exception:
            pass
        span.add_event("gen_ai.evaluation.result", self._evaluation_attributes(violation, event))
        if op is not None:
            op.error_type = error_type
        if run is not None and span is run.span:
            run.error_type = error_type
        if standalone:
            span.end()

    def on_event_end(self, event: Any) -> None:
        op = self._open.pop(event.id, None)
        if op is None:
            return
        end = op.end_ns if op.end_ns is not None else time.time_ns()
        if op.span is not None:
            op.span.end(end_time=end)
        if self._metrics is not None:
            self._record_metrics(event, op, (end - op.start_ns) / 1e9 if op.end_ns is not None else None)

    def on_session_end(self, state: Any) -> None:
        run = self._run_for(state)
        self._runs.pop(id(state), None)
        if run.span is None:
            return
        end = time.time_ns()
        run.span.set_attribute("gen_ai.usage.input_tokens", run.input_tokens)
        run.span.set_attribute("gen_ai.usage.output_tokens", run.output_tokens)
        run.span.end(end_time=end)
        if self._metrics is not None:
            attrs: dict[str, Any] = {}
            if run.agent_name:
                attrs["gen_ai.agent.name"] = run.agent_name
            self._metrics.agent_inference_calls.record(run.llm_calls, attrs)
            self._metrics.agent_tool_calls.record(run.tool_calls, attrs)
            if run.error_type:
                attrs["error.type"] = run.error_type
            self._metrics.agent_duration.record((end - run.start_ns) / 1e9, attrs)

    # -- Internal ----------------------------------------------------------

    def _run_for(self, state: Any) -> _Run:
        run = self._runs.get(id(state))
        if run is None:
            run = self._start_run(state)
            self._runs[id(state)] = run
        return run

    def _active_run(self) -> _Run | None:
        from pactrun.session import get_active_session

        session = get_active_session()
        return self._run_for(session.state) if session is not None else None

    def _start_run(self, state: Any) -> _Run:
        from pactrun.session import get_active_session

        # Observers have no session-start hook, so the run opens on the first
        # callback and takes the start time and contract from the active session.
        session = get_active_session()
        summary = session.summary() if session is not None and session.state is state else None
        agent_name = self._agent_name or (summary.contract_name if summary else None) or None
        metadata = getattr(state, "metadata", None) or {}
        conversation_id = self._conversation_id or metadata.get("conversation_id")
        start = int(summary.started_at * 1e9) if summary and summary.started_at else time.time_ns()
        span = None
        if self._emit_agent_span:
            attrs: dict[str, Any] = {"gen_ai.operation.name": "invoke_agent"}
            for key, value in (
                ("gen_ai.agent.name", agent_name),
                ("gen_ai.agent.id", self._agent_id),
                ("gen_ai.agent.description", self._agent_description),
                ("gen_ai.agent.version", self._agent_version),
                ("gen_ai.conversation.id", conversation_id),
            ):
                if value:
                    attrs[key] = str(value)
            if summary is not None:
                if summary.contract_name:
                    attrs["pactrun.contract.name"] = summary.contract_name
                attrs["pactrun.session.id"] = summary.session_id
            name = f"invoke_agent {agent_name}" if agent_name else "invoke_agent"
            span = self._tracer.start_span(name, kind=SpanKind.INTERNAL, start_time=start, attributes=attrs)
        return _Run(
            state=state, span=span, start_ns=start, agent_name=agent_name,
            conversation_id=str(conversation_id) if conversation_id else None,
        )

    def _set_attributes(self, span: Any, event: Any, run: _Run) -> None:
        if run.conversation_id:
            span.set_attribute("gen_ai.conversation.id", run.conversation_id)
        if event.kind == EventKind.LLM_CALL:
            span.set_attribute("gen_ai.operation.name", "chat")
            model = _model_of(event)
            if model:
                span.set_attribute("gen_ai.request.model", model)
            # provider.name is Required on inference spans, so resolve it even
            # for a model-less event; omit entirely when it cannot be derived
            # rather than emitting a non-enum placeholder.
            provider = self._provider_for(event)
            if provider:
                span.set_attribute("gen_ai.provider.name", provider)
                if self._emit_legacy_system:
                    _set_system(span, provider)
            span.set_attribute("gen_ai.usage.input_tokens", int(event.prompt_tokens or 0))
            span.set_attribute("gen_ai.usage.output_tokens", int(event.completion_tokens or 0))
            for attr, value in (
                ("gen_ai.usage.cache_read.input_tokens", event.cache_read_tokens),
                ("gen_ai.usage.cache_write.input_tokens", event.cache_write_tokens),
                ("gen_ai.usage.reasoning.output_tokens", event.reasoning_tokens),
            ):
                if value:
                    span.set_attribute(attr, int(value))
            if event.cost_usd:
                # Cost is NOT a GenAI convention attribute - the registry only
                # defines gen_ai.usage.* token counts - so keep it in
                # pactrun's own namespace instead of squatting gen_ai.*.
                span.set_attribute("pactrun.usage.cost", float(event.cost_usd))
        elif event.kind == EventKind.TOOL_CALL:
            span.set_attribute("gen_ai.operation.name", "execute_tool")
            if event.tool_name:
                span.set_attribute("gen_ai.tool.name", event.tool_name)
            call_id = (event.metadata or {}).get("tool_call_id")
            if call_id:
                span.set_attribute("gen_ai.tool.call.id", str(call_id))
            if run.agent_name:
                span.set_attribute("gen_ai.agent.name", run.agent_name)

    def _provider_for(self, event: Any) -> str | None:
        return self._provider or _provider_of(_model_of(event) or "")

    def _evaluation_attributes(self, violation: Any, event: Any) -> dict[str, Any]:
        attrs: dict[str, Any] = {
            "gen_ai.evaluation.name": violation.predicate_name or violation.clause_description or violation.kind.value,
            # monitor mode detected the failure but enforced nothing.
            "gen_ai.evaluation.score.label": "fail" if violation.enforced else "would_fail",
            "gen_ai.evaluation.score.value": 0.0,
            "gen_ai.evaluation.explanation": violation.message,
            "pactrun.violation.severity": violation.severity.value,
            "pactrun.violation.on_fail": violation.on_fail.value,
            "pactrun.violation.enforced": bool(violation.enforced),
        }
        response_id = (getattr(event, "metadata", None) or {}).get("response_id")
        if response_id:
            attrs["gen_ai.response.id"] = str(response_id)
        return attrs

    def _record_metrics(self, event: Any, op: _Op, duration_s: float | None) -> None:
        m = self._metrics
        assert m is not None
        if event.kind == EventKind.LLM_CALL:
            attrs: dict[str, Any] = {"gen_ai.operation.name": "chat"}
            provider = self._provider_for(event)
            if provider:
                attrs["gen_ai.provider.name"] = provider
            model = _model_of(event)
            if model:
                attrs["gen_ai.request.model"] = model
            usage_attrs = {**attrs, "gen_ai.token.modality": "unknown"}
            for key, value in (
                ("input", event.prompt_tokens),
                ("output", event.completion_tokens),
                ("cache_read", event.cache_read_tokens),
                ("cache_write", event.cache_write_tokens),
                ("reasoning", event.reasoning_tokens),
            ):
                m.usage[key].add(int(value or 0), usage_attrs)
            if op.error_type:
                attrs["error.type"] = op.error_type
            m.op_input.record(int(event.prompt_tokens or 0), attrs)
            m.op_output.record(int(event.completion_tokens or 0), attrs)
            if duration_s is not None:
                m.inference_duration.record(duration_s, attrs)
        elif event.kind == EventKind.TOOL_CALL and duration_s is not None:
            attrs = {"gen_ai.tool.name": event.tool_name or "unknown"}
            if op.run is not None and op.run.agent_name:
                attrs["gen_ai.agent.name"] = op.run.agent_name
            if op.error_type:
                attrs["error.type"] = op.error_type
            m.tool_duration.record(duration_s, attrs)


def assert_gen_ai_span(
    span: Any, *, name: str | None = None, model: str | None = None, has_violation: bool | None = None
) -> None:
    """Test helper: assert a captured span matches the gen_ai.* shape pactrun emits."""
    attrs = dict(getattr(span, "attributes", {}) or {})
    if name is not None:
        assert span.name == name, f"span name {span.name!r} != {name!r}"
    if model is not None:
        assert attrs.get("gen_ai.request.model") == model
    if has_violation is not None:
        got = bool(attrs.get("pactrun.violation"))
        assert got == has_violation, f"pactrun.violation={got}, expected {has_violation}"
