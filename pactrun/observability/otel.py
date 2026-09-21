"""OpenTelemetry GenAI emitter (experimental).

Emits standard ``gen_ai.*`` CLIENT spans for the LLM and tool calls a pactrun
Session records, and marks a span ERROR when a contract clause is violated — so
your runtime contracts also show up in Langfuse / Arize Phoenix / Datadog / any
OTLP backend. Attach it to a session:

    from pactrun import Contract, cost_under
    from pactrun.observability import OTelObserver

    contract = Contract("agent").require(cost_under(0.50))
    with contract.session(observers=[OTelObserver()]):
        ...

Status: experimental. The GenAI semantic conventions moved out of the core
OpenTelemetry semconv repo (core v1.42.0, June 2026) into a dedicated
repository that has **not cut a release yet** — so there is no stable version
to target. Every ``gen_ai.*`` attribute is still Development status, and the
Python constants for them are all marked deprecated pending that release, so
this module writes the attribute names as literals on purpose. Not
"stable / standard-compliant".
"""

from __future__ import annotations

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


class OTelObserver:
    """Emits ``gen_ai.*`` CLIENT spans per event; sets ERROR status on violations."""

    def __init__(
        self,
        tracer_provider: Any = None,
        semconv_version: str = "unreleased",
        *,
        provider: str | None = None,
        emit_legacy_system: bool = True,
    ) -> None:
        tp = tracer_provider or trace.get_tracer_provider()
        self._tracer = tp.get_tracer("pactrun")
        self._semconv_version = semconv_version
        self._provider = provider
        self._emit_legacy_system = emit_legacy_system
        self._open: dict[str, Any] = {}

    def on_event(self, event: Any, state: Any) -> None:
        if event.kind == EventKind.LLM_CALL:
            name = f"chat {event.model or 'unknown'}"
            kind = SpanKind.CLIENT
        elif event.kind == EventKind.TOOL_CALL:
            name = f"execute_tool {event.tool_name or 'unknown'}"
            # The GenAI conventions say execute_tool spans SHOULD be INTERNAL.
            kind = SpanKind.INTERNAL
        else:
            return
        span = self._tracer.start_span(name, kind=kind)
        self._set_attributes(span, event)
        self._open[event.id] = span

    def on_violation(self, violation: Any, event: Any) -> None:
        span = self._open.get(event.id)
        standalone = span is None
        if standalone:
            span = self._tracer.start_span("pactrun.violation", kind=SpanKind.CLIENT)
        span.set_status(Status(StatusCode.ERROR, violation.message))
        span.set_attribute("pactrun.violation", True)
        span.set_attribute("error.type", violation.clause_description or violation.kind.value)
        try:
            span.record_exception(ViolationError(violation))
        except Exception:
            pass
        if standalone:
            span.end()

    def on_event_end(self, event: Any) -> None:
        span = self._open.pop(event.id, None)
        if span is not None:
            span.end()

    def _set_attributes(self, span: Any, event: Any) -> None:
        if event.kind == EventKind.LLM_CALL:
            span.set_attribute("gen_ai.operation.name", "chat")
            if event.model:
                span.set_attribute("gen_ai.request.model", event.model)
            # provider.name is Required on inference spans, so resolve it even
            # for a model-less event; omit entirely when it cannot be derived
            # rather than emitting a non-enum placeholder.
            provider = self._provider or _provider_of(event.model or "")
            if provider:
                span.set_attribute("gen_ai.provider.name", provider)
                if self._emit_legacy_system:
                    _set_system(span, provider)
            span.set_attribute("gen_ai.usage.input_tokens", int(event.prompt_tokens or 0))
            span.set_attribute("gen_ai.usage.output_tokens", int(event.completion_tokens or 0))
            if event.cost_usd:
                # Cost is NOT a GenAI convention attribute - the registry only
                # defines gen_ai.usage.{input,output}_tokens - so keep it in
                # pactrun's own namespace instead of squatting gen_ai.*.
                span.set_attribute("pactrun.usage.cost", float(event.cost_usd))
        elif event.kind == EventKind.TOOL_CALL:
            span.set_attribute("gen_ai.operation.name", "execute_tool")
            if event.tool_name:
                span.set_attribute("gen_ai.tool.name", event.tool_name)


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
