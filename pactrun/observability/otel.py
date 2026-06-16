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

Status: experimental — tracks the OpenTelemetry GenAI semantic conventions
(Development status), pinned to v1.27. Not "stable / standard-compliant".
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


def _provider_of(model: str) -> str:
    m = (model or "").lower()
    if m.startswith(("gpt", "o1", "o3", "o4")):
        return "openai"
    if m.startswith("claude"):
        return "anthropic"
    if m.startswith("gemini"):
        return "gcp.gemini"
    return "unknown"


def _set_system(span: Any, provider: str) -> None:
    # gen_ai.system was renamed to gen_ai.provider.name in semconv v1.37; the
    # installed 0.48b0 ships only gen_ai.system. Set the old constant too.
    try:
        from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as ga

        span.set_attribute(ga.GEN_AI_SYSTEM, provider)
    except Exception:
        span.set_attribute("gen_ai.system", provider)


class OTelObserver:
    """Emits ``gen_ai.*`` CLIENT spans per event; sets ERROR status on violations."""

    def __init__(self, tracer_provider: Any = None, semconv_version: str = "1.27") -> None:
        provider = tracer_provider or trace.get_tracer_provider()
        self._tracer = provider.get_tracer("pactrun")
        self._semconv_version = semconv_version
        self._open: dict[str, Any] = {}

    def on_event(self, event: Any, state: Any) -> None:
        if event.kind == EventKind.LLM_CALL:
            name = f"chat {event.model or 'unknown'}"
        elif event.kind == EventKind.TOOL_CALL:
            name = f"execute_tool {event.tool_name or 'unknown'}"
        else:
            return
        span = self._tracer.start_span(name, kind=SpanKind.CLIENT)
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
                provider = _provider_of(event.model)
                span.set_attribute("gen_ai.request.model", event.model)
                span.set_attribute("gen_ai.provider.name", provider)  # v1.37 name
                _set_system(span, provider)                            # v1.27 name
            span.set_attribute("gen_ai.usage.input_tokens", int(event.prompt_tokens or 0))
            span.set_attribute("gen_ai.usage.output_tokens", int(event.completion_tokens or 0))
            if event.cost_usd:
                # pactrun extra (the GenAI spec omits cost); from the real
                # post-call usage, not the pre-call worst-case estimate.
                span.set_attribute("gen_ai.usage.cost", float(event.cost_usd))
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
