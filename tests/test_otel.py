"""Tests for the OpenTelemetry GenAI emitter (OTelObserver)."""

import pytest

pytest.importorskip("opentelemetry.sdk")

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from pactrun import Contract, cost_under
from pactrun.observability import OTelObserver, assert_gen_ai_span


@pytest.fixture
def pact_spans():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return OTelObserver(tracer_provider=provider), exporter


def test_llm_call_emits_gen_ai_span(pact_spans):
    observer, exporter = pact_spans
    with Contract("t").session(observers=[observer]) as s:
        s.emit_llm_response(
            model="gpt-4.1", output="hi", prompt_tokens=30, completion_tokens=12, cost=0.001
        )
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert_gen_ai_span(span, name="chat gpt-4.1", model="gpt-4.1", has_violation=False)
    attrs = dict(span.attributes)
    assert attrs["gen_ai.operation.name"] == "chat"
    assert attrs["gen_ai.usage.input_tokens"] == 30
    assert attrs["gen_ai.usage.output_tokens"] == 12
    assert attrs["gen_ai.provider.name"] == "openai"
    assert attrs["pactrun.usage.cost"] == pytest.approx(0.001)
    assert "gen_ai.usage.cost" not in attrs  # not a GenAI convention attribute


def test_tool_call_emits_span(pact_spans):
    observer, exporter = pact_spans
    with Contract("t").session(observers=[observer]) as s:
        s.emit_tool_call("search")
    assert any(sp.name == "execute_tool search" for sp in exporter.get_finished_spans())


def test_violation_sets_error_status(pact_spans):
    observer, exporter = pact_spans
    contract = Contract("t").require(cost_under(0.0001), on_fail="log")
    with contract.session(observers=[observer]) as s:
        s.emit_llm_response(model="gpt-4.1", output="x", prompt_tokens=1000, completion_tokens=1000, cost=0.05)
    span = exporter.get_finished_spans()[0]
    assert span.status.status_code == StatusCode.ERROR
    assert dict(span.attributes).get("pactrun.violation") is True


def test_no_observer_is_a_clean_noop():
    with Contract("t").session() as s:
        s.emit_llm_response(model="gpt-4.1", output="hi", cost=0.001)
    assert s.is_compliant


# ---------------------------------------------------------------------------
# GenAI-convention conformance
# ---------------------------------------------------------------------------

def _observer(**kw):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return OTelObserver(tracer_provider=provider, **kw), exporter


def test_execute_tool_span_is_internal_not_client():
    from opentelemetry.trace import SpanKind

    observer, exporter = _observer()
    with Contract("t").session(observers=[observer]) as s:
        s.emit_tool_call("search", args={"q": "x"})
    span = exporter.get_finished_spans()[0]
    # The conventions say execute_tool spans SHOULD be INTERNAL.
    assert span.kind is SpanKind.INTERNAL
    assert dict(span.attributes)["gen_ai.operation.name"] == "execute_tool"


def test_chat_span_stays_client():
    from opentelemetry.trace import SpanKind

    observer, exporter = _observer()
    with Contract("t").session(observers=[observer]) as s:
        s.emit_llm_response(model="gpt-4.1", output="hi")
    assert exporter.get_finished_spans()[0].kind is SpanKind.CLIENT


def test_unknown_provider_is_omitted_not_invalid():
    observer, exporter = _observer()
    with Contract("t").session(observers=[observer]) as s:
        s.emit_llm_response(model="some-inhouse-model", output="hi")
    attrs = dict(exporter.get_finished_spans()[0].attributes)
    # provider.name is an enum; a placeholder would be invalid, so omit it.
    assert "gen_ai.provider.name" not in attrs
    assert attrs.get("gen_ai.system") is None


def test_explicit_provider_override_is_used():
    observer, exporter = _observer(provider="aws.bedrock")
    with Contract("t").session(observers=[observer]) as s:
        s.emit_llm_response(model="some-inhouse-model", output="hi")
    attrs = dict(exporter.get_finished_spans()[0].attributes)
    assert attrs["gen_ai.provider.name"] == "aws.bedrock"


def test_legacy_system_attribute_can_be_disabled():
    observer, exporter = _observer(emit_legacy_system=False)
    with Contract("t").session(observers=[observer]) as s:
        s.emit_llm_response(model="gpt-4.1", output="hi")
    attrs = dict(exporter.get_finished_spans()[0].attributes)
    assert attrs["gen_ai.provider.name"] == "openai"
    assert "gen_ai.system" not in attrs  # removed from the GenAI registry
