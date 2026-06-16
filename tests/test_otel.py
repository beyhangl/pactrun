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
    assert attrs["gen_ai.usage.cost"] == pytest.approx(0.001)


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
