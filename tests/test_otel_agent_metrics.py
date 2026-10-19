"""OTelObserver: invoke_agent span tree, evaluation events and GenAI metrics."""

import pytest

pytest.importorskip("opentelemetry.sdk")

from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode

from pactrun import Contract, cost_under, must_not_call
from pactrun.observability import OTelObserver


def _observer(metrics=True, **kw):
    exporter = InMemorySpanExporter()
    tp = TracerProvider()
    tp.add_span_processor(SimpleSpanProcessor(exporter))
    reader = InMemoryMetricReader() if metrics else None
    if reader is not None:
        kw["meter_provider"] = MeterProvider(metric_readers=[reader])
    return OTelObserver(tracer_provider=tp, **kw), exporter, reader


def _by_name(exporter):
    return {sp.name: sp for sp in exporter.get_finished_spans()}


def _points(reader):
    """{metric name: [(attributes, point)]} from one collection."""
    out = {}
    data = reader.get_metrics_data()
    for rm in data.resource_metrics if data else []:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                assert metric.name not in out
                out[metric.name] = [(dict(p.attributes), p) for p in metric.data.data_points]
                out[metric.name + ".unit"] = metric.unit
    return out


def _evaluations(span):
    return [dict(e.attributes) for e in span.events if e.name == "gen_ai.evaluation.result"]


# ---------------------------------------------------------------------------
# Span tree
# ---------------------------------------------------------------------------

def test_invoke_agent_root_parents_chat_and_tool_spans():
    observer, exporter, _ = _observer()
    with Contract("support").session(observers=[observer]) as s:
        s.emit_llm_response(model="gpt-4.1", output="hi", prompt_tokens=3, completion_tokens=2)
        s.emit_tool_call("search")
    spans = _by_name(exporter)
    root = spans["invoke_agent support"]
    assert root.kind is SpanKind.INTERNAL
    assert root.parent is None
    for name in ("chat gpt-4.1", "execute_tool search"):
        child = spans[name]
        assert child.parent.span_id == root.context.span_id
        assert child.context.trace_id == root.context.trace_id
    attrs = dict(root.attributes)
    assert attrs["gen_ai.operation.name"] == "invoke_agent"
    assert attrs["gen_ai.agent.name"] == "support"  # contract name by default
    assert attrs["pactrun.session.id"] == s.session_id
    assert attrs["gen_ai.usage.input_tokens"] == 3
    assert attrs["gen_ai.usage.output_tokens"] == 2
    assert "gen_ai.conversation.id" not in attrs  # never a made-up fallback


def test_agent_and_conversation_attributes():
    observer, exporter, _ = _observer(
        agent_name="Math Tutor", agent_id="asst_1", agent_description="Helps", agent_version="2.0",
        conversation_id="conv_1",
    )
    with Contract("t").session(observers=[observer]) as s:
        s.emit_llm_response(model="claude-sonnet-4-6", output="x")
        s.emit_tool_call("calc", metadata={"tool_call_id": "call_abc"})
        s.emit_tool_call("calc2")
    spans = _by_name(exporter)
    root = dict(spans["invoke_agent Math Tutor"].attributes)
    assert root["gen_ai.agent.name"] == "Math Tutor"
    assert root["gen_ai.agent.id"] == "asst_1"
    assert root["gen_ai.agent.description"] == "Helps"
    assert root["gen_ai.agent.version"] == "2.0"
    assert root["gen_ai.conversation.id"] == "conv_1"
    assert dict(spans["chat claude-sonnet-4-6"].attributes)["gen_ai.conversation.id"] == "conv_1"
    tool = dict(spans["execute_tool calc"].attributes)
    assert tool["gen_ai.tool.call.id"] == "call_abc"
    assert tool["gen_ai.agent.name"] == "Math Tutor"
    assert tool["gen_ai.conversation.id"] == "conv_1"
    # Only when the provider supplied one.
    assert "gen_ai.tool.call.id" not in dict(spans["execute_tool calc2"].attributes)


def test_conversation_id_from_session_metadata():
    observer, exporter, _ = _observer()
    with Contract("t").session(observers=[observer], metadata={"conversation_id": "thread_9"}) as s:
        s.emit_llm_response(model="gpt-4.1", output="x")
    root = dict(_by_name(exporter)["invoke_agent t"].attributes)
    assert root["gen_ai.conversation.id"] == "thread_9"


def test_unnamed_contract_gives_bare_invoke_agent():
    observer, exporter, _ = _observer()
    with Contract().session(observers=[observer]) as s:
        s.emit_tool_call("x")
    root = _by_name(exporter)["invoke_agent"]
    assert "gen_ai.agent.name" not in dict(root.attributes)


@pytest.mark.parametrize("model", [None, "", "unknown"])
def test_chat_span_without_model_is_named_chat(model):
    observer, exporter, _ = _observer()
    with Contract("t").session(observers=[observer]) as s:
        s.emit_llm_response(model=model, output="x")
    spans = _by_name(exporter)
    assert "chat" in spans
    assert not any(name.startswith("chat ") for name in spans)
    assert "gen_ai.request.model" not in dict(spans["chat"].attributes)


def test_emit_llm_spans_false_keeps_agent_tool_and_metrics():
    observer, exporter, reader = _observer(emit_llm_spans=False)
    contract = Contract("t").require(cost_under(0.0001), on_fail="log")
    with contract.session(observers=[observer]) as s:
        s.emit_llm_response(model="gpt-4.1", output="x", prompt_tokens=10, completion_tokens=5,
                            cost=0.05, duration_ms=200)
        s.emit_tool_call("search")
    spans = _by_name(exporter)
    assert not any(name.startswith("chat") for name in spans)
    assert "execute_tool search" in spans
    root = spans["invoke_agent t"]
    # The chat span is not ours to annotate, so the evaluation lands on the agent span.
    (ev,) = _evaluations(root)
    assert ev["gen_ai.evaluation.name"] == "cost_under"
    assert root.status.status_code == StatusCode.ERROR
    points = _points(reader)
    assert points["gen_ai.client.inference.usage.input_tokens"][0][1].value == 10
    assert points["gen_ai.client.inference.duration"][0][1].sum == pytest.approx(0.2)


def test_emit_agent_span_false_keeps_old_flat_shape():
    from pactrun.predicates import max_tool_calls

    observer, exporter, _ = _observer(emit_agent_span=False)
    contract = Contract("t").require(max_tool_calls(0), on_fail="log", check_on="session_end")
    with contract.session(observers=[observer]) as s:
        s.emit_tool_call("x")
    chat, standalone = exporter.get_finished_spans()
    assert chat.name == "execute_tool x"
    assert chat.parent is None
    # Session-level violations fall back to the old standalone span.
    assert standalone.name == "pactrun.violation"
    assert _evaluations(standalone)


def test_observer_reused_across_sessions_gets_one_root_each():
    observer, exporter, _ = _observer()
    for _ in range(2):
        with Contract("t").session(observers=[observer]) as s:
            s.emit_tool_call("x")
    roots = [sp for sp in exporter.get_finished_spans() if sp.name == "invoke_agent t"]
    assert len(roots) == 2
    assert roots[0].context.trace_id != roots[1].context.trace_id


def test_chat_span_covers_reported_duration():
    observer, exporter, _ = _observer()
    with Contract("t").session(observers=[observer]) as s:
        s.emit_llm_response(model="gpt-4.1", output="x", duration_ms=1500)
    span = _by_name(exporter)["chat gpt-4.1"]
    assert (span.end_time - span.start_time) / 1e9 == pytest.approx(1.5, abs=0.05)


def test_failed_llm_call_sets_error_type():
    observer, exporter, reader = _observer()
    with Contract("t").session(observers=[observer]) as s:
        s.emit_llm_response(model="gpt-4.1", output="", metadata={"error": "rate limited"})
    span = _by_name(exporter)["chat gpt-4.1"]
    assert span.status.status_code == StatusCode.ERROR
    assert dict(span.attributes)["error.type"] == "_OTHER"
    # Token histograms are recorded for failed operations too, with 0.
    ((attrs, point),) = _points(reader)["gen_ai.client.inference.operation.input_tokens"]
    assert attrs["error.type"] == "_OTHER"
    assert point.sum == 0


# ---------------------------------------------------------------------------
# gen_ai.evaluation.result
# ---------------------------------------------------------------------------

def test_enforced_violation_emits_failing_evaluation_on_its_span():
    observer, exporter, _ = _observer()
    contract = Contract("t").require(cost_under(0.0001), on_fail="log")
    with contract.session(observers=[observer]) as s:
        s.emit_llm_response(model="gpt-4.1", output="x", cost=0.05)
    spans = _by_name(exporter)
    chat = spans["chat gpt-4.1"]
    (ev,) = _evaluations(chat)
    assert ev["gen_ai.evaluation.name"] == "cost_under"
    assert ev["gen_ai.evaluation.score.label"] == "fail"
    assert ev["gen_ai.evaluation.score.value"] == 0.0
    assert isinstance(ev["gen_ai.evaluation.score.value"], float)
    assert ev["gen_ai.evaluation.explanation"] == s.violations[0].message
    assert ev["pactrun.violation.enforced"] is True
    # Backward compatible: still ERROR + exception event + pactrun.violation.
    assert chat.status.status_code == StatusCode.ERROR
    assert dict(chat.attributes)["pactrun.violation"] is True
    assert any(e.name == "exception" for e in chat.events)
    assert not _evaluations(spans["invoke_agent t"])


def test_monitor_violation_is_labelled_would_fail():
    observer, exporter, _ = _observer()
    contract = Contract("t").forbid(must_not_call("delete_user"))
    with contract.session(observers=[observer], mode="monitor") as s:
        s.emit_tool_call("delete_user")
    (ev,) = _evaluations(_by_name(exporter)["execute_tool delete_user"])
    assert ev["gen_ai.evaluation.name"] == "must_not_call"
    assert ev["gen_ai.evaluation.score.label"] == "would_fail"
    assert ev["pactrun.violation.enforced"] is False


def test_blocking_violation_still_closes_spans():
    from pactrun.core.errors import ViolationError

    observer, exporter, reader = _observer()
    contract = Contract("t").forbid(must_not_call("delete_user"))
    with pytest.raises(ViolationError):
        with contract.session(observers=[observer]) as s:
            s.emit_tool_call("delete_user", duration_ms=10)
    spans = _by_name(exporter)
    assert _evaluations(spans["execute_tool delete_user"])[0]["gen_ai.evaluation.score.label"] == "fail"
    assert "invoke_agent t" in spans
    ((attrs, _),) = _points(reader)["gen_ai.execute_tool.duration"]
    assert attrs["error.type"]


def test_session_end_violation_lands_on_agent_span():
    from pactrun.predicates import max_tool_calls

    observer, exporter, reader = _observer()
    contract = Contract("t").require(max_tool_calls(0), on_fail="log", check_on="session_end")
    with contract.session(observers=[observer]) as s:
        s.emit_tool_call("x")
    spans = _by_name(exporter)
    assert "pactrun.violation" not in spans
    root = spans["invoke_agent t"]
    assert root.status.status_code == StatusCode.ERROR
    assert _evaluations(root)
    ((attrs, _),) = _points(reader)["gen_ai.invoke_agent.duration"]
    assert "error.type" in attrs


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_usage_counters_and_histograms():
    observer, _, reader = _observer()
    with Contract("t").session(observers=[observer]) as s:
        s.emit_llm_response(model="claude-sonnet-4-6", output="", prompt_tokens=102_000,
                            completion_tokens=1_000, cache_read_tokens=60_000,
                            cache_write_tokens=40_000, reasoning_tokens=300, duration_ms=2500)
        s.emit_llm_response(model="claude-sonnet-4-6", output="", prompt_tokens=10, completion_tokens=1)
    p = _points(reader)
    expected = {
        "gen_ai.client.inference.usage.input_tokens": 102_010,
        "gen_ai.client.inference.usage.output_tokens": 1_001,
        "gen_ai.client.inference.usage.cache_read.input_tokens": 60_000,
        "gen_ai.client.inference.usage.cache_write.input_tokens": 40_000,
        "gen_ai.client.inference.usage.reasoning.output_tokens": 300,
    }
    for name, total in expected.items():
        ((attrs, point),) = p[name]
        assert point.value == total, name
        assert p[name + ".unit"] == "{token}"
        assert attrs == {
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": "anthropic",
            "gen_ai.request.model": "claude-sonnet-4-6",
            "gen_ai.token.modality": "unknown",
        }
    ((attrs, hist),) = p["gen_ai.client.inference.operation.input_tokens"]
    assert (hist.count, hist.sum) == (2, 102_010)
    assert "gen_ai.token.modality" not in attrs
    ((_, out),) = p["gen_ai.client.inference.operation.output_tokens"]
    assert (out.count, out.sum) == (2, 1_001)
    # Only the call that reported a duration; seconds, not ms.
    ((_, dur),) = p["gen_ai.client.inference.duration"]
    assert p["gen_ai.client.inference.duration.unit"] == "s"
    assert dur.count == 1
    assert dur.sum == pytest.approx(2.5, abs=0.05)
    # Replaced metrics are not emitted.
    assert "gen_ai.client.token.usage" not in p
    assert "gen_ai.client.operation.duration" not in p


def test_invoke_agent_metrics():
    observer, _, reader = _observer(agent_name="bot")
    with Contract("t").session(observers=[observer]) as s:
        s.emit_llm_response(model="gpt-4.1", output="")
        s.emit_llm_response(model="gpt-4.1", output="")
        s.emit_tool_call("a")
    p = _points(reader)
    ((attrs, calls),) = p["gen_ai.invoke_agent.inference_calls"]
    assert attrs == {"gen_ai.agent.name": "bot"}
    assert (calls.count, calls.sum) == (1, 2)
    ((_, tools),) = p["gen_ai.invoke_agent.tool_calls"]
    assert tools.sum == 1
    ((attrs, dur),) = p["gen_ai.invoke_agent.duration"]
    assert attrs == {"gen_ai.agent.name": "bot"}
    assert p["gen_ai.invoke_agent.duration.unit"] == "s"
    assert dur.count == 1


def test_no_metrics_without_a_meter_provider():
    # No meter_provider= and no global one configured: the observer falls back
    # to the API's global proxy, which records nothing, and spans still work.
    unrelated = InMemoryMetricReader()
    MeterProvider(metric_readers=[unrelated])
    observer, exporter, _ = _observer(metrics=False)
    with Contract("t").session(observers=[observer]) as s:
        s.emit_llm_response(model="gpt-4.1", output="", prompt_tokens=5, duration_ms=10)
    assert exporter.get_finished_spans()
    assert _points(unrelated) == {}


def test_no_op_meter_provider_creates_no_instruments():
    from opentelemetry.metrics import NoOpMeterProvider

    observer = OTelObserver(tracer_provider=TracerProvider(), meter_provider=NoOpMeterProvider())
    assert observer._metrics is None
    with Contract("t").session(observers=[observer]) as s:
        s.emit_llm_response(model="gpt-4.1", output="", duration_ms=10)
