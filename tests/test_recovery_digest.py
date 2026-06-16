"""Tests for the digest recovery wrapper."""

import pytest

from pactrun import Contract, Severity, cost_under, digest
from pactrun.core.enums import OnFail
from pactrun.core.models import Violation
from pactrun.recovery import Digest


def _v(clause_id="c1", ts=0.0, msg="over budget", sev=Severity.ERROR):
    return Violation(clause_id=clause_id, clause_description="cost under $1",
                     severity=sev, on_fail=OnFail.ESCALATE, timestamp=ts, message=msg)


def test_window_flush_fires_inner_once_after_boundary():
    sink = []
    d = digest(sink.append, window="30s")
    d(_v(ts=0.0))
    d(_v(ts=10.0))
    d(_v(ts=20.0))
    assert sink == []                 # still within window, buffered
    d(_v(ts=40.0))                    # crosses 30s boundary -> flush prior 3
    assert len(sink) == 1
    agg = sink[0]
    assert agg.context_snapshot["count"] == 3
    assert agg.context_snapshot["digest"] is True


def test_aggregate_fields():
    sink = []
    d = digest(sink.append, window="10s", samples=2)
    for i in range(4):
        d(_v(ts=float(i), msg=f"m{i}"))
    d.flush()
    agg = sink[0]
    snap = agg.context_snapshot
    assert snap["count"] == 4
    assert snap["first_ts"] == 0.0 and snap["last_ts"] == 3.0
    assert snap["samples"] == ["m0", "m1"]      # capped at samples=2
    assert "4" in agg.actual


def test_explicit_flush():
    sink = []
    d = digest(sink.append, window="1h")
    d(_v(ts=0.0))
    d(_v(ts=1.0))
    assert sink == []
    d.flush()
    assert len(sink) == 1 and sink[0].context_snapshot["count"] == 2


def test_flush_empty_is_noop():
    sink = []
    digest(sink.append).flush()
    assert sink == []


def test_group_by_clause_default():
    sink = []
    d = digest(sink.append, window="1h")
    d(_v(clause_id="a", ts=0.0))
    d(_v(clause_id="b", ts=1.0))
    d(_v(clause_id="a", ts=2.0))
    d.flush()
    by_group = {a.context_snapshot["group"]: a.context_snapshot["count"] for a in sink}
    assert by_group == {"a": 2, "b": 1}


def test_group_by_callable():
    sink = []
    d = digest(sink.append, window="1h", group_by=lambda v: v.severity.value)
    d(_v(ts=0.0, sev=Severity.ERROR))
    d(_v(ts=1.0, sev=Severity.CRITICAL))
    d(_v(ts=2.0, sev=Severity.ERROR))
    d.flush()
    groups = {a.context_snapshot["group"] for a in sink}
    assert groups == {"error", "critical"}


def test_max_buffer_reports_omitted():
    sink = []
    d = digest(sink.append, window="1h", max_buffer=2)
    for i in range(5):
        d(_v(ts=float(i)))
    d.flush()
    agg = sink[0]
    assert agg.context_snapshot["count"] == 2
    assert agg.context_snapshot["omitted"] == 3
    assert "dropped" in agg.message


def test_worst_severity_propagates():
    sink = []
    d = digest(sink.append, window="1h")
    d(_v(ts=0.0, sev=Severity.WARNING))
    d(_v(ts=1.0, sev=Severity.CRITICAL))
    d.flush()
    assert sink[0].severity == Severity.CRITICAL


def test_is_usable_as_observer_run_end():
    sink = []
    d = digest(sink.append, window="run_end")
    c = Contract("t").require(cost_under(1.0), on_fail="escalate").on_escalate(d)
    with c.session(observers=[d]) as s:
        # Three escalate violations buffer; window='run_end' never flushes mid-run.
        try:
            s.emit_llm_response(model="m", output="x", cost=5.0)
        except Exception:
            pass
    # on_session_end flushed the buffer exactly once.
    assert len(sink) == 1
    assert sink[0].context_snapshot["count"] >= 1


def test_inner_failure_does_not_propagate():
    def boom(v):
        raise RuntimeError("delivery down")

    d = digest(boom, window="1h")
    d(_v(ts=0.0))
    d.flush()  # must not raise


def test_parse_window_units():
    from pactrun.recovery.digest import _parse_window
    assert _parse_window("30s") == 30
    assert _parse_window("5m") == 300
    assert _parse_window("1h") == 3600
    assert _parse_window(45) == 45
    assert _parse_window("run_end") is None


def test_factory_returns_digest():
    assert isinstance(digest(lambda v: None), Digest)


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("httpx") is None, reason="httpx not installed"
)
def test_digest_wraps_webhook():
    import json

    import httpx

    from pactrun import webhook_handler

    sink = []

    def handle(request):
        sink.append(json.loads(request.content))
        return httpx.Response(200)

    transport = httpx.MockTransport(handle)
    d = digest(webhook_handler("https://hook.test/in", transport=transport, throttle_s=0), window="1h")
    d(_v(ts=0.0))
    d(_v(ts=1.0))
    d.flush()
    assert len(sink) == 1
    assert sink[0]["actual"].startswith("2 violations")
