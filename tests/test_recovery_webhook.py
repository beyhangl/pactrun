"""Tests for the built-in webhook escalation handler.

httpx is optional — these run only when it is installed (adapters CI job /
``pip install pactrun[webhook]``).
"""

import json

import pytest

httpx = pytest.importorskip("httpx")

from pactrun import Severity, webhook_handler
from pactrun.core.enums import OnFail
from pactrun.core.models import Violation


def _violation(**kw):
    base = dict(
        clause_id="c1",
        clause_description="cost under $5",
        severity=Severity.ERROR,
        on_fail=OnFail.ESCALATE,
        message="spend exceeded",
        expected="<= $5",
        actual="$7",
    )
    base.update(kw)
    return Violation(**base)


def _capture():
    """Return (transport, sink) where sink collects (url, parsed_body) tuples."""
    sink = []

    def handle(request):
        sink.append((str(request.url), json.loads(request.content)))
        return httpx.Response(200, json={"ok": True})

    return httpx.MockTransport(handle), sink


def test_generic_mode_posts_violation_dict():
    transport, sink = _capture()
    handler = webhook_handler("https://hook.test/in", transport=transport)
    handler(_violation())
    assert len(sink) == 1
    url, body = sink[0]
    assert url == "https://hook.test/in"
    assert body["clause_id"] == "c1"
    assert body["actual"] == "$7"
    assert body["on_fail"] == "escalate"


def test_chat_mode_shapes_attachments():
    transport, sink = _capture()
    handler = webhook_handler("https://hook.test/in", mode="chat", transport=transport)
    handler(_violation())
    _, body = sink[0]
    assert "text" in body
    att = body["attachments"][0]
    assert att["color"] == "#d00000"  # error → red
    titles = {f["title"]: f["value"] for f in att["fields"]}
    assert titles["expected"] == "<= $5"
    assert titles["actual"] == "$7"


def test_chat_color_varies_by_severity():
    transport, sink = _capture()
    handler = webhook_handler("https://hook.test/in", mode="chat", transport=transport, throttle_s=0)
    handler(_violation(severity=Severity.CRITICAL, clause_id="a"))
    handler(_violation(severity=Severity.WARNING, clause_id="b"))
    assert sink[0][1]["attachments"][0]["color"] == "#7b001c"  # critical
    assert sink[1][1]["attachments"][0]["color"] == "#daa038"  # warning


def test_throttles_repeat_clause():
    transport, sink = _capture()
    handler = webhook_handler("https://hook.test/in", transport=transport, throttle_s=300)
    for _ in range(5):
        handler(_violation())  # same clause_id "c1"
    assert len(sink) == 1  # only the first POST goes out


def test_throttle_zero_sends_every_time():
    transport, sink = _capture()
    handler = webhook_handler("https://hook.test/in", transport=transport, throttle_s=0)
    for _ in range(3):
        handler(_violation())
    assert len(sink) == 3


def test_distinct_clauses_not_throttled_together():
    transport, sink = _capture()
    handler = webhook_handler("https://hook.test/in", transport=transport, throttle_s=300)
    handler(_violation(clause_id="x"))
    handler(_violation(clause_id="y"))
    assert len(sink) == 2


def test_delivery_failure_swallowed_by_default():
    def boom(request):
        return httpx.Response(500)

    transport = httpx.MockTransport(boom)
    handler = webhook_handler("https://hook.test/in", transport=transport)
    handler(_violation())  # must not raise


def test_delivery_failure_raises_when_strict():
    def boom(request):
        return httpx.Response(500)

    transport = httpx.MockTransport(boom)
    handler = webhook_handler("https://hook.test/in", transport=transport, strict=True)
    with pytest.raises(httpx.HTTPStatusError):
        handler(_violation())


def test_custom_headers_passed():
    sink = []

    def handle(request):
        sink.append(request.headers.get("authorization"))
        return httpx.Response(200)

    transport = httpx.MockTransport(handle)
    handler = webhook_handler(
        "https://hook.test/in", transport=transport, headers={"Authorization": "Bearer t0ken"}
    )
    handler(_violation())
    assert sink[0] == "Bearer t0ken"


def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        webhook_handler("https://hook.test/in", mode="bogus")


def test_wires_into_contract_escalation():
    """End-to-end: an escalate violation fires the webhook then raises."""
    from pactrun import Contract, cost_under
    from pactrun.recovery import EscalationError

    transport, sink = _capture()
    c = Contract("t").require(cost_under(1.0), on_fail="escalate").on_escalate(
        webhook_handler("https://hook.test/in", transport=transport)
    )
    with pytest.raises(EscalationError):
        with c.session() as s:
            s.emit_llm_response(model="m", output="x", cost=5.0)
    assert len(sink) == 1
    assert sink[0][1]["clause_description"]
