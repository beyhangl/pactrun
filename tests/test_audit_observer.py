"""Tests for the tamper-evident AuditLogObserver + verify_audit_log."""

import json

from pactrun import Contract, cost_under
from pactrun.observability import AuditLogObserver, verify_audit_log


def _run(path, *, secret=None, **kw):
    obs = AuditLogObserver(path, secret=secret, **kw)
    c = Contract("t").require(cost_under(1.0), on_fail="log")
    with c.session(observers=[obs]) as s:
        s.emit_llm_response(model="m", output="hello", cost=0.1)
        s.emit_tool_call("search", args={"q": "x", "api_key": "sk-SECRET-VALUE"})
        s.emit_llm_response(model="m", output="over budget", cost=5.0)  # trips cost_under
    return path


def test_happy_path_chain_verifies(tmp_path):
    p = _run(str(tmp_path / "audit.jsonl"))
    report = verify_audit_log(p)
    assert report.intact
    assert report.records > 0
    assert report.first_break is None


def test_records_include_events_violation_and_end(tmp_path):
    p = _run(str(tmp_path / "audit.jsonl"))
    types = [json.loads(line)["type"] for line in open(p)]
    assert "event" in types
    assert "violation" in types
    assert types[-1] == "session_end"


def test_tamper_detected(tmp_path):
    p = _run(str(tmp_path / "audit.jsonl"))
    lines = open(p).read().splitlines()
    rec = json.loads(lines[0])
    rec["model"] = "tampered"  # alter a field, keep its (now-stale) hash
    lines[0] = json.dumps(rec)
    open(p, "w").write("\n".join(lines) + "\n")
    report = verify_audit_log(p)
    assert not report.intact
    assert report.first_break == 0


def test_deletion_detected(tmp_path):
    p = _run(str(tmp_path / "audit.jsonl"))
    lines = open(p).read().splitlines()
    del lines[1]  # drop a record -> chain breaks at the next one
    open(p, "w").write("\n".join(lines) + "\n")
    assert not verify_audit_log(p).intact


def test_reorder_detected(tmp_path):
    p = _run(str(tmp_path / "audit.jsonl"))
    lines = open(p).read().splitlines()
    lines[1], lines[2] = lines[2], lines[1]
    open(p, "w").write("\n".join(lines) + "\n")
    assert not verify_audit_log(p).intact


def test_hmac_secret_binds(tmp_path):
    p = _run(str(tmp_path / "audit.jsonl"), secret="k3y")
    assert verify_audit_log(p, secret="k3y").intact
    assert not verify_audit_log(p, secret="wrong").intact
    assert not verify_audit_log(p).intact  # no secret can't verify an HMAC chain


def test_redaction_and_no_secret_leak(tmp_path):
    p = _run(str(tmp_path / "audit.jsonl"))
    blob = open(p).read()
    assert "sk-SECRET-VALUE" not in blob   # api_key value redacted
    assert "***redacted***" in blob


def test_outputs_hashed_by_default(tmp_path):
    p = _run(str(tmp_path / "audit.jsonl"))
    blob = open(p).read()
    assert "hello" not in blob              # output text not stored
    assert "output_sha256" in blob


def test_include_outputs_writes_text(tmp_path):
    p = _run(str(tmp_path / "audit.jsonl"), include_outputs=True)
    assert "hello" in open(p).read()


def test_append_resume_across_sessions(tmp_path):
    p = str(tmp_path / "audit.jsonl")
    _run(p)
    n1 = len(open(p).read().splitlines())
    _run(p)  # second session appends, chain continues
    assert len(open(p).read().splitlines()) > n1
    assert verify_audit_log(p).intact  # the joined chain is still intact


def test_missing_file_reports_not_intact(tmp_path):
    report = verify_audit_log(str(tmp_path / "nope.jsonl"))
    assert not report.intact


def test_nested_redaction(tmp_path):
    p = str(tmp_path / "audit.jsonl")
    obs = AuditLogObserver(p)
    c = Contract("t")
    with c.session(observers=[obs]) as s:
        s.emit_tool_call("login", args={"creds": {"password": "hunter2"}})
    assert "hunter2" not in open(p).read()
    assert verify_audit_log(p).intact
