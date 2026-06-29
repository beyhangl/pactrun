"""Tamper-evident audit log — an append-only, hash-chained JSONL ledger.

``AuditLogObserver`` writes one record per event (and per violation) to a JSONL
file, each record carrying the hash of the previous record. Altering, deleting,
or reordering any record breaks the chain, which :func:`verify_audit_log`
detects offline. With a ``secret`` the per-record digest is an HMAC, so an
attacker who can't forge the MAC can't rewrite history undetectably.

This supports record-keeping obligations (e.g. EU AI Act Art. 12 logging) that
the ephemeral OTel spans and the in-memory digest can't: a durable, verifiable
trail. Sensitive argument keys are redacted and model outputs are stored as a
hash by default. Wire it in with ``Session(observers=[AuditLogObserver(path)])``.

stdlib only (``hashlib`` / ``hmac`` / ``json``).
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass

_DEFAULT_REDACT = ("password", "api_key", "apikey", "token", "secret", "authorization")


def _digest(payload: str, secret) -> str:
    if secret:
        key = secret if isinstance(secret, (bytes, bytearray)) else str(secret).encode("utf-8")
        return hmac.new(key, payload.encode("utf-8"), hashlib.sha256).hexdigest()
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class AuditReport:
    """Result of :func:`verify_audit_log`."""
    intact: bool
    records: int
    first_break: int | None = None
    reason: str = ""


class AuditLogObserver:
    """Append a tamper-evident, hash-chained record per event/violation to JSONL.

    Parameters
    ----------
    path: destination JSONL file (appended to; the chain resumes across sessions).
    secret: if set, records are HMAC-chained instead of plain-SHA-256-chained.
    redact_args: tool-argument keys whose values are replaced with a redaction
        marker before writing (recursively).
    include_outputs: if ``False`` (default) model outputs are stored as a SHA-256
        hash, not text.
    """

    def __init__(
        self,
        path,
        *,
        secret=None,
        redact_args=_DEFAULT_REDACT,
        include_outputs: bool = False,
    ) -> None:
        self.path = str(path)
        self._secret = secret
        self._redact = set(redact_args or ())
        self._include_outputs = include_outputs
        self._prev_hash = self._tail_hash()

    # -- chain management ---------------------------------------------------

    def _tail_hash(self) -> str:
        """Resume the chain from the last record of an existing file."""
        last = ""
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        last = line
        except FileNotFoundError:
            return ""
        if last:
            try:
                return json.loads(last).get("hash", "")
            except json.JSONDecodeError:
                return ""
        return ""

    def _redact_value(self, value):
        if isinstance(value, dict):
            return {k: ("***redacted***" if k in self._redact else self._redact_value(v)) for k, v in value.items()}
        if isinstance(value, list):
            return [self._redact_value(v) for v in value]
        return value

    def _write(self, record_type: str, body: dict) -> None:
        record = {"type": record_type, "prev_hash": self._prev_hash}
        record.update(body)
        canonical = json.dumps(record, sort_keys=True, default=str)
        record["hash"] = _digest(canonical, self._secret)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
        self._prev_hash = record["hash"]

    # -- observer protocol --------------------------------------------------

    def on_event(self, event, state) -> None:  # noqa: ARG002 - state unused
        body = {
            "event_id": event.id,
            "kind": getattr(event.kind, "value", str(event.kind)),
            "timestamp": event.timestamp,
            "model": event.model,
            "tool_name": event.tool_name,
            "tool_args": self._redact_value(event.tool_args) if event.tool_args else None,
            "cost_usd": event.cost_usd,
        }
        if event.output is not None and str(event.output) != "":
            if self._include_outputs:
                body["output"] = str(event.output)
            else:
                body["output_sha256"] = hashlib.sha256(str(event.output).encode("utf-8")).hexdigest()
        self._write("event", body)

    def on_violation(self, violation, event=None) -> None:  # noqa: ARG002 - event unused
        self._write("violation", {
            "clause_id": violation.clause_id,
            "clause_description": violation.clause_description,
            "severity": getattr(violation.severity, "value", str(violation.severity)),
            "message": violation.message,
            "timestamp": violation.timestamp,
        })

    def on_session_end(self, state) -> None:
        self._write("session_end", {
            "turn_number": state.turn_number,
            "total_cost_usd": state.total_cost_usd,
            "total_tool_calls": state.total_tool_calls,
        })


def verify_audit_log(path, secret=None) -> AuditReport:
    """Re-walk a ledger and verify its hash chain is intact.

    Returns an :class:`AuditReport`; ``intact`` is ``False`` (with ``first_break``
    set to the 0-based record index) if any record was altered, deleted, or
    reordered, or if ``secret`` doesn't match the one used to write it.
    """
    records = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except FileNotFoundError:
        return AuditReport(intact=False, records=0, first_break=0, reason="file not found")

    prev = ""
    for i, rec in enumerate(records):
        stored = rec.get("hash")
        if rec.get("prev_hash") != prev:
            return AuditReport(False, len(records), i, "chain break (prev_hash mismatch)")
        recomputed = {k: v for k, v in rec.items() if k != "hash"}
        canonical = json.dumps(recomputed, sort_keys=True, default=str)
        if _digest(canonical, secret) != stored:
            return AuditReport(False, len(records), i, "hash mismatch (record altered or wrong secret)")
        prev = stored
    return AuditReport(intact=True, records=len(records), first_break=None, reason="")
