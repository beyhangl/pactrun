"""Record a session's events to a JSONL trace, for replay and policy tests.

``TraceRecorder`` writes each event as one JSON line in the same shape
``Event.to_dict()`` produces, so :func:`pactrun.replay.load_trace` can read it
back and :func:`pactrun.replay.replay` can re-evaluate it against any contract.

Record a real run once, then test contract changes against it offline:

    from pactrun.observability import TraceRecorder

    with contract.session(observers=[TraceRecorder("run.jsonl")]) as s:
        ...

    # later, or in CI
    pactrun replay new_contract.yaml run.jsonl

Unlike :class:`~pactrun.observability.audit.AuditLogObserver`, a trace keeps
model outputs and tool results as text - that is what makes it replayable.
Credential-looking argument keys are still redacted by default. Treat trace
files as sensitive and keep them out of version control unless you have
scrubbed them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_DEFAULT_REDACT = ("password", "api_key", "apikey", "token", "secret", "authorization")


class TraceRecorder:
    """Observer that appends every event to ``path`` as a JSON line.

    Parameters
    ----------
    path: destination file; truncated when the recorder is created unless
        ``append=True``.
    redact_args: tool-argument keys whose values are replaced before writing
        (recursively). Redacted values change what a replay sees, so predicates
        that inspect those arguments cannot be re-tested from the trace.
    append: add to an existing file instead of starting a new one.
    """

    def __init__(self, path, *, redact_args=_DEFAULT_REDACT, append: bool = False) -> None:
        self.path = Path(path)
        self._redact = set(redact_args or ())
        if not append:
            self.path.write_text("", encoding="utf-8")

    def _redact_value(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                k: ("***redacted***" if k in self._redact else self._redact_value(v))
                for k, v in value.items()
            }
        if isinstance(value, list):
            return [self._redact_value(v) for v in value]
        return value

    def on_event(self, event, state) -> None:  # noqa: ARG002 - state unused
        record = event.to_dict()
        if record.get("tool_args"):
            record["tool_args"] = self._redact_value(record["tool_args"])
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
