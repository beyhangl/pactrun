"""Digest recovery wrapper — collect violations and emit one aggregated alert.

The inverse of :func:`~pactrun.recovery.webhook.webhook_handler`'s throttle:
throttle *drops* repeat violations (and loses the count); a digest *buffers*
them and fires the inner action once per window with an aggregate — total
count, first/last timestamp, a capped sample of messages, and how many were
omitted. Wrap any ``escalation_handler`` (a ``Callable[[Violation], None]``):

    from pactrun import Contract, cost_under, webhook_handler
    from pactrun.recovery import digest

    contract = (
        Contract("agent")
        .require(cost_under(0.05), on_fail="escalate")
        .on_escalate(digest(webhook_handler(url), window="30s"))
    )

Window mode flushes lazily on the next violation that crosses the (event-time)
window boundary — no Session change needed. For a single end-of-run summary,
use ``window="run_end"`` and register the digest as an observer
(``Session(observers=[d])``); it flushes from ``on_session_end``. You can also
call :meth:`Digest.flush` explicitly at any time.
"""

from __future__ import annotations

import logging
from typing import Callable

from pactrun.core.enums import OnFail, Severity
from pactrun.core.models import Violation

logger = logging.getLogger("pactrun")

_SEVERITY_ORDER = {Severity.INFO: 0, Severity.WARNING: 1, Severity.ERROR: 2, Severity.CRITICAL: 3}


def _parse_window(window) -> float | None:
    """Seconds for '30s'/'5m'/'1h'/number, or None for the 'run_end' sentinel."""
    if window in ("run_end", "until_end", None):
        return None
    if isinstance(window, (int, float)):
        return float(window)
    text = str(window).strip().lower()
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    if text and text[-1] in units:
        return float(text[:-1]) * units[text[-1]]
    return float(text)


def _bucket_key(v: Violation, group_by) -> str:
    if callable(group_by):
        return str(group_by(v))
    if group_by == "predicate":
        return getattr(v, "predicate_name", "") or v.clause_id or "ungrouped"
    return v.clause_id or "ungrouped"  # default: by clause


class Digest:
    """Callable escalation handler that batches violations into aggregate alerts.

    Instances are also valid observers (they expose ``on_session_end``), so the
    same object can be passed to ``Session(observers=[...])`` for run-end flush.
    """

    def __init__(
        self,
        inner_action: Callable[[Violation], None],
        *,
        window="30s",
        group_by="clause",
        max_buffer: int = 1000,
        samples: int = 5,
    ) -> None:
        self._inner = inner_action
        self._window_s = _parse_window(window)
        self._group_by = group_by
        self._max_buffer = max_buffer
        self._samples = samples
        self._buffer: list[Violation] = []
        self._omitted = 0
        self._window_start: float | None = None

    # -- escalation-handler protocol ---------------------------------------

    def __call__(self, violation: Violation) -> None:
        ts = violation.timestamp
        if (
            self._window_s is not None
            and self._buffer
            and self._window_start is not None
            and (ts - self._window_start) >= self._window_s
        ):
            self.flush()
        if not self._buffer:
            self._window_start = ts
        if len(self._buffer) < self._max_buffer:
            self._buffer.append(violation)
        else:
            self._omitted += 1

    # -- observer protocol --------------------------------------------------

    def on_event(self, event, state) -> None:  # noqa: ARG002 - observer no-op
        """No-op: a digest reacts to violations via __call__, not raw events."""

    def on_session_end(self, state) -> None:  # noqa: ARG002 - state unused
        self.flush()

    # -- aggregation --------------------------------------------------------

    def flush(self) -> None:
        """Emit one aggregate alert per group for the buffered violations, then clear."""
        if not self._buffer:
            return
        buffer, omitted = self._buffer, self._omitted
        self._buffer, self._omitted, self._window_start = [], 0, None

        groups: dict[str, list[Violation]] = {}
        for v in buffer:
            groups.setdefault(_bucket_key(v, self._group_by), []).append(v)

        for key, items in groups.items():
            try:
                self._inner(self._aggregate(key, items, omitted))
            except Exception as exc:  # noqa: BLE001 - one bad delivery must not drop the rest
                logger.error("digest inner action failed for group %s: %s", key, exc)

    def _aggregate(self, key: str, items: list[Violation], omitted: int) -> Violation:
        count = len(items)
        first = min(i.timestamp for i in items)
        last = max(i.timestamp for i in items)
        worst = max(items, key=lambda i: _SEVERITY_ORDER.get(i.severity, 0))
        sample_msgs = [i.message for i in items[: self._samples]]
        omitted_note = f"; {omitted} dropped (buffer cap)" if omitted else ""
        summary = (
            f"{count} '{key}' violation(s) in {last - first:.0f}s{omitted_note}. "
            f"Samples: " + " | ".join(m for m in sample_msgs if m)
        )
        return Violation(
            clause_id=key,
            clause_description=f"digest: {count} x {worst.clause_description or key}",
            kind=worst.kind,
            severity=worst.severity,
            on_fail=OnFail.ESCALATE,
            timestamp=last,
            message=summary,
            expected="0 violations",
            actual=f"{count} violations (first {first:.0f}, last {last:.0f})",
            context_snapshot={
                "digest": True,
                "group": key,
                "count": count,
                "first_ts": first,
                "last_ts": last,
                "omitted": omitted,
                "samples": sample_msgs,
            },
        )


def digest(
    inner_action: Callable[[Violation], None],
    *,
    window="30s",
    group_by="clause",
    max_buffer: int = 1000,
    samples: int = 5,
) -> Digest:
    """Build a :class:`Digest` wrapping ``inner_action`` (see module docstring)."""
    return Digest(
        inner_action,
        window=window,
        group_by=group_by,
        max_buffer=max_buffer,
        samples=samples,
    )
