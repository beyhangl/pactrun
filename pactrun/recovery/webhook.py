"""Built-in webhook escalation handler.

``escalate``-action violations call an escalation handler before raising. This
ships a ready-made handler that POSTs the violation to an HTTP endpoint — wire
it straight into ``Contract(... escalation_handler=webhook_handler(url))`` or
``Session(observers=...)`` instead of hand-rolling a requests call.

Two payload shapes:

- ``mode="generic"`` — POSTs ``violation.to_dict()`` as JSON. For your own
  ingest endpoint, a queue, or an automation platform.
- ``mode="chat"`` — POSTs a chat-webhook-shaped body (``text`` + colored
  ``attachments`` with expected/actual fields) that the common team-chat
  "incoming webhook" integrations render as a rich message.

A burst of identical violations (same ``clause_id``) is throttled to one POST
per ``throttle_s`` so a tight agent loop can't flood the channel.

``httpx`` is an optional dependency (``pip install pactrun[webhook]``). The
handler is created eagerly but only imports httpx when first called, so merely
importing this module never requires httpx.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable

from pactrun.core.models import Violation

logger = logging.getLogger("pactrun")

# Chat attachment bar color per severity.
_SEVERITY_COLOR = {
    "info": "#36a64f",      # green
    "warning": "#daa038",   # amber
    "error": "#d00000",     # red
    "critical": "#7b001c",  # dark red
}


def _chat_payload(v: Violation) -> dict:
    color = _SEVERITY_COLOR.get(getattr(v.severity, "value", str(v.severity)), "#d00000")
    title = v.clause_description or v.clause_id or "contract violation"
    return {
        "text": f":warning: Contract violation — {title}",
        "attachments": [
            {
                "color": color,
                "title": title,
                "text": v.message or "",
                "fields": [
                    {"title": "expected", "value": str(v.expected), "short": False},
                    {"title": "actual", "value": str(v.actual), "short": False},
                    {"title": "severity", "value": getattr(v.severity, "value", str(v.severity)), "short": True},
                    {"title": "action", "value": getattr(v.on_fail, "value", str(v.on_fail)), "short": True},
                ],
            }
        ],
    }


def webhook_handler(
    url: str,
    *,
    mode: str = "generic",
    timeout: float = 5.0,
    headers: dict | None = None,
    throttle_s: float = 300.0,
    transport: Any = None,
    strict: bool = False,
) -> Callable[[Violation], None]:
    """Build an escalation handler that POSTs violations to ``url``.

    Parameters
    ----------
    url:
        Destination endpoint.
    mode:
        ``"generic"`` POSTs ``violation.to_dict()``; ``"chat"`` POSTs a
        chat-webhook-shaped body with colored attachments.
    timeout:
        Per-request timeout (seconds).
    headers:
        Extra HTTP headers (``Content-Type: application/json`` is set for you).
    throttle_s:
        Suppress repeat POSTs for the same ``clause_id`` within this many
        seconds. ``0`` disables throttling.
    transport:
        Optional ``httpx`` transport — pass ``httpx.MockTransport(...)`` in
        tests to capture the request without real network I/O.
    strict:
        If ``True``, a delivery failure re-raises; if ``False`` (default), it
        is logged and swallowed so a flaky webhook never masks the underlying
        violation or breaks the run.

    Returns a ``Callable[[Violation], None]`` suitable as ``escalation_handler``.
    """
    if mode not in ("generic", "chat"):
        raise ValueError(f"mode must be 'generic' or 'chat', got {mode!r}")

    last_sent: dict[str, float] = {}

    def handler(violation: Violation) -> None:
        key = violation.clause_id or violation.id
        now = time.time()
        if throttle_s:
            prev = last_sent.get(key)
            if prev is not None and (now - prev) < throttle_s:
                logger.debug("webhook throttled for clause %s", key)
                return

        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - exercised only without httpx
            msg = "pactrun webhook handler needs httpx — `pip install pactrun[webhook]`"
            if strict:
                raise RuntimeError(msg) from exc
            logger.error(msg)
            return

        payload = _chat_payload(violation) if mode == "chat" else violation.to_dict()
        hdrs = {"Content-Type": "application/json"}
        if headers:
            hdrs.update(headers)

        try:
            with httpx.Client(transport=transport, timeout=timeout) as client:
                resp = client.post(url, content=json.dumps(payload, default=str), headers=hdrs)
                resp.raise_for_status()
            last_sent[key] = now
        except Exception as exc:  # noqa: BLE001 - delivery is best-effort by default
            if strict:
                raise
            logger.error("webhook delivery to %s failed: %s", url, exc)

    return handler
