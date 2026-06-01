"""Recovery engine — routes a contract violation to its recovery action.

Each clause carries an ``on_fail`` action. When a clause is violated the
session records the :class:`~pactrun.core.models.Violation` and then calls
:func:`apply_recovery`, which reacts according to that action:

Event-level actions (resolved the instant a clause is violated):

- ``LOG``       — record only (no raise).
- ``WARN``      — record + emit a ``UserWarning`` (no raise).
- ``BLOCK``     — record + raise :class:`~pactrun.core.errors.ViolationError`,
                  halting the run immediately.
- ``ESCALATE``  — record + invoke an escalation handler (e.g. page a human /
                  fire a webhook), then raise :class:`EscalationError`.

Execution-level actions (need control of the call, so they surface here as
control-flow signals that :meth:`Contract.enforce <pactrun.Contract.enforce>`
catches and acts on):

- ``RETRY``     — raise :class:`RetrySignal`; ``@enforce`` re-runs the wrapped
                  function up to ``Contract.max_retries`` times.
- ``FALLBACK``  — raise :class:`FallbackSignal`; ``@enforce`` calls the
                  registered fallback function instead.

Outside ``@enforce`` (a plain ``with contract.session()``), ``RETRY`` and
``FALLBACK`` simply surface as those signals for the caller to handle.
"""

from __future__ import annotations

import logging
import warnings
from typing import Callable

from pactrun.core.enums import OnFail
from pactrun.core.errors import ViolationError
from pactrun.core.models import Violation

logger = logging.getLogger("pactrun")


class EscalationError(ViolationError):
    """Raised after an ``escalate``-action violation has notified its handler."""


class RetrySignal(ViolationError):
    """Control-flow signal: ``@enforce`` should retry the wrapped call."""


class FallbackSignal(ViolationError):
    """Control-flow signal: ``@enforce`` should run the fallback instead."""


def apply_recovery(
    violation: Violation,
    *,
    escalation_handler: Callable[[Violation], None] | None = None,
) -> None:
    """React to a single recorded violation according to its ``on_fail`` action.

    Returns ``None`` for non-halting actions (``LOG`` / ``WARN``). Raises for
    halting or control-flow actions (``BLOCK`` / ``ESCALATE`` / ``RETRY`` /
    ``FALLBACK``). The violation is assumed already recorded by the caller.
    """
    action = violation.on_fail

    if action == OnFail.LOG:
        logger.info("contract violation [log]: %s", violation.message)
        return

    if action == OnFail.WARN:
        warnings.warn(f"contract violation: {violation.message}", stacklevel=2)
        logger.warning("contract violation [warn]: %s", violation.message)
        return

    if action == OnFail.BLOCK:
        raise ViolationError(violation)

    if action == OnFail.ESCALATE:
        logger.error("contract violation [escalate]: %s", violation.message)
        if escalation_handler is not None:
            escalation_handler(violation)
        raise EscalationError(violation)

    if action == OnFail.RETRY:
        raise RetrySignal(violation)

    if action == OnFail.FALLBACK:
        raise FallbackSignal(violation)

    # Unknown action — fail safe by recording only (already done by caller).
    return
