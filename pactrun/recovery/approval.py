"""Approval handlers for the ``approve`` recovery action.

An ``approve``-action clause hands its :class:`~pactrun.core.models.Violation`
to an approval handler that returns truthy to let the run continue or falsy to
block. This ships the simplest one — an interactive terminal prompt — plus an
``auto_approver`` for tests / policy stubs.

Wire one in with ``Contract(...).on_approve(handler)`` or
``wrap(..., approval_handler=handler)``.
"""

from __future__ import annotations

import logging
from typing import Callable

from pactrun.core.models import Violation

logger = logging.getLogger("pactrun")


def cli_approver(*, default_deny: bool = True) -> Callable[[Violation], bool]:
    """Build a synchronous approval handler that prompts on the terminal.

    Prints the violation and reads a ``y/N`` answer from stdin. ``default_deny``
    (the default) blocks on an empty answer or an unreadable stdin — fail closed.
    Not suitable for headless runs; use a webhook/policy handler there.
    """

    def approve(violation: Violation) -> bool:
        prompt = (
            "\n[pactrun] contract violation requires approval:\n"
            f"  {violation.clause_description or violation.clause_id}\n"
            f"  {violation.message}\n"
            f"  expected: {violation.expected}\n"
            f"  actual:   {violation.actual}\n"
            f"  Allow this to proceed? [y/N] "
        )
        try:
            answer = input(prompt)
        except (EOFError, KeyboardInterrupt):
            logger.warning("approval prompt aborted — denying")
            return False
        answer = (answer or "").strip().lower()
        if not answer:
            return not default_deny
        return answer in ("y", "yes")

    return approve


def auto_approver(decision: bool = True) -> Callable[[Violation], bool]:
    """Build a non-interactive approval handler that always returns ``decision``.

    Handy for tests, dry-runs, or a policy stub you replace later.
    """

    def approve(violation: Violation) -> bool:
        return decision

    return approve
