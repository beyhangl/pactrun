"""Recovery — route contract violations to log / warn / block / escalate / retry / fallback."""

from pactrun.recovery.engine import (
    EscalationError,
    FallbackSignal,
    RetrySignal,
    apply_recovery,
)
from pactrun.recovery.webhook import webhook_handler
from pactrun.recovery.approval import cli_approver, auto_approver
from pactrun.recovery.digest import digest, Digest

__all__ = [
    "apply_recovery",
    "EscalationError",
    "RetrySignal",
    "FallbackSignal",
    "webhook_handler",
    "cli_approver",
    "auto_approver",
    "digest",
    "Digest",
]
