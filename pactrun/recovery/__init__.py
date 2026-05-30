"""Recovery — route contract violations to log / warn / block / escalate / retry / fallback."""

from pactrun.recovery.engine import (
    EscalationError,
    FallbackSignal,
    RetrySignal,
    apply_recovery,
)

__all__ = [
    "apply_recovery",
    "EscalationError",
    "RetrySignal",
    "FallbackSignal",
]
