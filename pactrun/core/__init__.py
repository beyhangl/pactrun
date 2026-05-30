"""Core types and models for pactrun."""

from pactrun.core.enums import ClauseKind, EventKind, OnFail, Severity
from pactrun.core.errors import ContractLoadError, SessionError, ViolationError
from pactrun.core.models import (
    Clause,
    Event,
    PredicateResult,
    SessionState,
    SessionSummary,
    Violation,
)

__all__ = [
    "ClauseKind", "EventKind", "OnFail", "Severity",
    "ContractLoadError", "SessionError", "ViolationError",
    "Clause", "Event", "PredicateResult", "SessionState", "SessionSummary", "Violation",
]
