"""Custom exceptions for pactrun."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pactrun.core.models import Violation


class ViolationError(Exception):
    """Raised when a contract clause is violated in BLOCK mode."""

    def __init__(self, violation: Violation) -> None:
        self.violation = violation
        super().__init__(f"Contract violation: {violation.message}")


class ContractLoadError(Exception):
    """Raised when a contract YAML/dict cannot be parsed."""
    pass


class SessionError(Exception):
    """Raised for session lifecycle errors."""
    pass
