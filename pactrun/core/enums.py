"""Core enumerations for pactrun."""

from enum import Enum


class Severity(str, Enum):
    """Severity level of a clause violation."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class OnFail(str, Enum):
    """Action to take when a clause is violated."""
    LOG = "log"
    WARN = "warn"
    BLOCK = "block"
    ESCALATE = "escalate"
    RETRY = "retry"
    FALLBACK = "fallback"


class EventKind(str, Enum):
    """Type of event in an agent session."""
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    OUTPUT = "output"
    INPUT = "input"
    ERROR = "error"
    TURN_END = "turn_end"


class ClauseKind(str, Enum):
    """When a clause is evaluated."""
    REQUIRE = "require"       # Must be satisfied (checked per-event or session-end)
    FORBID = "forbid"         # Must never be violated (checked per-event)
    PRECONDITION = "precondition"   # Checked at session start
    POSTCONDITION = "postcondition"  # Checked at session end
