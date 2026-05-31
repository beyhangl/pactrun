"""pactrun — behavioral contracts for AI agents.

Declare what an agent must / must not do across a whole session — cost, tool,
output, timing, and drift limits — and enforce them at runtime.
"""

__version__ = "0.1.0"

from pactrun.core.enums import ClauseKind, EventKind, OnFail, Severity
from pactrun.core.errors import ContractLoadError, ViolationError
from pactrun.core.models import (
    Clause, Event, PredicateResult, SessionState, SessionSummary, Violation,
)
from pactrun.contract import Contract
from pactrun.session import Session, get_active_session
from pactrun.recovery import EscalationError, RetrySignal, FallbackSignal
from pactrun.predicates.base import predicate, get_predicate, list_predicates
from pactrun.predicates import (
    cost_under, cost_per_turn_under, token_budget,
    must_call, must_not_call, tool_order, tools_allowed, max_tool_calls,
    no_pii, output_contains, output_matches, max_output_length, output_must_not_contain,
    max_latency, session_timeout, max_turns,
    no_loops, max_retries, drift_bounds, no_repeated_output,
)
from pactrun.wrap import wrap

__all__ = [
    "ClauseKind", "EventKind", "OnFail", "Severity",
    "ContractLoadError", "ViolationError",
    "EscalationError", "RetrySignal", "FallbackSignal",
    "Clause", "Event", "PredicateResult", "SessionState", "SessionSummary", "Violation",
    "Contract", "Session", "get_active_session", "wrap",
    "predicate", "get_predicate", "list_predicates",
    # Built-in predicates
    "cost_under", "cost_per_turn_under", "token_budget",
    "must_call", "must_not_call", "tool_order", "tools_allowed", "max_tool_calls",
    "no_pii", "output_contains", "output_matches", "max_output_length", "output_must_not_contain",
    "max_latency", "session_timeout", "max_turns",
    "no_loops", "max_retries", "drift_bounds", "no_repeated_output",
]
