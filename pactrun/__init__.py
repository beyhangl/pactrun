"""pactrun — Agent Behavioral Contracts.

Design-by-Contract for AI agents. Declare what agents must/must not do,
enforce at runtime, detect behavioral drift, generate compliance docs.
"""

__version__ = "0.1.0"

from pactrun.core.enums import ClauseKind, EventKind, OnFail, Severity
from pactrun.core.errors import ContractLoadError, ViolationError
from pactrun.core.models import (
    Clause, Event, PredicateResult, SessionState, SessionSummary, Violation,
)
from pactrun.contract import Contract
from pactrun.session import Session, get_active_session
from pactrun.predicates.base import predicate, get_predicate, list_predicates
from pactrun.predicates import (
    cost_under, cost_per_turn_under, token_budget,
    must_call, must_not_call, tool_order, tools_allowed, max_tool_calls,
    no_pii, output_contains, output_matches, max_output_length, output_must_not_contain,
    max_latency, session_timeout, max_turns,
    no_loops, max_retries, drift_bounds, no_repeated_output,
)

__all__ = [
    "ClauseKind", "EventKind", "OnFail", "Severity",
    "ContractLoadError", "ViolationError",
    "Clause", "Event", "PredicateResult", "SessionState", "SessionSummary", "Violation",
    "Contract", "Session", "get_active_session",
    "predicate", "get_predicate", "list_predicates",
    # Built-in predicates
    "cost_under", "cost_per_turn_under", "token_budget",
    "must_call", "must_not_call", "tool_order", "tools_allowed", "max_tool_calls",
    "no_pii", "output_contains", "output_matches", "max_output_length", "output_must_not_contain",
    "max_latency", "session_timeout", "max_turns",
    "no_loops", "max_retries", "drift_bounds", "no_repeated_output",
]
