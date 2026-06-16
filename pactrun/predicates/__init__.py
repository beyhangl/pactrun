"""Predicate system — registry, base types, and all built-in predicates."""

from pactrun.predicates.base import predicate, get_predicate, list_predicates

# Import all built-in predicates to register them
from pactrun.predicates.cost import cost_under, cost_per_turn_under, token_budget
from pactrun.predicates.tools import (
    must_call, must_not_call, tool_order, tools_allowed, max_tool_calls,
    tool_args_match, no_destructive_args, tool_path_within,
    tool_arg_value_guard, required_disclosure, tool_host_within,
)
from pactrun.predicates.output import (
    no_pii, output_contains, output_matches, max_output_length, output_must_not_contain,
    valid_json, json_schema_valid, no_secrets,
)
from pactrun.predicates.timing import max_latency, session_timeout, max_turns
from pactrun.predicates.behavioral import no_loops, max_retries, drift_bounds, no_repeated_output
from pactrun.predicates.ratelimit import (
    spend_rate_under, call_rate_under, tool_rate_limit,
    per_key_rate_limit, tool_quota_per_period,
)

__all__ = [
    "predicate", "get_predicate", "list_predicates",
    # Cost
    "cost_under", "cost_per_turn_under", "token_budget",
    # Tools
    "must_call", "must_not_call", "tool_order", "tools_allowed", "max_tool_calls",
    "tool_args_match", "no_destructive_args", "tool_path_within",
    "tool_arg_value_guard", "required_disclosure", "tool_host_within",
    # Output
    "no_pii", "output_contains", "output_matches", "max_output_length", "output_must_not_contain",
    "valid_json", "json_schema_valid", "no_secrets",
    # Timing
    "max_latency", "session_timeout", "max_turns",
    # Behavioral
    "no_loops", "max_retries", "drift_bounds", "no_repeated_output",
    # Rate limits
    "spend_rate_under", "call_rate_under", "tool_rate_limit",
    "per_key_rate_limit", "tool_quota_per_period",
]
