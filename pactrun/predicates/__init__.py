"""Predicate system — registry, base types, and all built-in predicates."""

from pactrun.predicates.base import get_predicate, list_predicates, predicate
from pactrun.predicates.behavioral import (
    bounded_error_retries,
    drift_bounds,
    max_retries,
    no_duplicate_side_effect,
    no_loops,
    no_progress_stall,
    no_redundant_reads,
    no_repeated_output,
    tool_error_rate_under,
)
from pactrun.predicates.compliance import ai_disclosure_in_output
from pactrun.predicates.consent import (
    consent_token_required,
    mint_approval_token,
    mint_consent_token,
    multi_party_approval_required,
)
from pactrun.predicates.content_security import (
    canary_not_leaked,
    mint_canary,
    no_injection_phrases,
)

# Import all built-in predicates to register them
from pactrun.predicates.cost import cost_per_turn_under, cost_under, token_budget
from pactrun.predicates.exfil import (
    lethal_trifecta_guard,
    no_exfiltration_after_untrusted,
    untrusted_taint_to_sink,
)
from pactrun.predicates.flow import flow_progression
from pactrun.predicates.network import tool_host_within
from pactrun.predicates.output import (
    json_schema_valid,
    max_output_length,
    no_exfil_links,
    no_invisible_text,
    no_pii,
    no_secrets,
    output_contains,
    output_matches,
    output_must_not_contain,
    tenant_response_isolation,
    valid_json,
)
from pactrun.predicates.ratelimit import (
    approval_request_rate_under,
    call_rate_under,
    per_key_rate_limit,
    spend_rate_under,
    tool_quota_per_period,
    tool_rate_limit,
)
from pactrun.predicates.supply_chain import tool_definitions_stable
from pactrun.predicates.timing import max_latency, max_turns, session_timeout
from pactrun.predicates.tool_args import (
    no_destructive_args,
    required_disclosure,
    tool_arg_value_guard,
    tool_args_match,
    tool_path_within,
)
from pactrun.predicates.tools import (
    max_tool_calls,
    must_call,
    must_not_call,
    tool_order,
    tools_allowed,
)

__all__ = [
    "predicate", "get_predicate", "list_predicates",
    # Cost
    "cost_under", "cost_per_turn_under", "token_budget",
    # Tool-name policy
    "must_call", "must_not_call", "tool_order", "tools_allowed", "max_tool_calls",
    # Tool args
    "tool_args_match", "no_destructive_args", "tool_path_within",
    "tool_arg_value_guard", "required_disclosure",
    # Network egress
    "tool_host_within",
    # Consent / approval
    "consent_token_required", "mint_consent_token",
    "multi_party_approval_required", "mint_approval_token",
    # Exfiltration / cross-run
    "no_exfiltration_after_untrusted", "lethal_trifecta_guard", "untrusted_taint_to_sink",
    # Compliance
    "ai_disclosure_in_output",
    # Supply chain
    "tool_definitions_stable",
    # Output
    "no_pii", "output_contains", "output_matches", "max_output_length", "output_must_not_contain",
    "valid_json", "json_schema_valid", "no_secrets", "tenant_response_isolation",
    "no_invisible_text", "no_exfil_links",
    # Timing
    "max_latency", "session_timeout", "max_turns",
    # Behavioral
    "no_loops", "max_retries", "drift_bounds", "no_repeated_output", "tool_error_rate_under",
    "bounded_error_retries", "no_redundant_reads", "no_progress_stall", "no_duplicate_side_effect",
    # Flow
    "flow_progression",
    # Content security
    "no_injection_phrases", "canary_not_leaked", "mint_canary",
    # Rate limits
    "spend_rate_under", "call_rate_under", "tool_rate_limit",
    "per_key_rate_limit", "tool_quota_per_period", "approval_request_rate_under",
]
