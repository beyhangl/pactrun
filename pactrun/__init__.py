"""pactrun — behavioral contracts for AI agents.

Declare what an agent must / must not do across a whole session — cost, tool,
output, timing, and drift limits — and enforce them at runtime.
"""

__version__ = "0.1.0"

from pactrun.contract import Contract
from pactrun.core.enums import ClauseKind, EventKind, OnFail, Severity
from pactrun.core.errors import ContractLoadError, ViolationError
from pactrun.core.models import (
    Clause,
    Event,
    PredicateResult,
    SessionState,
    SessionSummary,
    Violation,
)
from pactrun.predicates import (
    ai_disclosure_in_output,
    approval_request_rate_under,
    bounded_error_retries,
    call_rate_under,
    canary_not_leaked,
    consent_token_required,
    cost_per_turn_under,
    cost_under,
    drift_bounds,
    flow_progression,
    json_schema_valid,
    lethal_trifecta_guard,
    max_latency,
    max_output_length,
    max_retries,
    max_tool_calls,
    max_turns,
    mint_approval_token,
    mint_canary,
    mint_consent_token,
    multi_party_approval_required,
    must_call,
    must_not_call,
    no_destructive_args,
    no_duplicate_side_effect,
    no_exfil_links,
    no_exfiltration_after_untrusted,
    no_injection_phrases,
    no_invisible_text,
    no_loops,
    no_pii,
    no_progress_stall,
    no_redundant_reads,
    no_repeated_output,
    no_secrets,
    output_contains,
    output_matches,
    output_must_not_contain,
    per_key_rate_limit,
    required_disclosure,
    session_timeout,
    spend_rate_under,
    tenant_response_isolation,
    token_budget,
    tool_arg_value_guard,
    tool_args_match,
    tool_definitions_stable,
    tool_error_rate_under,
    tool_host_within,
    tool_order,
    tool_path_within,
    tool_quota_per_period,
    tool_rate_limit,
    tools_allowed,
    untrusted_taint_to_sink,
    valid_json,
)
from pactrun.predicates.base import get_predicate, list_predicates, predicate
from pactrun.recovery import (
    EscalationError,
    FallbackSignal,
    RetrySignal,
    auto_approver,
    cli_approver,
    digest,
    webhook_handler,
)
from pactrun.session import Session, get_active_session
from pactrun.wrap import wrap

__all__ = [
    "ClauseKind", "EventKind", "OnFail", "Severity",
    "ContractLoadError", "ViolationError",
    "EscalationError", "RetrySignal", "FallbackSignal",
    "webhook_handler", "cli_approver", "auto_approver", "digest",
    "Clause", "Event", "PredicateResult", "SessionState", "SessionSummary", "Violation",
    "Contract", "Session", "get_active_session", "wrap",
    "predicate", "get_predicate", "list_predicates",
    # Built-in predicates
    "cost_under", "cost_per_turn_under", "token_budget",
    "must_call", "must_not_call", "tool_order", "tools_allowed", "max_tool_calls",
    "tool_args_match", "no_destructive_args", "tool_path_within",
    "tool_arg_value_guard", "required_disclosure", "tool_host_within",
    "consent_token_required", "mint_consent_token",
    "no_exfiltration_after_untrusted", "lethal_trifecta_guard", "untrusted_taint_to_sink",
    "multi_party_approval_required", "mint_approval_token",
    "ai_disclosure_in_output",
    "tool_definitions_stable",
    "no_pii", "output_contains", "output_matches", "max_output_length", "output_must_not_contain",
    "valid_json", "json_schema_valid", "no_secrets", "tenant_response_isolation",
    "no_invisible_text", "no_exfil_links",
    "max_latency", "session_timeout", "max_turns",
    "no_loops", "max_retries", "drift_bounds", "no_repeated_output", "tool_error_rate_under",
    "bounded_error_retries", "no_redundant_reads", "no_progress_stall", "no_duplicate_side_effect",
    "spend_rate_under", "call_rate_under", "tool_rate_limit",
    "per_key_rate_limit", "tool_quota_per_period", "approval_request_rate_under",
    "flow_progression",
    "no_injection_phrases", "canary_not_leaked", "mint_canary",
]
