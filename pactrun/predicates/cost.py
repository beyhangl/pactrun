"""Cost predicates — budget caps and rate limits."""

from __future__ import annotations

from pactrun.core.models import Event, PredicateResult, SessionState
from pactrun.predicates.base import predicate


@predicate("cost_under")
def cost_under(max_usd: float):
    """Session total cost must stay under budget."""
    def check(event: Event, state: SessionState) -> PredicateResult:
        return PredicateResult(
            passed=state.total_cost_usd <= max_usd,
            expected=f"<= ${max_usd:.4f}",
            actual=f"${state.total_cost_usd:.4f}",
            message=f"Cost ${state.total_cost_usd:.4f} exceeds budget ${max_usd:.4f}",
        )
    check.predicate_name = "cost_under"  # type: ignore[attr-defined]
    return check


@predicate("cost_per_turn_under")
def cost_per_turn_under(max_usd: float):
    """Per-turn cost must stay under limit."""
    def check(event: Event, state: SessionState) -> PredicateResult:
        if not state.cost_per_turn:
            return PredicateResult(passed=True)
        last_cost = state.cost_per_turn[-1]
        return PredicateResult(
            passed=last_cost <= max_usd,
            expected=f"<= ${max_usd:.4f}/turn",
            actual=f"${last_cost:.4f}",
            message=f"Turn cost ${last_cost:.4f} exceeds ${max_usd:.4f}/turn",
        )
    check.predicate_name = "cost_per_turn_under"  # type: ignore[attr-defined]
    return check


@predicate("token_budget")
def token_budget(max_tokens: int):
    """Session total tokens must stay under budget."""
    def check(event: Event, state: SessionState) -> PredicateResult:
        return PredicateResult(
            passed=state.total_tokens <= max_tokens,
            expected=f"<= {max_tokens} tokens",
            actual=f"{state.total_tokens} tokens",
            message=f"Token count {state.total_tokens} exceeds budget {max_tokens}",
        )
    check.predicate_name = "token_budget"  # type: ignore[attr-defined]
    return check
