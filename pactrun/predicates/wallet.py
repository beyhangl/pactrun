"""Denial-of-wallet and money-movement guards.

An agent can run up a bill without ever breaking a flat budget early: tool
output kept in the history is billed again on every later model call, so the
prompt grows each step, and a loop of tool-only steps never hands control back
to the user. ``context_growth_under`` and ``max_steps_per_turn`` catch that
shape before the total budget is gone.

``tool_spend_under`` is about money the agent moves through tools (payments,
refunds, transfers), not model spend: per-call and per-run caps, a currency, a
merchant allowlist and an expiry, the same limits a delegated-payment allowance
carries.
"""

from __future__ import annotations

import math
from typing import Any

from pactrun.core.amounts import is_valid_amount, require_limit
from pactrun.core.enums import EventKind
from pactrun.core.models import Event, PredicateResult, SessionState
from pactrun.predicates._argpath import _resolve_path
from pactrun.predicates.base import predicate
from pactrun.predicates.cost import _reject_invalid

_TURN_BOUNDARIES = (EventKind.INPUT, EventKind.OUTPUT, EventKind.TURN_END)


@predicate("context_growth_under", owasp=("ASI08",))
def context_growth_under(
    max_prompt_tokens: int | None = None,
    max_growth_ratio: float | None = None,
    min_baseline_tokens: int = 1000,
):
    """Cap how large a single prompt gets, and how fast prompts grow in a run.

    Each model call's ``prompt_tokens`` (total input, cached included) is
    checked against ``max_prompt_tokens``, and against ``max_growth_ratio``
    times the run's first prompt. The first prompt is raised to
    ``min_baseline_tokens`` before the ratio is taken, so a tiny opening prompt
    doesn't make ordinary growth look like an anomaly.

    Runaway growth is the cost pattern of tool output piling up in the history:
    every later call pays for it again, so spend rises with the square of the
    step count while each single step still looks cheap.
    """
    if max_prompt_tokens is None and max_growth_ratio is None:
        raise ValueError("context_growth_under: set max_prompt_tokens and/or max_growth_ratio")
    if max_prompt_tokens is not None:
        require_limit("max_prompt_tokens", max_prompt_tokens, allow_zero=False)
    if max_growth_ratio is not None:
        require_limit("max_growth_ratio", max_growth_ratio, allow_zero=False)
    require_limit("min_baseline_tokens", min_baseline_tokens, allow_zero=False)

    def _first_prompt(state: SessionState) -> int | None:
        for e in state.events:
            if e.kind == EventKind.LLM_CALL and is_valid_amount(e.prompt_tokens) and e.prompt_tokens > 0:
                return int(e.prompt_tokens)
        return None

    def check(event: Event, state: SessionState) -> PredicateResult:
        if event.kind != EventKind.LLM_CALL:
            return PredicateResult(passed=True)
        rejected = _reject_invalid(event, "prompt_tokens")
        if rejected:
            return rejected
        prompt = event.prompt_tokens
        if not is_valid_amount(prompt):
            return PredicateResult(
                passed=False,
                expected="a valid prompt token count",
                actual=repr(prompt),
                message="Prompt token count is invalid; cannot check context growth (fail-closed)",
            )
        if max_prompt_tokens is not None and prompt > max_prompt_tokens:
            return PredicateResult(
                passed=False,
                expected=f"prompt <= {max_prompt_tokens:,} tokens",
                actual=f"{int(prompt):,} tokens",
                message=f"Prompt of {int(prompt):,} tokens exceeds the {max_prompt_tokens:,}-token cap",
            )
        if max_growth_ratio is not None:
            first = _first_prompt(state)
            if first is not None:
                baseline = max(first, min_baseline_tokens)
                ratio = prompt / baseline
                if ratio > max_growth_ratio:
                    return PredicateResult(
                        passed=False,
                        expected=f"prompt <= {max_growth_ratio:g}x the first prompt ({baseline:,} tokens)",
                        actual=f"{ratio:.1f}x ({int(prompt):,} tokens)",
                        message=(
                            f"Context grew {ratio:.1f}x since the first call "
                            f"(limit {max_growth_ratio:g}x): tool output may be piling up in the history"
                        ),
                    )
        return PredicateResult(passed=True)

    check.predicate_name = "context_growth_under"  # type: ignore[attr-defined]
    return check


@predicate("max_steps_per_turn", owasp=("ASI08",))
def max_steps_per_turn(n: int):
    """Cap the model calls the agent makes before it hands control back.

    Counts model calls since the last turn boundary: user input, agent output
    to the user, or a turn end (``session.emit_input`` / ``emit_output`` /
    ``end_turn``). A loop of tool-only steps never reaches a boundary, so this
    stops it where ``max_turns`` can't. If the host never marks turns, the
    whole run counts as one turn and this caps model calls per run.
    """
    require_limit("n", n, allow_zero=False)

    def check(event: Event, state: SessionState) -> PredicateResult:
        if event.kind != EventKind.LLM_CALL:
            return PredicateResult(passed=True)
        steps = 0
        for e in reversed(state.events):
            if e.kind in _TURN_BOUNDARIES:
                break
            if e.kind == EventKind.LLM_CALL:
                steps += 1
        if steps > n:
            return PredicateResult(
                passed=False,
                expected=f"<= {n} model calls per turn",
                actual=f"{steps} model calls without returning to the user",
                message=f"{steps} model calls in one turn (limit {n}): the agent may be stuck in a tool loop",
            )
        return PredicateResult(passed=True)

    check.predicate_name = "max_steps_per_turn"  # type: ignore[attr-defined]
    return check


def _parse_amount(value: Any) -> float | None:
    """A finite, non-negative amount from a number or numeric string, else None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError:
            return None
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        return None
    return float(value)


@predicate("tool_spend_under", owasp=("ASI02", "ASI03"))
def tool_spend_under(
    tools,
    *,
    amount_arg: str = "amount",
    per_call: float | None = None,
    per_session: float | None = None,
    currency: str | None = None,
    currency_arg: str = "currency",
    merchants=None,
    merchant_arg: str = "merchant_id",
    expires_at: float | None = None,
):
    """Cap the money an agent moves through tools.

    Applies to calls of ``tools`` (glob patterns, e.g. ``["create_payment",
    "issue_refund"]``). The amount is read from the argument at ``amount_arg``
    (a dotted path such as ``"payment.amount"``); limits are in the same unit
    as that argument, so a caller that passes minor units (cents) sets limits
    in minor units too.

    - ``per_call``: the largest single amount.
    - ``per_session``: the most the run may move in total, counting this call.
    - ``currency``: every call must name this currency at ``currency_arg``
      (case-insensitive).
    - ``merchants``: the merchant at ``merchant_arg`` must be on this list.
    - ``expires_at``: Unix time after which no call is allowed (judged on the
      event's timestamp).

    Fails closed: a call with a missing, non-numeric, negative, NaN or infinite
    amount fails, so a malformed or "negative refund" call can't slip past or
    lower the running total. The run total counts every earlier matching call
    that reached pactrun, including ones a clause blocked. That over-counts
    rather than under-counts, and in enforcement a block normally ends the run.
    """
    patterns = (tools,) if isinstance(tools, str) else tuple(tools)
    if not patterns:
        raise ValueError("tool_spend_under: tools must name at least one tool")
    if per_call is None and per_session is None and currency is None and merchants is None and expires_at is None:
        raise ValueError("tool_spend_under: set at least one of per_call, per_session, currency, merchants, expires_at")
    if per_call is not None:
        require_limit("per_call", per_call)
    if per_session is not None:
        require_limit("per_session", per_session)
    if expires_at is not None:
        require_limit("expires_at", expires_at)
    allowed_merchants = None if merchants is None else {str(m) for m in merchants}
    want_currency = None if currency is None else currency.strip().upper()

    from pactrun.predicates.exfil import _name_matches

    def _amount_of(e: Event) -> float | None:
        found, value = _resolve_path(e.tool_args or {}, amount_arg)
        return _parse_amount(value) if found else None

    def _fail(event: Event, expected: str, actual: str, message: str) -> PredicateResult:
        return PredicateResult(
            passed=False, expected=expected, actual=actual, message=f"'{event.tool_name}': {message}"
        )

    def check(event: Event, state: SessionState) -> PredicateResult:
        if event.kind != EventKind.TOOL_CALL or not _name_matches(event.tool_name, patterns):
            return PredicateResult(passed=True)
        args = event.tool_args or {}
        amount = _amount_of(event)
        if amount is None:
            found, raw = _resolve_path(args, amount_arg)
            shown = repr(raw) if found else "missing"
            return _fail(event, f"a valid non-negative amount at '{amount_arg}'", shown,
                         f"amount at '{amount_arg}' is {shown}; refusing (fail-closed)")
        if want_currency is not None:
            found, cur = _resolve_path(args, currency_arg)
            if not found or not isinstance(cur, str) or cur.strip().upper() != want_currency:
                shown = repr(cur) if found else "missing"
                return _fail(event, f"currency {want_currency}", shown, f"currency is {shown}, only {want_currency} is allowed")
        if allowed_merchants is not None:
            found, merchant = _resolve_path(args, merchant_arg)
            if not found or str(merchant) not in allowed_merchants:
                shown = repr(merchant) if found else "missing"
                return _fail(event, "an allowed merchant", shown, f"merchant {shown} is not on the allowlist")
        if expires_at is not None and event.timestamp > expires_at:
            return _fail(event, f"before {expires_at}", str(event.timestamp), "the spending allowance has expired")
        if per_call is not None and amount > per_call:
            return _fail(event, f"<= {per_call:g} per call", f"{amount:g}",
                         f"amount {amount:g} exceeds the {per_call:g} per-call cap")
        if per_session is not None:
            earlier = 0.0
            for e in state.events:
                if e.id == event.id or e.kind != EventKind.TOOL_CALL or not _name_matches(e.tool_name, patterns):
                    continue
                prior = _amount_of(e)
                if prior is not None:
                    earlier += prior
            total = earlier + amount
            if total > per_session:
                return _fail(event, f"<= {per_session:g} per run", f"{total:g}",
                             f"this call brings the run to {total:g}, over the {per_session:g} cap")
        return PredicateResult(passed=True)

    check.predicate_name = "tool_spend_under"  # type: ignore[attr-defined]
    return check
