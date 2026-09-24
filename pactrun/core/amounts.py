"""Validation for the numeric amounts that budgets are computed from.

A budget is only as trustworthy as the arithmetic behind it. Accumulating an
unvalidated value is a known way for a runtime guardrail to fail open:

- a **negative** cost or token count lowers the running total, so one bad
  event "refunds" the budget and later spend passes the cap;
- a **NaN** makes every later comparison false, poisoning the total for the rest
  of the run (fail-closed, but a denial of service);
- an **infinite** limit in a contract silently means "no limit".

The session sanitizes amounts once, at ingestion, so every reader (totals,
windowed rate predicates, drift, telemetry, audit) sees the same sane value,
and records what it rejected so the budget predicates can fail closed on that
event instead of trusting it.
"""

from __future__ import annotations

import math
from typing import Any

# Event fields that feed budget arithmetic.
AMOUNT_FIELDS: tuple[str, ...] = ("cost_usd", "prompt_tokens", "completion_tokens", "duration_ms")

# event.metadata key holding {field: repr(raw_value)} for rejected amounts.
INVALID_AMOUNTS_KEY = "pactrun.invalid_amounts"

# state.metadata key counting events that carried a rejected amount.
INVALID_AMOUNT_EVENTS_KEY = "pactrun.invalid_amount_events"


def is_valid_amount(value: Any) -> bool:
    """True for a finite, non-negative real number (bools are not amounts)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value) and value >= 0


def invalid_amounts(event: Any, *fields: str) -> dict[str, str]:
    """The subset of ``fields`` the session rejected on this event."""
    rejected = (getattr(event, "metadata", None) or {}).get(INVALID_AMOUNTS_KEY) or {}
    return {f: rejected[f] for f in fields if f in rejected}


def require_limit(name: str, value: Any, *, allow_zero: bool = True) -> None:
    """Reject a non-finite, negative (or zero, if disallowed) contract limit.

    Raised at construction time: a budget of ``nan``/``inf``/``-1`` is always a
    configuration bug, and failing loudly beats enforcing something nobody meant.
    """
    if not is_valid_amount(value) or (not allow_zero and value == 0):
        bound = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"{name} must be a finite number {bound}, got {value!r}")
