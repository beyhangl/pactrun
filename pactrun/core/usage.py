"""Normalised token usage and cache-aware pricing shared by the adapters.

Every provider reports prompt caching differently (Anthropic's ``input_tokens``
excludes cache reads and writes, OpenAI's ``prompt_tokens`` includes them,
Gemini reports thinking tokens outside ``candidates_token_count``). Adapters
convert each provider's usage object into one :class:`TokenUsage` that follows
the pactrun convention documented on :class:`pactrun.core.models.Event`:

- ``prompt_tokens`` is the TOTAL input, including cache reads and cache writes;
  ``cache_read_tokens`` and ``cache_write_tokens`` are subsets of it.
- ``completion_tokens`` is the TOTAL output, including reasoning/thinking;
  ``reasoning_tokens`` is a subset of it.

Pricing then charges the three slices of input separately:
``uncached * input + reads * input * read_multiplier + writes * input * write_multiplier``.

Invalid counts (negative, NaN, inf, non-numbers) are never silently repaired
here. They are passed through so the session's ingestion check rejects and
records them, and a cost computed from them is ``nan`` so the cost predicates
fail closed on that event instead of trusting a made-up number.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeVar

from pactrun.core.amounts import is_valid_amount

T = TypeVar("T")


@dataclass(frozen=True)
class CacheRates:
    """Cache price multipliers, relative to the model's base input price.

    ``write_1h`` prices the part of ``cache_write_tokens`` the provider says
    went to a one-hour cache (Anthropic only today); it defaults to ``write``.
    """

    read: float = 1.0
    write: float = 1.0
    write_1h: float | None = None

    @property
    def write_1h_rate(self) -> float:
        return self.write if self.write_1h is None else self.write_1h


NO_CACHE_DISCOUNT = CacheRates()


@dataclass(frozen=True)
class TokenUsage:
    """One response's usage in pactrun's convention (see module docstring)."""

    prompt_tokens: Any = 0
    completion_tokens: Any = 0
    cache_read_tokens: Any = 0
    cache_write_tokens: Any = 0
    # Subset of cache_write_tokens billed at the one-hour cache-write rate.
    # Used for pricing only; not an Event field.
    cache_write_1h_tokens: Any = 0
    reasoning_tokens: Any = 0

    def event_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for ``Session.emit_llm_response``."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "reasoning_tokens": self.reasoning_tokens,
        }

    def is_valid(self) -> bool:
        return all(
            is_valid_amount(v)
            for v in (
                self.prompt_tokens,
                self.completion_tokens,
                self.cache_read_tokens,
                self.cache_write_tokens,
                self.cache_write_1h_tokens,
                self.reasoning_tokens,
            )
        )


def read_field(obj: Any, name: str) -> Any:
    """``obj.name`` or ``obj[name]``; ``None`` when absent."""
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def count(value: Any) -> Any:
    """A reported token count: ``None`` (not reported) becomes 0.

    Anything else is returned unchanged, including invalid values, so the
    session can reject and record them instead of an adapter hiding them.
    """
    return 0 if value is None else value


def total(*parts: Any) -> Any:
    """Sum of token counts, or the first invalid part unchanged.

    Returning the invalid part (rather than a sum it would corrupt, e.g. a
    negative cache count lowering the total) makes the session reject the
    total itself, so token budgets fail closed on that event.
    """
    values = [count(p) for p in parts]
    for v in values:
        if not is_valid_amount(v):
            return v
    return sum(values)


def price(usage: TokenUsage, input_per_m: float, output_per_m: float, rates: CacheRates) -> float:
    """USD cost of ``usage`` at per-1M-token prices, charging cache slices separately.

    Cache counts larger than the input they are part of are clamped to it (a
    provider inconsistency must not produce a negative uncached slice). Returns
    ``nan`` when any count is invalid, so the session flags the cost.
    """
    if not usage.is_valid():
        return math.nan
    prompt = usage.prompt_tokens
    read = min(usage.cache_read_tokens, prompt)
    write = min(usage.cache_write_tokens, prompt - read)
    write_1h = min(usage.cache_write_1h_tokens, write)
    uncached = prompt - read - write
    input_cost = input_per_m * (
        uncached
        + read * rates.read
        + (write - write_1h) * rates.write
        + write_1h * rates.write_1h_rate
    )
    return (input_cost + usage.completion_tokens * output_per_m) / 1_000_000


def lookup(table: Mapping[str, T], model: str) -> T | None:
    """Exact key, else the first key the model name starts with (table order)."""
    if model in table:
        return table[model]
    for key, value in table.items():
        if model.startswith(key):
            return value
    return None
