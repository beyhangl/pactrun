"""The one model price table, and how a model name and its usage are priced.

USD per 1M tokens, standard tier, checked 2026-10-19 against:

- OpenAI: https://developers.openai.com/api/docs/pricing
- Anthropic: https://platform.claude.com/docs/en/about-claude/pricing
- Google: https://ai.google.dev/gemini-api/docs/pricing

and cross-checked against litellm 1.82.6's price map. Cache prices are
multipliers on the input price (see :class:`pactrun.core.usage.CacheRates`);
they apply unchanged to a long-context tier and to Anthropic fast mode, as the
pricing pages state.

Lookup (:func:`find`): the name is normalised (lowercase; provider paths such
as ``models/`` or ``openai/``; Bedrock ``anthropic.`` with its region prefix
and ``-v1:0`` suffix; Vertex ``@date``; a ``[1m]`` tag), then matched exactly,
then with trailing date stamps removed, then against the LONGEST key it starts
with on a ``-`` boundary. A prefix match is refused when the rest of the name
names a variant that bills differently (``-pro``, ``-tts``, ``-audio``...).

Unknown models are never priced at $0: a $0 (or any low) price lets real spend
pass ``cost_under``. They are priced at the provider's ceiling, the highest
input, output and cache-write rates of any model of that provider in the
table (any provider when it cannot be told), with no cache discount, and are
flagged with ``metadata["cost_source"] = "unknown_model"``.

Not modelled, all of which over-count except where noted: batch (half price,
never visible per response); cache storage fees; scheduled price changes
(noted per entry; they under-count once they take effect until this table is
updated); OpenAI priority tier and Gemini audio input (both under-count).
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Any

from pactrun.core.amounts import is_valid_amount
from pactrun.core.usage import NO_CACHE_DISCOUNT, CacheRates, TokenUsage, price

logger = logging.getLogger("pactrun")

COST_SOURCE = "cost_source"
UNKNOWN_MODEL = "unknown_model"

# Anthropic data residency: inference_geo "us" is 1.1x on every token price.
US_INFERENCE_MULTIPLIER = 1.1


@dataclass(frozen=True)
class LongContext:
    """Higher rates for a request whose TOTAL prompt (cache included) exceeds ``threshold``."""

    threshold: int
    input: float
    output: float


@dataclass(frozen=True)
class ModelPrice:
    provider: str
    input: float
    output: float
    cache: CacheRates = NO_CACHE_DISCOUNT
    long_context: LongContext | None = None
    # (input, output) when the response reports fast mode (Anthropic usage.speed).
    fast: tuple[float, float] | None = None
    retired: bool = False

    def rates(self, prompt_tokens: Any, *, fast: bool = False) -> tuple[float, float]:
        if fast and self.fast is not None:
            return self.fast
        lc = self.long_context
        if lc is not None and is_valid_amount(prompt_tokens) and prompt_tokens > lc.threshold:
            return lc.input, lc.output
        return self.input, self.output


def _openai(inp: float, out: float, read: float = 1.0, write: float = 1.0,
            lc: tuple[float, float] | None = None) -> ModelPrice:
    # OpenAI long context is "> 272K input tokens"; 272,000 is the lower reading.
    return ModelPrice("openai", inp, out, CacheRates(read=read, write=write),
                      LongContext(272_000, *lc) if lc else None)


def _claude(inp: float, out: float, read: float = 0.10, *, fast: tuple[float, float] | None = None,
            lc: LongContext | None = None, retired: bool = False) -> ModelPrice:
    return ModelPrice("anthropic", inp, out, CacheRates(read=read, write=1.25, write_1h=2.0),
                      lc, fast, retired)


def _gemini(inp: float, out: float, read: float = 0.10, lc: tuple[float, float] | None = None,
            retired: bool = False) -> ModelPrice:
    # No per-request cache-write charge (explicit caches bill storage per hour).
    return ModelPrice("google", inp, out, CacheRates(read=read),
                      LongContext(200_000, *lc) if lc else None, retired=retired)


PRICES: dict[str, ModelPrice] = {
    # OpenAI. GPT-5.6+ bills cache writes at 1.25x; earlier models bill them as input.
    "gpt-6-astra": _openai(10.00, 50.00, 0.10, 1.25, lc=(20.00, 75.00)),
    "gpt-6.1-sol": _openai(2.00, 10.00, 0.05, 1.25, lc=(4.00, 15.00)),
    "gpt-6-sol": _openai(2.00, 10.00, 0.10, 1.25, lc=(4.00, 15.00)),
    "gpt-6-luna": _openai(0.10, 0.50, 0.10, 1.25, lc=(0.20, 0.75)),
    # Shown on the page during a promotion "available at least through November
    # 21, 2026"; the page states no other price.
    "gpt-5.6-sol": _openai(4.00, 20.00, 0.10, 1.25, lc=(8.00, 30.00)),
    "gpt-5.6-terra": _openai(2.00, 12.00, 0.10, 1.25, lc=(4.00, 18.00)),
    "gpt-5.6-luna": _openai(0.20, 1.20, 0.10, 1.25, lc=(0.40, 1.80)),
    "gpt-5.5": _openai(5.00, 30.00, 0.10, lc=(10.00, 45.00)),
    "gpt-5.5-pro": _openai(30.00, 180.00, lc=(60.00, 270.00)),
    "gpt-5.4": _openai(2.50, 15.00, 0.10, lc=(5.00, 22.50)),
    "gpt-5.4-mini": _openai(0.75, 4.50, 0.10),
    "gpt-5.4-nano": _openai(0.20, 1.25, 0.10),
    "gpt-5.4-pro": _openai(30.00, 180.00, lc=(60.00, 270.00)),
    "gpt-5.2": _openai(1.75, 14.00, 0.10),
    "gpt-5.2-pro": _openai(21.00, 168.00),
    "gpt-5.1": _openai(1.25, 10.00, 0.10),
    "gpt-5": _openai(1.25, 10.00, 0.10),
    "gpt-5-mini": _openai(0.25, 2.00, 0.10),
    "gpt-5-nano": _openai(0.05, 0.40, 0.10),
    "gpt-5-pro": _openai(15.00, 120.00),
    "gpt-4.1": _openai(2.00, 8.00, 0.25),
    "gpt-4.1-mini": _openai(0.40, 1.60, 0.25),
    "gpt-4.1-nano": _openai(0.10, 0.40, 0.25),
    "gpt-4o": _openai(2.50, 10.00, 0.50),
    "gpt-4o-2024-05-13": _openai(5.00, 15.00),
    "gpt-4o-mini": _openai(0.15, 0.60, 0.50),
    "o1": _openai(15.00, 60.00, 0.50),
    "o1-pro": _openai(150.00, 600.00),
    "o3": _openai(2.00, 8.00, 0.25),
    "o3-pro": _openai(20.00, 80.00),
    "o3-mini": _openai(1.10, 4.40, 0.50),
    "o4-mini": _openai(1.10, 4.40, 0.25),
    # Anthropic. Cache writes 1.25x (5 minutes) / 2x (1 hour); reads 0.1x unless noted.
    "claude-fable-5-1": _claude(10.00, 50.00, 0.025),
    "claude-mythos-5-1": _claude(10.00, 50.00, 0.025),
    "claude-fable-5": _claude(10.00, 50.00),
    "claude-mythos-5": _claude(10.00, 50.00),
    "claude-opus-5-5": _claude(4.00, 20.00, 0.05, fast=(8.00, 40.00)),
    "claude-opus-5": _claude(5.00, 25.00, fast=(10.00, 50.00)),
    "claude-opus-4-8": _claude(5.00, 25.00, fast=(10.00, 50.00)),
    "claude-opus-4-7": _claude(5.00, 25.00),
    "claude-opus-4-6": _claude(5.00, 25.00),
    "claude-opus-4-5": _claude(5.00, 25.00),
    "claude-opus-4-1": _claude(15.00, 75.00, retired=True),
    "claude-opus-4": _claude(15.00, 75.00, retired=True),
    "claude-sonnet-5-5": _claude(2.00, 10.00, 0.05),
    "claude-sonnet-5": _claude(2.00, 10.00),
    "claude-sonnet-4-6": _claude(3.00, 15.00),
    "claude-sonnet-4-5": _claude(3.00, 15.00),
    "claude-sonnet-4": _claude(3.00, 15.00, retired=True),
    # "For prompts over 100,000 tokens", counting cache reads and writes.
    "claude-haiku-5-5": _claude(0.10, 0.50, lc=LongContext(100_000, 0.50, 2.50)),
    "claude-haiku-4-5": _claude(1.00, 5.00),
    # Google. Context-cache reads 0.1x unless noted; long context is "> 200k".
    # Rises to $4 / $20 (cache $0.20) on 2026-12-01.
    "gemini-4-argon": _gemini(2.00, 10.00, 0.05),
    # Both rise to $1.50 / $7.50 (cache $0.15) on 2027-01-01.
    "gemini-3.8-flash": _gemini(0.75, 3.75),
    "gemini-3.6-flash": _gemini(0.75, 3.75),
    "gemini-3.5-flash-lite": _gemini(0.30, 2.50),
    "gemini-3.1-flash-lite": _gemini(0.25, 1.50),
    "gemini-3.1-pro-preview": _gemini(2.00, 12.00, lc=(4.00, 18.00)),
    "gemini-2.5-pro": _gemini(1.25, 10.00, lc=(2.50, 15.00)),
    "gemini-2.5-flash": _gemini(0.30, 2.50),
    "gemini-2.5-flash-lite": _gemini(0.10, 0.40),
    # Retired: last published prices, kept for old traces (genai-prices 0.1.9).
    "gemini-2.0-flash": _gemini(0.10, 0.40, 0.25, retired=True),
    "gemini-1.5-pro": _gemini(1.25, 5.00, 1.0, retired=True),
    "gemini-1.5-flash": _gemini(0.075, 0.30, 0.25, retired=True),
}

# Name segments after a matched prefix that mean a differently billed model.
_PRICIER_VARIANTS = frozenset(
    {"pro", "max", "ultra", "tts", "audio", "image", "realtime", "live", "search", "research", "transcribe"}
)
_BEDROCK_PREFIX = re.compile(r"^(?:[a-z][a-z-]*\.)?anthropic\.")
_BEDROCK_VERSION = re.compile(r"-v\d+(?::\d+)?$")
_STAMPS = re.compile(r"(?:-(?:\d{4}-\d{2}-\d{2}|\d{8}|\d{2}-\d{4}|\d{2}-\d{2}|\d{4}|\d{3}|latest))+$")
_OPENAI_NAME = re.compile(r"^(?:gpt|chatgpt|codex|o\d(?:-|$))")


def normalize(model: Any) -> str:
    """Canonical form of a model name for lookup (see module docstring)."""
    name = str(model or "").strip().lower()
    name = re.sub(r"\[[^\]]*\]$", "", name)
    name = name.rsplit("/", 1)[-1]
    name = _BEDROCK_PREFIX.sub("", name)
    name = _BEDROCK_VERSION.sub("", name)
    return name.split("@", 1)[0]


def find(model: Any) -> tuple[str, ModelPrice] | None:
    """``(table key, price)`` for ``model``, or ``None`` when the table does not know it."""
    name = normalize(model)
    undated = _STAMPS.sub("", name)
    for candidate in (name, undated):
        if candidate in PRICES:
            return candidate, PRICES[candidate]
    best = max((k for k in PRICES if undated.startswith(k + "-")), key=len, default=None)
    if best is None or _PRICIER_VARIANTS.intersection(undated[len(best) + 1:].split("-")):
        return None
    return best, PRICES[best]


def provider_of(model: Any) -> str | None:
    name = normalize(model).removeprefix("ft:")
    if name.startswith("claude") or "anthropic" in str(model).lower():
        return "anthropic"
    if name.startswith("gemini"):
        return "google"
    if _OPENAI_NAME.match(name):
        return "openai"
    return None


def _ceiling_of(entries: list[ModelPrice], provider: str) -> ModelPrice:
    rates = [(p.input, p.output) for p in entries]
    rates += [(p.long_context.input, p.long_context.output) for p in entries if p.long_context]
    rates += [p.fast for p in entries if p.fast]
    write = max(p.cache.write for p in entries)
    write_1h = max(p.cache.write_1h_rate for p in entries)
    return ModelPrice(
        provider,
        max(r[0] for r in rates),
        max(r[1] for r in rates),
        CacheRates(read=1.0, write=write, write_1h=write_1h),
    )


def _ceilings() -> dict[str | None, ModelPrice]:
    out: dict[str | None, ModelPrice] = {
        p: _ceiling_of([m for m in PRICES.values() if m.provider == p], p)
        for p in {m.provider for m in PRICES.values()}
    }
    out[None] = _ceiling_of(list(PRICES.values()), "unknown")
    return out


_CEILINGS = _ceilings()


def register_price(
    model: str,
    input: float,
    output: float,
    *,
    provider: str | None = None,
    cache_read: float = 1.0,
    cache_write: float = 1.0,
) -> None:
    """Add or replace a model's price (USD per 1M tokens).

    Use it for a fine-tuned, private or newly released model the table lacks,
    which would otherwise be priced at the provider ceiling. ``cache_read`` and
    ``cache_write`` are multipliers on ``input``. The name is normalised like a
    lookup, and longest-prefix matching applies to it. Ceilings are recomputed,
    so registering a pricier model raises them.
    """
    from pactrun.core.amounts import require_limit

    for name, value in (("input", input), ("output", output), ("cache_read", cache_read),
                        ("cache_write", cache_write)):
        require_limit(f"register_price({name})", value)
    key = normalize(model)
    if not key:
        raise ValueError("register_price: model name is empty")
    global _CEILINGS
    PRICES[key] = ModelPrice(
        provider or provider_of(model) or "custom",
        float(input),
        float(output),
        CacheRates(read=cache_read, write=cache_write),
    )
    _CEILINGS = _ceilings()
    with _warned_lock:
        _warned.discard(str(model))


def ceiling(provider: str | None) -> ModelPrice:
    """The conservative price for an unknown model of ``provider`` (``None``: any provider)."""
    return _CEILINGS.get(provider, _CEILINGS[None])


_warned: set[str] = set()
_warned_lock = threading.Lock()


def _warn_unknown(model: Any, entry: ModelPrice) -> None:
    key = str(model)
    with _warned_lock:
        if key in _warned:
            return
        _warned.add(key)
    logger.warning(
        "pactrun has no price for model %r; pricing it at the %s ceiling ($%g in / $%g out per 1M "
        "tokens, no cache discount) so cost limits fail closed", key, entry.provider, entry.input, entry.output,
    )


def resolve(model: Any) -> tuple[ModelPrice, bool]:
    """``(price, known)``: the table price, or the provider ceiling (logged once per name)."""
    found = find(model)
    if found is not None:
        return found[1], True
    entry = ceiling(provider_of(model))
    _warn_unknown(model, entry)
    return entry, False


def modifier(usage: TokenUsage) -> float:
    """Multiplier on the whole price from modifiers reported with the usage."""
    geo = usage.inference_geo
    return US_INFERENCE_MULTIPLIER if isinstance(geo, str) and geo.lower() == "us" else 1.0


def is_fast(usage: TokenUsage) -> bool:
    speed = usage.speed
    return isinstance(speed, str) and speed.lower() == "fast"


@dataclass(frozen=True)
class Quote:
    usd: float
    known: bool
    metadata: dict[str, Any] = field(default_factory=dict)


def cost(model: Any, usage: TokenUsage) -> Quote:
    """USD cost of one response. ``nan`` for invalid counts, never 0 for an unknown model."""
    entry, known = resolve(model)
    input_rate, output_rate = entry.rates(usage.prompt_tokens, fast=is_fast(usage))
    usd = price(usage, input_rate, output_rate, entry.cache) * modifier(usage)
    return Quote(usd, known, {} if known else {COST_SOURCE: UNKNOWN_MODEL})
