"""The single price table (pactrun.core.pricing) and everything that uses it.

Expected prices are the provider pricing pages as checked 2026-10-19, USD per
1M tokens; see the module docstring of pactrun.core.pricing for the sources.
"""

from __future__ import annotations

import logging
import math
import sys
from types import SimpleNamespace as NS

import pytest

from pactrun import Contract
from pactrun.core import pricing
from pactrun.core.enums import EventKind
from pactrun.core.usage import TokenUsage


def _usd(model, prompt=1_000_000, completion=1_000_000, **kw):
    return pricing.cost(model, TokenUsage(prompt_tokens=prompt, completion_tokens=completion, **kw)).usd


def _rates(model):
    """(input, output) per 1M, read back from 10k tokens (below every long-context tier)."""
    return (
        round(_usd(model, 10_000, 0) * 100, 6),
        round(_usd(model, 0, 10_000) * 100, 6),
    )


def _key(model):
    found = pricing.find(model)
    return found and found[0]


def _llm_events(session):
    return [e for e in session.state.events if e.kind == EventKind.LLM_CALL]


# ---------------------------------------------------------------------------
# Entries that were wrong before 2026-10-19
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gpt-4.1-mini", (0.40, 1.60)),  # was 0.20 / 0.80
        ("o4-mini", (1.10, 4.40)),  # was 0.55 / 2.20
        ("claude-opus-4-6", (5.00, 25.00)),  # was 15 / 75
        ("claude-haiku-4-5", (1.00, 5.00)),  # was 0.80 / 4
        ("claude-haiku-4-5-20251001", (1.00, 5.00)),
        ("claude-opus-4-5-20251101", (5.00, 25.00)),  # was 15 / 75 via "claude-opus-4"
        ("claude-opus-4-7", (5.00, 25.00)),
        ("claude-opus-4-8", (5.00, 25.00)),
        ("gemini-2.5-flash-lite", (0.10, 0.40)),  # was priced as 2.5-flash
        ("gpt-5.4-pro", (30.00, 180.00)),  # was priced as gpt-5.4
        ("o3-pro", (20.00, 80.00)),  # was priced as o3
        ("gpt-4o-2024-05-13", (5.00, 15.00)),  # was priced as gpt-4o
    ],
)
def test_corrected_prices(model, expected):
    assert _rates(model) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("claude-fable-5-1", (10.00, 50.00)),
        ("claude-mythos-5-1", (10.00, 50.00)),
        ("claude-opus-5-5", (4.00, 20.00)),
        ("claude-opus-5", (5.00, 25.00)),
        ("claude-sonnet-5-5", (2.00, 10.00)),
        ("claude-sonnet-5", (2.00, 10.00)),
        ("claude-haiku-5-5", (0.10, 0.50)),
        ("gemini-4-argon", (2.00, 10.00)),
        ("gemini-3.8-flash", (0.75, 3.75)),
        ("gemini-3.6-flash", (0.75, 3.75)),
        ("gemini-3.5-flash-lite", (0.30, 2.50)),
        ("gemini-3.1-flash-lite", (0.25, 1.50)),
        ("gemini-3.1-pro-preview", (2.00, 12.00)),
        ("gpt-5.5", (5.00, 30.00)),
        ("gpt-5.6-sol", (4.00, 20.00)),
        ("gpt-5.6-terra", (2.00, 12.00)),
        ("gpt-5.6-luna", (0.20, 1.20)),
        ("gpt-6-astra", (10.00, 50.00)),
        ("gpt-6.1-sol", (2.00, 10.00)),
        ("gpt-6-sol", (2.00, 10.00)),
        ("gpt-6-luna", (0.10, 0.50)),
    ],
)
def test_models_that_used_to_cost_zero(model, expected):
    assert _rates(model) == pytest.approx(expected)


def test_cache_rates_follow_the_pages():
    def read(model):
        return _usd(model, 10_000, 0, cache_read_tokens=10_000) * 100

    def write(model, one_hour=False):
        return _usd(model, 10_000, 0, cache_write_tokens=10_000,
                    cache_write_1h_tokens=10_000 if one_hour else 0) * 100

    assert read("claude-fable-5-1") == pytest.approx(0.25)
    assert read("claude-opus-5-5") == pytest.approx(0.20)
    assert read("claude-sonnet-5") == pytest.approx(0.20)
    assert read("claude-haiku-4-5") == pytest.approx(0.10)
    assert write("claude-opus-4-6") == pytest.approx(6.25)
    assert write("claude-opus-4-6", one_hour=True) == pytest.approx(10.00)
    assert write("claude-haiku-5-5", one_hour=True) == pytest.approx(0.20)
    assert read("gpt-6.1-sol") == pytest.approx(0.10)
    assert write("gpt-6-astra") == pytest.approx(12.50)
    assert write("gpt-6-luna") == pytest.approx(0.125)
    assert write("gpt-5.4") == pytest.approx(2.50)  # no write charge before GPT-5.6
    assert read("gpt-5.4-pro") == pytest.approx(30.00)  # no cached-input price
    assert read("o4-mini") == pytest.approx(0.275)
    assert read("gemini-4-argon") == pytest.approx(0.10)
    assert read("gemini-2.5-flash-lite") == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# Lookup: normalisation and longest prefix
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("model", "key"),
    [
        ("gpt-5.4", "gpt-5.4"),
        ("gpt-5.4-mini", "gpt-5.4-mini"),
        ("gpt-5.4-nano", "gpt-5.4-nano"),
        ("gpt-5.4-pro", "gpt-5.4-pro"),
        ("gpt-5.4-2026-03-05", "gpt-5.4"),
        ("gpt-5.4-mini-2026-03-17", "gpt-5.4-mini"),
        ("gpt-5.4-nano-2026-03-17", "gpt-5.4-nano"),
        ("gpt-5.4-pro-2026-03-05", "gpt-5.4-pro"),
        ("o3", "o3"),
        ("o3-pro", "o3-pro"),
        ("o3-pro-2025-06-10", "o3-pro"),
        ("o3-2025-04-16", "o3"),
        ("gpt-4o", "gpt-4o"),
        ("gpt-4o-mini", "gpt-4o-mini"),
        ("gpt-4o-mini-2024-07-18", "gpt-4o-mini"),
        ("gpt-4o-2024-05-13", "gpt-4o-2024-05-13"),
        ("gpt-4o-2024-08-06", "gpt-4o"),
        ("gpt-5-chat-latest", "gpt-5"),
        ("claude-opus-4", "claude-opus-4"),
        ("claude-opus-4-20250514", "claude-opus-4"),
        ("claude-opus-4-1-20250805", "claude-opus-4-1"),
        ("claude-opus-4-5", "claude-opus-4-5"),
        ("claude-opus-4-6", "claude-opus-4-6"),
        ("claude-opus-4-7", "claude-opus-4-7"),
        ("claude-opus-4-8", "claude-opus-4-8"),
        ("claude-sonnet-5", "claude-sonnet-5"),
        ("claude-sonnet-5-5", "claude-sonnet-5-5"),
        ("claude-sonnet-4-20250514", "claude-sonnet-4"),
        ("gemini-2.5-flash", "gemini-2.5-flash"),
        ("gemini-2.5-flash-lite", "gemini-2.5-flash-lite"),
        ("gemini-2.5-flash-lite-preview-09-2025", "gemini-2.5-flash-lite"),
        ("gemini-2.5-flash-preview-05-20", "gemini-2.5-flash"),
        ("gemini-3.1-pro-preview-customtools", "gemini-3.1-pro-preview"),
    ],
)
def test_longest_prefix_and_date_stamps(model, key):
    assert _key(model) == key


@pytest.mark.parametrize(
    ("model", "key"),
    [
        ("GPT-4o", "gpt-4o"),
        ("openai/gpt-4.1-mini", "gpt-4.1-mini"),
        ("models/gemini-2.5-pro", "gemini-2.5-pro"),
        ("publishers/google/models/gemini-2.5-flash-001", "gemini-2.5-flash"),
        ("gemini/gemini-1.5-pro-latest", "gemini-1.5-pro"),
        ("gemini-2.0-flash-001", "gemini-2.0-flash"),
        ("anthropic/claude-sonnet-4-6", "claude-sonnet-4-6"),
        ("anthropic.claude-haiku-4-5-20251001-v1:0", "claude-haiku-4-5"),
        ("us.anthropic.claude-sonnet-4-5-20250929-v1:0", "claude-sonnet-4-5"),
        ("global.anthropic.claude-opus-4-6-v1", "claude-opus-4-6"),
        ("apac.anthropic.claude-opus-5-5", "claude-opus-5-5"),
        ("bedrock/us.anthropic.claude-opus-4-8", "claude-opus-4-8"),
        ("claude-sonnet-4-5@20250929", "claude-sonnet-4-5"),
        ("claude-opus-4-6[1m]", "claude-opus-4-6"),
    ],
)
def test_names_are_normalised(model, key):
    assert _key(model) == key


@pytest.mark.parametrize(
    "model",
    ["gpt-4o-mini-tts", "gpt-4o-audio-preview", "gpt-4o-realtime-preview", "gpt-4o-search-preview",
     "o4-mini-deep-research", "gemini-2.5-flash-preview-tts", "gemini-2.5-flash-image",
     "gpt-6-sol-pro", "gpt-5.2-pro-2025-12-11"],
)
def test_differently_billed_variants_are_not_prefix_matched(model):
    if model == "gpt-5.2-pro-2025-12-11":
        assert _key(model) == "gpt-5.2-pro"  # its own entry, not gpt-5.2
    else:
        assert _key(model) is None


def test_every_key_finds_itself():
    for key in pricing.PRICES:
        assert pricing.normalize(key) == key
        assert _key(key) == key


# ---------------------------------------------------------------------------
# Long-context tiers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("model", "threshold", "base", "long"),
    [
        ("gpt-5.4", 272_000, (2.50, 15.00), (5.00, 22.50)),
        ("gpt-5.4-pro", 272_000, (30.00, 180.00), (60.00, 270.00)),
        ("gpt-5.5", 272_000, (5.00, 30.00), (10.00, 45.00)),
        ("gpt-6-sol", 272_000, (2.00, 10.00), (4.00, 15.00)),
        ("gemini-2.5-pro", 200_000, (1.25, 10.00), (2.50, 15.00)),
        ("gemini-3.1-pro-preview", 200_000, (2.00, 12.00), (4.00, 18.00)),
        ("claude-haiku-5-5", 100_000, (0.10, 0.50), (0.50, 2.50)),
    ],
)
def test_long_context_tier_starts_above_the_threshold(model, threshold, base, long):
    at = _usd(model, threshold, 1_000)
    above = _usd(model, threshold + 1, 1_000)
    assert at == pytest.approx((threshold * base[0] + 1_000 * base[1]) / 1e6)
    assert above == pytest.approx(((threshold + 1) * long[0] + 1_000 * long[1]) / 1e6)


def test_haiku_5_5_threshold_counts_cached_tokens():
    # 1 fresh token + 100k cache reads is a 100,001-token prompt: the higher tier.
    above = _usd("claude-haiku-5-5", 100_001, 1_000, cache_read_tokens=100_000)
    assert above == pytest.approx((1 * 0.50 + 100_000 * 0.05 + 1_000 * 2.50) / 1e6)
    at = _usd("claude-haiku-5-5", 100_000, 1_000, cache_read_tokens=99_999)
    assert at == pytest.approx((1 * 0.10 + 99_999 * 0.01 + 1_000 * 0.50) / 1e6)


def test_gemini_long_context_prices_cache_reads_at_the_tier_rate():
    usd = _usd("gemini-2.5-pro", 300_000, 0, cache_read_tokens=250_000)
    assert usd == pytest.approx((50_000 * 2.50 + 250_000 * 0.25) / 1e6)


# ---------------------------------------------------------------------------
# Anthropic modifiers: fast mode and US-only inference
# ---------------------------------------------------------------------------

class TestAnthropicModifiers:
    def test_fast_mode_rates(self):
        assert _usd("claude-opus-5-5", speed="fast") == pytest.approx(8.00 + 40.00)
        assert _usd("claude-opus-5", speed="fast") == pytest.approx(10.00 + 50.00)
        assert _usd("claude-opus-4-8", speed="fast") == pytest.approx(10.00 + 50.00)
        assert _usd("claude-opus-5-5", speed="standard") == pytest.approx(4.00 + 20.00)

    def test_cache_multipliers_apply_on_top_of_fast_mode(self):
        usd = _usd("claude-opus-5-5", 11_000, 1_000, cache_read_tokens=10_000, speed="fast")
        # 1k*8 + 10k*(8*0.05) + 1k*40
        assert usd == pytest.approx((8_000 + 4_000 + 40_000) / 1e6)

    def test_us_inference_is_ten_percent_on_everything(self):
        base = _usd("claude-sonnet-4-6", 11_000, 1_000, cache_read_tokens=5_000, cache_write_tokens=5_000)
        us = _usd("claude-sonnet-4-6", 11_000, 1_000, cache_read_tokens=5_000, cache_write_tokens=5_000,
                  inference_geo="us")
        assert us == pytest.approx(base * 1.1)
        assert _usd("claude-sonnet-4-6", inference_geo="global") == pytest.approx(18.00)

    def test_fast_and_us_stack(self):
        assert _usd("claude-opus-4-8", speed="fast", inference_geo="us") == pytest.approx(60.00 * 1.1)

    def test_anthropic_adapter_reads_speed_and_geo_from_usage(self):
        from pactrun.adapters.anthropic import AnthropicAdapter

        usage = NS(input_tokens=1_000, output_tokens=1_000, cache_read_input_tokens=0,
                   cache_creation_input_tokens=0, speed="fast", inference_geo="us")
        response = NS(model="claude-opus-5-5", content=[NS(type="text", text="ok")], usage=usage)
        with Contract("t").session() as s:
            AnthropicAdapter()._emit_response({}, response, 1.0)
        (e,) = _llm_events(s)
        assert e.cost_usd == pytest.approx((1_000 * 8 + 1_000 * 40) * 1.1 / 1e6)

    def test_langchain_reads_modifiers_from_raw_usage(self):
        pytest.importorskip("langchain_core")
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, LLMResult

        from pactrun.adapters.langchain import PactrunCallbackHandler

        message = AIMessage(
            content="ok",
            usage_metadata={"input_tokens": 1_000, "output_tokens": 1_000, "total_tokens": 2_000},
            response_metadata={"model_name": "claude-opus-4-8",
                               "usage": {"input_tokens": 1_000, "output_tokens": 1_000, "speed": "fast"}},
        )
        result = LLMResult(generations=[[ChatGeneration(message=message)]], llm_output={})
        with Contract("t").session() as s:
            PactrunCallbackHandler().on_llm_end(result)
        (e,) = _llm_events(s)
        assert e.model == "claude-opus-4-8"  # from response_metadata when llm_output has none
        assert e.cost_usd == pytest.approx((1_000 * 10 + 1_000 * 50) / 1e6)

    def test_cost_model_and_wrap_apply_modifiers(self):
        import pactrun
        from pactrun import cost_model as cm

        cost, tag = cm.actual_cost("claude-opus-5", 1_000, 1_000, speed="fast", inference_geo="us")
        assert tag == cm.EXACT
        assert cost == pytest.approx(60_000 * 1.1 / 1e6)

        usage = NS(input_tokens=1_000, output_tokens=1_000, speed="fast", inference_geo="us")
        response = NS(model="claude-opus-5", content=[NS(type="text", text="ok")], usage=usage)
        client = pactrun.wrap(NS(messages=NS(create=lambda **kw: response)))
        client.messages.create(model="claude-opus-5", max_tokens=10, messages=[], speed="fast")
        (e,) = _llm_events(client.session)
        assert e.cost_usd == pytest.approx(60_000 * 1.1 / 1e6)

    def test_precall_bound_uses_request_speed_and_geo(self):
        from pactrun import cost_model as cm

        messages = [{"role": "user", "content": "word " * 500}]
        standard, _ = cm.precall_worstcase("claude-opus-4-8", messages, 100)
        fast, _ = cm.precall_worstcase("claude-opus-4-8", messages, 100, speed="fast")
        fast_us, _ = cm.precall_worstcase("claude-opus-4-8", messages, 100, speed="fast", inference_geo="us")
        assert fast == pytest.approx(standard * 2)
        assert fast_us == pytest.approx(fast * 1.1)


# ---------------------------------------------------------------------------
# Unknown models: priced at the provider ceiling, flagged, never $0
# ---------------------------------------------------------------------------

class TestUnknownModels:
    @pytest.mark.parametrize(
        ("model", "provider"),
        [("gpt-9-hypothetical", "openai"), ("o9-mini", "openai"), ("ft:gpt-4o-mini:org::abc", "openai"),
         ("claude-3-5-sonnet-20241022", "anthropic"), ("gemini-flash-latest", "google"),
         ("mistral-large-latest", None)],
    )
    def test_priced_at_provider_ceiling(self, model, provider):
        assert pricing.provider_of(model) == provider
        quote = pricing.cost(model, TokenUsage(prompt_tokens=1_000_000, completion_tokens=1_000_000))
        ceiling = pricing.ceiling(provider)
        assert not quote.known
        assert quote.metadata == {"cost_source": "unknown_model"}
        assert quote.usd == pytest.approx(ceiling.input + ceiling.output)
        assert quote.usd > 0

    def test_ceilings(self):
        assert (pricing.ceiling("openai").input, pricing.ceiling("openai").output) == (150.00, 600.00)
        assert (pricing.ceiling("anthropic").input, pricing.ceiling("anthropic").output) == (15.00, 75.00)
        assert (pricing.ceiling("google").input, pricing.ceiling("google").output) == (4.00, 18.00)
        assert pricing.ceiling(None) == pricing.ceiling("no-such-provider")

    def test_ceiling_is_at_least_every_known_model_of_its_provider(self):
        usage = TokenUsage(prompt_tokens=500_000, completion_tokens=100_000, cache_read_tokens=200_000,
                           cache_write_tokens=200_000, cache_write_1h_tokens=100_000)
        for key, entry in pricing.PRICES.items():
            for fast in ("standard", "fast"):
                u = TokenUsage(**{**vars(usage), "speed": fast})
                known = pricing.cost(key, u).usd
                for provider in (entry.provider, None):
                    top = pricing.ceiling(provider)
                    ceiling = (
                        100_000 * top.input + 200_000 * top.input * top.cache.read
                        + 100_000 * top.input * top.cache.write + 100_000 * top.input * top.cache.write_1h_rate
                        + 100_000 * top.output
                    ) / 1e6
                    assert ceiling >= known, (key, provider)

    def test_cache_reads_get_no_discount_at_the_ceiling(self):
        usd = _usd("claude-unknown-9", 1_000_000, 0, cache_read_tokens=1_000_000)
        assert usd == pytest.approx(15.00)

    def test_one_warning_per_model_name(self, caplog):
        with caplog.at_level(logging.WARNING, logger="pactrun"):
            for _ in range(3):
                _usd("gpt-warn-once-test")
            _usd("gpt-warn-once-test-2")
        messages = [r.getMessage() for r in caplog.records if "no price for model" in r.getMessage()]
        assert len(messages) == 2
        assert "'gpt-warn-once-test'" in messages[0]

    def test_known_models_do_not_warn(self, caplog):
        with caplog.at_level(logging.WARNING, logger="pactrun"):
            _usd("gpt-5.4-mini-2026-03-17")
        assert not caplog.records

    @pytest.mark.parametrize("adapter", ["openai", "gemini"])
    def test_adapters_flag_unknown_models(self, adapter):
        with Contract("t").session() as s:
            if adapter == "openai":
                from pactrun.adapters.openai import OpenAIAdapter

                usage = NS(prompt_tokens=1_000, completion_tokens=1_000)
                response = NS(model="gpt-9-adapter-test", choices=[NS(message=NS(content="", tool_calls=None))],
                              usage=usage)
                OpenAIAdapter()._emit_response({}, response, 1.0)
            else:
                from pactrun.adapters.gemini import GeminiAdapter

                usage = NS(prompt_token_count=1_000, candidates_token_count=1_000)
                response = NS(usage_metadata=usage, function_calls=[], text="")
                GeminiAdapter()._emit_response({"model": "gemini-9-adapter-test"}, response, 1.0)
        (e,) = _llm_events(s)
        assert e.cost_usd > 0
        assert e.metadata["cost_source"] == "unknown_model"

    def test_known_model_events_carry_no_cost_source(self):
        from pactrun.adapters.openai import OpenAIAdapter

        response = NS(model="gpt-4.1-mini", choices=[NS(message=NS(content="", tool_calls=None))],
                      usage=NS(prompt_tokens=1_000_000, completion_tokens=1_000_000))
        with Contract("t").session() as s:
            OpenAIAdapter()._emit_response({}, response, 1.0)
        (e,) = _llm_events(s)
        assert e.cost_usd == pytest.approx(2.00)
        assert "cost_source" not in e.metadata

    def test_wrap_flags_unknown_model(self):
        import pactrun

        response = NS(model="gpt-9-wrap-test", choices=[NS(message=NS(content="ok", tool_calls=[]))],
                      usage=NS(prompt_tokens=1_000, completion_tokens=1_000))
        client = pactrun.wrap(NS(chat=NS(completions=NS(create=lambda **kw: response))))
        client.chat.completions.create(model="gpt-9-wrap-test", max_tokens=10, messages=[])
        (e,) = _llm_events(client.session)
        assert e.cost_usd == pytest.approx((1_000 * 150 + 1_000 * 600) / 1e6)
        assert e.metadata["cost_source"] == "unknown_model"

    def test_cost_model_unknown_model_uses_ceiling(self):
        from pactrun import cost_model as cm

        cost, tag = cm.actual_cost("claude-unknown-cost-model", 1_000_000, 0)
        assert tag == cm.HEURISTIC
        assert cost == pytest.approx(15.00)
        assert cm.cache_write_multiplier("gpt-9-unknown", [{"role": "user", "content": "x"}]) == 1.25

    def test_litellm_adapter_falls_back_to_table_when_litellm_cannot_price(self, monkeypatch):
        from pactrun.adapters.litellm import LiteLLMAdapter

        def completion_cost(**_):
            raise ValueError("model not mapped")

        monkeypatch.setitem(sys.modules, "litellm", NS(completion_cost=completion_cost))
        for model, expected, flagged in (("gpt-4.1-mini", 2.00, False), ("gpt-9-litellm-test", 750.00, True)):
            response = NS(model=model, choices=[NS(message=NS(content="", tool_calls=None))],
                          usage=NS(prompt_tokens=1_000_000, completion_tokens=1_000_000))
            with Contract("t").session() as s:
                LiteLLMAdapter()._emit_response({}, response, 1.0)
            (e,) = _llm_events(s)
            assert e.cost_usd == pytest.approx(expected)
            assert (e.metadata.get("cost_source") == "unknown_model") is flagged


# ---------------------------------------------------------------------------
# Invalid counts still fail closed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [-1, float("nan"), float("inf"), "12"])
@pytest.mark.parametrize("model", ["gpt-5.4", "claude-haiku-5-5", "gemini-2.5-pro", "no-such-model"])
def test_invalid_counts_give_nan_cost(model, bad):
    for field in ("prompt_tokens", "completion_tokens", "cache_read_tokens"):
        usage = TokenUsage(**{"prompt_tokens": 300_000, "completion_tokens": 10, field: bad})
        assert math.isnan(pricing.cost(model, usage).usd)
    assert math.isnan(pricing.cost(model, TokenUsage(prompt_tokens=bad, speed="fast", inference_geo="us")).usd)


@pytest.fixture
def restore_prices():
    from pactrun.core import pricing as P

    saved, ceilings = dict(P.PRICES), P._CEILINGS
    yield P
    P.PRICES.clear()
    P.PRICES.update(saved)
    P._CEILINGS = ceilings


class TestRegisterPrice:
    def test_registered_model_is_known_and_priced(self, restore_prices):
        from pactrun.core.usage import TokenUsage

        P = restore_prices
        usage = TokenUsage(prompt_tokens=1_000_000, completion_tokens=1_000_000)
        assert P.cost("ft:gpt-4o-mini:acme::x1", usage).known is False
        P.register_price("ft:gpt-4o-mini:acme::x1", 0.30, 1.20, cache_read=0.5)
        quote = P.cost("ft:gpt-4o-mini:acme::x1", usage)
        assert quote.known and quote.usd == pytest.approx(1.50) and quote.metadata == {}

    def test_pricier_registration_raises_the_ceiling(self, restore_prices):
        P = restore_prices
        before = P.ceiling("openai").input
        P.register_price("gpt-9-colossus", before * 2, 10_000)
        assert P.ceiling("openai").input == before * 2

    @pytest.mark.parametrize("args", [("m", -1, 1), ("m", 1, float("nan")), ("", 1, 1)])
    def test_bad_registration(self, restore_prices, args):
        with pytest.raises(ValueError):
            restore_prices.register_price(*args)
