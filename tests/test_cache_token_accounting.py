"""Prompt-cache and reasoning token accounting across every adapter.

Convention under test (see ``pactrun.core.models.Event``): ``prompt_tokens`` is
the TOTAL input including cache reads and writes, ``completion_tokens`` the
TOTAL output including reasoning, and the cache/reasoning counts are subsets.
Costs price cache reads, cache writes and uncached input separately.

The running example is a long agent turn: 100k cached input + 2k fresh input,
1k output. Expected costs are written out from the adapters' pricing tables.
"""

from __future__ import annotations

import math
import os
from types import SimpleNamespace as NS

import pytest

from pactrun import Contract, cost_under, token_budget
from pactrun.core.amounts import INVALID_AMOUNT_EVENTS_KEY, INVALID_AMOUNTS_KEY
from pactrun.core.enums import EventKind
from pactrun.core.models import Event, SessionState
from pactrun.core.usage import CacheRates, TokenUsage, price


def _llm_events(session):
    return [e for e in session.state.events if e.kind == EventKind.LLM_CALL]


# ---------------------------------------------------------------------------
# Shared pricing helper
# ---------------------------------------------------------------------------

class TestPriceHelper:
    def test_slices_priced_separately(self):
        usage = TokenUsage(
            prompt_tokens=102_000, completion_tokens=1_000,
            cache_read_tokens=60_000, cache_write_tokens=40_000, cache_write_1h_tokens=10_000,
        )
        # 2k uncached*3 + 60k*0.3 + 30k*3.75 + 10k*6 + 1k*15  (per 1M)
        expected = (2_000 * 3 + 60_000 * 0.3 + 30_000 * 3.75 + 10_000 * 6 + 1_000 * 15) / 1e6
        assert price(usage, 3.0, 15.0, CacheRates(0.1, 1.25, 2.0)) == pytest.approx(expected)

    def test_cache_counts_larger_than_input_are_clamped(self):
        usage = TokenUsage(prompt_tokens=1_000, cache_read_tokens=5_000)
        assert price(usage, 3.0, 15.0, CacheRates(read=0.1)) == pytest.approx(1_000 * 0.3 / 1e6)

    @pytest.mark.parametrize("bad", [-1, float("nan"), float("inf"), "12"])
    def test_invalid_count_gives_nan_cost(self, bad):
        usage = TokenUsage(prompt_tokens=1_000, cache_read_tokens=bad)
        assert math.isnan(price(usage, 3.0, 15.0, CacheRates(read=0.1)))


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

def _anthropic_response(usage, model="claude-sonnet-4-6"):
    return NS(model=model, content=[NS(type="text", text="ok")], usage=usage)


def _emit_anthropic(usage, model="claude-sonnet-4-6", contract=None):
    from pactrun.adapters.anthropic import AnthropicAdapter

    with (contract or Contract("t")).session() as s:
        AnthropicAdapter()._emit_response({}, _anthropic_response(usage, model), 1.0)
    return s


class TestAnthropic:
    def test_cache_reads_counted_in_total_and_discounted(self):
        usage = NS(input_tokens=2_000, output_tokens=1_000,
                   cache_read_input_tokens=100_000, cache_creation_input_tokens=0)
        s = _emit_anthropic(usage)
        (e,) = _llm_events(s)
        assert e.prompt_tokens == 102_000
        assert e.cache_read_tokens == 100_000
        assert e.cache_write_tokens == 0
        assert s.state.total_tokens == 103_000
        # sonnet-4-6 $3/$15: 2k*3 + 100k*0.3 + 1k*15 = $0.051 (was $0.021 before)
        assert e.cost_usd == pytest.approx(0.051)

    def test_five_minute_cache_write_premium(self):
        usage = NS(input_tokens=2_000, output_tokens=1_000,
                   cache_read_input_tokens=0, cache_creation_input_tokens=100_000)
        (e,) = _llm_events(_emit_anthropic(usage))
        assert e.prompt_tokens == 102_000
        assert e.cache_write_tokens == 100_000
        # 2k*3 + 100k*3.75 + 1k*15
        assert e.cost_usd == pytest.approx(0.396)

    def test_one_hour_cache_write_premium_from_breakdown(self):
        usage = NS(
            input_tokens=2_000, output_tokens=1_000,
            cache_read_input_tokens=0, cache_creation_input_tokens=100_000,
            cache_creation=NS(ephemeral_5m_input_tokens=60_000, ephemeral_1h_input_tokens=40_000),
        )
        (e,) = _llm_events(_emit_anthropic(usage))
        # 2k*3 + 60k*3.75 + 40k*6 + 1k*15
        assert e.cost_usd == pytest.approx(0.486)

    def test_thinking_tokens_are_a_subset_of_output(self):
        usage = NS(input_tokens=10, output_tokens=1_000,
                   output_tokens_details=NS(thinking_tokens=700))
        (e,) = _llm_events(_emit_anthropic(usage))
        assert e.completion_tokens == 1_000
        assert e.reasoning_tokens == 700

    def test_old_sdk_usage_without_cache_fields_unchanged(self):
        usage = NS(input_tokens=40, output_tokens=15)
        (e,) = _llm_events(_emit_anthropic(usage))
        assert (e.prompt_tokens, e.completion_tokens, e.cache_read_tokens) == (40, 15, 0)
        assert e.cost_usd == pytest.approx((40 * 3 + 15 * 15) / 1e6)

    def test_token_budget_now_sees_cached_input(self):
        usage = NS(input_tokens=2_000, output_tokens=1_000,
                   cache_read_input_tokens=100_000, cache_creation_input_tokens=0)
        contract = Contract("t").require(token_budget(50_000), on_fail="log")
        assert not _emit_anthropic(usage, contract=contract).is_compliant

    def test_invalid_cache_count_fails_closed(self):
        usage = NS(input_tokens=2_000, output_tokens=1_000,
                   cache_read_input_tokens=-100_000, cache_creation_input_tokens=0)
        contract = (Contract("t")
                    .require(token_budget(10**9), on_fail="log")
                    .require(cost_under(100.0), on_fail="log"))
        s = _emit_anthropic(usage, contract=contract)
        (e,) = _llm_events(s)
        rejected = e.metadata[INVALID_AMOUNTS_KEY]
        # The negative read is rejected, and so are the total it would have
        # lowered and the cost computed from it - both budgets fail closed.
        assert set(rejected) == {"cache_read_tokens", "prompt_tokens", "cost_usd"}
        assert e.cache_read_tokens == 0 and e.prompt_tokens == 0 and e.cost_usd == 0
        assert {v.predicate_name for v in s.violations} == {"token_budget", "cost_under"}


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

def _openai_usage(prompt, completion, cached=None, write=None, reasoning=None):
    return NS(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
        prompt_tokens_details=NS(cached_tokens=cached, cache_write_tokens=write),
        completion_tokens_details=NS(reasoning_tokens=reasoning),
    )


def _emit_openai(usage, model):
    from pactrun.adapters.openai import OpenAIAdapter

    message = NS(content="ok", tool_calls=None)
    response = NS(model=model, choices=[NS(message=message)], usage=usage)
    with Contract("t").session() as s:
        OpenAIAdapter()._emit_response({}, response, 1.0)
    return s


class TestOpenAI:
    def test_cached_input_discounted_gpt_4_1(self):
        (e,) = _llm_events(_emit_openai(_openai_usage(102_000, 1_000, cached=100_000), "gpt-4.1"))
        assert e.prompt_tokens == 102_000  # OpenAI's total already includes cached
        assert e.cache_read_tokens == 100_000
        # $2/$8, cached 0.25x: 2k*2 + 100k*0.5 + 1k*8 = $0.062 (was $0.212)
        assert e.cost_usd == pytest.approx(0.062)

    def test_cached_input_discounted_gpt_5_4(self):
        (e,) = _llm_events(_emit_openai(_openai_usage(102_000, 1_000, cached=100_000), "gpt-5.4"))
        # $2.50/$15, cached 0.1x: 2k*2.5 + 100k*0.25 + 1k*15 = $0.045 (was $0.27)
        assert e.cost_usd == pytest.approx(0.045)

    def test_cached_input_discounted_gpt_4o_mini_family_prefix(self):
        (e,) = _llm_events(_emit_openai(_openai_usage(102_000, 1_000, cached=100_000), "gpt-4o-mini"))
        # $0.15/$0.60, cached 0.5x: 2k*0.15 + 100k*0.075 + 1k*0.6
        assert e.cost_usd == pytest.approx((300 + 7_500 + 600) / 1e6)

    def test_cache_writes_recorded_at_input_rate_before_gpt_5_6(self):
        usage = _openai_usage(102_000, 1_000, cached=50_000, write=50_000)
        (e,) = _llm_events(_emit_openai(usage, "gpt-4.1"))
        assert e.cache_write_tokens == 50_000
        # 2k*2 + 50k*0.5 + 50k*2 + 1k*8
        assert e.cost_usd == pytest.approx(0.137)

    def test_reasoning_tokens_subset_of_completion(self):
        (e,) = _llm_events(_emit_openai(_openai_usage(100, 1_000, reasoning=800), "o3"))
        assert e.completion_tokens == 1_000
        assert e.reasoning_tokens == 800

    def test_usage_without_details_unchanged(self):
        usage = NS(prompt_tokens=50, completion_tokens=20, total_tokens=70)
        (e,) = _llm_events(_emit_openai(usage, "gpt-5.4-nano"))
        assert (e.prompt_tokens, e.cache_read_tokens, e.reasoning_tokens) == (50, 0, 0)
        assert e.cost_usd == pytest.approx((50 * 0.20 + 20 * 1.25) / 1e6)


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------

def _emit_gemini(usage, model="gemini-2.5-flash"):
    from pactrun.adapters.gemini import GeminiAdapter

    response = NS(model_version=model, usage_metadata=usage, function_calls=[], text="ok")
    with Contract("t").session() as s:
        GeminiAdapter()._emit_response({"model": model}, response, 1.0)
    return s


class TestGemini:
    def test_thoughts_counted_as_output_and_cache_discounted(self):
        usage = NS(prompt_token_count=102_000, cached_content_token_count=100_000,
                   candidates_token_count=1_000, thoughts_token_count=3_000)
        s = _emit_gemini(usage)
        (e,) = _llm_events(s)
        assert e.prompt_tokens == 102_000  # prompt_token_count already includes the cache
        assert e.cache_read_tokens == 100_000
        assert e.completion_tokens == 4_000  # candidates + thoughts
        assert e.reasoning_tokens == 3_000
        assert s.state.total_tokens == 106_000
        # $0.30/$2.50, cached 0.1x: 2k*0.3 + 100k*0.03 + 4k*2.5 = $0.0136 (was $0.0331)
        assert e.cost_usd == pytest.approx(0.0136)

    def test_tool_use_prompt_tokens_are_input(self):
        usage = NS(prompt_token_count=1_000, tool_use_prompt_token_count=500,
                   candidates_token_count=100)
        (e,) = _llm_events(_emit_gemini(usage))
        assert e.prompt_tokens == 1_500
        assert e.cost_usd == pytest.approx((1_500 * 0.30 + 100 * 2.50) / 1e6)

    def test_gemini_2_0_flash_quarter_rate(self):
        usage = NS(prompt_token_count=102_000, cached_content_token_count=100_000,
                   candidates_token_count=1_000)
        (e,) = _llm_events(_emit_gemini(usage, model="gemini-2.0-flash"))
        # $0.10/$0.40, cached 0.25x: 2k*0.1 + 100k*0.025 + 1k*0.4
        assert e.cost_usd == pytest.approx((200 + 2_500 + 400) / 1e6)


# ---------------------------------------------------------------------------
# LiteLLM
# ---------------------------------------------------------------------------

class TestLiteLLM:
    def test_fields_extracted_from_normalised_usage(self):
        from pactrun.adapters.litellm import LiteLLMAdapter

        usage = NS(
            prompt_tokens=102_000, completion_tokens=1_000,
            prompt_tokens_details=NS(cached_tokens=60_000, cache_creation_tokens=40_000),
            completion_tokens_details=NS(reasoning_tokens=300),
        )
        response = NS(model="claude-sonnet-4-6", choices=[NS(message=NS(content="ok", tool_calls=None))],
                      usage=usage, _hidden_params={"response_cost": 0.1234})
        with Contract("t").session() as s:
            LiteLLMAdapter()._emit_response({}, response, 1.0)
        (e,) = _llm_events(s)
        assert (e.prompt_tokens, e.cache_read_tokens, e.cache_write_tokens, e.reasoning_tokens) == (
            102_000, 60_000, 40_000, 300)
        assert e.cost_usd == pytest.approx(0.1234)  # LiteLLM's own cost wins

    def test_real_litellm_usage_object(self):
        utils = pytest.importorskip("litellm.types.utils")
        from pactrun.adapters.litellm import usage_from_litellm

        usage = utils.Usage(prompt_tokens=102_000, completion_tokens=1_000, total_tokens=103_000,
                            cache_read_input_tokens=100_000)
        got = usage_from_litellm(usage)
        assert (got.prompt_tokens, got.cache_read_tokens) == (102_000, 100_000)


# ---------------------------------------------------------------------------
# LangChain
# ---------------------------------------------------------------------------

class TestLangChain:
    @pytest.fixture(autouse=True)
    def _need_langchain(self):
        pytest.importorskip("langchain_core")

    def _emit(self, result):
        from pactrun.adapters import PactrunCallbackHandler

        with Contract("t").session() as s:
            PactrunCallbackHandler().on_llm_end(result, run_id="r")
        return s

    def test_usage_metadata_cache_details(self):
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, LLMResult

        message = AIMessage(content="ok", usage_metadata={
            "input_tokens": 102_000, "output_tokens": 1_000, "total_tokens": 103_000,
            "input_token_details": {"cache_read": 100_000, "cache_creation": 0},
            "output_token_details": {"reasoning": 250},
        })
        result = LLMResult(generations=[[ChatGeneration(message=message)]],
                           llm_output={"model_name": "claude-sonnet-4-6"})
        (e,) = _llm_events(self._emit(result))
        assert (e.prompt_tokens, e.cache_read_tokens, e.reasoning_tokens) == (102_000, 100_000, 250)
        assert e.cost_usd == pytest.approx(0.051)

    def test_usage_metadata_ttl_split_from_langchain_anthropic(self):
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, LLMResult

        # langchain-anthropic zeroes cache_creation and reports the TTL split.
        message = AIMessage(content="ok", usage_metadata={
            "input_tokens": 102_000, "output_tokens": 1_000, "total_tokens": 103_000,
            "input_token_details": {"cache_read": 0, "cache_creation": 0,
                                    "ephemeral_5m_input_tokens": 60_000,
                                    "ephemeral_1h_input_tokens": 40_000},
        })
        result = LLMResult(generations=[[ChatGeneration(message=message)]],
                           llm_output={"model_name": "claude-sonnet-4-6"})
        (e,) = _llm_events(self._emit(result))
        assert e.cache_write_tokens == 100_000
        assert e.cost_usd == pytest.approx(0.486)

    def test_raw_anthropic_llm_output_adds_cache_back(self):
        from langchain_core.outputs import Generation, LLMResult

        result = LLMResult(generations=[[Generation(text="ok")]], llm_output={
            "model_name": "claude-sonnet-4-6",
            "usage": {"input_tokens": 2_000, "output_tokens": 1_000,
                      "cache_read_input_tokens": 100_000, "cache_creation_input_tokens": 0},
        })
        (e,) = _llm_events(self._emit(result))
        assert e.prompt_tokens == 102_000
        assert e.cost_usd == pytest.approx(0.051)

    def test_raw_openai_llm_output_cached_tokens(self):
        from langchain_core.outputs import Generation, LLMResult

        result = LLMResult(generations=[[Generation(text="ok")]], llm_output={
            "model_name": "gpt-4.1",
            "token_usage": {"prompt_tokens": 102_000, "completion_tokens": 1_000,
                            "prompt_tokens_details": {"cached_tokens": 100_000}},
        })
        (e,) = _llm_events(self._emit(result))
        assert e.cache_read_tokens == 100_000
        assert e.cost_usd == pytest.approx(0.062)


# ---------------------------------------------------------------------------
# Pydantic AI (already normalises input_tokens to include the cache)
# ---------------------------------------------------------------------------

class TestPydanticAI:
    def test_cache_and_reasoning_recorded(self):
        pytest.importorskip("pydantic_ai")
        os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
        from decimal import Decimal

        from pydantic_ai import Agent
        from pydantic_ai.messages import ModelResponse, TextPart
        from pydantic_ai.models.function import FunctionModel
        from pydantic_ai.usage import RequestUsage

        from pactrun.adapters import PactrunCapability

        usage = RequestUsage(input_tokens=102_000, cache_read_tokens=100_000, cache_write_tokens=0,
                             output_tokens=1_000, details={"thinking_tokens": 400}, cost=Decimal("0.051"))

        def fn(messages, info):
            return ModelResponse(parts=[TextPart("hi")], usage=usage)

        cap = PactrunCapability(Contract("t"))
        Agent(FunctionModel(fn, model_name="x"), capabilities=[cap]).run_sync("hi")
        (e,) = _llm_events(cap.last_session)
        assert (e.prompt_tokens, e.cache_read_tokens, e.cache_write_tokens) == (102_000, 100_000, 0)
        assert (e.completion_tokens, e.reasoning_tokens) == (1_000, 400)
        assert e.cost_usd == pytest.approx(0.051)
        assert cap.last_session.state.total_cache_read_tokens == 100_000


# ---------------------------------------------------------------------------
# Session ingestion, Event / SessionState serialisation
# ---------------------------------------------------------------------------

class TestSessionAndModels:
    @pytest.mark.parametrize("bad", [-5, float("nan"), float("inf"), float("-inf")])
    @pytest.mark.parametrize("field", ["cache_read_tokens", "cache_write_tokens", "reasoning_tokens"])
    def test_invalid_cache_amounts_sanitized(self, field, bad):
        with Contract("t").session() as s:
            s.emit_llm_response(model="m", output="", prompt_tokens=10, completion_tokens=5, **{field: bad})
        (e,) = _llm_events(s)
        assert getattr(e, field) == 0
        assert field in e.metadata[INVALID_AMOUNTS_KEY]
        assert s.state.metadata[INVALID_AMOUNT_EVENTS_KEY] == 1
        assert s.state.total_cache_read_tokens == 0
        assert s.state.total_cache_write_tokens == 0
        assert s.state.total_reasoning_tokens == 0

    def test_session_totals(self):
        with Contract("t").session() as s:
            s.emit_llm_response(model="m", output="", prompt_tokens=100, completion_tokens=10,
                                cache_read_tokens=60, cache_write_tokens=30, reasoning_tokens=4)
            s.emit_llm_response(model="m", output="", prompt_tokens=100, completion_tokens=10,
                                cache_read_tokens=90, reasoning_tokens=6)
        st = s.state
        assert (st.total_tokens, st.total_cache_read_tokens, st.total_cache_write_tokens,
                st.total_reasoning_tokens) == (220, 150, 30, 10)
        d = st.to_dict()
        assert (d["total_cache_read_tokens"], d["total_cache_write_tokens"],
                d["total_reasoning_tokens"]) == (150, 30, 10)

    def test_manual_emit_accepts_cache_fields(self):
        from pactrun.adapters.manual import emit_llm_call

        with Contract("t").session() as s:
            emit_llm_call("m", "x", prompt_tokens=100, cache_read_tokens=80, cache_write_tokens=10,
                          completion_tokens=5, reasoning_tokens=2)
        (e,) = _llm_events(s)
        assert (e.cache_read_tokens, e.cache_write_tokens, e.reasoning_tokens) == (80, 10, 2)

    def test_event_round_trip(self):
        e = Event(model="m", prompt_tokens=100, completion_tokens=10,
                  cache_read_tokens=60, cache_write_tokens=30, reasoning_tokens=4)
        back = Event.from_dict(e.to_dict())
        assert (back.cache_read_tokens, back.cache_write_tokens, back.reasoning_tokens) == (60, 30, 4)
        assert back.to_dict() == e.to_dict()

    def test_old_event_dict_still_loads(self):
        old = Event(model="m", prompt_tokens=7).to_dict()
        for key in ("cache_read_tokens", "cache_write_tokens", "reasoning_tokens"):
            del old[key]
        back = Event.from_dict(old)
        assert (back.prompt_tokens, back.cache_read_tokens, back.cache_write_tokens,
                back.reasoning_tokens) == (7, 0, 0, 0)

    def test_session_state_defaults(self):
        st = SessionState()
        assert (st.total_cache_read_tokens, st.total_cache_write_tokens, st.total_reasoning_tokens) == (0, 0, 0)


# ---------------------------------------------------------------------------
# OpenTelemetry
# ---------------------------------------------------------------------------

class TestOTel:
    @pytest.fixture
    def spans(self):
        pytest.importorskip("opentelemetry.sdk")
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        from pactrun.observability import OTelObserver

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        return OTelObserver(tracer_provider=provider), exporter

    def test_cache_and_reasoning_attributes(self, spans):
        observer, exporter = spans
        with Contract("t").session(observers=[observer]) as s:
            s.emit_llm_response(model="claude-sonnet-4-6", output="", prompt_tokens=102_000,
                                completion_tokens=1_000, cache_read_tokens=60_000,
                                cache_write_tokens=40_000, reasoning_tokens=300)
        attrs = dict(exporter.get_finished_spans()[0].attributes)
        assert attrs["gen_ai.usage.input_tokens"] == 102_000  # total, incl. cache
        assert attrs["gen_ai.usage.cache_read.input_tokens"] == 60_000
        assert attrs["gen_ai.usage.cache_write.input_tokens"] == 40_000
        assert attrs["gen_ai.usage.reasoning.output_tokens"] == 300
        assert attrs["gen_ai.usage.output_tokens"] == 1_000
        assert "gen_ai.usage.cache_creation.input_tokens" not in attrs  # renamed in the spec

    def test_breakdown_attributes_omitted_when_zero(self, spans):
        observer, exporter = spans
        with Contract("t").session(observers=[observer]) as s:
            s.emit_llm_response(model="gpt-4.1", output="", prompt_tokens=10, completion_tokens=5)
        attrs = dict(exporter.get_finished_spans()[0].attributes)
        assert not any(k.startswith(("gen_ai.usage.cache_", "gen_ai.usage.reasoning")) for k in attrs)

    def test_anthropic_adapter_total_reaches_span(self, spans):
        from pactrun.adapters.anthropic import AnthropicAdapter

        observer, exporter = spans
        usage = NS(input_tokens=2_000, output_tokens=1_000,
                   cache_read_input_tokens=100_000, cache_creation_input_tokens=0)
        with Contract("t").session(observers=[observer]):
            AnthropicAdapter()._emit_response({}, _anthropic_response(usage), 1.0)
        attrs = dict(exporter.get_finished_spans()[0].attributes)
        assert attrs["gen_ai.usage.input_tokens"] == 102_000
        assert attrs["gen_ai.usage.cache_read.input_tokens"] == 100_000


# ---------------------------------------------------------------------------
# pactrun.wrap() and cost_model
# ---------------------------------------------------------------------------

class _FakeAnthropic:
    def __init__(self, responses):
        responses = list(responses)

        def create(**kwargs):
            item = responses.pop(0)
            return item(kwargs) if callable(item) else item

        self.messages = NS(create=create)


class TestWrapAndCostModel:
    def test_wrap_anthropic_records_cache_tokens_and_cost(self):
        import pactrun

        usage = NS(input_tokens=2_000, output_tokens=1_000,
                   cache_read_input_tokens=100_000, cache_creation_input_tokens=0)
        client = pactrun.wrap(_FakeAnthropic([_anthropic_response(usage)]))
        client.messages.create(model="claude-sonnet-4-6", max_tokens=10, messages=[])
        (e,) = _llm_events(client.session)
        assert (e.prompt_tokens, e.cache_read_tokens) == (102_000, 100_000)
        # litellm and the static fallback agree on sonnet-4-6: $0.051
        assert e.cost_usd == pytest.approx(0.051)

    def test_wrap_openai_records_cached_tokens(self):
        import pactrun

        response = NS(model="gpt-4.1", choices=[NS(message=NS(content="ok", tool_calls=[]))],
                      usage=_openai_usage(102_000, 1_000, cached=100_000, reasoning=200))
        client = pactrun.wrap(NS(chat=NS(completions=NS(create=lambda **kw: response))))
        client.chat.completions.create(model="gpt-4.1", max_tokens=10, messages=[])
        (e,) = _llm_events(client.session)
        assert (e.cache_read_tokens, e.reasoning_tokens) == (100_000, 200)
        assert e.cost_usd == pytest.approx(0.062)

    def test_wrap_anthropic_stream_merges_start_and_delta(self):
        import pactrun

        events = [
            NS(type="message_start", message=NS(usage=NS(
                input_tokens=2_000, output_tokens=1, cache_read_input_tokens=0,
                cache_creation_input_tokens=100_000,
                cache_creation=NS(ephemeral_5m_input_tokens=0, ephemeral_1h_input_tokens=100_000)))),
            NS(type="content_block_delta", delta=NS(text="hi")),
            NS(type="message_delta", usage=NS(output_tokens=1_000)),
            NS(type="message_stop"),
        ]
        client = pactrun.wrap(_FakeAnthropic([lambda kw: iter(events)]))
        for _ in client.messages.create(model="claude-sonnet-4-6", max_tokens=10, messages=[], stream=True):
            pass
        (e,) = _llm_events(client.session)
        assert (e.prompt_tokens, e.cache_write_tokens, e.completion_tokens) == (102_000, 100_000, 1_000)
        # 1-hour write at 2x: 2k*3 + 100k*6 + 1k*15
        assert e.cost_usd == pytest.approx(0.621)

    def test_actual_cost_prices_cache_slices(self):
        from pactrun import cost_model as cm

        read, _ = cm.actual_cost("claude-sonnet-4-6", 102_000, 1_000, cache_read_tokens=100_000)
        assert read == pytest.approx(0.051)
        write_1h, _ = cm.actual_cost("claude-sonnet-4-6", 102_000, 1_000,
                                     cache_write_tokens=100_000, cache_write_1h_tokens=100_000)
        assert write_1h == pytest.approx(0.621)
        oa, _ = cm.actual_cost("gpt-4.1", 102_000, 1_000, cache_read_tokens=100_000)
        assert oa == pytest.approx(0.062)
        assert math.isnan(cm.actual_cost("gpt-4.1", 10, 1, cache_read_tokens=-1)[0])

    def test_actual_cost_static_fallback_prices_cache_slices(self, monkeypatch):
        import builtins

        from pactrun import cost_model as cm

        real_import = builtins.__import__

        def no_litellm(name, *args, **kwargs):
            if name == "litellm" or name.startswith("litellm."):
                raise ImportError(name)
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_litellm)
        cost, tag = cm.actual_cost("claude-sonnet-4-6", 102_000, 1_000, cache_read_tokens=100_000)
        assert tag == cm.HEURISTIC
        assert cost == pytest.approx(0.051)
        cost, _ = cm.actual_cost("gpt-4.1", 102_000, 1_000, cache_read_tokens=100_000)
        assert cost == pytest.approx(0.062)

    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        [
            ({}, 1.0),
            ({"cache_control": {"type": "ephemeral"}}, 1.25),
            ({"cache_control": {"type": "ephemeral", "ttl": "1h"}}, 2.0),
            ({"system": [{"type": "text", "text": "s", "cache_control": {"type": "ephemeral"}}]}, 1.25),
            ({"messages": [{"role": "user", "content": [
                {"type": "text", "text": "x", "cache_control": {"type": "ephemeral", "ttl": "1h"}}]}]}, 2.0),
            ({"tools": [{"name": "t", "cache_control": {"type": "ephemeral"}}]}, 1.25),
        ],
    )
    def test_precall_cache_write_multiplier_anthropic(self, kwargs, expected):
        from pactrun import cost_model as cm

        messages = kwargs.pop("messages", [{"role": "user", "content": "hi"}])
        assert cm.cache_write_multiplier("claude-sonnet-4-6", messages, **kwargs) == expected

    def test_precall_worstcase_includes_write_premium(self):
        from pactrun import cost_model as cm

        messages = [{"role": "user", "content": "word " * 2_000}]
        plain, _ = cm.precall_worstcase("claude-sonnet-4-6", messages, 100)
        cached, _ = cm.precall_worstcase("claude-sonnet-4-6", messages, 100,
                                         cache_control={"type": "ephemeral", "ttl": "1h"})
        in_tokens, _ = cm.count_input_tokens("claude-sonnet-4-6", messages)
        input_cost, _ = cm._price_tokens("claude-sonnet-4-6", in_tokens, 0)
        assert cached == pytest.approx(plain + input_cost)  # 2x input = +1x
        # A request that cannot write to the cache keeps the old bound.
        assert cm.cache_write_multiplier("gpt-4.1", messages) == 1.0
