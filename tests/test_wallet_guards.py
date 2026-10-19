"""context_growth_under, max_steps_per_turn and tool_spend_under."""

import math

import pytest

from pactrun import (
    Contract,
    Event,
    EventKind,
    ViolationError,
    context_growth_under,
    max_steps_per_turn,
    tool_spend_under,
)
from pactrun.predicates.base import predicate_owasp


def _llm(prompt, **kw):
    return Event(kind=EventKind.LLM_CALL, model="m", prompt_tokens=prompt, completion_tokens=10, **kw)


def _pay(amount, tool="create_payment", **args):
    return Event(kind=EventKind.TOOL_CALL, tool_name=tool, tool_args={"amount": amount, **args})


def _violated(predicate_fn, events):
    return Contract("t").require(predicate_fn).replay(events).violated


class TestContextGrowth:
    def test_absolute_cap(self):
        assert _violated(context_growth_under(max_prompt_tokens=50_000), [_llm(10_000), _llm(60_000)]) == [
            "context_growth_under"
        ]

    def test_growth_ratio_against_first_prompt(self):
        events = [_llm(2_000), _llm(8_000), _llm(30_000)]
        assert _violated(context_growth_under(max_growth_ratio=10), events) == ["context_growth_under"]
        assert _violated(context_growth_under(max_growth_ratio=20), events) == []

    def test_small_first_prompt_is_raised_to_the_baseline(self):
        # 50 -> 5,000 is 100x, but against the 1,000-token baseline it is 5x.
        assert _violated(context_growth_under(max_growth_ratio=10), [_llm(50), _llm(5_000)]) == []

    def test_cached_input_counts(self):
        # prompt_tokens is total input, cached included - a mostly-cached
        # prompt still grows the bill and must still trip the cap.
        big = _llm(120_000, cache_read_tokens=118_000)
        assert _violated(context_growth_under(max_prompt_tokens=100_000), [big]) == ["context_growth_under"]

    def test_invalid_count_fails_closed(self):
        e = _llm(1_000)
        e.prompt_tokens = math.nan
        assert _violated(context_growth_under(max_prompt_tokens=10), [e]) == ["context_growth_under"]

    def test_other_events_ignored(self):
        tool = Event(kind=EventKind.TOOL_CALL, tool_name="search")
        assert _violated(context_growth_under(max_prompt_tokens=10), [tool]) == []

    @pytest.mark.parametrize("kwargs", [
        {}, {"max_prompt_tokens": 0}, {"max_growth_ratio": -1}, {"max_growth_ratio": math.inf},
        {"max_prompt_tokens": 10, "min_baseline_tokens": 0},
    ])
    def test_bad_args(self, kwargs):
        with pytest.raises(ValueError):
            context_growth_under(**kwargs)


class TestMaxStepsPerTurn:
    def test_tool_loop_without_returning_to_user(self):
        events = [Event(kind=EventKind.INPUT, input="hi")] + [_llm(100) for _ in range(4)]
        assert _violated(max_steps_per_turn(3), events) == ["max_steps_per_turn"]

    def test_counter_resets_at_each_boundary(self):
        events = []
        for boundary in (EventKind.INPUT, EventKind.OUTPUT, EventKind.TURN_END):
            events += [Event(kind=boundary)] + [_llm(100) for _ in range(3)]
        assert _violated(max_steps_per_turn(3), events) == []

    def test_unmarked_run_is_one_turn(self):
        assert _violated(max_steps_per_turn(2), [_llm(1), _llm(1), _llm(1)]) == ["max_steps_per_turn"]

    def test_blocks_live(self):
        with pytest.raises(ViolationError), Contract("t").require(max_steps_per_turn(1)).session() as s:
            s.emit_llm_response(model="m", output="", prompt_tokens=1)
            s.emit_llm_response(model="m", output="", prompt_tokens=1)

    @pytest.mark.parametrize("n", [0, -1, math.nan])
    def test_bad_n(self, n):
        with pytest.raises(ValueError):
            max_steps_per_turn(n)


class TestToolSpend:
    def test_per_call_cap(self):
        assert _violated(tool_spend_under(["create_payment"], per_call=100), [_pay(150)]) == ["tool_spend_under"]
        assert _violated(tool_spend_under(["create_payment"], per_call=100), [_pay(100)]) == []

    def test_per_session_total_spans_tools_and_numeric_strings(self):
        g = tool_spend_under(["create_payment", "issue_*"], per_session=100)
        assert _violated(g, [_pay(40), _pay("35.50", tool="issue_refund")]) == []
        assert _violated(g, [_pay(40), _pay("35.50", tool="issue_refund"), _pay(30)]) == ["tool_spend_under"]

    @pytest.mark.parametrize("bad", [-50, math.nan, math.inf, "lots", True, None])
    def test_invalid_amount_fails_closed(self, bad):
        assert _violated(tool_spend_under("create_payment", per_call=100), [_pay(bad)]) == ["tool_spend_under"]

    def test_negative_amount_cannot_lower_the_total(self):
        g = tool_spend_under("create_payment", per_session=100)
        result = Contract("t").require(g, on_fail="log").replay([_pay(90), _pay(-80), _pay(50)])
        assert [result.event_index(v) for v in result.violations] == [2, 3]

    def test_missing_amount_fails(self):
        e = Event(kind=EventKind.TOOL_CALL, tool_name="create_payment", tool_args={"value": 1})
        assert _violated(tool_spend_under("create_payment", per_call=10), [e]) == ["tool_spend_under"]

    def test_nested_amount_path(self):
        e = Event(kind=EventKind.TOOL_CALL, tool_name="pay", tool_args={"payment": {"amount": 5000}})
        assert _violated(tool_spend_under("pay", amount_arg="payment.amount", per_call=2500), [e]) == [
            "tool_spend_under"
        ]

    def test_currency(self):
        g = tool_spend_under("create_payment", currency="eur")
        assert _violated(g, [_pay(5, currency="EUR")]) == []
        assert _violated(g, [_pay(5, currency="usd")]) == ["tool_spend_under"]
        assert _violated(g, [_pay(5)]) == ["tool_spend_under"]

    def test_merchant_allowlist(self):
        g = tool_spend_under("create_payment", merchants=["m_123"])
        assert _violated(g, [_pay(5, merchant_id="m_123")]) == []
        assert _violated(g, [_pay(5, merchant_id="m_999")]) == ["tool_spend_under"]

    def test_expiry_uses_event_time(self):
        g = tool_spend_under("create_payment", expires_at=1_000.0)
        assert _violated(g, [Event(kind=EventKind.TOOL_CALL, tool_name="create_payment",
                                   tool_args={"amount": 1}, timestamp=999.0)]) == []
        assert _violated(g, [Event(kind=EventKind.TOOL_CALL, tool_name="create_payment",
                                   tool_args={"amount": 1}, timestamp=1_001.0)]) == ["tool_spend_under"]

    def test_other_tools_ignored(self):
        e = Event(kind=EventKind.TOOL_CALL, tool_name="lookup_order", tool_args={"amount": 10_000})
        assert _violated(tool_spend_under("create_payment", per_call=1), [e]) == []

    def test_blocks_before_the_payment_runs(self):
        contract = Contract("t").forbid(tool_spend_under("create_payment", per_call=100))
        with pytest.raises(ViolationError, match="per-call cap"), contract.session() as s:
            s.emit_tool_call("create_payment", args={"amount": 500})

    @pytest.mark.parametrize("args,kwargs", [
        (["create_payment"], {}),
        ([[]], {"per_call": 1}),
        (["create_payment"], {"per_call": -1}),
        (["create_payment"], {"per_session": math.nan}),
    ])
    def test_bad_args(self, args, kwargs):
        with pytest.raises(ValueError):
            tool_spend_under(*args, **kwargs)


def test_yaml_and_owasp(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "name: x\nclauses:\n"
        "  - forbid: tool_spend_under\n    args: {tools: [create_payment], per_session: 200, currency: EUR}\n"
        "  - require: context_growth_under\n    args: {max_growth_ratio: 20}\n"
        "  - require: max_steps_per_turn\n    args: {n: 25}\n",
        encoding="utf-8",
    )
    assert [c.predicate_name for c in Contract.from_yaml(p).clauses] == [
        "tool_spend_under", "context_growth_under", "max_steps_per_turn"
    ]
    assert predicate_owasp("context_growth_under") == ("ASI08",)
    assert predicate_owasp("tool_spend_under") == ("ASI02", "ASI03")
