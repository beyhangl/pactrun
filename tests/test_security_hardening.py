"""Regression suite for guardrail-integrity fixes.

Each class below locks in a bypass class that has shipped as a real advisory
or CVE in a peer runtime guardrail. Every test here fails against the code
before the fix (verified when the suite was written).
"""

import math
from types import SimpleNamespace as NS

import pytest

from pactrun import (
    Contract,
    ViolationError,
    cost_per_turn_under,
    cost_under,
    no_exfil_links,
    spend_rate_under,
    token_budget,
    tool_host_within,
    tools_allowed,
    wrap,
)
from pactrun.predicates.base import predicate


def _llm(contract, costs=(), tokens=()):
    with contract.session() as s:
        for c in costs:
            s.emit_llm_response(model="m", output="x", cost=c)
        for t in tokens:
            s.emit_llm_response(model="m", output="x", prompt_tokens=t)
    return s


# ---------------------------------------------------------------------------
# 1. Budget integrity: NaN / Infinity / negative amounts
#    (the class of a published budget-kill-switch bypass in a peer toolkit)
# ---------------------------------------------------------------------------

class TestBudgetIntegrity:
    def test_negative_cost_cannot_refund_the_budget(self):
        s = _llm(Contract("t").require(cost_under(5.0), on_fail="log"), costs=[4.9, -100.0, 50.0])
        assert s.state.total_cost_usd == pytest.approx(54.9)  # real spend, never lowered
        assert not s.is_compliant

    def test_negative_tokens_cannot_refund_the_budget(self):
        s = _llm(Contract("t").require(token_budget(100), on_fail="log"), tokens=[90, -1000, 500])
        assert s.state.total_tokens == 590
        assert not s.is_compliant

    def test_invalid_amount_fails_closed_on_that_event(self):
        s = _llm(Contract("t").require(cost_under(5.0), on_fail="log"), costs=[-1.0])
        assert len(s.violations) == 1
        assert "invalid amount" in s.violations[0].message

    def test_nan_does_not_poison_the_rest_of_the_run(self):
        s = _llm(Contract("t").require(cost_under(5.0), on_fail="log"), costs=[1.0, math.nan, 0.01, 0.01])
        assert math.isfinite(s.state.total_cost_usd)
        assert s.state.total_cost_usd == pytest.approx(1.02)
        assert len(s.violations) == 1  # only the NaN event, not every later one

    @pytest.mark.parametrize("bad", [math.inf, -math.inf])
    def test_infinite_cost_is_rejected(self, bad):
        s = _llm(Contract("t").require(cost_under(5.0), on_fail="log"), costs=[bad])
        assert math.isfinite(s.state.total_cost_usd)
        assert not s.is_compliant

    def test_invalid_cost_blocks_the_spend_window_too(self):
        c = Contract("t").require(spend_rate_under(5.0, 60), on_fail="log")
        s = _llm(c, costs=[4.9, -100.0])
        assert not s.is_compliant

    def test_block_mode_halts_on_invalid_amount(self):
        with pytest.raises(ViolationError):
            _llm(Contract("t").require(cost_under(5.0)), costs=[-1.0])

    @pytest.mark.parametrize("factory", [cost_under, cost_per_turn_under, token_budget])
    @pytest.mark.parametrize("bad", [math.nan, math.inf, -1.0])
    def test_nonsense_limits_rejected_at_construction(self, factory, bad):
        with pytest.raises(ValueError):
            factory(bad)

    def test_zero_budget_is_legitimate(self):
        cost_under(0)  # "nothing allowed" is a valid, if strict, policy

    def test_spend_window_must_be_positive(self):
        with pytest.raises(ValueError):
            spend_rate_under(5.0, 0)

    def test_wrap_rejects_a_negative_budget(self):
        client = NS(chat=NS(completions=NS(create=lambda **kw: None)))
        with pytest.raises(ValueError):
            wrap(client, max_cost="$-5")


# ---------------------------------------------------------------------------
# 2. Failure posture: a predicate that raises is a FAILED check
# ---------------------------------------------------------------------------

@predicate("_test_exploding_predicate")
def _exploding():
    def check(event, state):
        raise RuntimeError("detector unavailable")
    check.predicate_name = "_test_exploding_predicate"
    return check


class TestFailurePosture:
    def test_raising_predicate_fails_closed_by_default(self):
        c = Contract("t").require(_exploding())
        with pytest.raises(ViolationError, match="treated as failing"):
            with c.session() as s:
                s.emit_llm_response(model="m", output="x")

    def test_raising_predicate_is_recorded_not_crashed(self):
        c = Contract("t").require(_exploding(), on_fail="log")
        with c.session() as s:
            s.emit_llm_response(model="m", output="x")
        assert len(s.violations) == 1
        assert "RuntimeError" in s.violations[0].actual

    def test_other_clauses_still_evaluated_after_a_crash(self):
        c = (
            Contract("t")
            .require(_exploding(), on_fail="log")
            .require(cost_under(1.0), on_fail="log")
        )
        s = _llm(c, costs=[5.0])
        # both the crashed clause and the real budget breach are recorded
        assert len(s.violations) == 2


# ---------------------------------------------------------------------------
# 3. Monitor (shadow) mode
# ---------------------------------------------------------------------------

class TestMonitorMode:
    def test_monitor_records_but_never_blocks(self):
        c = Contract("t").require(cost_under(1.0)).monitor()  # default on_fail=block
        s = _llm(c, costs=[5.0])  # would raise under enforcement
        assert len(s.violations) == 1
        assert s.violations[0].enforced is False
        assert not s.is_compliant  # compliance stays honest

    def test_session_level_mode_override(self):
        c = Contract("t").require(cost_under(1.0))
        with c.session(mode="monitor") as s:
            s.emit_llm_response(model="m", output="x", cost=5.0)
        assert s.violations[0].enforced is False

    def test_enforce_mode_marks_violations_enforced(self):
        s = _llm(Contract("t").require(cost_under(1.0), on_fail="log"), costs=[5.0])
        assert s.violations[0].enforced is True

    def test_enforced_flag_round_trips(self):
        from pactrun.core.models import Violation

        v = Violation(enforced=False)
        assert Violation.from_dict(v.to_dict()).enforced is False

    def test_unknown_mode_rejected(self):
        with pytest.raises(ValueError):
            Contract("t").session(mode="yolo")


# ---------------------------------------------------------------------------
# 4. Egress parser differentials (the SSRF-validator CVE class)
# ---------------------------------------------------------------------------

def _host_blocked(url, **kw):
    c = Contract("t").require(tool_host_within(**kw), on_fail="log")
    with c.session() as s:
        s.emit_tool_call("fetch", args={"url": url})
    return not s.is_compliant


LOOPBACK_ALIASES = [
    "http://2130706433/",      # decimal
    "http://0x7f000001/",      # hex
    "http://0177.0.0.1/",      # octal (libc inet_aton reading)
    "http://127.1/",           # short form
    "http://localhost./",      # trailing root dot
    "http://api.localhost/",   # RFC 6761 *.localhost
    "http://0.0.0.0:8080/",
    "http://[::ffff:127.0.0.1]/",
    "http://[::ffff:169.254.169.254]/",
]


class TestEgressParserDifferentials:
    @pytest.mark.parametrize("url", LOOPBACK_ALIASES)
    def test_loopback_aliases_are_blocked(self, url):
        assert _host_blocked(url, block_private=True)

    def test_backslash_differential_cannot_beat_an_allowlist(self):
        # RFC 3986 reads host=good.com; WHATWG reads host=evil.com
        assert _host_blocked("http://evil.com\\@good.com/", allow=["good.com"])

    @pytest.mark.parametrize("url", ["https://good.com/x", "https://good.com./x"])
    def test_legitimate_allowlisted_host_passes(self, url):
        assert not _host_blocked(url, allow=["good.com"])

    def test_public_ip_and_host_pass_block_private(self):
        assert not _host_blocked("https://example.com/x", block_private=True)
        assert not _host_blocked("http://8.8.8.8/", block_private=True)

    def test_output_image_exfil_uses_same_canonicalization(self):
        c = Contract("t").require(no_exfil_links(allow_hosts=["cdn.good.com"]), on_fail="log")
        with c.session() as s:
            s.emit_llm_response(model="m", output="![x](http://2130706433/leak?d=secret)")
        assert not s.is_compliant


# ---------------------------------------------------------------------------
# 5. Allowlists enforce at CALL time, not list time
#    (the class where restrictions applied only when tools were listed)
# ---------------------------------------------------------------------------

def test_unlisted_tool_is_blocked_when_called_directly():
    c = Contract("t").require(tools_allowed(["search"]), on_fail="log")
    with c.session() as s:
        s.emit_tool_call("delete_everything", args={})  # never "listed" anywhere
    assert not s.is_compliant
