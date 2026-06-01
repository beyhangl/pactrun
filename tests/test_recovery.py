"""Tests for the recovery engine: log / warn / block / escalate / retry / fallback."""

import pytest

from pactrun import (
    Contract,
    EscalationError,
    ViolationError,
    cost_under,
    get_active_session,
)


def _emit_cost(amount: float) -> None:
    """Emit one LLM event with a given cost into the active session."""
    get_active_session().emit_llm_response(model="m", output="x", cost=amount)


class TestEventLevelActions:
    def test_log_records_without_raising(self):
        c = Contract("t").require(cost_under(0.05), on_fail="log")
        with c.session() as s:
            s.emit_llm_response(model="m", output="x", cost=0.10)
        assert s.violation_count == 1
        assert not s.is_compliant

    def test_warn_emits_warning_without_raising(self):
        c = Contract("t").require(cost_under(0.05), on_fail="warn")
        with pytest.warns(UserWarning, match="contract violation"):
            with c.session() as s:
                s.emit_llm_response(model="m", output="x", cost=0.10)
        assert s.violation_count == 1

    def test_block_raises(self):
        c = Contract("t").require(cost_under(0.05), on_fail="block")
        with pytest.raises(ViolationError):
            with c.session() as s:
                s.emit_llm_response(model="m", output="x", cost=0.10)

    def test_escalate_calls_handler_then_raises(self):
        seen = []
        c = (
            Contract("t")
            .require(cost_under(0.05), on_fail="escalate")
            .on_escalate(lambda v: seen.append(v))
        )
        with pytest.raises(EscalationError):
            with c.session() as s:
                s.emit_llm_response(model="m", output="x", cost=0.10)
        assert len(seen) == 1
        assert seen[0].message


class TestEnforceRetry:
    def test_retry_until_success(self):
        calls = {"n": 0}
        c = Contract("t").require(cost_under(0.05), on_fail="retry").with_retries(3)

        @c.enforce
        def agent():
            calls["n"] += 1
            _emit_cost(0.10 if calls["n"] == 1 else 0.01)  # over budget once, then fine
            return f"ok on attempt {calls['n']}"

        assert agent() == "ok on attempt 2"
        assert calls["n"] == 2

    def test_retry_exhausted_raises(self):
        calls = {"n": 0}
        c = Contract("t").require(cost_under(0.05), on_fail="retry").with_retries(2)

        @c.enforce
        def agent():
            calls["n"] += 1
            _emit_cost(0.10)  # always over budget
            return "never"

        with pytest.raises(ViolationError):
            agent()
        assert calls["n"] == 3  # 1 initial attempt + 2 retries


class TestEnforceFallback:
    def test_fallback_returns_fallback_result(self):
        def safe(*args, **kwargs):
            return "fallback result"

        c = Contract("t").require(cost_under(0.05), on_fail="fallback").fallback(safe)

        @c.enforce
        def agent():
            _emit_cost(0.10)
            return "primary"

        assert agent() == "fallback result"

    def test_fallback_without_handler_raises(self):
        c = Contract("t").require(cost_under(0.05), on_fail="fallback")

        @c.enforce
        def agent():
            _emit_cost(0.10)
            return "primary"

        with pytest.raises(ViolationError):
            agent()


class TestEnforceHappyPath:
    def test_compliant_run_returns_result(self):
        c = Contract("t").require(cost_under(1.0), on_fail="block")

        @c.enforce
        def agent():
            _emit_cost(0.01)
            return "done"

        assert agent() == "done"
