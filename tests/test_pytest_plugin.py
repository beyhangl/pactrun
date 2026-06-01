"""Tests for the pactrun pytest plugin."""

import pytest

from pactrun import Contract, cost_under


# --- dogfood: a real @contracted test in this very suite -------------------
_budget = Contract("dogfood").require(cost_under(1.0), on_fail="log")


@pytest.mark.contracted(_budget)
def test_contracted_marker_dogfood(pact_session):
    pact_session.emit_llm_response(model="m", output="x", cost=0.02)
    assert pact_session.is_compliant


# --- isolated runs via pytester --------------------------------------------
def test_compliant_contracted_test_passes(pytester):
    pytester.makepyfile(
        """
        import pytest
        from pactrun import Contract, cost_under

        c = Contract("ok").require(cost_under(0.05), on_fail="log")

        @pytest.mark.contracted(c)
        def test_under_budget(pact_session):
            pact_session.emit_llm_response(model="m", output="x", cost=0.01)
        """
    )
    result = pytester.runpytest()
    result.assert_outcomes(passed=1)


def test_violating_contracted_test_fails(pytester):
    pytester.makepyfile(
        """
        import pytest
        from pactrun import Contract, cost_under, get_active_session

        c = Contract("toopricey").require(cost_under(0.05), on_fail="log")

        @pytest.mark.contracted(c)
        def test_over_budget():
            get_active_session().emit_llm_response(model="m", output="x", cost=0.10)
        """
    )
    result = pytester.runpytest()
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*contract 'toopricey' violated*"])


def test_block_mode_contracted_test_fails(pytester):
    pytester.makepyfile(
        """
        import pytest
        from pactrun import Contract, cost_under, get_active_session

        c = Contract("hardstop").require(cost_under(0.05), on_fail="block")

        @pytest.mark.contracted(c)
        def test_blocks():
            get_active_session().emit_llm_response(model="m", output="x", cost=0.10)
        """
    )
    result = pytester.runpytest()
    result.assert_outcomes(failed=1)


def test_pact_session_without_marker_errors(pytester):
    pytester.makepyfile(
        """
        def test_no_marker(pact_session):
            pass
        """
    )
    result = pytester.runpytest()
    result.assert_outcomes(errors=1)
