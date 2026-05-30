"""pytest plugin for pactrun.

Auto-loaded via the ``pytest11`` entry point once pactrun is installed. Provides:

- ``@pytest.mark.contracted(contract)`` — run a test under a pactrun ``Contract``.
  ``block``-mode violations fail the test as they happen; any other recorded
  violation fails the test at the end with a clear message.
- a ``pact_session`` fixture — the active enforcement ``Session`` for the test,
  so you can ``emit_*`` events (or let an adapter do it).
- a one-line terminal summary of contracted tests and violations.

Example::

    import pytest
    from pactrun import Contract, cost_under

    budget = Contract("agent").require(cost_under(0.50), on_fail="log")

    @pytest.mark.contracted(budget)
    def test_my_agent(pact_session):
        run_agent_emitting_into(pact_session)   # uses the active session
"""

from __future__ import annotations

import pytest

from pactrun.core.errors import ViolationError


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "contracted(contract): run this test under a pactrun Contract; "
        "the test fails if the contract is violated.",
    )
    if not hasattr(config, "_pactrun_contracted"):
        config._pactrun_contracted = 0  # type: ignore[attr-defined]
        config._pactrun_violated = 0  # type: ignore[attr-defined]


def _contract_from_marker(marker):
    if marker.args:
        return marker.args[0]
    return marker.kwargs.get("contract")


@pytest.fixture(autouse=True)
def _pactrun_contracted_session(request):
    """Open an enforcement session around any ``@contracted`` test."""
    marker = request.node.get_closest_marker("contracted")
    if marker is None:
        yield None
        return

    contract = _contract_from_marker(marker)
    if contract is None:
        raise pytest.UsageError("@pytest.mark.contracted requires a Contract argument")

    config = request.config
    config._pactrun_contracted = getattr(config, "_pactrun_contracted", 0) + 1

    session = contract.session()
    request.node._pactrun_session = session
    session.__enter__()
    try:
        yield session
    finally:
        # The call hookwrapper usually closes the session already; close here
        # too in case the test errored during setup/collection.
        if session.is_active:
            try:
                session.__exit__(None, None, None)
            except ViolationError:
                pass


@pytest.fixture
def pact_session(_pactrun_contracted_session):
    """The active pactrun Session for a ``@pytest.mark.contracted`` test."""
    if _pactrun_contracted_session is None:
        raise pytest.UsageError(
            "the 'pact_session' fixture requires @pytest.mark.contracted(contract) on the test"
        )
    return _pactrun_contracted_session


def _body_raised(outcome) -> bool:
    # pluggy >= 1.0 exposes Result.exception; older _Result used .excinfo.
    if hasattr(outcome, "exception"):
        return outcome.exception is not None
    return getattr(outcome, "excinfo", None) is not None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    outcome = yield
    session = getattr(item, "_pactrun_session", None)
    if session is None:
        return

    # Finalize now so compliance reflects session-end clauses (e.g. must_call)
    # before the autouse fixture tears the session down.
    if session.is_active:
        try:
            session.__exit__(None, None, None)
        except ViolationError:
            pass

    if session.is_compliant:
        return

    item.config._pactrun_violated = getattr(item.config, "_pactrun_violated", 0) + 1

    # If the test body already raised (a real error or a block-mode
    # ViolationError), keep that — don't mask it.
    if _body_raised(outcome):
        return

    marker = item.get_closest_marker("contracted")
    name = _contract_from_marker(marker).name if marker else "contract"
    messages = "; ".join(v.message for v in session.violations) or "contract violated"
    outcome.force_exception(
        AssertionError(f"pactrun: contract '{name}' violated: {messages}")
    )


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    total = getattr(config, "_pactrun_contracted", 0)
    if not total:
        return
    violated = getattr(config, "_pactrun_violated", 0)
    terminalreporter.write_sep(
        "-", f"pactrun: {total} contracted test(s), {violated} with violations"
    )
