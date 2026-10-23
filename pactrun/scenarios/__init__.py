"""Synthetic stateful environments; Evalcraft is required only for scenario runs.

These fake tools cannot issue real refunds. They exercise host-owned identity,
approval, idempotency and reconciliation boundaries in isolated local state.
"""

from pactrun.scenarios.refund_cases import refund_scenarios
from pactrun.scenarios.refund_environment import RefundScenarioEnvironment, refund_environment
from pactrun.scenarios.refunds import (
    DuplicateActionError,
    FakePaymentProvider,
    Identity,
    ProviderRejectedError,
    Refund,
    RefundWorkflow,
)

__all__ = [
    "DuplicateActionError",
    "FakePaymentProvider",
    "Identity",
    "ProviderRejectedError",
    "Refund",
    "RefundScenarioEnvironment",
    "RefundWorkflow",
    "refund_environment",
    "refund_scenarios",
]
