"""Reviewed synthetic refund scenarios with explicit, independent expectations.

The case pack uses fresh state per case and fixed amount boundaries. It never
learns expected decisions from the implementation being tested. Evalcraft is
imported only when constructing the optional scenario pack.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from evalcraft.scenarios import Scenario


def refund_scenarios() -> list[Scenario]:
    """Return fresh typed cases for ``run_scenarios(cases, refund_environment)``.

    Passing this finite synthetic pack does not establish payment safety. Add
    customer-owned rules and real-provider test-environment cases separately.
    """
    from evalcraft.scenarios import Scenario, ScenarioStep

    initial: dict[str, Any] = {
        "identity": {"tenant_id": "tenant-a", "customer_id": "customer-a", "principal_id": "beyhangl"},
        "orders": [{"order_id": "order-1", "tenant": "tenant-a", "customer": "customer-a",
                    "currency": "USD", "remaining": 10000, "version": 1}],
    }
    proposal = {"action_id": "refund-1", "order_id": "order-1", "amount_minor": 2500,
                "currency": "USD", "expected_version": 1}

    def state(remaining: int = 10000, version: int = 1, *, refunds: int = 0,
              consumed: int = 0, actions: list[dict] | None = None) -> dict:
        return {
            "orders": [{**initial["orders"][0], "remaining": remaining, "version": version}],
            "provider_refunds": refunds, "consumed_approvals": consumed,
            "actions": actions or [],
        }

    def approve(value: dict | None = None, *, alias: str = "grant", issued_at=None) -> ScenarioStep:
        args = {"proposal": copy.deepcopy(proposal if value is None else value), "alias": alias}
        if issued_at is not None:
            args["issued_at"] = issued_at
        return ScenarioStep("approve", args, "succeeded")

    def refund(value: dict | None = None, *,
               status: Literal["succeeded", "blocked", "unknown", "error"] = "succeeded", approval: str = "grant",
               fault: str | None = None, expected: dict | None = None) -> ScenarioStep:
        return ScenarioStep(
            "refund", {"proposal": copy.deepcopy(proposal if value is None else value), "approval": approval},
            status, expected_result=expected, fault=fault,
        )

    verified = [{"action_id": "refund-1", "status": "verified"}]
    unknown = [{"action_id": "refund-1", "status": "unknown"}]
    success = state(7500, 2, refunds=1, consumed=1, actions=verified)
    cases = [
        Scenario("refund-approved-partial", copy.deepcopy(initial), [
            approve(), refund(expected={"status": "verified", "receipt": {"amount_minor": 2500}}),
        ], copy.deepcopy(success)),
    ]
    second = {**proposal, "action_id": "refund-2", "amount_minor": 7500, "expected_version": 2}
    cases.append(Scenario("refund-partial-then-remainder", copy.deepcopy(initial), [
        approve(), refund(), approve(second, alias="second"), refund(second, approval="second"),
    ], state(0, 3, refunds=2, consumed=2, actions=[
        *verified, {"action_id": "refund-2", "status": "verified"},
    ])))

    # Fixed boundaries form separate isolated cases; bool and float are not
    # accepted as integer minor units even when Python compares their values.
    for label, amount, allowed in [
        ("negative", -1, False), ("zero", 0, False), ("minimum", 1, True),
        ("full", 10000, True), ("excess", 10001, False),
        ("boolean", True, False), ("fractional", 25.5, False),
    ]:
        value = {**proposal, "amount_minor": amount}
        final = (
            state(10000 - int(amount), 2, refunds=1, consumed=1, actions=verified)
            if allowed else state()
        )
        cases.append(Scenario(f"refund-amount-{label}", copy.deepcopy(initial), [
            approve(value), refund(value, status="succeeded" if allowed else "blocked"),
        ], final))

    # The business-rule clause runs before consent: these early rejections
    # never reach durable grant consumption.
    for field, foreign in [("tenant_id", "tenant-b"), ("customer_id", "customer-b")]:
        foreign_state = copy.deepcopy(initial)
        foreign_state["identity"][field] = foreign
        cases.append(Scenario(f"refund-foreign-{field}", foreign_state, [
            approve(), refund(status="blocked"),
        ], state()))
    currency = {**proposal, "currency": "EUR"}
    cases.append(Scenario("refund-wrong-currency", copy.deepcopy(initial), [
        approve(currency), refund(currency, status="blocked"),
    ], state()))
    cases.append(Scenario("refund-changed-approved-amount", copy.deepcopy(initial), [
        approve(), refund({**proposal, "amount_minor": 2501}, status="blocked"),
    ], state()))
    cases.append(Scenario("refund-missing-approval", copy.deepcopy(initial), [
        refund(status="blocked"),
    ], state()))
    cases.append(Scenario("refund-expired-approval", copy.deepcopy(initial), [
        approve(issued_at=1), refund(status="blocked"),
    ], state()))
    cases.append(Scenario("refund-tampered-timestamp", copy.deepcopy(initial), [
        approve(issued_at=1), ScenarioStep("tamper_approval", {
            "field": "issued_at", "use_current_time": True,
        }, "succeeded"),
        refund(status="blocked"),
    ], state()))
    cases.append(Scenario("refund-stale-order-version", copy.deepcopy(initial), [
        approve(), ScenarioStep("host_update_order", {"order_id": "order-1", "version": 2}, "succeeded"),
        refund(status="blocked"),
    ], state(version=2)))
    cases.append(Scenario("refund-balance-changed-after-approval", copy.deepcopy(initial), [
        approve(), ScenarioStep("host_update_order", {"order_id": "order-1", "remaining": 2000}, "succeeded"),
        refund(status="blocked"),
    ], state(remaining=2000)))
    cases.append(Scenario("refund-restart-replay-blocked", copy.deepcopy(initial), [
        approve(), refund(), ScenarioStep("restart", {}, "succeeded"), refund(status="blocked"),
    ], copy.deepcopy(success)))
    cases.append(Scenario("refund-concurrent-same-grant", copy.deepcopy(initial), [
        approve(), ScenarioStep("concurrent_refund", {"proposal": copy.deepcopy(proposal), "attempts": 2},
                                "succeeded", expected_result={"statuses": ["blocked", "succeeded"]}),
    ], copy.deepcopy(success)))
    first_large = {**proposal, "amount_minor": 7500}
    second_large = {**first_large, "action_id": "refund-2"}
    overspend_state = state(2500, 2, refunds=1)
    # Either action can win; assert the deterministic business invariant rather
    # than guessing thread order or whether the losing grant reached admission.
    del overspend_state["actions"], overspend_state["consumed_approvals"]
    cases.append(Scenario("refund-concurrent-distinct-grants", copy.deepcopy(initial), [
        approve(first_large), approve(second_large, alias="second"),
        ScenarioStep("concurrent_refund", {"requests": [
            {"proposal": first_large, "approval": "grant"},
            {"proposal": second_large, "approval": "second"},
        ]}, "succeeded", expected_result={"statuses": ["blocked", "succeeded"]}),
    ], overspend_state))
    cases.append(Scenario("refund-timeout-after-commit-reconciles", copy.deepcopy(initial), [
        approve(), refund(status="unknown", fault="timeout_after_commit"),
        ScenarioStep("restart", {}, "succeeded"),
        ScenarioStep("reconcile", {"action_id": "refund-1"}, "succeeded", expected_result={"status": "verified"}),
        refund(status="blocked"),
    ], copy.deepcopy(success)))
    cases.append(Scenario("refund-timeout-before-commit-never-resends", copy.deepcopy(initial), [
        approve(), refund(status="unknown", fault="timeout_before_commit"),
        ScenarioStep("restart", {}, "succeeded"),
        ScenarioStep("reconcile", {"action_id": "refund-1"}, "unknown"), refund(status="blocked"),
    ], state(consumed=1, actions=unknown)))
    cases.append(Scenario("refund-missing-provider-evidence", copy.deepcopy(initial), [
        approve(), refund(status="unknown", fault="missing_evidence"),
        ScenarioStep("reconcile", {"action_id": "refund-1"}, "succeeded", expected_result={"status": "verified"}),
    ], copy.deepcopy(success)))
    cases.append(Scenario("refund-mismatched-provider-evidence", copy.deepcopy(initial), [
        approve(), refund(status="error", fault="mismatched_evidence", expected={"status": "mismatch"}),
        ScenarioStep("reconcile", {"action_id": "refund-1"}, "succeeded", expected_result={"status": "verified"}),
    ], copy.deepcopy(success)))
    minimum = {**proposal, "amount_minor": 1}
    cases.append(Scenario("refund-malformed-provider-evidence", copy.deepcopy(initial), [
        approve(minimum), refund(minimum, status="error", fault="malformed_evidence",
                                 expected={"status": "mismatch"}),
    ], state(9999, 2, refunds=1, consumed=1, actions=[{"action_id": "refund-1", "status": "mismatch"}])))
    return cases
