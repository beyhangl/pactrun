"""Isolated refund environment for Evalcraft's optional scenario runner.

Scenario operations describe host test orchestration, not an agent's tool schema.
In particular, identity, approval issuance and fault injection are host-owned.
An agent proposes only the fields of :class:`Refund`. Each environment owns a
new temporary directory; restart reopens the same durable files within that case.
"""

from __future__ import annotations

import copy
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from pactrun.core.errors import ViolationError
from pactrun.scenarios.refunds import (
    DuplicateActionError,
    FakePaymentProvider,
    Identity,
    ProviderRejectedError,
    Refund,
    RefundWorkflow,
    database,
)

if TYPE_CHECKING:
    from evalcraft.scenarios import ScenarioAction, StepOutcome

_FAULTS = {
    None, "timeout_before_commit", "timeout_after_commit", "missing_evidence", "mismatched_evidence",
    "malformed_evidence",
}
_OUTCOMES: dict[str, Literal["succeeded", "blocked", "unknown", "error"]] = {
    "verified": "succeeded", "unknown": "unknown", "mismatch": "error",
}


class _FaultProvider(FakePaymentProvider):
    """Fault settings belong to the test host, never proposed refund arguments."""

    fault: str | None = None

    def refund(self, proposal: Refund, identity: Identity, *, timeout_after_commit=False) -> dict:
        if self.fault == "timeout_before_commit":
            raise TimeoutError("Synthetic timeout before provider commit")
        return super().refund(
            proposal, identity,
            timeout_after_commit=timeout_after_commit or self.fault == "timeout_after_commit",
        )

    def lookup(self, action_id: str) -> dict | None:
        if self.fault == "missing_evidence":
            return None
        receipt = super().lookup(action_id)
        if self.fault == "mismatched_evidence" and receipt is not None:
            receipt["amount_minor"] += 1
        if self.fault == "malformed_evidence" and receipt is not None:
            receipt["amount_minor"] = True
        return receipt


class RefundScenarioEnvironment:
    """Execute deterministic host actions against the real refund boundary.

    Initial state has ``identity`` and an ``orders`` list. Order fields are
    ``order_id``, ``tenant``, ``customer``, ``currency``, ``remaining`` and optional
    ``version``. Snapshot values come from durable provider/action/nonce ledgers;
    approval tokens, signatures and the signing secret are deliberately omitted.

    Actions: ``approve`` (proposal, optional alias), ``refund`` (proposal,
    approval alias), ``reconcile`` (action_id), ``restart``, ``concurrent_refund``
    (proposal, approval alias, attempts), ``host_update_order`` (order_id and
    changed fields), and ``tamper_approval`` (alias, field, value). Timestamp
    tampering can instead set ``use_current_time=True`` to isolate signature
    validation from expiry. The last two are explicit host test operations,
    not tools available to a model.
    """

    def __init__(self, initial_state: dict[str, Any]) -> None:
        self._temporary = tempfile.TemporaryDirectory(prefix="pactrun-scenario-")
        self.directory = Path(self._temporary.name)
        self._closed = False
        self._secret = "synthetic-refund-scenario-key"
        self._approvals: dict[str, dict] = {}
        try:
            self.identity = Identity(**copy.deepcopy(initial_state["identity"]))
            self.workflow = self._open_workflow()
            for order in copy.deepcopy(initial_state.get("orders", [])):
                self.workflow.provider.seed_order(**order)
        except BaseException:
            self.close()
            raise

    def _open_workflow(self) -> RefundWorkflow:
        recorder = None
        try:
            from evalcraft.capture.recorder import get_active_context
        except ImportError:
            pass
        else:
            context = get_active_context()
            if context is not None:
                from pactrun.integrations.evalcraft import EvalcraftRecorder

                recorder = EvalcraftRecorder(context.cassette)
        return RefundWorkflow(
            self.directory, secret=self._secret,
            provider=_FaultProvider(self.directory / "provider.sqlite"),
            recorder=recorder,
        )

    def _refund(self, args: dict[str, Any], *, workflow: RefundWorkflow | None = None) -> StepOutcome:
        from evalcraft.scenarios import StepOutcome

        flow = workflow or self.workflow
        proposal = Refund(**args["proposal"])
        token = copy.deepcopy(self._approvals.get(args.get("approval", "grant"), {}))
        try:
            result = flow.execute(proposal, self.identity, token)
        except (ViolationError, DuplicateActionError, ProviderRejectedError) as exc:
            return StepOutcome("blocked", error=str(exc))
        return StepOutcome(_OUTCOMES[result["status"]], result=result)

    def execute(self, step: ScenarioAction) -> StepOutcome:
        """Apply one isolated host action, restoring temporary faults afterwards."""
        from evalcraft.scenarios import StepOutcome

        if self._closed:
            raise RuntimeError("Refund scenario environment is closed")
        if step.fault not in _FAULTS:
            raise ValueError(f"Unknown synthetic provider fault: {step.fault!r}")
        if step.fault is not None and step.action not in {"refund", "reconcile"}:
            raise ValueError("Provider faults apply only to refund or reconcile actions")
        provider = cast(_FaultProvider, self.workflow.provider)
        provider.fault = step.fault
        try:
            args = step.args
            if step.action == "approve":
                # A new nonce is signed at the current time. It is never part of
                # the snapshot/oracle, so repeat runs compare stable state only.
                proposal = Refund(**args["proposal"])
                alias = args.get("alias", "grant")
                if alias in self._approvals:
                    raise ValueError(f"Approval alias already exists: {alias!r}")
                options = {"issued_at": args["issued_at"]} if "issued_at" in args else {}
                self._approvals[alias] = self.workflow.approve(proposal, self.identity, **options)
                return StepOutcome("succeeded", result={"approval": alias})
            if step.action == "refund":
                return self._refund(args)
            if step.action == "reconcile":
                result = self.workflow.reconcile(args["action_id"])
                return StepOutcome(_OUTCOMES[result["status"]], result=result)
            if step.action == "restart":
                self.workflow = self._open_workflow()
                return StepOutcome("succeeded", result={"restarted": True})
            if step.action == "host_update_order":
                allowed = {"tenant", "customer", "currency", "remaining", "version"}
                changes = {key: value for key, value in args.items() if key != "order_id"}
                if not changes or not set(changes).issubset(allowed):
                    raise ValueError("host_update_order requires known changed order fields")
                # Columns are selected exclusively from the fixed host allowlist.
                assignments = ", ".join(f"{column}=?" for column in changes)
                with self.workflow.provider.connect() as db:
                    changed = db.execute(
                        f"UPDATE orders SET {assignments} WHERE id=?",  # noqa: S608
                        (*changes.values(), args["order_id"]),
                    ).rowcount
                if changed != 1:
                    raise KeyError(args["order_id"])
                return StepOutcome("succeeded", result={"updated": args["order_id"]})
            if step.action == "tamper_approval":
                token = self._approvals[args.get("alias", "grant")]
                if args["field"] not in {"issued_at", "nonce", "sig", "context"}:
                    raise ValueError("Unknown approval field for tampering")
                if args["field"] == "issued_at" and args.get("use_current_time") is True:
                    token["issued_at"] = time.time()
                else:
                    token[args["field"]] = copy.deepcopy(args["value"])
                return StepOutcome("succeeded", result={"tampered": args["field"]})
            if step.action == "concurrent_refund":
                requests = args.get("requests")
                if requests is None:
                    attempts = args.get("attempts", 2)
                    if type(attempts) is not int or not 2 <= attempts <= 8:
                        raise ValueError("concurrent_refund attempts must be between 2 and 8")
                    requests = [copy.deepcopy(args) for _ in range(attempts)]
                elif not isinstance(requests, list) or not 2 <= len(requests) <= 8:
                    raise ValueError("concurrent_refund requests must contain between 2 and 8 entries")
                attempts = len(requests)
                flows = [self._open_workflow() for _ in range(attempts)]
                barrier = threading.Barrier(attempts)

                def attempt(pair: tuple[RefundWorkflow, dict]) -> str:
                    flow, request = pair
                    barrier.wait(timeout=10)
                    return self._refund(request, workflow=flow).status

                with ThreadPoolExecutor(max_workers=attempts) as pool:
                    statuses = sorted(pool.map(attempt, zip(flows, requests, strict=True)))
                return StepOutcome("succeeded", result={"statuses": statuses})
            raise ValueError(f"Unknown refund scenario action: {step.action!r}")
        finally:
            provider.fault = None

    def snapshot(self) -> dict[str, Any]:
        """Read the independent durable ledgers without trusting step responses."""
        if self._closed:
            raise RuntimeError("Refund scenario environment is closed")
        with self.workflow.provider.connect() as db:
            rows = db.execute(
                "SELECT id, tenant, customer, currency, remaining, version FROM orders ORDER BY id",
            ).fetchall()
            provider_refunds = db.execute("SELECT COUNT(*) FROM refunds").fetchone()[0]
        with database(self.workflow.actions) as db:
            actions = [
                {"action_id": row[0], "status": row[1]}
                for row in db.execute("SELECT id, status FROM actions ORDER BY id").fetchall()
            ]
        with database(self.directory / "approvals.sqlite") as db:
            consumed = db.execute("SELECT COUNT(*) FROM pactrun_used_approvals").fetchone()[0]
        fields = ("order_id", "tenant", "customer", "currency", "remaining", "version")
        return {
            "identity": asdict(self.identity),
            "orders": [dict(zip(fields, row, strict=True)) for row in rows],
            "actions": actions,
            "provider_refunds": provider_refunds,
            "consumed_approvals": consumed,
        }

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._temporary.cleanup()


def refund_environment(initial_state: dict[str, Any]) -> RefundScenarioEnvironment:
    """Factory for ``evalcraft.scenarios.run_scenario(s)``; each call is isolated."""
    return RefundScenarioEnvironment(initial_state)
