"""Synthetic refund reference workflow: no model keys, real payments, or network.

The host owns identity, signing keys and SQLite files. The provider simulator
implements idempotency and state checks; Pactrun checks/consumes approval before
its callback. A consumed grant is not proof of a successful external effect.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

from pactrun import (
    Contract,
    EventKind,
    PredicateResult,
    ViolationError,
    consent_token_required,
    mint_consent_token,
)
from pactrun.approval_store import SQLiteApprovalStore
from pactrun.integrations.evalcraft import EvalcraftRecorder, replay_cassette
from pactrun.predicates import max_tool_calls

FIELDS = ["action_id", "order_id", "amount_minor", "currency", "expected_version"]


@contextmanager
def database(path):
    connection = sqlite3.connect(path, timeout=5)
    try:
        with connection:
            yield connection
    finally:
        connection.close()


@dataclass(frozen=True)
class Identity:
    tenant_id: str
    customer_id: str
    principal_id: str


@dataclass(frozen=True)
class Refund:
    action_id: str
    order_id: str
    amount_minor: int
    currency: str
    expected_version: int = 1


class DuplicateActionError(ValueError):
    """The action was already admitted; reconcile instead of executing again."""


class FakePaymentProvider:
    """SQLite provider simulator with atomic state checks and idempotent refunds."""

    def __init__(self, path: Path) -> None:
        self.path = path
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS orders "
                       "(id TEXT PRIMARY KEY, tenant TEXT, customer TEXT, currency TEXT, "
                       "remaining INTEGER, version INTEGER)")
            db.execute("CREATE TABLE IF NOT EXISTS refunds "
                       "(id TEXT PRIMARY KEY, proposal TEXT, receipt TEXT)")

    def connect(self):
        return database(self.path)

    def seed_order(self, order_id="order-1", *, tenant="tenant-a", customer="customer-a",
                   currency="USD", remaining=10000) -> None:
        with self.connect() as db:
            db.execute("INSERT OR IGNORE INTO orders VALUES (?, ?, ?, ?, ?, 1)",
                       (order_id, tenant, customer, currency, remaining))

    def order(self, order_id: str) -> dict:
        with self.connect() as db:
            row = db.execute("SELECT tenant, customer, currency, remaining, version "
                             "FROM orders WHERE id=?", (order_id,)).fetchone()
        if row is None:
            raise KeyError(order_id)
        return dict(zip(("tenant", "customer", "currency", "remaining", "version"), row, strict=True))

    def refund(self, proposal: Refund, identity: Identity, *, timeout_after_commit=False) -> dict:
        if type(proposal.amount_minor) is not int or proposal.amount_minor <= 0:
            raise ValueError("Refund amount must be a positive integer in minor units")
        payload = json.dumps({"refund": asdict(proposal), "identity": asdict(identity)}, sort_keys=True)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute("SELECT proposal, receipt FROM refunds WHERE id=?",
                                  (proposal.action_id,)).fetchone()
            if existing:
                if existing[0] != payload:
                    raise ValueError("Provider idempotency key was reused for a different action")
                return json.loads(existing[1])
            row = db.execute("SELECT tenant, customer, currency, remaining, version "
                             "FROM orders WHERE id=?", (proposal.order_id,)).fetchone()
            if (row is None or row[:3] != (identity.tenant_id, identity.customer_id, proposal.currency)
                    or row[4] != proposal.expected_version or proposal.amount_minor > row[3]):
                raise ValueError("Authoritative state changed before provider commit")
            receipt = {"refund_id": proposal.action_id, "order_id": proposal.order_id,
                       "tenant_id": identity.tenant_id, "customer_id": identity.customer_id,
                       "amount_minor": proposal.amount_minor, "currency": proposal.currency,
                       "status": "confirmed"}
            db.execute("UPDATE orders SET remaining=remaining-?, version=version+1 WHERE id=?",
                       (proposal.amount_minor, proposal.order_id))
            db.execute("INSERT INTO refunds VALUES (?, ?, ?)",
                       (proposal.action_id, payload, json.dumps(receipt)))
        if timeout_after_commit:
            raise TimeoutError("Synthetic timeout after provider accepted the refund")
        return receipt

    def lookup(self, action_id: str) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT receipt FROM refunds WHERE id=?", (action_id,)).fetchone()
        return json.loads(row[0]) if row else None

    @property
    def committed_refunds(self) -> int:
        with self.connect() as db:
            return db.execute("SELECT COUNT(*) FROM refunds").fetchone()[0]


class RefundWorkflow:
    """Application-owned example boundary and durable action reconciliation."""

    def __init__(self, directory: Path, *, secret: str, provider=None) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self.secret = secret
        self.provider = provider or FakePaymentProvider(directory / "provider.sqlite")
        self.approvals = SQLiteApprovalStore(directory / "approvals.sqlite")
        self.actions = directory / "actions.sqlite"
        with database(self.actions) as db:
            db.execute("CREATE TABLE IF NOT EXISTS actions "
                       "(id TEXT PRIMARY KEY, proposal TEXT, identity TEXT, status TEXT, receipt TEXT)")
        self.recorder = EvalcraftRecorder(name="support-refund")

    def context(self, proposal: Refund, identity: Identity) -> dict:
        order = self.provider.order(proposal.order_id)
        return {**asdict(identity), "policy_version": "refund-policy-1",
                "order_version": order["version"]}

    def approve(self, proposal: Refund, identity: Identity, **token_options) -> dict:
        """Synthetic host issuance; production identity/approval UX is outside this demo."""
        return mint_consent_token(
            "refund", args=asdict(proposal), bind_args=FIELDS, secret=self.secret,
            context=self.context(proposal, identity), **token_options,
        )

    def contract(self, proposal: Refund, identity: Identity) -> Contract:
        def business_rules(event, state):
            if event.kind != EventKind.TOOL_CALL or event.tool_name != "refund":
                return PredicateResult(passed=True)
            try:
                order = self.provider.order(proposal.order_id)
            except KeyError:
                return PredicateResult(passed=False, message="Order was not found")
            valid = (
                isinstance(proposal.action_id, str) and bool(proposal.action_id)
                and type(proposal.amount_minor) is int and proposal.amount_minor > 0
                and type(proposal.expected_version) is int
                and order["tenant"] == identity.tenant_id
                and order["customer"] == identity.customer_id
                and order["currency"] == proposal.currency
                and proposal.amount_minor <= order["remaining"]
                and proposal.expected_version == order["version"]
                and event.tool_args == asdict(proposal)
            )
            return PredicateResult(passed=valid, message="Refund business state or identity does not match")

        business_rules.predicate_name = "refund_business_rules"
        return (
            Contract("support-refund")
            .require(business_rules)
            .require(consent_token_required(
                "refund", bind_args=FIELDS, secret=self.secret,
                context=lambda event, state: self.context(proposal, identity),
                replay_store=self.approvals,
            ))
        )

    def _reserve(self, proposal: Refund, identity: Identity) -> None:
        try:
            with database(self.actions) as db:
                db.execute("INSERT INTO actions VALUES (?, ?, ?, 'reserved', NULL)",
                           (proposal.action_id, json.dumps(asdict(proposal)), json.dumps(asdict(identity))))
        except sqlite3.IntegrityError as exc:
            raise DuplicateActionError("Action already admitted; call reconcile") from exc

    def _status(self, action_id: str, status: str, receipt=None) -> dict:
        with database(self.actions) as db:
            db.execute("UPDATE actions SET status=?, receipt=? WHERE id=?",
                       (status, json.dumps(receipt) if receipt else None, action_id))
        return {"action_id": action_id, "status": status, "receipt": receipt}

    def execute(self, proposal: Refund, identity: Identity, token: dict, *,
                timeout_after_commit=False) -> dict:
        with self.contract(proposal, identity).session(observers=[self.recorder]) as session:
            session.emit_tool_call("refund", args=asdict(proposal), metadata={"user_consent": token})
            self._reserve(proposal, identity)
            try:
                self.provider.refund(proposal, identity, timeout_after_commit=timeout_after_commit)
            except TimeoutError:
                outcome = self._status(proposal.action_id, "unknown")
            except ValueError:
                self._status(proposal.action_id, "rejected")
                raise
            else:
                outcome = self.reconcile(proposal.action_id)
            session.emit_output(json.dumps(outcome))
        return outcome

    def reconcile(self, action_id: str) -> dict:
        """Observe provider ledger only: never issue another refund automatically."""
        with database(self.actions) as db:
            row = db.execute("SELECT proposal, identity FROM actions WHERE id=?", (action_id,)).fetchone()
        if row is None:
            raise KeyError(action_id)
        proposal, identity = json.loads(row[0]), json.loads(row[1])
        receipt = self.provider.lookup(action_id)
        if receipt is None:
            return self._status(action_id, "unknown")
        matches = (
            receipt.get("refund_id") == action_id and receipt.get("status") == "confirmed"
            and receipt.get("order_id") == proposal["order_id"]
            and receipt.get("amount_minor") == proposal["amount_minor"]
            and receipt.get("currency") == proposal["currency"]
            and receipt.get("tenant_id") == identity["tenant_id"]
            and receipt.get("customer_id") == identity["customer_id"]
        )
        return self._status(action_id, "verified" if matches else "mismatch", receipt)


def run_demo(directory: Path) -> dict:
    workflow = RefundWorkflow(directory, secret="synthetic-demo-key")
    workflow.provider.seed_order()
    identity = Identity("tenant-a", "customer-a", "beyhangl")
    proposal = Refund("refund-1", "order-1", 2500, "USD")
    grant = workflow.approve(proposal, identity)
    unknown = workflow.execute(proposal, identity, grant, timeout_after_commit=True)
    restarted = RefundWorkflow(directory, secret="synthetic-demo-key")
    verified = restarted.reconcile(proposal.action_id)
    blocked = False
    try:
        restarted.execute(proposal, identity, grant)
    except ViolationError:
        blocked = True
    cassette = workflow.recorder.cassette.save(directory / "refund.json")
    # Policy replay evaluates saved observations; changed-code playback is tested separately.
    replay = replay_cassette(Contract("recorded-tools").require(max_tool_calls(1)), cassette)
    report = {"timeout_outcome": unknown["status"], "after_restart": verified["status"],
              "repeat_blocked": blocked, "provider_refunds": restarted.provider.committed_refunds,
              "recorded_policy_compliant": replay.compliant, "cassette": str(cassette)}
    (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="Keep synthetic artifacts in a new subdirectory")
    args = parser.parse_args()
    if args.output_dir:
        print(json.dumps(run_demo(args.output_dir / uuid.uuid4().hex), indent=2))
    else:
        with tempfile.TemporaryDirectory(prefix="pactrun-refund-") as directory:
            print(json.dumps(run_demo(Path(directory)), indent=2))


if __name__ == "__main__":
    main()
