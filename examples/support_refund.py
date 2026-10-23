"""Synthetic refund reference workflow: no model keys, real payments, or network.

The reusable implementation lives in pactrun.scenarios.refunds. This module
keeps the original example API and cassette demo available to existing users.
Use examples/refund_scenarios.py to run the full stateful case pack.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import uuid
from dataclasses import asdict as asdict
from pathlib import Path

from pactrun import Contract, ViolationError
from pactrun.integrations.evalcraft import EvalcraftRecorder, replay_cassette
from pactrun.predicates import max_tool_calls
from pactrun.scenarios.refunds import (
    FIELDS as FIELDS,
)
from pactrun.scenarios.refunds import (
    DuplicateActionError as DuplicateActionError,
)
from pactrun.scenarios.refunds import (
    FakePaymentProvider as FakePaymentProvider,
)
from pactrun.scenarios.refunds import (
    Identity as Identity,
)
from pactrun.scenarios.refunds import (
    Refund as Refund,
)
from pactrun.scenarios.refunds import (
    RefundWorkflow as _RefundWorkflow,
)
from pactrun.scenarios.refunds import (
    database as database,
)


class RefundWorkflow(_RefundWorkflow):
    """Compatibility example that records cassettes by default."""

    def __init__(self, directory: Path, *, secret: str, provider=None) -> None:
        super().__init__(
            directory, secret=secret, provider=provider,
            recorder=EvalcraftRecorder(name="support-refund"),
        )


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
