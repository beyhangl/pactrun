# Reusable refund scenarios

Run a reviewed pack of synthetic refund cases against the actual local refund
workflow. Each case starts with isolated state, executes real application code,
and checks independently authored decisions and business-state expectations.
Cassettes remain part of the evidence; they do not replace state assertions.

Maintainer: [beyhangl](https://github.com/beyhangl) — beyhangl@gmail.com.

## Install matching source checkouts

The scenario runner and refund pack are new source-checkout features. Install
both matching checkouts until releases include them; an older published
Evalcraft package may have the bridge but not `evalcraft.scenarios`.

```bash
python -m pip install -e /path/to/evalcraft -e /path/to/agentpact
```

From the Pactrun checkout:

```bash
python examples/refund_scenarios.py
python examples/refund_scenarios.py --list
python examples/refund_scenarios.py --case refund-approved-partial
python examples/refund_scenarios.py --case refund-concurrent-same-grant --case refund-concurrent-distinct-grants
python examples/refund_scenarios.py --output-dir /tmp/new-refund-scenario-run
```

`--case` selects exact names and can be repeated. `--output-dir` must name a
directory that does not yet exist. Without it, the command runs in memory and
does not print artifact paths for deleted temporary state. This example needs
no model SDK, API key, network connection, or real payment account.

The Python API is equally small:

```python
from evalcraft import run_scenarios
from pactrun.scenarios import refund_environment, refund_scenarios

suite = run_scenarios(refund_scenarios(), refund_environment)
assert suite.passed, suite.failures
```

Evalcraft is optional for Pactrun generally. It is required when constructing or
executing this scenario pack. The refund workflow and provider simulator remain
usable without installing the scenario runner.

## What executes and what is asserted

`RefundScenarioEnvironment` creates a fresh temporary directory with SQLite
provider, approval, and action ledgers for each case. `restart` reopens the same
case's ledgers. The next case receives new state. Cleanup removes those temporary
stores after the runner has observed final state.

The factory runs inside Evalcraft's `CaptureContext`. The workflow attaches its
`EvalcraftRecorder` to the active cassette, preserving Pactrun observations
alongside informational scenario attempts. No saved cassette is replayed in
place of executing the environment.

The runner keeps expectations separate from runtime inputs. Case authors write
`ScenarioStep` expectations; the environment receives only a `ScenarioAction`
with `action`, detached `args`, and optional host-controlled `fault`. Its
`snapshot()` reads the actual durable ledgers instead of trusting a returned
success string. Assertions compare those observations with the independently
authored expected state.

`succeeded`, `blocked`, `unknown`, and `error` describe the observed step outcome.
An expected block can pass a test; an unexpected exception always fails it.
`unknown` preserves uncertainty, including a provider timeout. Reconciliation
must inspect matching provider evidence before calling that effect verified.
For `concurrent_refund`, the outer `succeeded` means the batch completed; the
case separately checks the individual request statuses and resulting balance.

## Included cases

The initial pack contains 26 named cases. Use `--list` to see their exact names.

| Case group | Independent business assertion |
|---|---|
| Approved partial refund, then remaining balance | Correct amount, remaining balance, order version, receipt, and effect count |
| Negative, zero, minimum, full, excessive, boolean, and fractional amounts | Only positive integer minor units within the balance execute |
| Foreign tenant/customer or wrong currency | No provider refund and no balance change |
| Changed amount, missing/expired approval, refreshed timestamp without a new signature | Proposal or approval changes do not authorize execution |
| Changed order version or balance after approval | Current authoritative state still controls admission |
| Restart and reuse of a grant | A previously used approval does not admit the action again |
| Concurrent calls using one grant | One effect occurs; the competing request is blocked |
| Concurrent distinct grants exceeding the shared order balance | One effect occurs and the order cannot be over-refunded, regardless of the winning thread |
| Timeout before or after provider commit | Uncertainty is retained; reconciliation never blindly resends a refund |
| Missing, mismatched, or malformed provider evidence | Missing evidence remains uncertain; incorrect evidence is not accepted as verified |

Fault injection is explicit and host-owned. Supported refund/reconcile faults
are `timeout_before_commit`, `timeout_after_commit`, `missing_evidence`,
`mismatched_evidence`, and `malformed_evidence`. A fault is removed after its
step. The malformed-evidence case includes a boolean amount that must not match
the integer amount `1` merely because Python considers `True == 1`.

Concurrency expectations target stable business invariants. The distinct-grant
case does not guess which thread wins or whether the losing grant reached
admission. It checks the statuses, one provider effect, and remaining balance.

## Add a case or another environment

Cases use the standard Evalcraft models:

```python
from copy import deepcopy
from evalcraft import Scenario, ScenarioStep, run_scenario
from pactrun.scenarios import refund_environment, refund_scenarios

template = refund_scenarios()[0]
case = Scenario(
    name="customer-rule-no-approval",
    initial_state=deepcopy(template.initial_state),
    steps=[ScenarioStep(
        "refund",
        {"proposal": {
            "action_id": "customer-case-1", "order_id": "order-1",
            "amount_minor": 100, "currency": "USD", "expected_version": 1,
        }},
        "blocked",
        expected_state={"provider_refunds": 0, "consumed_approvals": 0},
    )],
    expected_state={
        "orders": [{
            "order_id": "order-1", "tenant": "tenant-a", "customer": "customer-a",
            "currency": "USD", "remaining": 10000, "version": 1,
        }],
        "provider_refunds": 0, "actions": [], "consumed_approvals": 0,
    },
)
report = run_scenario(case, refund_environment)
assert report.passed, report.failures
```

Expected mapping values are recursive subsets; lists match exactly. Write the
expected values from the customer rule. Do not run the implementation first to
generate its own expected answer. Add a step-level `expected_state` when an
intermediate invariant matters, and always supply the explicit final state.

The environment accepts `approve`, `refund`, `reconcile`, `restart`,
`concurrent_refund`, `host_update_order`, and `tamper_approval`. Identity,
approval issuance, host updates, tampering, and faults are test orchestration;
they are not tools available to a model. To support another domain, implement
`execute(ScenarioAction)`, `snapshot()`, and `close()` and reuse Evalcraft's
[`run_scenario` / `run_scenarios`](https://github.com/beyhangl/evalcraft/blob/main/docs/user-guide/stateful-scenarios.md).

## Reports, cassettes, and test boundaries

An optional new output directory holds a suite `report.json` and an indexed
directory per case containing `report.json` and `cassette.json`. Each report
retains actual before/after state, outcomes, final expectations, and failures.
These artifacts do not retain the temporary SQLite stores. Existing output
directories are refused, preserving earlier recordings and baselines.

Scenario attempts use versioned `evalcraft_scenario` metadata on informational
`AGENT_STEP` spans. Such a span does not prove a tool executed. Pactrun's observed
tool spans may themselves represent attempted calls; approval consumption also
does not establish an external effect. Inspect outcome annotations and the
authoritative simulated provider/action state for the effect being asserted.

The [cassette bridge](EVALCRAFT_BRIDGE.md) skips only strictly recognized version
1 informational scenario envelopes while preserving actual supported SDK and
Pactrun spans. Policy replay evaluates those recorded observations. It does not
rerun this environment or establish that a recording producer was truthful.

State snapshots omit raw approval tokens and signing keys, but recorded event
metadata can still contain approval tokens or other sensitive data. Review and
redact artifacts before sharing them.

| Test | Purpose |
|---|---|
| Synthetic stateful scenarios | Test current local business code across explicit states, actions, and injected faults |
| Recorded-run checks and policy replay | Evaluate saved observations without executing tools |
| SDK playback | Run current application code with recorded model replies and strict request matching |
| Live model evaluation | Sample actual model/prompt behavior; paid APIs cost money |

The existing [refund workflow guide](SUPPORT_REFUND.md) describes the application
boundary and actual SDK capture/playback test through a mock transport. The
stateful pack complements that test. It does not measure fresh model decisions,
authenticate a real human approver, or make a remote payment transaction atomic.
Real-provider test-environment integration remains separate customer work.
