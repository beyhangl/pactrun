# Synthetic support refund workflow

Maintainer: [beyhangl](https://github.com/beyhangl) — beyhangl@gmail.com.

This reference application combines Pactrun controls and Evalcraft evidence for
one refund workflow. It runs locally with fake payments and requires no API keys.
It is an example to adapt and test, not a production payment service.

## Run from this checkout

Install Evalcraft alongside Pactrun. For sibling local checkouts:

```bash
python -m pip install -e ../evalcraft -e ".[dev]"
python examples/support_refund.py --output-dir /tmp/pactrun-refund
python -m pytest -q tests/test_support_refund.py tests/test_evalcraft_bridge.py
```

The optional SDK playback test also needs `openai`; a current installed SDK is
called only through an HTTP mock, never a paid model. The normal demo does not
need a provider SDK. Each kept demo run creates a new subdirectory containing
synthetic SQLite state, `refund.json` and `report.json`.

Expected decision summary:

```json
{
  "timeout_outcome": "unknown",
  "after_restart": "verified",
  "repeat_blocked": true,
  "provider_refunds": 1,
  "recorded_policy_compliant": true
}
```

The output also includes the exact cassette path.

## What the application owns

The host supplies tenant/customer/principal identity and a signing key. The model
cannot set the trusted identity through proposed tool arguments. The example
checks ownership, currency, positive integer minor units, remaining refundable
amount and order version. Approval binds the action ID, all refund arguments,
host identity, policy version, current order version, timestamp and nonce.

A durable SQLite approval store consumes the grant before the provider callback.
A separate action ledger records admission and outcomes. The fake provider
atomically rechecks ownership/state and uses the action ID as an idempotency key.
It rejects the same key reused with different arguments.

A timeout after provider commit becomes `unknown`. On restart, reconciliation
reads the provider ledger and marks the matching effect `verified`; it never
blindly issues a second refund. Missing or mismatched evidence remains unknown
or mismatch. A reserved action surviving a simulated crash can be reconciled.

## Verified acceptance cases

The tests cover:

- correct refund and explicit persisted evidence;
- foreign tenant/customer, wrong currency and invalid or excessive amounts;
- changed amount after approval, edited timestamp and stale order state;
- concurrent use of one grant and restart replay;
- provider timeout, crash between commit and recording, missing/mismatched evidence;
- recording and policy replay with stable token/event accounting;
- changed application code against one recorded SDK reply: an incremented refund
  amount is rejected before a fake payment occurs.

The demo retains a synthetic approval token in its cassette. Production users
must review/redact sensitive evidence before sharing.

## Limits when adapting it

SQLite grant consumption proves authorization use in that store, not a named
human's identity or exactly-once external payment execution. A shared signing key
authenticates the issuing host. Production identity and approval UI are separate.

Every actual payment route must remain behind the host's enforced boundary.
Real providers need their own idempotency and reconciliation behavior; the fake
provider's transaction does not make a remote API transaction atomic. Keep the
store across restarts and use host-owned storage. Do not automatically release a
used grant after an uncertain attempt. A later clause failure can also consume a
grant; reconcile and issue a new approval deliberately.

Current-code model playback, policy replay, attempted action and observed outcome
are distinct forms of evidence. Passing this case set does not prove universal
agent safety.

## Customer pilot and contributor scope

A proposed first pilot covers one customer-owned refund workflow, one framework
integration, a reviewed set of real failure cases and executable regression tests.
The research suggests testing a fixed $1,500–$3,000 pilot price; no customers,
revenue or willingness to pay are claimed. Hosted rollout, retention and team
review features should follow demonstrated repeat usage.

Small useful contributions are a synthetic boundary case, deterministic provider
fault, one requested framework binding, or a cassette compatibility example.
See [contributor tasks](CONTRIBUTOR_TASKS.md). Start in this project; split an
engine-neutral case repository when outside teams need independent ownership.
