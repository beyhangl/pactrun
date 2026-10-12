# Approval tokens at a tool boundary

Use approval tokens when the host has independently checked whether a principal
may authorize a particular effect. A token is a bearer grant from that signing
host. A shared-secret HMAC does **not** verify a named human's identity: the host
must authenticate that person and check their authority before issuing a token.

Version 2 tokens authenticate the action, selected argument values, approver ID
(for multi-party approvals), `issued_at`, nonce, and exact business context.
Earlier token formats did not authenticate timestamps and are rejected. Reissue
existing approvals with the current minting functions; changing a timestamp is
not a renewal.

## Host-owned context and arguments

Bind every argument that can change the approved effect. Without `bind_args`, a
token covers the tool name only. Missing selected paths fail validation and
cannot stand in for a selected value of `None`. Selected arguments and context
must contain finite JSON values; arbitrary objects are never coerced to strings.

Read tenant, principal, policy version, and business state from host-owned
records. A callback can refresh those records at validation time. Returning a
model argument or event metadata as the authority makes the host trust an
attacker's assertion. A changed order version or policy should require a new
approval. This check is not a transaction with the business backend; the backend
must enforce its state/version conditions at execution as well.

```python
from pactrun import Contract, consent_token_required, mint_consent_token
from pactrun.approval_store import SQLiteApprovalStore

# Illustrative trusted values; obtain the real key and identity in the host.
secret = b"example-host-key-do-not-use-in-production"
approved_args = {"order_id": "order-17", "amount_cents": 500, "currency": "USD"}
bound = ["order_id", "amount_cents", "currency"]
host_context = {
    "tenant": "shop-a",
    "principal": "support-user-1",
    "policy": "refund-v3",
    "order_version": 8,
}
token = mint_consent_token(
    "refund", args=approved_args, bind_args=bound,
    secret=secret, context=host_context,
)
store = SQLiteApprovalStore("approvals.sqlite")
contract = Contract("refund-approval").require(
    consent_token_required(
        "refund", bind_args=bound, secret=secret,
        context=lambda event, state: host_context,
        replay_store=store,
    ),
    on_fail="block",
)

with contract.session() as session:
    # Emit before the host calls its refund backend. The host must propagate
    # a denial and enforce backend ownership, version and idempotency checks.
    session.emit_tool_call("refund", args=approved_args, metadata={"user_consent": token})
    # backend.refund(...) would run here after validation succeeds.
```

In real use, replace the static `host_context` with an authoritative lookup.
Keep the secret and SQLite file outside model-controlled storage. Other tool
arguments remain outside the signature unless explicitly selected.

## Consumption and failures

Passing `replay_store` opts into durable one-use authorization. It requires a
nonempty HMAC secret. `SQLiteApprovalStore` opens a host-owned file and consumes
nonces in an atomic transaction across instances and processes that share that
file. It retains consumed nonces indefinitely, because callers may disable
expiration. Deleting or restoring an older database loses the corresponding
replay history; retain it consistently with the host's grant lifecycle.

Consumption occurs **when this predicate authorizes the event**, before tool
execution. Another clause can subsequently reject the event; the authorization
will still have been spent. A tool can fail, time out, or succeed with an
uncertain response; the authorization remains spent in each case. Reconcile the
backend outcome and issue a new approval when appropriate. Never automatically
release a nonce after a tool error. Put nonmutating validation clauses before
the approval clause to avoid spending grants on already-invalid requests.

The store does not implement backend idempotency or exactly-once effects. Two
distinct valid approvals can authorize two attempts. The backend must use a
stable idempotency key and enforce remaining balance/state constraints within
its own transaction. A valid token says nothing about the final tool result.

Without `replay_store`, valid tokens can be presented repeatedly. Without
`secret`, minting uses an unkeyed digest for API compatibility and tests; anyone
can produce that digest, so it must not be treated as authentic approval.
Use `on_fail="block"` at an intercepted tool boundary. A log-only clause records
a failure but does not prevent execution. Approval recovery handlers can also
override a failure; an always-allow handler is a test stub, not authentication.

## Quorum approvals

`mint_approval_token` and `multi_party_approval_required` accept the same
`context`, `bind_args`, `secret`, and `replay_store` options. Tokens must name
distinct host-asserted approver IDs. An explicit empty `approvers=[]` allowlist
denies every approver. A quorum size must be a positive integer.

Only the first valid distinct quorum in presentation order is consumed.
Shortfalls and invalid tokens consume nothing. If any nonce in the selected
quorum has already been spent, the entire transaction rolls back and the call
is denied; other selected approvals remain available. Additional unselected
valid approvals are not consumed. Each token should use its own random nonce.

## Freshness and analysis

Timestamps must be finite numeric values, not booleans, and must not be in the
future. `max_age_s=None` disables expiration only. Live sessions use the wall
clock and do not trust an event-supplied old timestamp to revive an expired
grant. Hosts must maintain a consistent clock across issuers and consumers.

Monitor sessions and `Contract.replay` never read or consume the live durable
store. With a configured store, they simulate nonce use within their own
session, so repeated presentation can be detected without spending grants.
Monitor replay uses the declared event clock for expiration; enforcement always
uses wall-clock freshness even if its session duration uses an event clock.
This simulation does not
reconstruct approvals consumed before the recorded trace. Use a separate
authoritative snapshot if a historical investigation needs that information.

Tokens may carry sensitive business context and can authorize actions while
valid. Do not publish live grants or include them in public fixtures. Use only
synthetic identities, arguments, context, and signing keys in shared examples.
