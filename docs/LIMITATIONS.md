# Limitations and failure posture

pactrun is an in-process runtime guardrail. This page states plainly what it
does when something goes wrong, and what it cannot see. If you rely on pactrun
for a security property, read this first.

## Failure posture

| Situation | What pactrun does |
|---|---|
| **No contract, or no clauses** | Nothing is checked, so everything is allowed. pactrun enforces only what you declare. |
| **A predicate raises an exception** | Treated as a **failed** check (a check that cannot run has not passed). The failure goes through that clause's own `on_fail`: a default `block` clause halts the run, and the violation is recorded and audited. The remaining clauses are still evaluated. Set `on_fail="log"` on a clause to make a broken check fail open deliberately. |
| **An event reports a negative, NaN, or infinite cost/token count** | The amount is counted as `0` so it can never lower or poison a running total. The raw value is kept in the event's metadata, and the budget predicates (`cost_under`, `cost_per_turn_under`, `token_budget`, `spend_rate_under`) **fail closed** on that event. |
| **A contract limit is NaN, infinite, or negative** | Rejected with `ValueError` when the contract is built. |
| **An optional dependency is missing** (e.g. `jsonschema` for `tool_args_match` / `json_schema_valid`) | Detected on the first event the predicate evaluates, not when the contract loads, and handled like any raising predicate — a failed check. |
| **Monitor mode** (`Contract(...).monitor()` or `session(mode="monitor")`) | Every clause is evaluated and violations are recorded with `enforced=False`, but **no recovery action runs**. Nothing is blocked. `is_compliant` still reports the violations honestly. |
| **MCP adapter cannot load tool annotations** | A warning is logged. With `destructive_policy="hint"` nothing is blocked by annotation (there are no hints to act on); with `"strict"` every tool not marked read-only is blocked. Annotations are read once per session. |

## What pactrun cannot see

- **Calls that bypass it.** pactrun only sees events that flow through a
  wrapped client, an adapter, or an explicit `Session`. A tool the agent calls
  directly is invisible to it.
- **Code that runs without a tool call.** A side effect triggered outside the
  agent's tool calls — a git hook, a config file read by another process, a
  package install script — never produces an event.
- **Model internals.** pactrun judges inputs, outputs, and tool calls, not the
  model's reasoning.

## Known limits of specific checks

- **Egress checks match literal hosts; they do no DNS resolution.**
  `tool_host_within` and `no_exfil_links` canonicalize every literal form a
  client might use (legacy numeric IPv4, trailing dots, `*.localhost`,
  RFC 3986 vs. WHATWG parsing) and block if any interpretation is disallowed.
  But a **hostname that resolves to a private address** — including through DNS
  rebinding — is not caught, because resolving it at check time would itself be
  racy. Pair these checks with network-level egress controls.
- **Heuristic predicates are tripwires, not guarantees.** The injection-phrase
  bank, the system-prompt canary, taint-overlap detection, the PII and secret
  regexes, the destructive-argument denylist, and the invisible-text scanner can
  all be evaded by paraphrase, novel encodings, or content split across several
  tool calls. Treat any denylist as incomplete.
- **The cost check in `wrap()` is a worst-case bound.** Completion tokens can't
  be known before a call, and reasoning models can exceed the estimate. The
  recorded cost after the call is the real one.
- **Windowed rate limits trust `Event.timestamp`.** Adapters and `emit_*`
  stamp events with the wall clock. If you build `Event` objects yourself, the
  windows are only as honest as the timestamps you supply.
- **`tool_definitions_stable` compares only what is recorded.** It can only
  catch a tool definition that changes if the host records the definition on
  each call. The MCP adapter currently reads definitions once per session.

## What the audit log does and does not prove

`AuditLogObserver` writes a hash-chained ledger that `verify_audit_log()`
re-checks offline.

- It records what pactrun **observed and decided** in-process. It does not
  prove that every action the agent took went through pactrun.
- **Without a `secret`, the chain is plain SHA-256.** It detects a record that
  was edited, deleted, or reordered in place — but someone who rewrites the
  **entire** file can recompute every hash, and verification will pass. Only
  **HMAC mode** (`AuditLogObserver(path, secret=...)`) resists a full rewrite,
  and only while the secret stays secret.
- It is a control that *supports* record-keeping. It is not, on its own,
  compliance with any regulation. See the audit module's docstring for the EU
  AI Act specifics.
