# Security Policy

pactrun is a runtime guardrail. For a library like this, **a guardrail that
silently stops guarding is the worst kind of bug**, so we treat bypasses and
fail-open behavior as security issues, not ordinary defects.

## Supported versions

pactrun is pre-1.0. Only the latest release line receives security fixes.

| Version | Supported |
|---|---|
| `0.x` (latest minor) | ✅ |
| older `0.x` minors | ❌ — upgrade to the latest minor |

## Reporting a vulnerability

**Please do not open a public issue for security problems.**

Report privately through GitHub: go to the repository's **Security** tab and
choose **Report a vulnerability**
(<https://github.com/beyhangl/pactrun/security/advisories/new>).

### What to include

- pactrun version, Python version, and which adapter or entry point you used
  (`wrap()`, an adapter, a manual `Session`, the MCP adapter, …).
- The contract involved: the predicates, their parameters, and each clause's
  `on_fail`.
- A minimal reproduction, and what you expected pactrun to do versus what it did.
- Whether the path was sync or async, streaming or not.

### Redact sensitive data

Strip API keys, tokens, private prompts, and customer data from reports and
reproductions. If a credential was exposed while you investigated, rotate it.

## What we commit to

This is a small project maintained on a best-effort basis. We aim to:

- **acknowledge** your report within **3 business days**;
- keep you updated as we investigate;
- ship a fix or documented mitigation within **90 days**, under coordinated
  disclosure — sooner for anything that fails open;
- credit you in the advisory and the changelog, unless you prefer otherwise.

## Scope

### In scope

- **Predicate bypass or fail-open** — any input that makes a predicate pass
  when its documented guarantee says it should fail (e.g. a value that lowers a
  running budget, a URL form that reaches a private host past
  `tool_host_within`).
- **Enforcement skipped** — an adapter or code path through which a wrapped
  call reaches the model or a tool without the contract being evaluated.
- **Audit-log integrity** — altering, deleting, or reordering records without
  `verify_audit_log()` detecting it.
- **Redaction bypass** — secrets or redacted arguments leaking into the audit
  log, telemetry, or violation records despite redaction being enabled.
- **Parser differentials** — pactrun interpreting a URL, path, or argument
  differently from the code that actually executes it.

### Out of scope

- **Model jailbreaks with no pactrun bypass.** Getting a model to say something
  is not a pactrun vulnerability unless a configured predicate should have
  caught it and did not.
- **Documented heuristic limits.** Several predicates are deliberately
  best-effort tripwires (the injection-phrase bank, the canary, taint overlap).
  A paraphrase they don't catch is a known limit — see
  [docs/LIMITATIONS.md](docs/LIMITATIONS.md). A bypass of a guarantee the docs
  *do* make is in scope.
- **Calls that never go through pactrun.** If the agent calls a provider or
  tool directly rather than through a wrapped client, adapter, or session,
  pactrun cannot see it.
- **Deployment hardening** — authentication, TLS, rate limiting of your own
  services. pactrun is an in-process library, not a gateway.

## Past security fixes

These were found and fixed before the first PyPI release, so no published
version was affected. They are listed for transparency.

| Fixed in | Issue |
|---|---|
| `77ed67a` | Negative or non-finite costs/tokens could lower a running budget (fail-open) or poison it (denial of service). Legacy numeric IPv4 forms, `localhost.`, `*.localhost`, and a backslash URL-parser differential bypassed `tool_host_within` / `no_exfil_links`. |
| `4e53a0a` | With MCP SDK v2, the MCP adapter parsed no tool annotations, so `destructive_policy="hint"` allowed every destructive tool while reporting blocking as enabled (fail-open). |
