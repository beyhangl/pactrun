# Changelog

All notable changes to pactrun are documented here. The format follows
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/), and the project
follows [Semantic Versioning](https://semver.org/). While pactrun is `0.x`,
breaking changes can land in a minor release — they are always marked
**BREAKING:** below. See [docs/VERSION_POLICY.md](docs/VERSION_POLICY.md).

## [Unreleased]

pactrun has not yet been published to PyPI; everything below is unreleased.

### Security

- Budget predicates no longer fail open on invalid amounts. A negative cost or
  token count used to lower the running total, so one bad event could
  "refund" the budget ($54.90 of real spend passed a $5 `cost_under` with no
  violation), and a NaN poisoned the total for the rest of the run. Amounts are
  now sanitized at ingestion and the budget predicates fail closed on the
  offending event. (`77ed67a`)
- `tool_host_within` and `no_exfil_links` no longer let legacy numeric IPv4
  forms (`2130706433`, `0x7f000001`, `0177.0.0.1`, `127.1`), `localhost.`, or
  `*.localhost` reach loopback past `block_private`, and a backslash URL
  (`http://evil.com\@good.com/`) can no longer beat an allowlist. Both guards
  now check every host a real client might connect to. (`77ed67a`)
- The MCP adapter no longer fails open with MCP SDK v2. It read only camelCase
  annotation names, so on SDK v2 `destructive_policy="hint"` allowed every
  destructive tool while reporting blocking as enabled. It now reads both
  spellings, warns loudly when annotations can't load, and pins `mcp<3`.
  (`4e53a0a`)

### Added

- Monitor (shadow) mode: `Contract(...).monitor()` or `session(mode="monitor")`
  evaluates every clause and records violations with `enforced=False` without
  running recovery actions.
- `tool_definitions_stable` — detects a tool server changing what it
  advertises mid-run (OWASP ASI04).
- OWASP Top 10 for Agentic Applications (2026) mapping on every predicate, and
  `pactrun predicates --owasp`.
- Security predicates: `no_injection_phrases`, `canary_not_leaked` /
  `mint_canary`, `no_invisible_text` (including variation-selector
  smuggling), `no_exfil_links`, `no_exfiltration_after_untrusted`,
  `lethal_trifecta_guard`, `untrusted_taint_to_sink`, `tool_host_within`.
- Human-in-the-loop predicates: `consent_token_required`,
  `multi_party_approval_required`, `ai_disclosure_in_output` (with
  `on_behalf_of`).
- Reliability predicates: `bounded_error_retries`, `no_redundant_reads`,
  `no_progress_stall`, `no_duplicate_side_effect`, `tool_error_rate_under`.
- Rate and quota predicates, argument-level tool guards, flow tracking.
- `AuditLogObserver` / `verify_audit_log()` — a hash-chained JSONL ledger.
- `digest()` batched escalation, webhook escalation, `approve` recovery.
- Adapters for OpenAI, Anthropic, Gemini, LangChain/LangGraph, LiteLLM/CrewAI,
  and MCP; `pactrun.wrap()` with async and streaming support.
- `py.typed`, so downstream type checkers see pactrun's annotations.

### Changed

- **BREAKING:** a predicate that raises an exception is now recorded as a
  failed check and routed through the clause's `on_fail`, instead of the
  exception escaping `record_event`. A default `block` clause still halts the
  run, now with a `ViolationError`.
- **BREAKING:** budget limits that are NaN, infinite, or negative now raise
  `ValueError` when the contract is built (including `wrap(max_cost=...)`).
- **BREAKING:** the OTel observer emits cost as `pactrun.usage.cost` instead of
  `gen_ai.usage.cost`, which is not a GenAI convention attribute.
- The OTel observer emits `execute_tool` spans as `INTERNAL` rather than
  `CLIENT`, and omits `gen_ai.provider.name` when the provider is unknown
  rather than emitting an invalid value.
- Minimum Python is now 3.10 (3.9 reached end of life). Tested on 3.10–3.14.

### Fixed

- The EU AI Act framing in the audit log docs overstated what Art. 12
  requires; it now describes the log as a control that supports
  record-keeping, not compliance.
- A trailing-dot host (`good.com.`) no longer fails its own allowlist.
