# Changelog

All notable changes to pactrun are documented here. The format follows
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/), and the project
follows [Semantic Versioning](https://semver.org/). While pactrun is `0.x`,
breaking changes can land in a minor release — they are always marked
**BREAKING:** below. See [docs/VERSION_POLICY.md](docs/VERSION_POLICY.md).

## [Unreleased]

pactrun has not yet been published to PyPI; everything below is unreleased.

### Security

- Model pricing under-counted, which lets spend pass `cost_under`. An
  unknown model cost $0 in the adapters and $2.50 / $10 in the pre-call gate.
  Lookup took the first prefix, so `gpt-5.4-pro` (30 / 180) was priced as
  `gpt-5.4` (2.50 / 15), and `o3-pro` as `o3`. Several prices were stale
  (`gpt-4.1-mini`, `o4-mini`, `claude-haiku-4-5`), and every Claude 4.7+ /
  5.x, Gemini 3.x / 4 and GPT-5.5 / 5.6 / 6.x model was missing. All pricing
  now comes from one table, `pactrun.core.pricing`, checked 2026-10-19
  against the providers' pricing pages. Lookup normalises Bedrock, Vertex,
  LiteLLM and `models/` names, strips date stamps, and takes the longest
  prefix, refusing differently billed variants such as `-pro` or `-tts`. An
  unknown model is priced at its provider's highest rate in the table,
  flagged `cost_source="unknown_model"` and logged once. Long-context tiers,
  Anthropic fast mode and US-only inference (+10%) are priced. The LiteLLM
  adapter falls back to the table instead of $0.
- The MCP adapter failed open three ways under `block_destructive`: it read
  only the first page of `tools/list`, so a destructive tool on page 2 was
  allowed; it read annotations once, so a tool re-annotated as destructive
  later was allowed; and a tool missing from the listing was allowed under the
  `hint` policy. It now follows cursors (failing closed past
  `max_tool_pages`), lists again when the listing expires (server `ttlMs`,
  capped by the new `tool_cache_max_age`, default 60s), after
  `invalidate_tools()`, and once for any unknown tool, and blocks a tool that
  is still unlisted. A failed listing now blocks every call instead of none.
  Each MCP tool call records the advertised definition in
  `metadata["tool_definition"]`, so `tool_definitions_stable()` works with MCP.
- **BREAKING:** approval token envelopes use version 2; reissue older tokens.
  Signatures now authenticate timestamps, nonces, exact actions/selected
  arguments, approver IDs and optional host-owned context. Invalid/future
  timestamps and altered context fail closed. SQLite replay stores atomically
  consume grants/quorums across sessions and restarts. Monitor/replay never
  consume the live store; enforcement freshness always uses wall time.
- The wrapper refuses unsupported OpenAI Responses API access and reserves
  estimated cost atomically across calls sharing one wrapper. Streams, missing
  or invalid usage, failures and cancellation settle once, conservatively when
  billed usage is unknown. Estimates include tool definitions and supported
  system/tool-history text. SDK async subclasses retain async enforcement.

- Prompt caching broke token and cost accounting in every adapter. Anthropic
  dropped cache reads and writes from the token total, so `token_budget`
  under-counted and cached runs looked cheap. OpenAI billed cached input at the
  full rate. Gemini dropped thinking and tool-use input tokens. The `wrap()`
  pre-call gate priced Anthropic cache writes (1.25x or 2x input) at 1x.
  `Event` now has `cache_read_tokens`, `cache_write_tokens` and
  `reasoning_tokens`, cached input is priced at each provider's rate, and the
  OTel span emits `gen_ai.usage.cache_read.input_tokens`,
  `gen_ai.usage.cache_write.input_tokens` and
  `gen_ai.usage.reasoning.output_tokens`.
- `no_injection_phrases(decode=("base64",))` missed an encoded payload whenever
  text was glued to its front (`payload=...`, `id_...`, a URL query) or it used
  the URL-safe alphabet. It now decodes at every alignment in both alphabets.
- `no_exfil_links` missed three kinds of image a renderer loads automatically:
  reference-style markdown images (`![x][1]` with `[1]: url`), images whose alt
  text contains brackets, and `srcset` on `<img>` / `<source>`.
- `no_exfiltration_after_untrusted` now also honours `metadata["untrusted"] =
  True`, the label `untrusted_taint_to_sink` and `lethal_trifecta_guard` read.
  A host that set only that label lost the label check here.
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

### Fixed

- TURN_END events advance state on ingestion, so policy replay retains turn
  limits instead of falsely passing them with a zero turn count.

### Added

- `pactrun.register_price(model, input, output, cache_read=, cache_write=)`
  gives a fine-tuned, private or newly released model its real price instead
  of the provider ceiling.
- Session restore: `contract.session(restore_from=...)` rebuilds budgets,
  histories, turn counts and rate windows from a `TraceRecorder` file or a
  list of events, so a process restart no longer resets every budget to zero.
  Restored events are not re-checked, recovery doesn't run, and observers
  aren't called. `restore_violations=` carries earlier violations over
  (marked `Violation.restored`), and `restore_elapsed_ms=` sets the restored
  wall-clock time. Restoring never consumes an approval grant, and a grant
  used before the restart stays used. A `TraceRecorder` without
  `append=True` on the file being restored is refused, because it would empty
  the trace before the restore reads it.
- `Session.pause()` / `resume()` / `is_paused` leave suspended time out of
  wall-clock `elapsed_ms` and `session_timeout`. Recording an event resumes
  the clock.
- `context_growth_under` and `max_steps_per_turn` (OWASP ASI08) catch the
  denial-of-wallet pattern where tool output piles up in the history and is
  billed again on every later call: a cap on one prompt's size and on growth
  against the run's first prompt, and a cap on model calls before the agent
  returns to the user.
- `tool_spend_under` (OWASP ASI02, ASI03) caps money moved through payment or
  refund tools: per call, per run, one currency, a merchant allowlist and an
  expiry. A missing, negative, NaN or non-numeric amount fails closed, so a
  "negative refund" can't lower the running total.
- Optional Evalcraft cassette bridge: normalized cache tiers, stable identities,
  observed tool results/provenance, and monitor-mode policy replay. Missing
  costs and ambiguous result relationships are refused by default.
- Synthetic refund reference workflow and end-to-end tests for ownership,
  signed approvals, concurrent attempts, restart recovery, uncertain provider
  outcomes, cassette evidence and changed-code playback. No real payments or
  model calls are made.
- Application coverage, approval migration, bridge and contributor guides.

- `no_untrusted_memory_write` (OWASP ASI06): flags a memory-write tool call
  that copies untrusted content (`mode="copy"`), or any memory write after
  untrusted input (`mode="any"`), so injected text can't persist into later
  runs.
- Repeated runs in policy tests: a test's `traces:` (a list or a glob) replays
  several runs of one task. The test passes only if every run gets the
  expected verdict, and the report gives Pass^k and Mean@k.
- Pydantic AI adapter: `PactrunCapability(contract)` plugs into
  `Agent(capabilities=[...])`, records model responses and tool calls, and
  stops a blocked tool before it runs. `on_tool_block="return_to_model"`
  returns the refusal to the model instead. New extra: `pactrun[pydantic-ai]`.
- Trace replay: `TraceRecorder` writes every session event to a JSONL file
  (credential-looking arguments redacted), and `Contract.replay()` /
  `replay_trace()` / `pactrun replay CONTRACT TRACE` evaluate a recorded run
  against any contract without enforcing. Replays run on the event clock, so
  `session_timeout` judges how long the original run took.
- Policy tests: a contract's `tests:` block pins the exact set of predicates
  each trace (or inline event list) should trip; `pactrun test` and
  `run_contract_tests()` fail on missed detections and on over-blocking.
- `Violation.predicate_name`, so violations name the predicate that fired even
  when a clause has a custom description.
- `session(clock="event")` measures elapsed time from event timestamps instead
  of the wall clock.
- A top-level `mode: monitor` in YAML contracts.
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

- **BREAKING:** `pactrun.core.usage.lookup` is removed (use
  `pactrun.core.pricing.find` / `cost`). Budgets set against the old,
  under-counted prices trip earlier. An unknown OpenAI model is now priced at
  the most expensive OpenAI rate ($150 / $600 per 1M tokens), so register a
  price for any model the table lacks. `cost_model` prefers the table over
  litellm's price map: "exact" / "estimated" mean a known price from either,
  "heuristic-fallback" a rough token count or an unknown model.
- Corrected prices: claude-opus-4-6 and Opus 4.5–4.8 (5 / 25, were 15 / 75),
  gemini-2.5-flash-lite (0.10 / 0.40, was priced as 2.5 Flash), and the
  Gemini 2.5 Pro over-200k tier.
- OTel (experimental): `OTelObserver` emits an `invoke_agent` root span per
  session, with chat and tool spans as children (`emit_agent_span=False` keeps
  the old flat shape). Violations also emit `gen_ai.evaluation.result` span
  events (`fail`, or `would_fail` in monitor mode); ERROR status and exception
  events are unchanged. New metrics with a `meter_provider=`:
  `gen_ai.client.inference.usage.*` (cache and reasoning included),
  `gen_ai.client.inference.operation.{input,output}_tokens`,
  `gen_ai.client.inference.duration`, `gen_ai.execute_tool.duration` and
  `gen_ai.invoke_agent.{duration,inference_calls,tool_calls}`. New options:
  `agent_name=`, `agent_id=`, `agent_description=`, `agent_version=`,
  `conversation_id=`, `emit_llm_spans=`. `execute_tool` spans carry
  `gen_ai.tool.call.id`, a chat span with no model is named `chat`, and failed
  calls set `error.type`. Targets semantic-conventions-genai main `547934a`
  (2026-10-18).
- **BREAKING (MCP):** under `block_destructive`, a failed tool listing blocks
  every call, and an unlisted tool is blocked. Re-listing adds round trips;
  with `ttlMs=0` (the SDK 2.x server default) that is one per call.
- **BREAKING:** `prompt_tokens`, `total_tokens`, `token_budget` and
  `gen_ai.usage.input_tokens` now count all input, including cached input. On
  Anthropic with a large cached prefix, existing token budgets trip much
  sooner. On Gemini they also include thinking and tool-use input tokens.
- Cost changes direction by provider: cached Anthropic runs cost more than
  before (reads and writes are now counted), cached OpenAI and Gemini runs cost
  less (cached input is discounted). `cost_under` can trip at a different point.
- The LangChain adapter reads `usage_metadata` before `llm_output`.
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

- `pactrun validate` crashed with a traceback on a clause with bad arguments
  (an unknown argument name, or a NaN/infinite/negative limit). It now reports
  the clause as invalid and exits 1.
- The EU AI Act framing in the audit log docs overstated what Art. 12
  requires; it now describes the log as a control that supports
  record-keeping, not compliance.
- A trailing-dot host (`good.com.`) no longer fails its own allowlist.
