<p align="center">
  <strong>pactrun</strong>
</p>
<p align="center"><strong>One line that refuses the call that would blow your AI agent's run</strong> — total cost, tool use, loops, drift — on the OpenAI/Anthropic SDKs or LangGraph/CrewAI.<br>Observability records what happened; pactrun refuses before it does.</p>

[![Tests](https://github.com/beyhangl/pactrun/actions/workflows/test.yml/badge.svg)](https://github.com/beyhangl/pactrun/actions/workflows/test.yml)
[![License](https://img.shields.io/github/license/beyhangl/pactrun)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-blue)](https://github.com/beyhangl/pactrun)
[![Status](https://img.shields.io/badge/status-alpha-orange)](#status)

---

## What is this?

Guardrails check individual **messages**. pactrun checks an agent's behavior across an entire **session** — enforcing limits on accumulated cost, tool usage, call ordering, loops, and drift, and raising (or recording) a violation the moment a contract is broken.

**The one-liner — wrap any OpenAI/Anthropic client and every call is checked _before_ it bills:**

```python
import openai, pactrun

client = pactrun.wrap(
    openai.OpenAI(),
    max_cost="$0.50",                 # whole-run budget — refused BEFORE the call that would cross it
    no_loops=True,                    # stop repeating tool loops
    forbid_tools=["delete_account"],  # never let the model call this
    max_drift=0.5,                    # flag cost-per-turn creep
)

client.chat.completions.create(model="gpt-4.1", messages=[{"role": "user", "content": "..."}])
# raises ViolationError before billing if the worst-case cost would blow the run budget
```

> The cost check is a **worst-case bound** — you can't know completion tokens before a call, and reasoning models can exceed the estimate. It's an in-process circuit-breaker *between* calls, not a proxy to deploy. Demo: [`examples/three_in_one.py`](examples/three_in_one.py).

For full control — any provider, or a non-SDK agent loop — build a `Contract` and drive it yourself:

```python
from pactrun import Contract, cost_under, max_turns, no_loops, must_not_call

contract = (
    Contract("support_agent")
    .require(cost_under(0.50))                  # whole-run budget
    .require(max_turns(20))
    .require(no_loops())                        # catch infinite tool loops
    .forbid(must_not_call("delete_account"))
    .on_violation("block")
)

with contract.session() as session:
    session.emit_llm_response(model="gpt-4.1", output="Looking that up…", cost=0.003)
    session.emit_tool_call("lookup_order", args={"id": "123"})
    # ... the rest of your agent loop ...

print(session.summary().is_compliant)   # True
```

An agent can pass every per-message guardrail and still run up a $50 bill, loop forever, or call a tool it never should. pactrun is the layer that catches **session-level** behavior.

---

## Status

> **pactrun is alpha (v0.1.0).** This README documents only what actually ships today. The core below works and is covered by **450 passing tests**. A few capabilities that belong to the longer-term vision — compliance-document export, one more framework adapter, and formal composition — are **not built yet**; they live in the [Roadmap](#roadmap), not in the feature list.

| Works today ✅ | Not built yet 🚧 (see Roadmap) |
|---|---|
| One-line `pactrun.wrap()` pre-call gate — real-tokenizer cost (tiktoken/litellm), **async + streaming** | EU AI Act / compliance document export |
| Fluent `Contract` builder + YAML loader | Pre-call gate for Gemini / LiteLLM clients (today: OpenAI, Anthropic) |
| Session-level runtime enforcement (sync + async) | Pydantic-AI adapter; native CrewAI tool events |
| 38 built-in predicates (cost, tools, **tool-args**, output, **schema/secrets**, timing, behavioral, **rate-limit**, **flow**) | Formal multi-agent composition |
| Recovery: log / warn / block / escalate / **approve** / retry / fallback | |
| **Argument-level tool guards** — JSON-Schema match, destructive-command block, path-sandbox, **per-field value allow/deny**, **required disclosure** | |
| **Egress / SSRF guard** — host allow/deny + CIDR + block-private (`tool_host_within`) | |
| **Consent gating** — fresh, action-bound, HMAC-signable consent tokens per tool call | |
| **Output integrity** — `valid_json` / `json_schema_valid` / `no_secrets` + **cross-tenant isolation** | |
| **Rate & quota** — rolling spend/call/tool windows, **per-recipient buckets**, **calendar quotas** | |
| **Flow tracking** — ordered-stage drop-off diagnostics + out-of-order phase gate | |
| **Escalation handlers** — built-in webhook (generic / chat) + **digest** (batched alerts) | |
| Drift detection (Page-Hinkley + EWMA) | |
| OpenAI + Anthropic + Gemini + LangChain/LangGraph + LiteLLM/CrewAI + **MCP** adapters | |
| `@contract.enforce` decorator | |
| CLI (`init` / `validate` / `show` / `predicates`) | |
| pytest plugin (`@pytest.mark.contracted`) | |
| **OpenTelemetry GenAI** span emitter (experimental) | |

---

## Install

The package is named **`pactrun`**. A PyPI release is planned (see Roadmap); until then, install from source:

```bash
pip install "git+https://github.com/beyhangl/pactrun"

# with the OpenAI adapter extra:
pip install "git+https://github.com/beyhangl/pactrun#egg=pactrun[openai]"

# local development:
git clone https://github.com/beyhangl/pactrun && cd pactrun
pip install -e ".[dev]"
pytest
```

```python
import pactrun   # the import name is `pactrun`
```

---

## How it works

A **Contract** is a set of **clauses**, each evaluated at the right moment in a session:

```
Contract
├── precondition   — checked at session start
├── require        — must hold (per-event, or at session end for ordering/output checks)
├── forbid         — must never happen (checked per-event)
└── postcondition  — checked at session end
```

Predicates that can only be judged once the run is over — `must_call`, `tool_order`, `output_contains`, `output_matches` — automatically defer to **session end**. Everything else (cost, token, loop, latency checks) is evaluated **per event**, so a `block`-mode violation stops the run the instant a limit is crossed.

There are three ways to feed events into a session:

**1. Manually** — call `emit_*` as your agent runs:

```python
with contract.session() as session:
    session.emit_llm_response(model="gpt-4.1", output="...", cost=0.003, completion_tokens=120)
    session.emit_tool_call("search", args={"q": "weather"})
    session.emit_output("Here is your answer.")
```

**2. Auto-instrument** — wrap your provider call in an adapter that emits events for you:

```python
import openai
from pactrun import Contract, cost_under, must_not_call
from pactrun.adapters import OpenAIAdapter

contract = Contract("agent").require(cost_under(0.25)).forbid(must_not_call("transfer_funds"))
client = openai.OpenAI()

with contract.session() as session:
    with OpenAIAdapter():            # patches client.chat.completions.create for the block
        client.chat.completions.create(
            model="gpt-5.4-nano",
            messages=[{"role": "user", "content": "Hello"}],
        )

print(session.summary().total_cost_usd)
```

`AnthropicAdapter` and `GeminiAdapter` work the same way — wrap the provider call in `with AnthropicAdapter():` or `with GeminiAdapter():`. `LiteLLMAdapter` patches `litellm.completion`, so it instruments **CrewAI** and anything else routed through LiteLLM: `with LiteLLMAdapter(): crew.kickoff()`.

**3. As a decorator** — `@contract.enforce` opens a session around a function (emit events inside it via an adapter or `emit_*`):

```python
@contract.enforce
def run_agent(query: str) -> str:
    with OpenAIAdapter():
        ...
    return answer

run_agent("refund my order")   # raises ViolationError if a block-mode clause is breached
```

When a clause set to `block` is violated, pactrun raises `ViolationError`. Other modes (`log`, `warn`) record the violation and let the run continue; you inspect them via `session.violations` and `session.summary()`.

### LangChain / LangGraph

LangChain and LangGraph instrument via callbacks, so pactrun ships a `PactrunCallbackHandler` you pass through the run config. It records every LLM and tool event the graph produces into the active session:

```python
from pactrun import Contract, cost_under, max_turns
from pactrun.adapters import PactrunCallbackHandler

contract = Contract("graph_agent").require(cost_under(0.50)).require(max_turns(15))
handler = PactrunCallbackHandler()

with contract.session() as session:
    graph.invoke(state, config={"callbacks": [handler]})   # any LangGraph graph or LangChain runnable

print(session.summary().is_compliant)
```

For async or multi-threaded runs where the active-session contextvar may not propagate to the callback, pass the session explicitly: `PactrunCallbackHandler(session=session)`.

### MCP (Model Context Protocol)

Wrap an MCP `ClientSession` so your tool contracts apply to its tool calls — the whole tool-predicate suite works for free, plus an optional `block_destructive` policy driven by the server's own annotations:

```python
from pactrun import Contract, must_not_call
from pactrun.adapters import GuardedMCPSession

contract = Contract("mcp_agent").forbid(must_not_call("delete_file"))
guarded = GuardedMCPSession(client_session, contract, block_destructive=True)
await guarded.initialize()
await guarded.call_tool("read_file", {"path": "a.txt"})    # ok
await guarded.call_tool("delete_file", {"path": "a.txt"})  # raises ViolationError
```

`destructiveHint` is an advisory, possibly-untrusted hint, so `block_destructive` is defense-in-depth — pair it with an explicit `tools_allowed` for high assurance. (`pip install "pactrun[mcp]"`)

### OpenTelemetry (experimental)

Attach an `OTelObserver` and pactrun emits standard `gen_ai.*` spans for every LLM and tool call, marking the span `ERROR` when a contract is violated — so your runtime contracts show up in Langfuse / Arize Phoenix / Datadog / any OTLP backend:

```python
from pactrun.observability import OTelObserver

with contract.session(observers=[OTelObserver()]) as session:
    ...
```

Experimental — tracks the OpenTelemetry GenAI semantic conventions (Development status), pinned to v1.27. (`pip install "pactrun[otel]"`)

---

## Drift detection

`DriftMonitor` runs streaming change-point detectors (Page-Hinkley or EWMA) over per-turn metrics to flag when an agent's behavior is shifting mid-session — cost creep, token inflation, tool-pattern changes.

```python
from pactrun.drift import DriftMonitor

monitor = DriftMonitor(threshold=0.3, detector_type="page_hinkley")

for turn in turns:
    report = monitor.record_turn(
        cost=turn.cost,
        tokens=turn.tokens,
        tool_calls=turn.tool_count,
    )

if report.is_drifting:
    print(f"drift detected: score {report.overall_drift_score:.2f} over {report.turn_count} turns")
```

> Drift detection needs a minimum number of turns before it activates (`min_turns=5` by default) and is most meaningful on longer-running sessions. On short 3–5 turn sessions it deliberately stays quiet.

You can also use `drift_bounds(...)` as an inline predicate inside a contract to fail a run when a turn deviates too far from the session average.

---

## Recovery actions

Every clause has an `on_fail` action (set per-clause via `on_fail=...`, or for the whole contract via `.on_violation(...)`). When the clause is violated, pactrun reacts:

| Action | What happens |
|---|---|
| `log` | record the violation and continue |
| `warn` | record + emit a `UserWarning`, then continue |
| `block` | record + raise `ViolationError`, halting the run immediately |
| `escalate` | record + call an escalation handler (page a human / webhook), then raise `EscalationError` |
| `approve` | record + ask an approval handler whether to proceed; continue if it allows, else raise `ViolationError` (fail-closed) |
| `retry` | under `@contract.enforce`, re-run the wrapped call up to `max_retries` times |
| `fallback` | under `@contract.enforce`, call a registered fallback function instead |

```python
from pactrun import Contract, cost_under, get_active_session

# Retry the agent up to 3 times if it busts the budget; fall back if it keeps failing.
def safe_agent(*args, **kwargs):
    return "served by the safe fallback agent"

contract = (
    Contract("resilient_agent")
    .require(cost_under(0.05), on_fail="retry")
    .with_retries(3)
    .fallback(safe_agent)
)

# Or escalate to a human/webhook and halt. webhook_handler() ships ready to wire:
from pactrun import webhook_handler

contract = (
    Contract("supervised_agent")
    .require(cost_under(0.05), on_fail="escalate")
    .on_escalate(webhook_handler("https://hooks.example.com/agents", mode="chat"))
)

# Or gate a violation on a human/policy decision instead of hard-blocking:
from pactrun import cli_approver

contract = (
    Contract("reviewed_agent")
    .require(cost_under(0.05), on_fail="approve")
    .on_approve(cli_approver())          # returns truthy → proceed, falsy → raise
)
```

`webhook_handler(url, mode="generic"|"chat")` POSTs the violation (your own JSON endpoint or a chat-webhook shape) with per-clause throttling; `cli_approver()` prompts on the terminal, and any `(violation) -> bool` works as a headless policy gate (a missing or erroring approver fails closed). Wrap any escalation handler in `digest(...)` to batch a flood of violations into one aggregated alert per window instead of one message each:

```python
from pactrun import webhook_handler
from pactrun.recovery import digest

# One summarized alert per 30s (count, first/last, samples) instead of a storm:
contract = Contract("agent").require(cost_under(0.05), on_fail="escalate").on_escalate(
    digest(webhook_handler("https://hooks.example.com/agents"), window="30s")
)
```

`retry` and `fallback` are control-flow actions handled by `@contract.enforce` (which owns the call); outside the decorator they surface as `RetrySignal` / `FallbackSignal` for you to handle. See [`examples/recovery.py`](examples/recovery.py).

---

## Built-in predicates

All 38 ship today. Pass any of them to `.require(...)` / `.forbid(...)` (or reference them by name in YAML).

| Group | Predicate | What it checks |
|---|---|---|
| **Cost** | `cost_under(max_usd)` | session total cost stays under budget |
| | `cost_per_turn_under(max_usd)` | latest turn's cost under a limit |
| | `token_budget(max_tokens)` | session total tokens under budget |
| **Tools** | `must_call(tool)` | tool was called by session end |
| | `must_not_call(tool)` | tool is never called |
| | `tool_order(expected, strict=False)` | tools called in a given order |
| | `tools_allowed(whitelist)` | only whitelisted tools are called |
| | `max_tool_calls(limit)` | total tool calls capped |
| **Tool args** | `tool_args_match(tool, schema)` | a tool's call arguments validate against a JSON Schema |
| | `no_destructive_args(tool=None, extra=None)` | tool args don't carry a destructive command (`rm -rf`, `DROP TABLE`, …) |
| | `tool_path_within(root, tool=None, arg_keys=None)` | path-like args stay inside a sandbox root (no `..` escape) |
| | `tool_arg_value_guard(tool, field, deny=/allow=, …)` | a dotted-path arg field is allow/deny-listed (exact/ci/glob/regex), with optional once-per-session dedupe |
| | `required_disclosure(tool, arg, must_contain, …)` | a tool arg must contain required disclosure phrase(s) before the call fires (fail-closed) |
| | `tool_host_within(allow=/deny=/block_private=, …)` | URL/host args reach only allowed hosts (globs + CIDR; blocks private/metadata IPs) |
| | `consent_token_required(tools, bind_args=, secret=, …)` | a fresh, action-bound (HMAC-signable) consent token was presented for the call |
| **Output** | `no_pii()` | no email / SSN / phone / card number in output |
| | `output_contains(substring, case_sensitive=True)` | final output contains a string |
| | `output_matches(pattern)` | final output matches a regex |
| | `max_output_length(max_chars)` | output length capped |
| | `output_must_not_contain(pattern)` | output does not match a forbidden regex |
| | `valid_json()` | the final output parses as JSON |
| | `json_schema_valid(schema)` | the final output is JSON that validates against a schema |
| | `no_secrets(scan_tool_args=False)` | output (and optionally tool args) carries no API keys / tokens / private keys |
| | `tenant_response_isolation(tenant_key="tenant", …)` | a response tagged for another tenant never surfaces in this run (fail-closed) |
| **Timing** | `max_latency(max_ms)` | no single event exceeds a latency |
| | `session_timeout(max_ms)` | whole session completes within a time budget |
| | `max_turns(n)` | session does not exceed N turns |
| **Rate limit** | `spend_rate_under(max_usd, window_s)` | LLM spend within a rolling time window stays under a cap |
| | `call_rate_under(max_calls, window_s)` | LLM-call count within a rolling window stays under a cap |
| | `tool_rate_limit(tool, max_calls, per_seconds)` | one tool's invocation rate within a window stays under a cap |
| | `per_key_rate_limit(tool, key_path, max_calls, per_seconds)` | independent rolling windows **per extracted arg value** (e.g. per recipient) |
| | `tool_quota_per_period(tool, limit, period="month", …)` | calendar-anchored quota that **resets** on the day/week/month boundary |
| **Behavioral** | `no_loops(window=5, threshold=0.8)` | recent tool calls aren't a repeating loop |
| | `max_retries(n, tool=None)` | no more than N consecutive identical tool calls |
| | `drift_bounds(cost_pct=None, tokens_pct=None)` | per-turn metrics stay within N% of the session average |
| | `no_repeated_output(window=3)` | agent doesn't repeat identical outputs |
| | `tool_error_rate_under(max_rate=0.3, window=10, min_calls=3)` | rolling tool-failure fraction stays under a ceiling |
| **Flow** | `flow_progression(stages, mode="diagnostic"\|"gate", …)` | run reaches ordered milestones (drop-off report) or is gated against out-of-order phases |

Custom predicates are a small function — register one with `@predicate("my_check")` returning a `(event, state) -> PredicateResult` checker.

---

## YAML contracts

Contracts can be declared as data and loaded with `Contract.from_yaml(...)`:

```yaml
# contracts/support_agent.yaml
name: support_agent
version: "1.0"
description: Customer support agent
on_fail: block

clauses:
  - require: cost_under
    args: { max_usd: 0.50 }
  - require: must_call
    args: { tool: lookup_order }
  - require: max_turns
    args: { n: 20 }
  - forbid: must_not_call
    args: { tool: delete_account }
    on_fail: block
  - require: no_pii
    severity: warning
    on_fail: warn
```

```python
from pactrun import Contract
contract = Contract.from_yaml("contracts/support_agent.yaml")
```

Each clause names a predicate (`require` / `forbid` / `precondition` / `postcondition`), its `args`, and optionally `severity`, `on_fail`, and `check_on`.

---

## CLI

Installing pactrun adds a `pactrun` command:

```bash
pactrun init --name support_agent      # scaffold contracts/support_agent.yaml
pactrun validate contracts/            # validate one file or a whole directory
pactrun show contracts/support_agent.yaml   # pretty-print a contract's clauses
pactrun predicates                     # list the 38 built-in predicates
```

```text
$ pactrun show contracts/support_agent.yaml
support_agent  v1.0
default on_fail: block

┏━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┓
┃ kind    ┃ predicate     ┃ check_on    ┃ severity ┃ on_fail ┃
┡━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━┩
│ require │ cost_under    │ every_event │ error    │ block   │
│ forbid  │ must_not_call │ every_event │ critical │ block   │
└─────────┴───────────────┴─────────────┴──────────┴─────────┘
```

---

## How pactrun fits

pactrun is intentionally small, dependency-light, and framework-agnostic. It is **complementary to** — not a replacement for — your agent framework and observability stack.

| Tool | Focus | Scope |
|---|---|---|
| NeMo Guardrails | dialog flows / topical rails (Colang) | per message |
| Guardrails AI | input/output validation, structured output | per message |
| Microsoft Agent Governance Toolkit | enterprise governance, policy, drift, compliance evidence | per call + platform |
| LangGraph / LangSmith | durable agent state + tracing / evals | session state + observability |
| **pactrun** | session-accumulated limits (cost / turns / tool-order / loops) + statistical drift, as a tiny `@contract` | whole session, any provider |

*Star counts and capabilities of other projects change quickly; check their repos for current status. pactrun does not claim to be the only tool that does drift or session-level work — several of the above do parts of it. What pactrun offers is a single, declarative, framework-agnostic contract over an agent's entire run, with no heavy platform to adopt.*

---

## Roadmap

Planned, **not yet implemented** (tracked in `docs/IMPLEMENTATION_PLAN.md`):

- **More adapters** — Pydantic AI, and native CrewAI tool-event integration (today: OpenAI, Anthropic, Gemini, LangChain/LangGraph, LiteLLM/CrewAI, manual).
- **Compliance export** — mapping contract specs to EU AI Act Annex IV / OWASP Agentic Top-10 evidence. (This produces *machine-readable inputs* to a technical file, not a complete compliance package.)
- **Formal composition** — provable composition of contracts across multi-agent pipelines. This is a research direction, not a current feature.

Contributions toward any of these are very welcome.

---

## Research background

pactrun is an independent implementation informed by recent work on agent behavioral specification and runtime enforcement. These papers shaped the design; pactrun is **not** an official implementation of any of them, and the ideas it borrows (e.g. probabilistic `(p, δ, k)`-satisfaction, formal composition) are partly on the [Roadmap](#roadmap) rather than shipped.

| Paper | Venue | Note |
|-------|-------|------|
| [Agent Behavioral Contracts (ABC)](https://arxiv.org/abs/2602.22302) | arXiv preprint, Feb 2026 | Maps Design-by-Contract to agents; defines `(p, δ, k)`-satisfaction. Its AgentAssert prototype reports detecting **5.2–6.8 soft violations per session** that uncontracted baselines miss. |
| [AgentSpec](https://arxiv.org/abs/2503.18666) | **ICSE 2026** (peer-reviewed) | DSL for runtime constraints; >90% prevention, ~95.56% precision for auto-generated rules. |
| [Agent Contracts (Resource-Bounded)](https://arxiv.org/abs/2601.08815) | **COINE / AAMAS 2026 workshop** | Unifies resource, temporal, and quality governance for multi-agent delegation. |
| [Pro2Guard](https://arxiv.org/abs/2508.00500) | arXiv preprint, Aug 2025 | Predictive enforcement via DTMCs (later revised as "ProbGuard"). |
| [Agent-C](https://arxiv.org/abs/2512.23738) | arXiv preprint, Dec 2025 | Temporal safety constraints via SMT solving. |
| [Runtime Governance: Policies on Paths](https://arxiv.org/abs/2603.16586) | arXiv preprint, Mar 2026 | Treats the execution path as a governance object. |
| [DbC Neurosymbolic Layer](https://arxiv.org/abs/2508.03665) | arXiv preprint, Aug 2025 | A contract layer mediating LLM calls with probabilistic remediation. |

*Most of the above are preprints; AgentSpec and Agent Contracts are peer-reviewed. Quantitative figures are the original authors'.*

---

## Testing your agents (pytest plugin)

Installing pactrun registers a pytest plugin. Mark a test with a contract and it runs under enforcement — `block` violations fail the test as they happen, and any other recorded violation fails it at the end with a clear message:

```python
import pytest
from pactrun import Contract, cost_under, must_not_call

budget = (
    Contract("support_agent")
    .require(cost_under(0.50))
    .forbid(must_not_call("delete_account"))
)

@pytest.mark.contracted(budget)
def test_support_agent(pact_session):
    run_my_agent(pact_session)        # emit events via the active session / an adapter
    # no assert needed — the contract is checked automatically
```

The `pact_session` fixture gives you the active `Session` to emit into (or let an adapter do it). At the end of the run you get a one-line summary: `pactrun: N contracted test(s), M with violations`. This is the same `Contract` object you enforce in production — write it once, test offline and enforce online.

---

## Relationship to evalcraft

pactrun is the **runtime-enforcement** companion to [evalcraft](https://github.com/beyhangl/evalcraft) (the **testing** companion):

| | evalcraft | pactrun |
|---|---|---|
| When | Post-hoc (after the run) | Real-time (during the run) |
| Question | "Did the agent behave correctly?" | "Is the agent behaving correctly right now?" |
| How | Cassette replay + assertions | Contract enforcement + drift detection |

They share design patterns (`contextvars`-based session tracking, the same dependency stack) and are intended to converge on a single contract artifact you can both test offline and enforce online.

---

## Contributing

```bash
git clone https://github.com/beyhangl/pactrun
cd agentpact
pip install -e ".[dev]"
pytest        # 450 tests
```

PRs welcome — please open an issue first for significant changes.

---

## License

MIT © 2026 Beyhan Gul. See [LICENSE](LICENSE).
