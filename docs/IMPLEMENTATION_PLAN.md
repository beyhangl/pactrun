# pactrun — Implementation Plan

## Timeline

| Phase | Week | What | LOC | Tests | Cumulative |
|-------|------|------|-----|-------|------------|
| 1. Foundation | 1 | Core models, Contract builder, Session, YAML loader | 1,223 | 55 | 55 |
| 2. Predicates | 1-2 | 20 built-in predicates (cost, tools, output, timing, behavioral) | 860 | 85 | 140 |
| **MVP v0.1.0** | | **Ship after Phase 2** | **2,083** | **140** | |
| 3. Adapters | 2 | OpenAI, Anthropic, LangGraph, Pydantic AI, CrewAI | 1,140 | 60 | 200 |
| 4. Drift | 2-3 | Per-turn + cross-session drift detection (EWMA, z-score, window) | 720 | 40 | 240 |
| 5. Recovery | 3 | 6 strategies (log, warn, block, escalate, retry, fallback) | 650 | 35 | 275 |
| 6. Compliance | 3-4 | EU AI Act Annex IV + OWASP Agentic Top 10 mapping | 1,095 | 30 | 305 |
| 7. CLI | 4 | init, validate, report, doctor commands | 645 | 35 | 340 |
| 8. pytest Plugin | 4 | Markers, fixtures, terminal summary | 355 | 25 | 365 |
| **Full v0.3.0** | | | **~6,700** | **365** | |

## MVP Definition (v0.1.0) — Ship after Phase 2

Foundation + 20 built-in predicates. Users can:
- Define contracts in Python (`Contract.require()/.forbid()`) or YAML
- Enforce at runtime via `contract.session()` context manager or `@contract.enforce` decorator
- Use 20 predicates out of the box (cost, tools, output, timing, behavioral)
- Get violations with clause, event, timestamp, message

## Phase 1: Foundation (Week 1)

### Files
```
pactrun/
├── __init__.py              # Public API exports
├── contract.py              # Contract class (fluent builder + YAML)
├── session.py               # Session runtime context manager
├── loader.py                # YAML/dict contract loader
├── core/
│   ├── models.py            # Contract, Clause, Event, Violation dataclasses
│   ├── enums.py             # Severity, OnFail, EventKind, ClauseKind
│   └── errors.py            # ViolationError, ContractLoadError
└── predicates/
    ├── __init__.py           # Predicate registry
    └── base.py               # PredicateResult, @predicate decorator
```

### Target API
```python
from pactrun import Contract, cost_under, must_not_call

contract = (
    Contract("support_agent")
    .require(cost_under(0.10))
    .forbid(must_not_call("delete_user"))
    .on_violation("block")
)

with contract.session() as session:
    session.emit_llm_response(model="gpt-4.1", output="Hello", cost=0.003)
    session.emit_tool_call("search", args={"q": "help"})

assert session.is_compliant
```

### Acceptance Criteria
- `pip install -e .` works
- Contract builder with `.require()` / `.forbid()` / `.on_violation()`
- YAML loading with `Contract.from_yaml()`
- Session context manager (sync + async)
- `@contract.enforce` decorator
- Violation detection with clause, event, timestamp, message
- 55 tests passing

## Phase 2: Built-in Predicates (Week 1-2)

### Predicate Catalog (20 predicates)

**Cost (3):** `cost_under`, `cost_per_turn_under`, `cost_per_tool_under`

**Tools (5):** `must_call`, `must_not_call`, `tool_order`, `tools_allowed`, `tool_args_match`

**Output (5):** `no_pii`, `matches_schema`, `output_contains`, `output_matches`, `max_output_length`

**Timing (3):** `max_latency`, `session_timeout`, `max_turns`

**Behavioral (4):** `no_loops`, `max_retries`, `drift_bounds`, `no_repeated_output`

### Acceptance Criteria
- All 20 predicates work in real-time (per-event) and post-hoc (session-end)
- Composable: `contract.require(cost_under(0.10)).require(must_call("search"))`
- YAML loading resolves predicate names from registry
- 85 new tests (140 total)

## Phase 3: Framework Adapters (Week 2)

Auto-instrument OpenAI, Anthropic, LangGraph, Pydantic AI, CrewAI via context managers that monkey-patch SDK methods (same pattern as evalcraft).

```python
with contract.session() as session:
    with OpenAIAdapter():
        response = client.chat.completions.create(...)
    # Events automatically emitted to session
```

60 new tests (200 total).

## Phase 4: Drift Detection (Week 2-3)

Per-turn and cross-session behavioral drift using EWMA, z-score, and sliding window strategies. Detects: cost creep, latency creep, token inflation, tool pattern divergence.

40 new tests (240 total).

## Phase 5: Recovery System (Week 3)

6 built-in strategies: log, warn, block, escalate (webhook), retry (with backoff), fallback (to safe agent). Per-clause and per-severity routing.

35 new tests (275 total).

## Phase 6: Compliance & Reporting (Week 3-4)

EU AI Act Annex IV documentation generator. OWASP Agentic Top 10 coverage mapping. Markdown + JSON report output.

30 new tests (305 total).

## Phase 7: CLI (Week 4)

`pactrun init`, `pactrun validate`, `pactrun report`, `pactrun doctor`.

35 new tests (340 total).

## Phase 8: pytest Plugin (Week 4)

`@pytest.mark.contracted`, `session` fixture, terminal summary.

25 new tests (365 total).

## What to SKIP in v0.1.0

- PDF reports (Markdown/JSON sufficient)
- LLM-judge predicates (requires API keys at runtime)
- Dashboard/web UI
- Custom policy DSL (YAML + Python sufficient)
- Formal verification
- Multi-agent contract composition
- Streaming enforcement

## Riskiest Technical Challenges

1. **Adapter monkey-patching** — SDKs change internal APIs. Mitigate with defensive patching + version matrix.
2. **Enforcement latency** — Predicates on every event. Mitigate with pure functions + optional async evaluation.
3. **Drift false positives** — Small samples are noisy. Mitigate with configurable thresholds + minimum sample size.
4. **Thread safety** — Use `contextvars.ContextVar` (same as evalcraft).
