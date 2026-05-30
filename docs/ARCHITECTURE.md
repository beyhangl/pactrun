# pactrun Architecture

## Design Philosophy

pactrun is "Design by Contract for AI agents" — inspired by Bertrand Meyer's DbC in Eiffel, not input/output guardrails.

| | Guardrails (NeMo, Guardrails AI) | pactrun |
|---|---|---|
| Scope | Per-message filtering | Session-level behavioral specification |
| State | Stateless | Stateful — tracks cumulative behavior |
| Drift | Not addressed | First-class drift detection across turns |
| Composition | Single agent | Multi-agent contract composition |
| Recovery | Block or allow | Recovery-as-specification |
| Compliance | Not addressed | EU AI Act Annex IV report generation |

## Module Structure

```
pactrun/
├── core/           # Contract, Clause, Violation, Session, Types
├── clauses/        # Built-in clause factories (cost, tools, output, timing, tokens, content, sequence)
├── enforcement/    # Runtime enforcement engine with hook points
├── drift/          # Behavioral drift detection (Page-Hinkley, CUSUM)
├── recovery/       # Recovery strategies (log, warn, block, escalate, retry, pipeline)
├── adapters/       # Framework integrations (OpenAI, Anthropic, LangGraph, Pydantic AI, manual)
├── loaders/        # Contract loading (@decorator, YAML, composition)
├── evalcraft_bridge/ # evalcraft integration (scorer, capture, assertions)
├── compliance/     # EU AI Act Annex IV report generation
├── cli/            # CLI commands (init, validate, check, compliance)
└── pytest_plugin/  # pytest integration (markers, fixtures, terminal summary)
```

## Core Data Model

### Contract = (Clauses, Recovery, Metadata)
- **Clause**: predicate function + kind + phase + severity + recovery
- **ClauseKind**: PRECONDITION, INVARIANT, POSTCONDITION, GOVERNANCE
- **ClausePhase**: SESSION_START, PRE_LLM, POST_LLM, PRE_TOOL, POST_TOOL, EVERY_TURN, SESSION_END

### ContractSession (stateful tracker)
- Binds a Contract to a running agent
- Maintains cumulative state (cost, tokens, tool history, turn count)
- Evaluates clauses at correct phases
- Records violations, triggers recovery
- Feeds DriftMonitor

### EnforcementEngine
- Sits between adapters and session
- Hook points: pre_llm, post_llm, pre_tool, post_tool, on_turn_end
- Modes: ENFORCE (block), MONITOR (record), AUDIT (log), DISABLED

## Data Flow

```
User Agent → @contract decorator or YAML → Contract
    → ContractSession (stateful) ← DriftMonitor
    → EnforcementEngine
    → Adapters / Manual Instrumentation
    → Hook Points (pre_llm → post_llm → pre_tool → post_tool → turn_end)
    → Clause Evaluation (predicate(ctx) → bool)
    → Violation? → Recovery Strategy (log/warn/block/escalate/retry)
    → SessionReport → evalcraft EvalResult / pytest summary / Compliance Report
```

## Academic References

- [Agent Behavioral Contracts (ABC)](https://arxiv.org/abs/2602.22302) — Feb 2026
- [AgentSpec](https://arxiv.org/abs/2503.18666) — ICSE 2026
- [Pro2Guard](https://arxiv.org/abs/2508.00500) — Aug 2025
- [Agent-C](https://arxiv.org/abs/2512.23738) — Dec 2025
- [Policies on Paths](https://arxiv.org/abs/2603.16586) — Mar 2026
