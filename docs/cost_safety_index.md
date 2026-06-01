# The pactrun Agent Cost & Safety Index

A reproducible benchmark of the **cost and safety profile** of an agentic
workload across providers and models — measured with pactrun's own
instrumentation. The workload is identical everywhere, so the numbers are
comparable.

Harness: [`scripts/cost_safety_index.py`](../scripts/cost_safety_index.py).

## What it measures

A **fixed workload** — a set number of model calls with an identical, sizeable
prompt — run through a non-blocking `pactrun.wrap()`. For each provider/model it
reports:

| Column | Meaning |
|---|---|
| LLM calls | model calls made |
| Tool calls | tool/function calls the model requested |
| Tokens | total prompt + completion tokens |
| Cost (USD) | total cost (from real usage, priced by pactrun) |
| $/call | average cost per model call |
| Loop | did pactrun flag a repeating tool loop? |
| Drift | did pactrun flag cost-per-turn drift? |

## Run it

Keyless — illustrative numbers via a local stand-in client:

```bash
python scripts/cost_safety_index.py
```

Real — install the SDKs and set the keys:

```bash
pip install openai anthropic
OPENAI_API_KEY=... ANTHROPIC_API_KEY=... \
  python scripts/cost_safety_index.py --providers openai:gpt-4.1,anthropic:claude-sonnet-4-6,mock

# machine-readable
python scripts/cost_safety_index.py --providers openai,anthropic --json index.json
```

## Illustrative output (mock runner — not real prices)

```
| Provider | Model      | LLM calls | Tool calls | Tokens | Cost (USD) | $/call    | Loop | Drift |
|----------|------------|----------:|-----------:|-------:|-----------:|----------:|:----:|:-----:|
| mock     | mock-model |         6 |          6 |   2550 | $0.020100  | $0.003350 |  ⚠️  |  ⚠️   |
```

> The `mock` row uses a scripted local client (escalating tokens + a repeated
> tool) to demonstrate the loop/drift detection without a network. **Real rows
> require API keys** and reflect each provider's real pricing.

## Why this exists

No authoritative source publishes "agent cost & safety overhead" numbers across
providers. pactrun's predicates already measure exactly this, so the project can
own the open benchmark — and re-run it on every model / price / framework change.
