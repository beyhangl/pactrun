"""The pactrun Agent Cost & Safety Index — a reproducible cost/safety benchmark.

Runs a FIXED agentic workload across providers using pactrun's own
instrumentation and reports, per provider/model: LLM calls, tool calls, tokens,
total cost, cost-per-call, and whether pactrun flagged a tool loop or cost
drift. The workload is identical everywhere, so the numbers are comparable.

    # keyless — illustrative numbers via a local stand-in client
    python scripts/cost_safety_index.py

    # real — install the SDKs and set the keys
    OPENAI_API_KEY=... ANTHROPIC_API_KEY=... \
      python scripts/cost_safety_index.py --providers openai:gpt-4.1,anthropic:claude-sonnet-4-6,mock

    # machine-readable
    python scripts/cost_safety_index.py --json index.json

This is the harness behind the public "Agent Cost & Safety Index". Re-run it on
every model/price/framework change and publish the table.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass

import pactrun

# A fixed, sizeable prompt so per-call cost is comparable across providers.
FIXED_PROMPT = "Summarize the following in one sentence:\n" + ("context line. " * 60)
STEPS = 6  # number of model calls in the workload

_DEFAULT_MODEL = {
    "openai": "gpt-4.1",
    "anthropic": "claude-sonnet-4-6",
    "mock": "mock-model",
}


@dataclass
class Row:
    provider: str
    model: str
    llm_calls: int
    tool_calls: int
    total_tokens: int
    cost_usd: float
    cost_per_call: float
    loop_flagged: bool
    drift_flagged: bool


# ---------------------------------------------------------------------------
# A keyless local stand-in (OpenAI-shaped), scripted to loop + drift so the
# harness demonstrates pactrun's safety detection without any network.
# ---------------------------------------------------------------------------
class _Usage:
    def __init__(self, p, c):
        self.prompt_tokens, self.completion_tokens = p, c


class _Fn:
    def __init__(self, name):
        self.name, self.arguments = name, "{}"


class _ToolCall:
    def __init__(self, name):
        self.function = _Fn(name)


class _Message:
    def __init__(self, tools):
        self.content = "ok"
        self.tool_calls = [_ToolCall(t) for t in tools]


class _Choice:
    def __init__(self, message):
        self.message = message


class _Response:
    def __init__(self, prompt_tokens, completion_tokens, tools, model):
        self.model = model
        self.usage = _Usage(prompt_tokens, completion_tokens)
        self.choices = [_Choice(_Message(tools))]


class _Completions:
    def __init__(self):
        self.step = 0

    def create(self, **kwargs):
        i = self.step
        self.step += 1
        # escalating completion tokens (=> cost drift) + a repeated tool (=> loop)
        return _Response(120, 80 + i * 90, ["search"], kwargs.get("model", "mock-model"))


class _Chat:
    def __init__(self, completions):
        self.completions = completions


class MockClient:
    def __init__(self):
        self.chat = _Chat(_Completions())


def make_client(provider: str):
    if provider == "mock":
        return MockClient()
    if provider == "openai":
        import openai

        return openai.OpenAI()
    if provider == "anthropic":
        import anthropic

        return anthropic.Anthropic()
    raise ValueError(f"unknown provider '{provider}' (try mock / openai / anthropic)")


def run_workload(client, model: str) -> Row:
    """Run the fixed workload through a measuring (non-blocking) wrap()."""
    guarded = pactrun.wrap(
        client,
        max_cost="$1000",      # high cap: we measure, we don't block
        no_loops=True,
        max_drift=0.5,
        on_violation="log",    # record loop/drift, don't raise
        default_max_tokens=256,
    )
    messages = [{"role": "user", "content": FIXED_PROMPT}]
    for _ in range(STEPS):
        if hasattr(guarded, "chat"):
            guarded.chat.completions.create(model=model, messages=messages, max_tokens=256)
        else:
            guarded.messages.create(model=model, messages=messages, max_tokens=256)

    state = guarded.session.state
    violations = guarded.session.violations
    calls = state.total_llm_calls or 1
    return Row(
        provider="",
        model=model,
        llm_calls=state.total_llm_calls,
        tool_calls=state.total_tool_calls,
        total_tokens=state.total_tokens,
        cost_usd=round(state.total_cost_usd, 6),
        cost_per_call=round(state.total_cost_usd / calls, 6),
        loop_flagged=any("loop" in v.message.lower() for v in violations),
        drift_flagged=any("drift" in v.message.lower() for v in violations),
    )


def render_markdown(rows: list[Row]) -> str:
    header = (
        "| Provider | Model | LLM calls | Tool calls | Tokens | Cost (USD) | $/call | Loop | Drift |\n"
        "|---|---|--:|--:|--:|--:|--:|:--:|:--:|"
    )
    body = "\n".join(
        f"| {r.provider} | {r.model} | {r.llm_calls} | {r.tool_calls} | {r.total_tokens} | "
        f"${r.cost_usd:.6f} | ${r.cost_per_call:.6f} | {'⚠️' if r.loop_flagged else '—'} | "
        f"{'⚠️' if r.drift_flagged else '—'} |"
        for r in rows
    )
    return header + "\n" + body


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="The pactrun Agent Cost & Safety Index.")
    parser.add_argument(
        "--providers", default="mock",
        help="comma-separated provider[:model] specs, e.g. openai:gpt-4.1,anthropic,mock",
    )
    parser.add_argument("--json", dest="json_out", help="also write the rows as JSON to this path")
    args = parser.parse_args(argv)

    rows: list[Row] = []
    for spec in (s.strip() for s in args.providers.split(",") if s.strip()):
        provider, _, model = spec.partition(":")
        model = model or _DEFAULT_MODEL.get(provider, "model")
        try:
            row = run_workload(make_client(provider), model)
            row.provider = provider
            rows.append(row)
        except Exception as exc:  # noqa: BLE001 - report and continue per provider
            print(f"# skipped {provider}: {type(exc).__name__}: {exc}", file=sys.stderr)

    print("# pactrun Agent Cost & Safety Index")
    print(f"# fixed workload: {STEPS} model calls, prompt ~{len(FIXED_PROMPT)} chars\n")
    print(render_markdown(rows))
    if any(r.provider == "mock" for r in rows):
        print("\n# note: the 'mock' row uses a local stand-in (illustrative). Real rows need API keys.")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump([asdict(r) for r in rows], f, indent=2)
        print(f"\n# wrote {args.json_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
