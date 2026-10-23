# Evalcraft cassette bridge

Maintainer: [beyhangl](https://github.com/beyhangl) — beyhangl@gmail.com.

Install both local projects, or install the optional `pactrun[evalcraft]` dependency
when using a published distribution. The bridge imports Evalcraft only when used;
the base Pactrun package does not require it.

```python
from pactrun import (
    Contract, EvalcraftRecorder, cassette_to_events, cost_under, replay_cassette,
)

recorder = EvalcraftRecorder(name="support")
with Contract("support").require(cost_under(0.50)).session(observers=[recorder]) as session:
    session.emit_llm_response(
        "synthetic", "propose refund", cost=0.01,
        prompt_tokens=100, cache_read_tokens=60, completion_tokens=10,
    )
    session.emit_tool_call("refund", args={"amount_minor": 2500})
path = recorder.cassette.save("support.json")
result = replay_cassette(Contract("review").require(cost_under(0.10)), path)
assert result.compliant
```

## Accounting and observed events

Evalcraft's `prompt_tokens` counts fresh input only. Pactrun's counts total input,
including cache reads/writes. The bridge adds the cache subsets on import and
subtracts them on export. Cost is carried as recorded; it is not repriced from a
current model catalog. Missing LLM cost refuses conversion by default.
`missing_cost="zero"` is an explicit caller decision and can weaken budget checks.

LLM request spans do not become extra response events. Tool attempts remain
attempts unless a result is actually observed. A separate tool-result span must
immediately follow its identified call, using `parent_id` or
`metadata["tool_call_id"]`. Result metadata is preserved. Conflicting annotations,
duplicate results/IDs, unknown span kinds and interleaved results are refused:
one combined Pactrun tool event cannot faithfully represent those relationships.

Input, output, error and turn events round-trip. Session turn accounting now
advances on ingestion, so saved TURN_END events retain their original turn counts.

## Stateful scenario cassettes

The optional [refund scenario pack](STATEFUL_SCENARIOS.md) uses Evalcraft's
`run_scenarios` to execute fresh environments against independently authored
expectations. Install both matching source checkouts until releases contain the
new runner and pack; the optional bridge dependency alone can select an older
Evalcraft version without `evalcraft.scenarios`.

The environment factory runs inside `CaptureContext` and attaches its
`EvalcraftRecorder` to the active cassette. This preserves actual Pactrun events
alongside the runner's informational action-attempt and outcome spans. Generic
scenario actions do not become fabricated tool calls.

The bridge skips only strictly recognized version 1 `evalcraft_scenario`
envelopes on `AGENT_STEP` spans. An unknown or malformed envelope remains an
unsupported conversion rather than silently disappearing. Supported observed
SDK and Pactrun spans retain their normal conversion and accounting.

The informational marker is not a verified execution receipt. Likewise, a
Pactrun tool-call event can be an attempted call that was blocked. Assert both
the outcome and authoritative application/provider state in the scenario report
when testing a business effect. Policy replay of the cassette evaluates saved
observations; it neither reruns state transitions nor imports scenario
expectations as evidence of success.

A compliant policy replay can accompany a failed scenario. Check
`ScenarioReport.passed` or `ScenarioSuiteReport.passed` for the result of the
outcome and business-state assertions.

## Policy replay versus changed-code playback

`replay_cassette()` evaluates saved observations in monitor mode with the event
clock. It does not execute tools or rerun changed application code.
Approval checks simulate per-trace nonce use without consuming the live store.

Evalcraft's `playback()` runs application code against recorded provider replies
and checks request matching. The bridge alone does not create provider request/
response payloads needed for playback. Capture a model recording with an Evalcraft
SDK adapter, optionally sending tool events to an `EvalcraftRecorder` attached to
that same cassette. Avoid recording a model call twice through both instruments.

The synthetic [refund example](SUPPORT_REFUND.md) and
`tests/test_support_refund.py` cover a real SDK call through a mock transport,
capture, unchanged-code playback and changed-amount refusal.

## Trust and sensitive data

The bridge copies observed host metadata; it does not invent approvals, verified
identities, trust labels or successful external effects. A cassette is not proof
that a recording producer told the truth. Inspect coverage and source provenance.

The recorder retains metadata, which may include approval tokens and customer
data. Redact and review recordings before sharing. This observer does not
automatically transmit or upload them.
