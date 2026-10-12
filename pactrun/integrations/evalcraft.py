"""Optional Evalcraft cassette bridge.

Recorded events are evidence of observation, not proof of authorization or an
external side effect. Conversion never creates approval/trust metadata and
does not execute application code; use Evalcraft playback to test changed code.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pactrun.core.enums import EventKind
from pactrun.core.models import Event, SessionState
from pactrun.replay import ReplayResult, replay_trace

if TYPE_CHECKING:
    from evalcraft.core.models import Cassette

    from pactrun.contract import Contract

_BRIDGE_KEY = "pactrun_bridge"


def _models():
    try:
        from evalcraft.core.models import Cassette, Span, SpanKind, TokenUsage
    except ImportError as exc:
        raise ImportError("Install pactrun[evalcraft] to use the cassette bridge.") from exc
    return Cassette, Span, SpanKind, TokenUsage


def cassette_to_events(
    cassette: Cassette | str | Path,
    *,
    missing_cost: Literal["error", "zero"] = "error",
) -> list[Event]:
    """Convert observed spans, preserving cache tiers and stable event identity.

    LLM request spans are not counted as responses. A separately recorded tool
    result must immediately follow and identify its call through parent_id or
    metadata.tool_call_id. Conflicting result annotations and interleaved
    results are refused rather than inventing a causal/trust relationship.
    Unreported LLM cost is an error unless missing_cost="zero" is explicit.
    """
    if missing_cost not in ("error", "zero"):
        raise ValueError("missing_cost must be 'error' or 'zero'")
    CassetteType, _, SpanKind, _ = _models()
    recording: Cassette
    if isinstance(cassette, (str, Path)):
        recording = CassetteType.load(cassette)
    else:
        recording = cassette
    events: list[Event] = []
    tool_calls: dict[str, Event] = {}
    seen_ids: set[str] = set()
    result_calls: set[str] = set()
    for span in recording.spans:
        if span.id in seen_ids:
            raise ValueError(f"Duplicate cassette span id: {span.id!r}")
        seen_ids.add(span.id)
        metadata = copy.deepcopy(span.metadata)
        marker = metadata.pop(_BRIDGE_KEY, {})
        event = Event(
            id=span.id, timestamp=span.timestamp, input=copy.deepcopy(span.input),
            output=copy.deepcopy(span.output), duration_ms=span.duration_ms,
            error=span.error, metadata=metadata,
        )
        if span.kind == SpanKind.LLM_RESPONSE:
            if span.cost_usd is None and missing_cost == "error":
                raise ValueError(f"LLM span {span.id!r} has no recorded cost")
            usage = span.token_usage
            event.kind = EventKind.LLM_CALL
            event.model = span.model
            event.cost_usd = span.cost_usd if span.cost_usd is not None else 0.0
            if usage is not None:
                event.prompt_tokens = (
                    usage.prompt_tokens + usage.cache_read_tokens + usage.cache_write_tokens
                )
                event.completion_tokens = usage.completion_tokens
                event.cache_read_tokens = usage.cache_read_tokens
                event.cache_write_tokens = usage.cache_write_tokens
            event.reasoning_tokens = marker.get("reasoning_tokens", 0)
        elif span.kind == SpanKind.TOOL_CALL:
            event.kind = EventKind.TOOL_CALL
            event.tool_name = span.tool_name
            event.tool_args = copy.deepcopy(span.tool_args)
            event.tool_result = copy.deepcopy(span.tool_result)
            tool_calls[span.id] = event
        elif span.kind == SpanKind.TOOL_RESULT:
            call_id = span.parent_id or metadata.get("tool_call_id")
            if call_id not in tool_calls:
                raise ValueError(f"Tool result {span.id!r} has no preceding identified call")
            call = tool_calls[call_id]
            if call_id in result_calls or call.tool_result is not None or call.error is not None:
                raise ValueError(f"Duplicate tool result for call {call_id!r}")
            result_calls.add(call_id)
            if not events or events[-1] is not call:
                raise ValueError("Interleaved tool results cannot be represented as one tool event")
            for key, value in metadata.items():
                if key == "tool_call_id":
                    continue
                if key in call.metadata and call.metadata[key] != value:
                    raise ValueError(f"Conflicting tool result metadata for {key!r}")
                call.metadata[key] = copy.deepcopy(value)
            call.tool_result = copy.deepcopy(
                span.tool_result if span.tool_result is not None else span.output
            )
            call.error = span.error
            continue
        elif span.kind == SpanKind.USER_INPUT:
            event.kind = EventKind.INPUT
        elif span.kind == SpanKind.AGENT_OUTPUT:
            event.kind = EventKind.OUTPUT
        elif span.kind == SpanKind.AGENT_STEP and marker.get("event_kind") in (
            EventKind.ERROR.value, EventKind.TURN_END.value,
        ):
            event.kind = EventKind(marker["event_kind"])
        elif span.kind == SpanKind.LLM_REQUEST:
            continue
        else:
            raise ValueError(f"Unsupported cassette span kind: {span.kind.value}")
        events.append(event)
    return events


def replay_cassette(
    contract: Contract,
    cassette: Cassette | str | Path,
    *,
    missing_cost: Literal["error", "zero"] = "error",
) -> ReplayResult:
    """Evaluate recorded events in monitor mode; does not rerun changed code."""
    return replay_trace(contract, cassette_to_events(cassette, missing_cost=missing_cost))


class EvalcraftRecorder:
    """Session observer recording Pactrun events into an optional cassette.

    Tool-call events may describe an attempted action before execution.
    Only an explicitly observed result is stored as a result. The bridge does
    not add provider payloads needed for model playback or independently verify
    approval identity. Cassettes can contain sensitive host metadata: redact
    and review before sharing.
    """

    def __init__(self, cassette: Cassette | None = None, *, name: str = "pactrun") -> None:
        Cassette, _, _, _ = _models()
        self.cassette = cassette if cassette is not None else Cassette(name=name, framework="pactrun")

    def on_event(self, event: Event, state: SessionState) -> None:
        _, Span, SpanKind, TokenUsage = _models()
        kinds = {
            EventKind.LLM_CALL: SpanKind.LLM_RESPONSE,
            EventKind.TOOL_CALL: SpanKind.TOOL_CALL,
            EventKind.INPUT: SpanKind.USER_INPUT,
            EventKind.OUTPUT: SpanKind.AGENT_OUTPUT,
            EventKind.ERROR: SpanKind.AGENT_STEP,
            EventKind.TURN_END: SpanKind.AGENT_STEP,
        }
        metadata = copy.deepcopy(event.metadata)
        metadata[_BRIDGE_KEY] = {
            "event_kind": event.kind.value, "reasoning_tokens": event.reasoning_tokens,
        }
        usage = None
        if event.kind == EventKind.LLM_CALL:
            fresh = event.prompt_tokens - event.cache_read_tokens - event.cache_write_tokens
            if fresh < 0:
                raise ValueError("Pactrun input token total is smaller than its cache subsets")
            usage = TokenUsage(
                prompt_tokens=fresh, completion_tokens=event.completion_tokens,
                cache_read_tokens=event.cache_read_tokens,
                cache_write_tokens=event.cache_write_tokens,
                total_tokens=event.prompt_tokens + event.completion_tokens,
            )
        self.cassette.add_span(Span(
            id=event.id, kind=kinds[event.kind], name=event.tool_name or event.kind.value,
            timestamp=event.timestamp, duration_ms=event.duration_ms,
            input=copy.deepcopy(event.input), output=copy.deepcopy(event.output),
            model=event.model, token_usage=usage,
            cost_usd=event.cost_usd if event.kind == EventKind.LLM_CALL else None,
            tool_name=event.tool_name, tool_args=copy.deepcopy(event.tool_args),
            tool_result=copy.deepcopy(event.tool_result), error=event.error, metadata=metadata,
        ))

    def on_session_end(self, state: SessionState) -> None:
        self.cassette.compute_metrics()
        self.cassette.compute_fingerprint()
