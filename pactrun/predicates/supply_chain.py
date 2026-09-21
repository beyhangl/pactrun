"""Agentic supply-chain predicates — detect tool definitions changing mid-run.

A tool server the agent already trusts can turn hostile *after* it has earned
that trust: serve benign descriptions while the agent builds reliance, then
swap in instructions that hunt credentials. Auditing a server once at connect
time misses this by construction, because the malicious definition only
appears after the first few calls.

These predicates fingerprint what each tool advertised and fail when it
changes within a single run. The host or adapter supplies the definition it
saw on ``event.metadata[metadata_key]`` — the same convention the taint and
approval predicates use — because the definition is not part of a tool-call
event by default.
"""

from __future__ import annotations

from pactrun.core.enums import EventKind
from pactrun.core.models import Event, PredicateResult, SessionState
from pactrun.predicates.base import predicate


def _fingerprint(value) -> str:
    """Stable digest of an advertised tool definition (str or structured)."""
    import hashlib
    import json

    if isinstance(value, str):
        blob = value
    else:
        try:
            blob = json.dumps(value, sort_keys=True, default=str)
        except (TypeError, ValueError):
            blob = str(value)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@predicate("tool_definitions_stable", owasp=("ASI04", "ASI02"))
def tool_definitions_stable(metadata_key: str = "tool_definition", tools=None):
    """A tool's advertised definition must not change during a run.

    Each tool call may carry what the server advertised for that tool at call
    time in ``event.metadata[metadata_key]`` — a description string, or a
    structured definition (schema + annotations) which is canonicalised before
    hashing. The first definition seen for a tool name in the run becomes the
    baseline; any later call presenting a different fingerprint fails.

    ``tools`` optionally restricts the check to specific tool names. Calls that
    carry no definition are ignored, so this degrades to a no-op rather than
    false-positives when an adapter does not record definitions.

    Detects the "rug-pull" shape where a server mutates what it advertises once
    the agent has come to rely on it. It compares only what the host recorded:
    if an adapter reads tool definitions once and caches them, this predicate
    can only ever see that cached value — the adapter must re-read them for the
    check to mean anything.
    """
    watch = set(tools) if tools else None

    def check(event: Event, state: SessionState) -> PredicateResult:
        if event.kind != EventKind.TOOL_CALL or not event.tool_name:
            return PredicateResult(passed=True)
        if watch is not None and event.tool_name not in watch:
            return PredicateResult(passed=True)
        current = (event.metadata or {}).get(metadata_key)
        if current is None:
            return PredicateResult(passed=True)
        current_fp = _fingerprint(current)

        for prior in state.events:
            if prior.id == event.id:
                continue
            if prior.kind != EventKind.TOOL_CALL or prior.tool_name != event.tool_name:
                continue
            seen = (prior.metadata or {}).get(metadata_key)
            if seen is None:
                continue
            baseline = _fingerprint(seen)
            if baseline != current_fp:
                return PredicateResult(
                    passed=False,
                    expected=f"'{event.tool_name}' definition stable for the run",
                    actual=f"definition changed ({baseline[:12]} -> {current_fp[:12]})",
                    message=(
                        f"Tool '{event.tool_name}' changed its advertised definition mid-run "
                        "— possible tool-server rug-pull"
                    ),
                )
            break  # compare against the first recorded definition only

        return PredicateResult(passed=True)

    check.predicate_name = "tool_definitions_stable"  # type: ignore[attr-defined]
    return check
