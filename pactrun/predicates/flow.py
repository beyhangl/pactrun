"""Flow predicates — track an agent through ordered stages toward a goal.

``tool_order`` checks tool *names* in order at session end; ``flow_progression``
generalizes that into a small stage machine:

- ``mode="diagnostic"`` (session end) reports **how far** a run got — the
  furthest milestone reached, ``k/N`` — so you can see *where* runs drop off,
  not just that they failed.
- ``mode="gate"`` (per event) pins the run to its current phase and **blocks a
  later-stage marker the moment it fires** before the prior stages are done
  (and before its optional enter-condition holds).

A stage is a tool name, an output marker (substring), or a
``Callable[[Event], bool]``.
"""

from __future__ import annotations

import uuid

from pactrun.core.models import Event, PredicateResult, SessionState
from pactrun.predicates.base import predicate


def _stage_label(stage) -> str:
    return stage if isinstance(stage, str) else getattr(stage, "__name__", "stage")


def _matches(stage, event: Event, match) -> bool:
    if callable(stage):
        try:
            return bool(stage(event))
        except Exception:  # noqa: BLE001 - a bad matcher never advances the flow
            return False
    if match in (None, "tool") and event.tool_name == stage:
        return True
    if match in (None, "output"):
        text = str(event.output or "")
        if text and stage in text:
            return True
    return False


@predicate("flow_progression")
def flow_progression(
    stages: list,
    mode: str = "diagnostic",
    terminal=None,
    enter: dict | None = None,
    allow_repeats: bool = True,
    match: str | None = None,
):
    """Track ordered ``stages`` toward a ``terminal`` goal (default ``stages[-1]``).

    ``mode="diagnostic"`` (session end): pass iff the run reached the terminal
    stage in order; on failure ``actual`` reports the furthest stage as
    ``reached_stage=<name> (k/N)``.

    ``mode="gate"`` (per event): advance a per-run ledger as each next stage is
    reached; fail when an event matches a *later* stage before the current one
    is satisfied, or when a stage's ``enter`` condition
    (``Callable[[Event, SessionState], bool]``) is not met.
    """
    if not stages:
        raise ValueError("flow_progression: stages must be non-empty")
    if mode not in ("diagnostic", "gate"):
        raise ValueError(f"flow_progression: mode must be diagnostic/gate, got {mode!r}")
    n = len(stages)
    terminal_idx = (n - 1) if terminal is None else None
    if terminal is not None:
        for i, s in enumerate(stages):
            if s == terminal or _stage_label(s) == _stage_label(terminal):
                terminal_idx = i
                break
        if terminal_idx is None:
            raise ValueError("flow_progression: terminal must be one of stages")
    term_label = _stage_label(stages[terminal_idx])
    ledger_key = f"_flow_progression_{uuid.uuid4().hex[:8]}"

    def _diagnostic(event: Event, state: SessionState) -> PredicateResult:
        p = 0
        for e in state.events:
            if p < n and _matches(stages[p], e, match):
                p += 1
        reached_label = _stage_label(stages[p - 1]) if p > 0 else "none"
        return PredicateResult(
            passed=p >= terminal_idx + 1,
            expected=f"reach terminal stage '{term_label}'",
            actual=f"reached_stage={reached_label} ({p}/{n})",
            message=f"Flow stopped at {reached_label} ({p}/{n}); needed '{term_label}'",
        )

    def _gate(event: Event, state: SessionState) -> PredicateResult:
        current = state.metadata.get(ledger_key, 0)
        for i, stage in enumerate(stages):
            if not _matches(stage, event, match):
                continue
            if i == current:
                if enter and stage in enter and not _safe_enter(enter[stage], event, state):
                    return PredicateResult(
                        passed=False,
                        expected=f"enter-condition for '{_stage_label(stage)}'",
                        actual="condition not met",
                        message=f"Entered stage '{_stage_label(stage)}' before its enter-condition held",
                    )
                state.metadata[ledger_key] = current + 1
                return PredicateResult(passed=True)
            if i > current:
                return PredicateResult(
                    passed=False,
                    expected=f"stage '{_stage_label(stages[current])}' before '{_stage_label(stage)}'",
                    actual=f"jumped to stage {i} at phase {current}",
                    message=f"Out-of-order: '{_stage_label(stage)}' fired before stage '{_stage_label(stages[current])}'",
                )
            # i < current: a repeat of an already-passed stage
            if not allow_repeats:
                return PredicateResult(
                    passed=False,
                    expected="no stage repeats",
                    actual=f"stage '{_stage_label(stage)}' repeated",
                    message=f"Stage '{_stage_label(stage)}' repeated after the flow advanced past it",
                )
            return PredicateResult(passed=True)
        return PredicateResult(passed=True)

    check = _diagnostic if mode == "diagnostic" else _gate
    check.predicate_name = "flow_progression"  # type: ignore[attr-defined]
    check._check_on = "session_end" if mode == "diagnostic" else "every_event"  # type: ignore[attr-defined]
    return check


def _safe_enter(cond, event: Event, state: SessionState) -> bool:
    try:
        return bool(cond(event, state))
    except Exception:  # noqa: BLE001 - a broken enter-condition fails closed
        return False
