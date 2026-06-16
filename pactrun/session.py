"""Session — runtime enforcement context manager.

A Session binds a Contract to a running agent, tracks cumulative state,
evaluates clauses per-event, and logs violations.

Usage::

    with contract.session() as session:
        session.emit_llm_response(model="gpt-4.1", output="Hello", cost=0.003)
        session.emit_tool_call("search", args={"q": "help"})

    assert session.is_compliant
"""

from __future__ import annotations

import contextvars
import time
import uuid
from typing import Any

from pactrun.core.enums import ClauseKind, EventKind, Severity
from pactrun.core.models import (
    Clause,
    Event,
    PredicateResult,
    SessionState,
    SessionSummary,
    Violation,
)
from pactrun.recovery.engine import apply_recovery


# Context variable for the active session
_active_session: contextvars.ContextVar["Session | None"] = contextvars.ContextVar(
    "pactrun_session", default=None
)


def get_active_session() -> "Session | None":
    """Get the currently active enforcement session, if any."""
    return _active_session.get()


class Session:
    """Runtime enforcement context that evaluates contract clauses per-event.

    Tracks cumulative state (cost, tokens, tool history, turn count),
    evaluates clauses at the right moments, and records violations.
    """

    def __init__(self, contract: Any, **kwargs: Any) -> None:
        from pactrun.contract import Contract
        self._contract: Contract = contract
        self._session_id = str(uuid.uuid4())
        self._started_at: float = 0.0
        self._ended_at: float = 0.0
        self._active = False
        self._state = SessionState()
        self._violations: list[Violation] = []
        self._token: contextvars.Token | None = None
        self._metadata = kwargs.get("metadata", {})
        # Optional handler invoked for `escalate`-action violations. Falls back
        # to one configured on the contract via Contract.on_escalate(...).
        self._escalation_handler = kwargs.get("escalation_handler") or getattr(
            contract, "escalation_handler", None
        )
        # Optional observers (e.g. the OTel span emitter) — pure consumers of
        # events and violations. No-op when none are registered.
        self._observers = list(kwargs.get("observers") or [])

    # -- Properties --------------------------------------------------------

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def violations(self) -> list[Violation]:
        return list(self._violations)

    @property
    def is_compliant(self) -> bool:
        """True if no ERROR or CRITICAL violations."""
        return not any(
            v.severity in (Severity.ERROR, Severity.CRITICAL)
            for v in self._violations
        )

    @property
    def state(self) -> SessionState:
        return self._state

    @property
    def violation_count(self) -> int:
        return len(self._violations)

    # -- Context manager ---------------------------------------------------

    def __enter__(self) -> Session:
        self._start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._end()

    async def __aenter__(self) -> Session:
        self._start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._end()

    # -- Lifecycle ---------------------------------------------------------

    def _start(self) -> None:
        """Begin the session. Evaluates preconditions."""
        self._started_at = time.time()
        self._active = True
        self._token = _active_session.set(self)

        # Check preconditions
        dummy_event = Event(kind=EventKind.INPUT)
        for clause in self._contract.get_clauses(check_on="session_start"):
            result = clause.evaluate(dummy_event, self._state)
            if not result.passed:
                self._record_violation(clause, dummy_event, result)

    def _end(self) -> None:
        """End the session. Evaluates postconditions."""
        self._ended_at = time.time()
        self._state.elapsed_ms = (self._ended_at - self._started_at) * 1000

        # Check postconditions and session-end clauses
        dummy_event = Event(kind=EventKind.OUTPUT)
        for clause in self._contract.get_clauses(check_on="session_end"):
            result = clause.evaluate(dummy_event, self._state)
            if not result.passed:
                self._record_violation(clause, dummy_event, result)

        self._active = False
        if self._token is not None:
            _active_session.reset(self._token)
            self._token = None

    # -- Event emission (user-facing API) ----------------------------------

    def emit_llm_response(
        self,
        model: str,
        output: str,
        *,
        input: Any = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost: float = 0.0,
        duration_ms: float = 0.0,
        metadata: dict | None = None,
    ) -> list[Violation]:
        """Record an LLM response event and evaluate applicable clauses."""
        event = Event(
            kind=EventKind.LLM_CALL,
            model=model,
            input=input,
            output=output,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
            duration_ms=duration_ms,
            metadata=metadata or {},
        )
        return self.record_event(event)

    def emit_tool_call(
        self,
        tool_name: str,
        *,
        args: dict | None = None,
        result: Any = None,
        duration_ms: float = 0.0,
        error: str | None = None,
        metadata: dict | None = None,
    ) -> list[Violation]:
        """Record a tool call event and evaluate applicable clauses."""
        event = Event(
            kind=EventKind.TOOL_CALL,
            tool_name=tool_name,
            tool_args=args,
            tool_result=result,
            duration_ms=duration_ms,
            error=error,
            metadata=metadata or {},
        )
        return self.record_event(event)

    def emit_output(self, text: str) -> list[Violation]:
        """Record the final agent output."""
        event = Event(kind=EventKind.OUTPUT, output=text)
        return self.record_event(event)

    def advance_turn(self) -> list[Violation]:
        """Advance the turn counter."""
        self._state.turn_number += 1
        event = Event(kind=EventKind.TURN_END)
        return self.record_event(event)

    # -- Core event processing ---------------------------------------------

    def record_event(self, event: Event) -> list[Violation]:
        """Record an event, update state, evaluate clauses.

        This is the heart of the enforcement engine.
        """
        # Update cumulative state
        self._update_state(event)
        self._state.events.append(event)

        for observer in self._observers:
            observer.on_event(event, self._state)

        # Evaluate applicable clauses
        violations: list[Violation] = []
        try:
            for clause in self._contract.get_clauses(check_on="every_event"):
                result = clause.evaluate(event, self._state)
                if not result.passed:
                    v = self._record_violation(clause, event, result)
                    violations.append(v)
        finally:
            for observer in self._observers:
                end = getattr(observer, "on_event_end", None)
                if end is not None:
                    end(event)

        return violations

    # -- Internal ----------------------------------------------------------

    def _update_state(self, event: Event) -> None:
        """Update cumulative session state from an event."""
        if event.kind == EventKind.LLM_CALL:
            self._state.total_cost_usd += event.cost_usd
            tokens = event.prompt_tokens + event.completion_tokens
            self._state.total_tokens += tokens
            self._state.total_llm_calls += 1
            if event.output:
                self._state.output_history.append(str(event.output))
            self._state.cost_per_turn.append(event.cost_usd)
            self._state.tokens_per_turn.append(tokens)

        elif event.kind == EventKind.TOOL_CALL:
            self._state.total_tool_calls += 1
            if event.tool_name:
                self._state.tool_call_history.append(event.tool_name)

        elif event.kind == EventKind.OUTPUT:
            if event.output:
                self._state.output_history.append(str(event.output))

        # Update elapsed time
        if self._started_at:
            self._state.elapsed_ms = (time.time() - self._started_at) * 1000

    def _record_violation(
        self,
        clause: Clause,
        event: Event,
        result: PredicateResult,
    ) -> Violation:
        """Create and store a violation record."""
        violation = Violation(
            clause_id=clause.id,
            clause_description=clause.description,
            kind=clause.kind,
            severity=clause.severity,
            on_fail=clause.on_fail,
            timestamp=time.time(),
            turn_number=self._state.turn_number,
            event_id=event.id,
            message=result.message or f"Clause violated: {clause.description}",
            expected=result.expected,
            actual=result.actual,
            context_snapshot=self._state.to_dict(),
        )
        self._violations.append(violation)

        for observer in self._observers:
            notify = getattr(observer, "on_violation", None)
            if notify is not None:
                notify(violation, event)

        # Route to the recovery action (log / warn / block / escalate / retry /
        # fallback). Halting and control-flow actions raise; log/warn return.
        apply_recovery(violation, escalation_handler=self._escalation_handler)

        return violation

    # -- Summary -----------------------------------------------------------

    def summary(self) -> SessionSummary:
        """Generate a complete session summary."""
        return SessionSummary(
            session_id=self._session_id,
            contract_name=self._contract.name,
            contract_version=self._contract.version,
            started_at=self._started_at,
            ended_at=self._ended_at or time.time(),
            duration_ms=self._state.elapsed_ms,
            turn_count=self._state.turn_number,
            total_cost_usd=self._state.total_cost_usd,
            total_tokens=self._state.total_tokens,
            total_tool_calls=self._state.total_tool_calls,
            total_llm_calls=self._state.total_llm_calls,
            tool_call_history=list(self._state.tool_call_history),
            violations=list(self._violations),
            is_compliant=self.is_compliant,
            metadata=self._metadata,
        )
