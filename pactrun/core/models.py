"""Core data models for pactrun."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from pactrun.core.enums import ClauseKind, EventKind, OnFail, Severity


# ---------------------------------------------------------------------------
# Event — something that happened during an agent session
# ---------------------------------------------------------------------------

@dataclass
class Event:
    """A single event in an agent session."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    kind: EventKind = EventKind.LLM_CALL
    timestamp: float = field(default_factory=time.time)

    # LLM call fields
    model: str | None = None
    input: Any = None
    output: Any = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    duration_ms: float = 0.0

    # Tool call fields
    tool_name: str | None = None
    tool_args: dict | None = None
    tool_result: Any = None

    # Error fields
    error: str | None = None

    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind.value,
            "timestamp": self.timestamp,
            "model": self.model,
            "input": self.input,
            "output": self.output,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": self.cost_usd,
            "duration_ms": self.duration_ms,
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "tool_result": self.tool_result,
            "error": self.error,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Event:
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            kind=EventKind(data.get("kind", "llm_call")),
            timestamp=data.get("timestamp", time.time()),
            model=data.get("model"),
            input=data.get("input"),
            output=data.get("output"),
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
            cost_usd=data.get("cost_usd", 0.0),
            duration_ms=data.get("duration_ms", 0.0),
            tool_name=data.get("tool_name"),
            tool_args=data.get("tool_args"),
            tool_result=data.get("tool_result"),
            error=data.get("error"),
            metadata=data.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# SessionState — cumulative state of a running session
# ---------------------------------------------------------------------------

@dataclass
class SessionState:
    """Cumulative state tracked across an agent session.

    Passed to predicate functions so they can make decisions based on
    the full session history, not just the current event.
    """
    turn_number: int = 0
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    total_tool_calls: int = 0
    total_llm_calls: int = 0
    tool_call_history: list[str] = field(default_factory=list)
    output_history: list[str] = field(default_factory=list)
    cost_per_turn: list[float] = field(default_factory=list)
    tokens_per_turn: list[int] = field(default_factory=list)
    elapsed_ms: float = 0.0
    events: list[Event] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "turn_number": self.turn_number,
            "total_cost_usd": self.total_cost_usd,
            "total_tokens": self.total_tokens,
            "total_tool_calls": self.total_tool_calls,
            "total_llm_calls": self.total_llm_calls,
            "tool_call_history": self.tool_call_history,
            "elapsed_ms": self.elapsed_ms,
        }


# ---------------------------------------------------------------------------
# PredicateResult — outcome of evaluating a predicate
# ---------------------------------------------------------------------------

@dataclass
class PredicateResult:
    """Result of evaluating a single predicate."""
    passed: bool = True
    message: str = ""
    expected: Any = None
    actual: Any = None

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "message": self.message,
            "expected": self.expected,
            "actual": self.actual,
        }


# ---------------------------------------------------------------------------
# Clause — a single behavioral requirement
# ---------------------------------------------------------------------------

@dataclass
class Clause:
    """A single behavioral requirement within a contract."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    kind: ClauseKind = ClauseKind.REQUIRE
    predicate: Callable[[Event, SessionState], PredicateResult] | None = None
    predicate_name: str = ""
    description: str = ""
    severity: Severity = Severity.ERROR
    on_fail: OnFail = OnFail.BLOCK
    check_on: str = "every_event"  # "every_event", "session_end", "session_start"
    metadata: dict = field(default_factory=dict)

    def evaluate(self, event: Event, state: SessionState) -> PredicateResult:
        """Evaluate this clause against an event and session state."""
        if self.predicate is None:
            return PredicateResult(passed=True)
        return self.predicate(event, state)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind.value,
            "predicate_name": self.predicate_name,
            "description": self.description,
            "severity": self.severity.value,
            "on_fail": self.on_fail.value,
            "check_on": self.check_on,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Violation — a breach of a clause
# ---------------------------------------------------------------------------

@dataclass
class Violation:
    """Record of a single contract clause breach."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    clause_id: str = ""
    clause_description: str = ""
    kind: ClauseKind = ClauseKind.REQUIRE
    severity: Severity = Severity.ERROR
    on_fail: OnFail = OnFail.BLOCK
    timestamp: float = field(default_factory=time.time)
    turn_number: int = 0
    event_id: str | None = None
    message: str = ""
    expected: Any = None
    actual: Any = None
    context_snapshot: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "clause_id": self.clause_id,
            "clause_description": self.clause_description,
            "kind": self.kind.value,
            "severity": self.severity.value,
            "on_fail": self.on_fail.value,
            "timestamp": self.timestamp,
            "turn_number": self.turn_number,
            "event_id": self.event_id,
            "message": self.message,
            "expected": self.expected,
            "actual": self.actual,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Violation:
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            clause_id=data.get("clause_id", ""),
            clause_description=data.get("clause_description", ""),
            kind=ClauseKind(data.get("kind", "require")),
            severity=Severity(data.get("severity", "error")),
            on_fail=OnFail(data.get("on_fail", "block")),
            timestamp=data.get("timestamp", time.time()),
            turn_number=data.get("turn_number", 0),
            event_id=data.get("event_id"),
            message=data.get("message", ""),
            expected=data.get("expected"),
            actual=data.get("actual"),
        )


# ---------------------------------------------------------------------------
# SessionSummary — final summary of a session
# ---------------------------------------------------------------------------

@dataclass
class SessionSummary:
    """Complete summary of a contract enforcement session."""
    session_id: str = ""
    contract_name: str = ""
    contract_version: str = ""
    started_at: float = 0.0
    ended_at: float = 0.0
    duration_ms: float = 0.0
    turn_count: int = 0
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    total_tool_calls: int = 0
    total_llm_calls: int = 0
    tool_call_history: list[str] = field(default_factory=list)
    violations: list[Violation] = field(default_factory=list)
    is_compliant: bool = True
    metadata: dict = field(default_factory=dict)

    @property
    def violation_count(self) -> int:
        return len(self.violations)

    @property
    def has_critical(self) -> bool:
        return any(v.severity == Severity.CRITICAL for v in self.violations)

    @property
    def has_errors(self) -> bool:
        return any(v.severity in (Severity.ERROR, Severity.CRITICAL) for v in self.violations)

    def violations_by_severity(self, severity: Severity) -> list[Violation]:
        return [v for v in self.violations if v.severity == severity]

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "contract_name": self.contract_name,
            "contract_version": self.contract_version,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_ms": self.duration_ms,
            "turn_count": self.turn_count,
            "total_cost_usd": self.total_cost_usd,
            "total_tokens": self.total_tokens,
            "total_tool_calls": self.total_tool_calls,
            "total_llm_calls": self.total_llm_calls,
            "tool_call_history": self.tool_call_history,
            "violations": [v.to_dict() for v in self.violations],
            "is_compliant": self.is_compliant,
            "violation_count": self.violation_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SessionSummary:
        violations = [Violation.from_dict(v) for v in data.get("violations", [])]
        return cls(
            session_id=data.get("session_id", ""),
            contract_name=data.get("contract_name", ""),
            contract_version=data.get("contract_version", ""),
            started_at=data.get("started_at", 0.0),
            ended_at=data.get("ended_at", 0.0),
            duration_ms=data.get("duration_ms", 0.0),
            turn_count=data.get("turn_count", 0),
            total_cost_usd=data.get("total_cost_usd", 0.0),
            total_tokens=data.get("total_tokens", 0),
            total_tool_calls=data.get("total_tool_calls", 0),
            total_llm_calls=data.get("total_llm_calls", 0),
            tool_call_history=data.get("tool_call_history", []),
            violations=violations,
            is_compliant=data.get("is_compliant", True),
        )
