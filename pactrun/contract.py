"""Contract — the core specification object.

A Contract is a named collection of clauses that define behavioral
requirements for an agent. Contracts can be built fluently in Python
or loaded from YAML files.

Usage::

    from pactrun import Contract, cost_under, must_not_call

    contract = (
        Contract("support_agent")
        .require(cost_under(0.10))
        .forbid(must_not_call("delete_user"))
        .on_violation("block")
    )
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from pactrun.core.enums import ClauseKind, OnFail, Severity
from pactrun.core.models import Clause, Event, PredicateResult, SessionState


@dataclass
class Contract:
    """A behavioral specification for an agent.

    Build fluently::

        contract = (
            Contract("my_agent")
            .require(cost_under(0.10))
            .forbid(must_not_call("delete"))
        )

    Or load from YAML::

        contract = Contract.from_yaml("contracts/agent.yaml")
    """
    name: str = ""
    version: str = "1.0"
    description: str = ""
    clauses: list[Clause] = field(default_factory=list)
    default_on_fail: OnFail = OnFail.BLOCK
    metadata: dict = field(default_factory=dict)

    # -- Fluent builder API ------------------------------------------------

    def require(
        self,
        predicate_fn: Callable[[Event, SessionState], PredicateResult],
        *,
        description: str = "",
        severity: Severity = Severity.ERROR,
        on_fail: OnFail | str | None = None,
        check_on: str | None = None,
    ) -> Contract:
        """Add a must-satisfy clause.

        When ``check_on`` is not given explicitly, the predicate's own
        ``_check_on`` hint is honored (e.g. ``must_call`` / ``tool_order`` /
        ``output_contains`` declare ``"session_end"`` because they can only be
        satisfied once the whole session has run). It falls back to
        ``"every_event"`` for predicates that should be checked continuously.
        """
        if isinstance(on_fail, str):
            on_fail = OnFail(on_fail)
        pred_name = getattr(predicate_fn, "predicate_name", "")
        resolved_check_on = check_on or getattr(predicate_fn, "_check_on", None) or "every_event"
        self.clauses.append(Clause(
            kind=ClauseKind.REQUIRE,
            predicate=predicate_fn,
            predicate_name=pred_name,
            description=description or pred_name or "require clause",
            severity=severity,
            on_fail=on_fail or self.default_on_fail,
            check_on=resolved_check_on,
        ))
        return self

    def forbid(
        self,
        predicate_fn: Callable[[Event, SessionState], PredicateResult],
        *,
        description: str = "",
        severity: Severity = Severity.CRITICAL,
        on_fail: OnFail | str | None = None,
        check_on: str | None = None,
    ) -> Contract:
        """Add a must-not-violate clause. Forbid always uses CRITICAL by default.

        Like :meth:`require`, an unset ``check_on`` is resolved from the
        predicate's ``_check_on`` hint, falling back to ``"every_event"`` so
        forbidden behavior (e.g. ``must_not_call``) is caught the moment it
        happens.
        """
        if isinstance(on_fail, str):
            on_fail = OnFail(on_fail)
        pred_name = getattr(predicate_fn, "predicate_name", "")
        resolved_check_on = check_on or getattr(predicate_fn, "_check_on", None) or "every_event"
        self.clauses.append(Clause(
            kind=ClauseKind.FORBID,
            predicate=predicate_fn,
            predicate_name=pred_name,
            description=description or pred_name or "forbid clause",
            severity=severity,
            on_fail=on_fail or OnFail.BLOCK,
            check_on=resolved_check_on,
        ))
        return self

    def precondition(
        self,
        predicate_fn: Callable[[Event, SessionState], PredicateResult],
        *,
        description: str = "",
        severity: Severity = Severity.ERROR,
    ) -> Contract:
        """Add a precondition checked at session start."""
        pred_name = getattr(predicate_fn, "predicate_name", "")
        self.clauses.append(Clause(
            kind=ClauseKind.PRECONDITION,
            predicate=predicate_fn,
            predicate_name=pred_name,
            description=description or pred_name or "precondition",
            severity=severity,
            on_fail=self.default_on_fail,
            check_on="session_start",
        ))
        return self

    def postcondition(
        self,
        predicate_fn: Callable[[Event, SessionState], PredicateResult],
        *,
        description: str = "",
        severity: Severity = Severity.ERROR,
        on_fail: OnFail | str | None = None,
    ) -> Contract:
        """Add a postcondition checked at session end."""
        if isinstance(on_fail, str):
            on_fail = OnFail(on_fail)
        pred_name = getattr(predicate_fn, "predicate_name", "")
        self.clauses.append(Clause(
            kind=ClauseKind.POSTCONDITION,
            predicate=predicate_fn,
            predicate_name=pred_name,
            description=description or pred_name or "postcondition",
            severity=severity,
            on_fail=on_fail or self.default_on_fail,
            check_on="session_end",
        ))
        return self

    def on_violation(self, action: str | OnFail) -> Contract:
        """Set the default recovery action for all clauses."""
        if isinstance(action, str):
            action = OnFail(action)
        self.default_on_fail = action
        return self

    # -- Query API ---------------------------------------------------------

    def get_clauses(self, *, kind: ClauseKind | None = None, check_on: str | None = None) -> list[Clause]:
        """Filter clauses by kind and/or check_on phase."""
        result = self.clauses
        if kind is not None:
            result = [c for c in result if c.kind == kind]
        if check_on is not None:
            result = [c for c in result if c.check_on == check_on]
        return result

    # -- Session factory ---------------------------------------------------

    def session(self, **kwargs: Any) -> "Session":
        """Create an enforcement session for this contract."""
        from pactrun.session import Session
        return Session(self, **kwargs)

    def enforce(self, fn: Callable) -> Callable:
        """Decorator that wraps a function with contract enforcement."""
        import asyncio
        import functools

        if asyncio.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with self.session() as session:
                    result = await fn(*args, **kwargs)
                if not session.is_compliant:
                    from pactrun.core.errors import ViolationError
                    errors = [v for v in session.violations if v.severity in (Severity.ERROR, Severity.CRITICAL)]
                    if errors:
                        raise ViolationError(errors[0])
                return result
            async_wrapper._pactrun_session = None  # type: ignore[attr-defined]
            return async_wrapper
        else:
            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                with self.session() as session:
                    result = fn(*args, **kwargs)
                if not session.is_compliant:
                    from pactrun.core.errors import ViolationError
                    errors = [v for v in session.violations if v.severity in (Severity.ERROR, Severity.CRITICAL)]
                    if errors:
                        raise ViolationError(errors[0])
                return result
            sync_wrapper._pactrun_session = None  # type: ignore[attr-defined]
            return sync_wrapper

    # -- Serialization -----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "default_on_fail": self.default_on_fail.value,
            "clauses": [c.to_dict() for c in self.clauses],
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> Path:
        """Save contract to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        return path

    @classmethod
    def from_yaml(cls, path: str | Path) -> Contract:
        """Load a contract from a YAML file."""
        from pactrun.loader import load_contract_yaml
        return load_contract_yaml(path)

    @classmethod
    def from_dict(cls, data: dict) -> Contract:
        """Load a contract from a dictionary."""
        from pactrun.loader import load_contract_dict
        return load_contract_dict(data)
