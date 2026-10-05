"""Replay recorded traces against a contract, and run policy tests.

A contract is code, and code needs tests. This module lets you:

- **replay** a recorded run (a JSONL trace, e.g. from
  :class:`~pactrun.observability.trace.TraceRecorder`) against any contract and
  see exactly which clauses it would violate - before you ship the contract;
- write **policy tests** in the contract's own YAML: a trace that must pass,
  and attack traces that must be caught by specific predicates.

Replays always run in monitor mode (nothing is blocked, nothing raises) and on
the **event clock**, so time-based checks such as ``session_timeout`` judge the
original run's duration rather than how long the replay took.

    from pactrun.replay import load_trace, replay_trace

    result = replay_trace(contract, load_trace("run.jsonl"))
    result.compliant       # False
    result.violated        # ["no_exfil_links"]
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from pactrun.core.errors import ContractLoadError
from pactrun.core.models import Event, SessionSummary, Violation

if TYPE_CHECKING:
    from pactrun.contract import Contract


class TraceLoadError(ValueError):
    """A trace file (or inline test events) could not be parsed."""


def _event_from(data: Any, where: str) -> Event:
    if not isinstance(data, dict):
        raise TraceLoadError(f"{where}: an event must be a mapping, got {type(data).__name__}")
    try:
        return Event.from_dict(data)
    except (TypeError, ValueError) as exc:
        raise TraceLoadError(f"{where}: {exc}") from exc


def load_trace(path: str | Path) -> list[Event]:
    """Read a JSONL trace (one ``Event.to_dict()`` object per line)."""
    path = Path(path)
    events: list[Event] = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TraceLoadError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            events.append(_event_from(data, f"{path}:{lineno}"))
    return events


@dataclass
class ReplayResult:
    """What a contract would have done to one recorded run."""

    contract_name: str
    events: list[Event]
    violations: list[Violation]
    summary: SessionSummary

    @property
    def compliant(self) -> bool:
        return not self.violations

    @property
    def violated(self) -> list[str]:
        """Sorted, de-duplicated names of the predicates that failed."""
        return sorted({v.predicate_name or v.clause_description for v in self.violations})

    def event_index(self, violation: Violation) -> int | None:
        """1-based position of the event that triggered ``violation``.

        ``None`` for clauses evaluated at session start or end, which are not
        tied to a recorded event.
        """
        for i, event in enumerate(self.events, 1):
            if event.id == violation.event_id:
                return i
        return None


def replay_trace(contract: Contract, events: list[Event]) -> ReplayResult:
    """Evaluate ``events`` against ``contract`` without enforcing anything.

    Events are shallow-copied first, so the caller's list is left untouched.
    """
    replayed = [copy.copy(e) for e in events]
    with contract.session(mode="monitor", clock="event") as session:
        for event in replayed:
            session.record_event(event)
    return ReplayResult(
        contract_name=contract.name,
        events=replayed,
        violations=list(session.violations),
        summary=session.summary(),
    )


# ---------------------------------------------------------------------------
# Policy tests declared in the contract YAML
# ---------------------------------------------------------------------------

@dataclass
class PolicyTestResult:
    name: str
    passed: bool
    expected: list[str]
    violated: list[str]
    detail: str = ""
    runs: int = 1
    runs_passed: int = 0


@dataclass
class PolicyTestReport:
    contract_name: str
    results: list[PolicyTestResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def repeated(self) -> bool:
        """True when any test replays more than one run of the same task."""
        return any(r.runs > 1 for r in self.results)

    @property
    def pass_k(self) -> float:
        """Share of tests whose verdict was right on **every** run (Pass^k)."""
        return sum(r.passed for r in self.results) / len(self.results) if self.results else 0.0

    @property
    def mean_k(self) -> float:
        """Average per-test share of runs with the right verdict (Mean@k)."""
        if not self.results:
            return 0.0
        return sum(r.runs_passed / r.runs for r in self.results) / len(self.results)


def _parse_expect(name: str, expect: Any) -> list[str]:
    """``pass`` -> []; ``{violated: [a, b]}`` -> [a, b]."""
    if expect == "pass":
        return []
    if isinstance(expect, dict) and set(expect) == {"violated"}:
        names = expect["violated"]
        if isinstance(names, str):
            names = [names]
        if not isinstance(names, list) or not all(isinstance(n, str) for n in names) or not names:
            raise ContractLoadError(f"Test {name!r}: 'violated' must be a non-empty list of predicate names")
        return sorted(set(names))
    raise ContractLoadError(
        f"Test {name!r}: 'expect' must be 'pass' or {{violated: [predicate, ...]}}, got {expect!r}"
    )


def _mismatch(expected: list[str], violated: list[str]) -> str:
    missing = sorted(set(expected) - set(violated))
    extra = sorted(set(violated) - set(expected))
    parts = []
    if missing:
        parts.append(f"expected but did not fire: {', '.join(missing)}")
    if extra:
        parts.append(f"fired unexpectedly: {', '.join(extra)}")
    return "; ".join(parts)


def _trace_paths(name: str, spec: Any, base: Path) -> list[Path]:
    """``traces:`` as a list of paths or one glob, relative to the YAML file."""
    if isinstance(spec, str):
        paths = sorted(base.glob(spec))
        if not paths:
            raise ContractLoadError(f"Test {name!r}: 'traces' glob {spec!r} matched no files")
        return paths
    if isinstance(spec, list) and spec and all(isinstance(p, str) for p in spec):
        return [base / p for p in spec]
    raise ContractLoadError(f"Test {name!r}: 'traces' must be a glob or a non-empty list of paths")


def run_contract_tests(path: str | Path) -> PolicyTestReport:
    """Run the ``tests:`` block of a contract YAML file.

    Each test supplies events - a ``trace:`` path (relative to the YAML file),
    ``traces:`` (several recorded runs of the same task, as a list or a glob),
    or inline ``events:`` - and an ``expect:``, either ``pass`` or
    ``{violated: [predicate, ...]}``. The violated set must match **exactly**:
    a test fails if an expected predicate did not fire *or* an unexpected one
    did, so over-blocking is caught as well as under-blocking.

    With ``traces:`` a test passes only if **every** run gets the expected
    verdict. The report's :attr:`PolicyTestReport.pass_k` and
    :attr:`~PolicyTestReport.mean_k` then measure how consistently the agent
    stays inside the contract, not just whether one run did.
    """
    from pactrun.loader import load_contract_dict

    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ContractLoadError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ContractLoadError(f"Contract YAML must be a mapping, got {type(data).__name__}")

    tests = data.get("tests")
    if not tests:
        raise ContractLoadError(f"{path} has no 'tests:' block")
    if not isinstance(tests, list):
        raise ContractLoadError(f"{path}: 'tests' must be a list")

    contract = load_contract_dict(data)
    report = PolicyTestReport(contract_name=contract.name)
    for i, test in enumerate(tests, 1):
        if not isinstance(test, dict):
            raise ContractLoadError(f"Test #{i} must be a mapping")
        name = str(test.get("name", f"test #{i}"))
        expected = _parse_expect(name, test.get("expect"))

        runs: list[tuple[str, list[Event]]]
        if "traces" in test:
            runs = [(p.name, load_trace(p)) for p in _trace_paths(name, test["traces"], path.parent)]
        elif "trace" in test:
            runs = [("", load_trace(path.parent / str(test["trace"])))]
        elif "events" in test:
            raw = test["events"]
            if not isinstance(raw, list):
                raise ContractLoadError(f"Test {name!r}: 'events' must be a list")
            runs = [("", [_event_from(e, f"test {name!r} event #{j}") for j, e in enumerate(raw, 1)])]
        else:
            raise ContractLoadError(f"Test {name!r} needs a 'trace:' path, 'traces:', or inline 'events:'")

        violated_union: set[str] = set()
        failures: list[str] = []
        runs_passed = 0
        for label, events in runs:
            violated = replay_trace(contract, events).violated
            violated_union.update(violated)
            if violated == expected:
                runs_passed += 1
            else:
                mismatch = _mismatch(expected, violated)
                failures.append(f"{label}: {mismatch}" if len(runs) > 1 else mismatch)
        report.results.append(PolicyTestResult(
            name,
            passed=runs_passed == len(runs),
            expected=expected,
            violated=sorted(violated_union),
            detail="; ".join(failures),
            runs=len(runs),
            runs_passed=runs_passed,
        ))
    return report
