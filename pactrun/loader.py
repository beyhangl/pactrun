"""Contract loader — parse YAML and dict contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from pactrun.core.enums import ClauseKind, OnFail, Severity
from pactrun.core.errors import ContractLoadError
from pactrun.core.models import Clause
from pactrun.predicates.base import get_predicate


def load_contract_yaml(path: str | Path) -> "Contract":
    """Load a Contract from a YAML file."""
    from pactrun.contract import Contract

    path = Path(path)
    if not path.exists():
        raise ContractLoadError(f"Contract file not found: {path}")

    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ContractLoadError(f"Invalid YAML in {path}: {e}") from e

    if not isinstance(data, dict):
        raise ContractLoadError(f"Contract YAML must be a mapping, got {type(data).__name__}")

    return load_contract_dict(data)


def load_contract_dict(data: dict) -> "Contract":
    """Load a Contract from a dictionary."""
    from pactrun.contract import Contract

    name = data.get("name", "")
    version = data.get("version", "1.0")
    description = data.get("description", "")
    default_on_fail = OnFail(data.get("default_on_fail", data.get("on_fail", "block")))
    metadata = data.get("metadata", {})

    contract = Contract(
        name=name,
        version=str(version),
        description=description,
        default_on_fail=default_on_fail,
        metadata=metadata,
    )

    # Parse clauses
    for clause_data in data.get("clauses", []):
        clause = _parse_clause(clause_data, default_on_fail)
        contract.clauses.append(clause)

    return contract


def _parse_clause(data: dict, default_on_fail: OnFail) -> Clause:
    """Parse a single clause from a dict."""
    # Determine kind and predicate name
    if "require" in data:
        kind = ClauseKind.REQUIRE
        pred_name = data["require"]
    elif "forbid" in data:
        kind = ClauseKind.FORBID
        pred_name = data["forbid"]
    elif "precondition" in data:
        kind = ClauseKind.PRECONDITION
        pred_name = data["precondition"]
    elif "postcondition" in data:
        kind = ClauseKind.POSTCONDITION
        pred_name = data["postcondition"]
    else:
        # Try generic "type" field
        kind = ClauseKind(data.get("kind", "require"))
        pred_name = data.get("predicate", data.get("type", ""))

    if not pred_name:
        raise ContractLoadError(f"Clause has no predicate name: {data}")

    # Resolve predicate from registry
    try:
        predicate_factory = get_predicate(pred_name)
    except KeyError as e:
        raise ContractLoadError(str(e)) from e

    # Build predicate with args
    args = data.get("args", {})
    if isinstance(args, dict):
        predicate_fn = predicate_factory(**args)
    elif isinstance(args, list):
        predicate_fn = predicate_factory(*args)
    else:
        predicate_fn = predicate_factory(args)

    severity = Severity(data.get("severity", "critical" if kind == ClauseKind.FORBID else "error"))
    on_fail = OnFail(data.get("on_fail", default_on_fail.value))

    # Resolve when the clause is evaluated. An explicit ``check_on`` always
    # wins; otherwise pre/postconditions map to their session phase and
    # require/forbid honor the predicate's own ``_check_on`` hint (so
    # ``must_call`` / ``tool_order`` / ``output_contains`` defer to session end
    # instead of failing on the first event).
    if "check_on" in data:
        check_on = data["check_on"]
    elif kind == ClauseKind.POSTCONDITION:
        check_on = "session_end"
    elif kind == ClauseKind.PRECONDITION:
        check_on = "session_start"
    else:
        check_on = getattr(predicate_fn, "_check_on", None) or "every_event"

    return Clause(
        kind=kind,
        predicate=predicate_fn,
        predicate_name=pred_name,
        description=data.get("description", pred_name),
        severity=severity,
        on_fail=on_fail,
        check_on=check_on,
        metadata=data.get("metadata", {}),
    )
