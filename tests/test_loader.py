"""Tests for pactrun YAML/dict contract loader."""

import pytest

from pactrun import Contract, ContractLoadError, PredicateResult
from pactrun.predicates.base import predicate, _PREDICATE_REGISTRY


# Register a test predicate
@predicate("test_cost_under")
def test_cost_under(max_usd: float = 1.0):
    def check(event, state):
        return PredicateResult(
            passed=state.total_cost_usd <= max_usd,
            message=f"Cost exceeds ${max_usd}",
        )
    return check


@predicate("test_must_not_call")
def _test_must_not_call(tool: str):
    def check(event, state):
        if event.tool_name == tool:
            return PredicateResult(passed=False, message=f"{tool} forbidden")
        return PredicateResult(passed=True)
    return check


class TestDictLoader:
    def test_basic_contract(self):
        data = {
            "name": "test",
            "version": "2.0",
            "clauses": [
                {"require": "test_cost_under", "args": {"max_usd": 0.50}},
            ],
        }
        c = Contract.from_dict(data)
        assert c.name == "test"
        assert c.version == "2.0"
        assert len(c.clauses) == 1

    def test_forbid_clause(self):
        data = {
            "name": "test",
            "clauses": [
                {"forbid": "test_must_not_call", "args": {"tool": "delete"}},
            ],
        }
        c = Contract.from_dict(data)
        assert c.clauses[0].kind.value == "forbid"

    def test_unknown_predicate_raises(self):
        data = {
            "name": "test",
            "clauses": [{"require": "nonexistent_predicate", "args": {}}],
        }
        with pytest.raises(ContractLoadError, match="Unknown predicate"):
            Contract.from_dict(data)

    def test_custom_severity(self):
        data = {
            "name": "test",
            "clauses": [
                {"require": "test_cost_under", "args": {"max_usd": 1.0}, "severity": "warning"},
            ],
        }
        c = Contract.from_dict(data)
        assert c.clauses[0].severity.value == "warning"

    def test_custom_on_fail(self):
        data = {
            "name": "test",
            "clauses": [
                {"require": "test_cost_under", "args": {"max_usd": 1.0}, "on_fail": "log"},
            ],
        }
        c = Contract.from_dict(data)
        assert c.clauses[0].on_fail.value == "log"


class TestYamlLoader:
    def test_load_yaml(self, tmp_path):
        yaml_content = """
name: support_agent
version: "1.0"
clauses:
  - require: test_cost_under
    args:
      max_usd: 0.50
  - forbid: test_must_not_call
    args:
      tool: delete
"""
        path = tmp_path / "contract.yaml"
        path.write_text(yaml_content)

        c = Contract.from_yaml(path)
        assert c.name == "support_agent"
        assert len(c.clauses) == 2

    def test_missing_file_raises(self):
        with pytest.raises(ContractLoadError, match="not found"):
            Contract.from_yaml("nonexistent.yaml")

    def test_invalid_yaml_raises(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text(": : : invalid")
        with pytest.raises(ContractLoadError):
            Contract.from_yaml(path)


class TestPredicateRegistry:
    def test_registered_predicates(self):
        from pactrun.predicates.base import list_predicates
        names = list_predicates()
        assert "test_cost_under" in names
        assert "test_must_not_call" in names

    def test_get_unknown_raises(self):
        from pactrun.predicates.base import get_predicate
        with pytest.raises(KeyError, match="Unknown predicate"):
            get_predicate("does_not_exist")
