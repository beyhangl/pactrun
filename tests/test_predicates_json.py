"""Tests for the valid_json / json_schema_valid output predicates."""

import pytest

from pactrun import Contract, json_schema_valid, valid_json


def _have(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


HAS_JSONSCHEMA = _have("jsonschema")


class TestValidJson:
    def test_valid_json_passes(self):
        c = Contract("t").require(valid_json(), on_fail="log")
        with c.session() as s:
            s.emit_output('{"ok": true}')
        assert s.is_compliant

    def test_invalid_json_fails(self):
        c = Contract("t").require(valid_json(), on_fail="log")
        with c.session() as s:
            s.emit_output("not json at all")
        assert not s.is_compliant

    def test_resolves_to_session_end(self):
        c = Contract("t").require(valid_json(), on_fail="log")
        assert c.clauses[0].check_on == "session_end"


SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer", "maximum": 120}},
    "required": ["name", "age"],
}


@pytest.mark.skipif(not HAS_JSONSCHEMA, reason="needs the jsonschema extra")
class TestJsonSchemaValid:
    def _run(self, output):
        c = Contract("t").require(json_schema_valid(SCHEMA), on_fail="log")
        with c.session() as s:
            s.emit_output(output)
        return s.is_compliant

    def test_matching_passes(self):
        assert self._run('{"name": "Ada", "age": 36}') is True

    def test_missing_required_fails(self):
        assert self._run('{"name": "Ada"}') is False

    def test_over_maximum_fails(self):
        assert self._run('{"name": "Ada", "age": 999}') is False

    def test_not_json_fails(self):
        assert self._run("nope") is False
