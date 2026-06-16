"""Tests for argument-level tool predicates."""

from types import SimpleNamespace as NS

import pytest

import pactrun
from pactrun import (
    Contract,
    ViolationError,
    no_destructive_args,
    tool_args_match,
    tool_path_within,
)


def _have(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


HAS_JSONSCHEMA = _have("jsonschema")


class TestNoDestructiveArgs:
    def test_blocks_rm_rf(self):
        c = Contract("t").forbid(no_destructive_args())
        with pytest.raises(ViolationError, match="rm -rf"):
            with c.session() as s:
                s.emit_tool_call("shell", args={"cmd": "rm -rf /data"})

    def test_blocks_drop_table(self):
        c = Contract("t").forbid(no_destructive_args())
        with pytest.raises(ViolationError, match="DROP"):
            with c.session() as s:
                s.emit_tool_call("sql", args={"query": "DROP TABLE users"})

    def test_clean_args_pass(self):
        c = Contract("t").forbid(no_destructive_args())
        with c.session() as s:
            s.emit_tool_call("sql", args={"query": "SELECT * FROM users WHERE id = 1"})
        assert s.is_compliant

    def test_extra_patterns(self):
        c = Contract("t").forbid(no_destructive_args(extra=[r"sudo\s"]))
        with pytest.raises(ViolationError):
            with c.session() as s:
                s.emit_tool_call("shell", args={"cmd": "sudo reboot"})

    def test_only_named_tool(self):
        c = Contract("t").forbid(no_destructive_args(tool="shell"))
        with c.session() as s:
            s.emit_tool_call("sql", args={"cmd": "rm -rf /"})  # different tool -> not checked
        assert s.is_compliant


@pytest.mark.skipif(not HAS_JSONSCHEMA, reason="needs the jsonschema extra")
class TestToolArgsMatch:
    SCHEMA = {
        "type": "object",
        "properties": {"amount": {"type": "number", "maximum": 1000}},
        "required": ["amount"],
    }

    def test_valid_args_pass(self):
        c = Contract("t").require(tool_args_match("pay", self.SCHEMA), on_fail="log")
        with c.session() as s:
            s.emit_tool_call("pay", args={"amount": 500})
        assert s.is_compliant

    def test_over_maximum_fails(self):
        c = Contract("t").require(tool_args_match("pay", self.SCHEMA), on_fail="log")
        with c.session() as s:
            s.emit_tool_call("pay", args={"amount": 5000})
        assert not s.is_compliant

    def test_missing_required_fails(self):
        c = Contract("t").require(tool_args_match("pay", self.SCHEMA), on_fail="log")
        with c.session() as s:
            s.emit_tool_call("pay", args={})
        assert not s.is_compliant

    def test_other_tool_ignored(self):
        c = Contract("t").require(tool_args_match("pay", self.SCHEMA), on_fail="log")
        with c.session() as s:
            s.emit_tool_call("search", args={"q": "hi"})
        assert s.is_compliant


class TestToolPathWithin:
    ROOT = "/tmp/sandbox"

    def _run(self, path):
        c = Contract("t").require(tool_path_within(self.ROOT), on_fail="log")
        with c.session() as s:
            s.emit_tool_call("read", args={"path": path})
        return s.is_compliant

    def test_outside_blocked(self):
        assert self._run("/etc/passwd") is False

    def test_traversal_blocked(self):
        assert self._run("/tmp/sandbox/../../etc/passwd") is False

    def test_sibling_prefix_blocked(self):
        assert self._run("/tmp/sandbox-evil/x") is False

    def test_inside_root_passes(self):
        assert self._run("/tmp/sandbox/notes.txt") is True

    def test_non_path_arg_ignored(self):
        c = Contract("t").require(tool_path_within(self.ROOT), on_fail="log")
        with c.session() as s:
            s.emit_tool_call("read", args={"label": "just a string"})
        assert s.is_compliant


def test_wrap_forbid_args_blocks_destructive_tool_arg():
    fn = NS(name="shell", arguments='{"cmd": "rm -rf /data"}')
    resp = NS(
        model="gpt-4.1",
        usage=NS(prompt_tokens=10, completion_tokens=5),
        choices=[NS(message=NS(content=None, tool_calls=[NS(function=fn)]))],
    )
    client = NS(chat=NS(completions=NS(create=lambda **kw: resp)))
    g = pactrun.wrap(client, max_cost="$1.00", forbid_args=True, default_max_tokens=10)
    with pytest.raises(ViolationError, match="rm -rf"):
        g.chat.completions.create(
            model="gpt-4.1", messages=[{"role": "user", "content": "clean up"}], max_tokens=10
        )
