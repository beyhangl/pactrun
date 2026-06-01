"""Tests for tool predicates."""

import pytest
from pactrun import (
    Contract, ViolationError,
    must_call, must_not_call, tool_order, tools_allowed, max_tool_calls,
)


class TestMustCall:
    def test_passes_when_called(self):
        c = Contract("test").postcondition(must_call("search"), on_fail="log")
        with c.session() as s:
            s.emit_tool_call("search")
        assert s.is_compliant

    def test_fails_when_not_called(self):
        c = Contract("test").postcondition(must_call("search"), on_fail="log")
        with c.session() as s:
            s.emit_tool_call("other")
        assert not s.is_compliant


class TestMustNotCall:
    def test_passes_when_not_called(self):
        c = Contract("test").forbid(must_not_call("delete"), on_fail="log")
        with c.session() as s:
            s.emit_tool_call("search")
        assert s.is_compliant

    def test_fails_when_called(self):
        c = Contract("test").forbid(must_not_call("delete"), on_fail="log")
        with c.session() as s:
            s.emit_tool_call("delete")
        assert not s.is_compliant

    def test_blocks_when_on_fail_block(self):
        c = Contract("test").forbid(must_not_call("delete"), on_fail="block")
        with pytest.raises(ViolationError, match="delete"):
            with c.session() as s:
                s.emit_tool_call("delete")


class TestToolOrder:
    def test_correct_order_passes(self):
        c = Contract("test").postcondition(tool_order(["search", "format"]), on_fail="log")
        with c.session() as s:
            s.emit_tool_call("search")
            s.emit_tool_call("format")
        assert s.is_compliant

    def test_wrong_order_fails(self):
        c = Contract("test").postcondition(tool_order(["search", "format"]), on_fail="log")
        with c.session() as s:
            s.emit_tool_call("format")
            s.emit_tool_call("search")
        assert not s.is_compliant

    def test_subsequence_passes(self):
        c = Contract("test").postcondition(tool_order(["search", "format"]), on_fail="log")
        with c.session() as s:
            s.emit_tool_call("search")
            s.emit_tool_call("other")
            s.emit_tool_call("format")
        assert s.is_compliant

    def test_strict_order(self):
        c = Contract("test").postcondition(tool_order(["search", "format"], strict=True), on_fail="log")
        with c.session() as s:
            s.emit_tool_call("search")
            s.emit_tool_call("other")
            s.emit_tool_call("format")
        assert not s.is_compliant  # Strict requires exact match


class TestToolsAllowed:
    def test_allowed_tool_passes(self):
        c = Contract("test").require(tools_allowed(["search", "format"]), on_fail="log")
        with c.session() as s:
            s.emit_tool_call("search")
        assert s.is_compliant

    def test_disallowed_tool_fails(self):
        c = Contract("test").require(tools_allowed(["search"]), on_fail="log")
        with c.session() as s:
            s.emit_tool_call("delete")
        assert not s.is_compliant


class TestMaxToolCalls:
    def test_under_limit_passes(self):
        c = Contract("test").require(max_tool_calls(5), on_fail="log")
        with c.session() as s:
            s.emit_tool_call("a")
            s.emit_tool_call("b")
        assert s.is_compliant

    def test_over_limit_fails(self):
        c = Contract("test").require(max_tool_calls(2), on_fail="log")
        with c.session() as s:
            s.emit_tool_call("a")
            s.emit_tool_call("b")
            s.emit_tool_call("c")
        assert not s.is_compliant
