"""Tests for the MCP adapter (GuardedMCPSession) against a real in-memory server."""

import pytest

pytest.importorskip("mcp")

from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import ToolAnnotations

from pactrun import Contract, ViolationError, must_not_call
from pactrun.adapters import GuardedMCPSession


def _server() -> FastMCP:
    server = FastMCP("test")

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
    def read_file(path: str) -> str:
        return f"contents of {path}"

    @server.tool(annotations=ToolAnnotations(destructiveHint=True))
    def delete_file(path: str) -> str:
        return f"deleted {path}"

    return server


async def test_contract_blocks_forbidden_mcp_tool():
    contract = Contract("t").forbid(must_not_call("delete_file"))
    async with create_connected_server_and_client_session(_server()._mcp_server) as cs:
        guarded = GuardedMCPSession(cs, contract)
        await guarded.initialize()

        ok = await guarded.call_tool("read_file", {"path": "a.txt"})
        assert not ok.isError

        with pytest.raises(ViolationError, match="delete_file"):
            await guarded.call_tool("delete_file", {"path": "a.txt"})


async def test_block_destructive_auto_denies():
    async with create_connected_server_and_client_session(_server()._mcp_server) as cs:
        guarded = GuardedMCPSession(cs, Contract("t"), block_destructive=True)
        await guarded.initialize()

        await guarded.call_tool("read_file", {"path": "a.txt"})  # read-only: allowed
        with pytest.raises(ViolationError, match="destructive"):
            await guarded.call_tool("delete_file", {"path": "a.txt"})


async def test_graceful_error_result_when_not_block():
    contract = Contract("t").forbid(must_not_call("delete_file"))
    async with create_connected_server_and_client_session(_server()._mcp_server) as cs:
        guarded = GuardedMCPSession(cs, contract, on_violation="warn")
        await guarded.initialize()

        res = await guarded.call_tool("delete_file", {"path": "a.txt"})
        assert res.isError is True
        assert "pactrun blocked" in res.content[0].text


async def test_passthrough_list_tools():
    async with create_connected_server_and_client_session(_server()._mcp_server) as cs:
        guarded = GuardedMCPSession(cs, Contract("t"))
        await guarded.initialize()
        listed = await guarded.list_tools()  # delegates to the wrapped session
        names = {t.name for t in listed.tools}
        assert {"read_file", "delete_file"} <= names


async def test_strict_policy_blocks_unannotated():
    server = FastMCP("t2")

    @server.tool()  # no annotations
    def mystery(x: str) -> str:
        return x

    async with create_connected_server_and_client_session(server._mcp_server) as cs:
        guarded = GuardedMCPSession(cs, Contract("t"), block_destructive=True, destructive_policy="strict")
        await guarded.initialize()
        with pytest.raises(ViolationError, match="destructive"):
            await guarded.call_tool("mystery", {"x": "1"})


# ---------------------------------------------------------------------------
# Annotation-spelling regression (MCP SDK v1 camelCase vs v2 snake_case)
#
# The v2 SDK (the 2026-07-28 spec line) renamed every model field to
# snake_case. Reading only `destructiveHint` silently yields None there, which
# made `destructive_policy="hint"` fail OPEN. These use plain fakes so they
# assert the parsing contract regardless of which SDK version is installed.
# ---------------------------------------------------------------------------

from types import SimpleNamespace as _NS  # noqa: E402

from pactrun.adapters.mcp import _hint  # noqa: E402


class _FakeSession:
    """Minimal stand-in exposing just the surface the adapter touches."""

    def __init__(self, tools, fail: bool = False):
        self._tools = tools
        self._fail = fail

    async def list_tools(self):
        if self._fail:
            raise RuntimeError("list_tools exploded")
        return _NS(tools=self._tools)


def _tool(name, **ann):
    return _NS(name=name, annotations=_NS(**ann))


def test_hint_reads_camel_and_snake():
    assert _hint(_NS(destructiveHint=True), "destructiveHint", "destructive_hint") is True
    assert _hint(_NS(destructive_hint=True), "destructiveHint", "destructive_hint") is True
    assert _hint(_NS(other=1), "destructiveHint", "destructive_hint") is None


@pytest.mark.parametrize(
    "destructive_ann, readonly_ann",
    [
        ({"destructiveHint": True}, {"readOnlyHint": True}),      # SDK v1
        ({"destructive_hint": True}, {"read_only_hint": True}),   # SDK v2
    ],
    ids=["sdk-v1-camelCase", "sdk-v2-snake_case"],
)
async def test_annotations_parsed_for_both_sdk_spellings(destructive_ann, readonly_ann):
    session = _FakeSession([_tool("delete_all", **destructive_ann), _tool("read_file", **readonly_ann)])
    guarded = GuardedMCPSession(session, Contract("t"), block_destructive=True)
    await guarded._load_annotations()

    assert guarded._destructive == {"delete_all"}
    assert guarded._readonly == {"read_file"}
    # 'hint' must block the destructive tool — this is the fail-open regression.
    assert guarded._is_blocked_destructive("delete_all") is True
    assert guarded._is_blocked_destructive("read_file") is False


@pytest.mark.parametrize("ann", [{"destructive_hint": True}, {"destructiveHint": True}])
async def test_strict_policy_still_allows_readonly(ann):
    session = _FakeSession([_tool("read_file", **{"read_only_hint": True})])
    guarded = GuardedMCPSession(
        session, Contract("t"), block_destructive=True, destructive_policy="strict"
    )
    await guarded._load_annotations()
    # strict presumes unannotated tools destructive, but a read-only tool must pass.
    assert guarded._is_blocked_destructive("read_file") is False
    assert guarded._is_blocked_destructive("unknown_tool") is True


async def test_annotation_load_failure_warns_instead_of_silently_passing(caplog):
    guarded = GuardedMCPSession(
        _FakeSession([], fail=True), Contract("t"), block_destructive=True
    )
    with caplog.at_level("WARNING", logger="pactrun"):
        await guarded._load_annotations()
    assert any("could not load MCP tool annotations" in r.message for r in caplog.records)
