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
