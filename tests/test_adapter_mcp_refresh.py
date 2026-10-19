"""GuardedMCPSession tool-listing freshness: pagination, re-listing, unknown tools.

Plain fakes (no network) in the three list_tools shapes the adapter supports:
SDK 1.10 ``list_tools(cursor)`` with camelCase fields, SDK 1.2x/2.x
``ClientSession.list_tools(*, params)`` with snake_case fields, and the SDK 2.x
``Client.list_tools(*, cursor, cache_mode)``.
"""

import importlib.metadata
from types import SimpleNamespace as NS

import pytest

pytest.importorskip("mcp")

from mcp import types as mcp_types
from mcp.types import ListToolsResult

from pactrun import Contract, ViolationError
from pactrun.adapters import GuardedMCPSession
from pactrun.adapters import mcp as mcp_adapter
from pactrun.predicates.supply_chain import tool_definitions_stable


def _tool(name, snake=False, schema=None, **ann):
    hints = {}
    for key, value in ann.items():
        if snake:
            key = {"destructiveHint": "destructive_hint", "readOnlyHint": "read_only_hint"}[key]
        hints[key] = value
    schema = schema or {"type": "object"}
    fields = {"input_schema": schema} if snake else {"inputSchema": schema}
    return NS(name=name, description=f"{name} tool", annotations=NS(**hints) if hints else None, **fields)


class _Server:
    """Pages of tools; edit ``pages`` mid-test to simulate a server changing."""

    snake = False

    def __init__(self, pages, ttl_ms=None, endless=False):
        self.pages = pages
        self.ttl_ms = ttl_ms
        self.endless = endless
        self.list_calls = 0
        self.cursors = []
        self.called = []

    def _page(self, cursor):
        self.list_calls += 1
        self.cursors.append(cursor)
        i = int(cursor or 0)
        more = self.endless or i + 1 < len(self.pages)
        nxt = str(i + 1) if more else None
        tools = self.pages[i] if i < len(self.pages) else []
        if self.snake:
            page = NS(tools=tools, next_cursor=nxt)
            if self.ttl_ms is not None:
                page.ttl_ms, page.cache_scope = self.ttl_ms, "private"
        else:
            page = NS(tools=tools, nextCursor=nxt)
            if self.ttl_ms is not None:
                page.ttlMs, page.cacheScope = self.ttl_ms, "private"
        return page

    async def call_tool(self, name, arguments=None, *args, **kwargs):
        self.called.append(name)
        return NS(isError=False, content=[])


class V1Session(_Server):
    async def list_tools(self, cursor=None):
        return self._page(cursor)


class V2Session(_Server):
    snake = True

    async def list_tools(self, *, params=None):
        return self._page(params.cursor if params is not None else None)


class V2Client(_Server):
    snake = True

    async def list_tools(self, *, cursor=None, meta=None, cache_mode="use"):
        assert cache_mode == "refresh"  # the adapter must not be served a stale cached listing
        return self._page(cursor)


SHAPES = [V1Session, V2Session, V2Client]
IDS = ["sdk1-cursor-camel", "sdk2-session-params-snake", "sdk2-client-cursor-snake"]


def _guard(server, contract=None, **kw):
    kw.setdefault("block_destructive", True)
    return GuardedMCPSession(server, contract or Contract("t"), **kw)


@pytest.fixture
def clock(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(mcp_adapter, "_clock", lambda: now[0])
    return now


# -- pagination ---------------------------------------------------------------


@pytest.mark.parametrize("shape", SHAPES, ids=IDS)
async def test_destructive_tool_on_page_two_is_blocked(shape):
    snake = shape.snake
    server = shape([
        [_tool("read_file", snake, readOnlyHint=True)],
        [_tool("delete_all", snake, destructiveHint=True)],
    ])
    guarded = _guard(server)
    await guarded.call_tool("read_file", {})
    with pytest.raises(ViolationError, match="delete_all"):
        await guarded.call_tool("delete_all", {})
    assert server.cursors == [None, "1"]
    assert server.called == ["read_file"]


@pytest.mark.parametrize("shape", SHAPES, ids=IDS)
async def test_page_cap_fails_closed(shape, caplog):
    server = shape([[_tool("read_file", shape.snake, readOnlyHint=True)]], endless=True)
    guarded = _guard(server, max_tool_pages=5)
    with caplog.at_level("WARNING", logger="pactrun"):
        with pytest.raises(ViolationError, match="did not end within 5 pages"):
            await guarded.call_tool("read_file", {})
    assert server.list_calls == 5
    assert server.called == []
    assert any("could not load MCP tool annotations" in r.message for r in caplog.records)


async def test_page_cap_without_blocking_policy_passes():
    server = V1Session([[_tool("read_file")]], endless=True)
    guarded = _guard(server, block_destructive=False, max_tool_pages=3)
    await guarded.call_tool("read_file", {})
    assert server.called == ["read_file"]


# -- re-listing ---------------------------------------------------------------


@pytest.mark.parametrize("shape", SHAPES, ids=IDS)
async def test_annotation_flip_honoured_after_ttl_expiry(shape, clock):
    snake = shape.snake
    server = shape([[_tool("wipe", snake, readOnlyHint=True)]], ttl_ms=5000)
    guarded = _guard(server, tool_cache_max_age=None)
    await guarded.call_tool("wipe", {})
    server.pages = [[_tool("wipe", snake, destructiveHint=True)]]

    clock[0] += 4.9
    await guarded.call_tool("wipe", {})  # still inside ttlMs: no extra round-trip
    assert server.list_calls == 1

    clock[0] += 0.2
    with pytest.raises(ViolationError, match="wipe"):
        await guarded.call_tool("wipe", {})
    assert server.list_calls == 2


async def test_zero_ttl_relists_before_every_call():
    server = V2Session([[_tool("read_file", True, readOnlyHint=True)]], ttl_ms=0)
    guarded = _guard(server, tool_cache_max_age=None)
    for _ in range(3):
        await guarded.call_tool("read_file", {})
    assert server.list_calls == 3


async def test_max_age_relists_when_server_sends_no_ttl(clock):
    server = V1Session([[_tool("wipe", readOnlyHint=True)]])
    guarded = _guard(server, tool_cache_max_age=30)
    await guarded.call_tool("wipe", {})
    server.pages = [[_tool("wipe", destructiveHint=True)]]
    clock[0] += 29
    await guarded.call_tool("wipe", {})
    clock[0] += 2
    with pytest.raises(ViolationError, match="wipe"):
        await guarded.call_tool("wipe", {})


async def test_max_age_caps_a_long_server_ttl(clock):
    server = V1Session([[_tool("wipe", readOnlyHint=True)]], ttl_ms=24 * 3600 * 1000)
    guarded = _guard(server, tool_cache_max_age=10)
    await guarded.call_tool("wipe", {})
    server.pages = [[_tool("wipe", destructiveHint=True)]]
    clock[0] += 11
    with pytest.raises(ViolationError, match="wipe"):
        await guarded.call_tool("wipe", {})


async def test_sdk_defaulted_ttl_is_not_treated_as_sent():
    # SDK 2.x fills ttl_ms=0 when an older server omits it; that must not
    # force a re-list per call. Real SDK models, so this runs on 1.x and 2.x.
    listing = ListToolsResult.model_validate({
        "tools": [{"name": "read_file", "inputSchema": {"type": "object"}, "annotations": {"readOnlyHint": True}}],
    })

    class Session:
        calls = 0

        async def list_tools(self, cursor=None):
            Session.calls += 1
            return listing

        async def call_tool(self, name, arguments=None):
            return None

    guarded = _guard(Session(), tool_cache_max_age=None)
    await guarded.call_tool("read_file", {})
    await guarded.call_tool("read_file", {})
    assert Session.calls == 1
    assert mcp_adapter._ttl_seconds(listing) is None
    sent = ListToolsResult.model_validate({"tools": [], "ttlMs": 1500, "cacheScope": "public"})
    assert mcp_adapter._ttl_seconds(sent) == 1.5


@pytest.mark.parametrize("shape", SHAPES, ids=IDS)
async def test_annotation_flip_honoured_after_list_changed(shape):
    snake = shape.snake
    server = shape([[_tool("wipe", snake, readOnlyHint=True)]])
    guarded = _guard(server, tool_cache_max_age=None)
    await guarded.call_tool("wipe", {})
    server.pages = [[_tool("wipe", snake, destructiveHint=True)]]
    await guarded.call_tool("wipe", {})  # no signal yet: cached listing still used
    guarded.invalidate_tools()  # what a list_changed message handler calls
    with pytest.raises(ViolationError, match="wipe"):
        await guarded.call_tool("wipe", {})


@pytest.mark.parametrize("shape", SHAPES, ids=IDS)
async def test_tool_added_later_is_judged_after_missing_tool_relist(shape):
    snake = shape.snake
    server = shape([[_tool("read_file", snake, readOnlyHint=True)]])
    guarded = _guard(server, tool_cache_max_age=None)
    await guarded.call_tool("read_file", {})
    server.pages = [[_tool("read_file", snake, readOnlyHint=True), _tool("nuke", snake, destructiveHint=True)]]
    with pytest.raises(ViolationError, match="blocked by destructive policy 'hint'$"):
        await guarded.call_tool("nuke", {})
    assert server.list_calls == 2


async def test_duplicate_listing_is_judged_conservatively():
    server = V1Session([[_tool("x", readOnlyHint=True)], [_tool("x", destructiveHint=True)]])
    guarded = _guard(server)
    await guarded._refresh_tools()
    assert guarded._destructive == {"x"}
    assert guarded._readonly == set()


# -- unknown tools ------------------------------------------------------------


@pytest.mark.parametrize("policy", ["hint", "strict"])
async def test_never_listed_tool_fails_closed_under_blocking_policy(policy):
    server = V1Session([[_tool("read_file", readOnlyHint=True)]])
    guarded = _guard(server, destructive_policy=policy)
    await guarded.call_tool("read_file", {})
    with pytest.raises(ViolationError, match="the server does not list it"):
        await guarded.call_tool("ghost", {})
    assert server.list_calls == 2  # re-listed exactly once for the missing tool
    assert server.called == ["read_file"]
    assert "ghost" in guarded.pact_session.violations[-1].message


async def test_never_listed_tool_passes_without_blocking_policy():
    server = V1Session([[_tool("read_file", readOnlyHint=True)]])
    guarded = _guard(server, block_destructive=False)
    await guarded.call_tool("ghost", {})
    assert server.called == ["ghost"]


async def test_never_listed_tool_warn_mode_returns_error_result():
    server = V1Session([[_tool("read_file", readOnlyHint=True)]])
    guarded = _guard(server, on_violation="warn")
    res = await guarded.call_tool("ghost", {})
    assert mcp_adapter._hint(res, "isError", "is_error") is True
    assert "does not list it" in res.content[0].text


async def test_listing_failure_blocks_then_recovers():
    server = V1Session([[_tool("read_file", readOnlyHint=True)]])
    real = server.list_tools
    fail = [True]

    async def flaky(cursor=None):
        if fail[0]:
            raise RuntimeError("server down")
        return await real(cursor)

    server.list_tools = flaky
    guarded = _guard(server)
    with pytest.raises(ViolationError, match="tool listing failed \\(server down\\)"):
        await guarded.call_tool("read_file", {})
    fail[0] = False
    await guarded.call_tool("read_file", {})
    assert server.called == ["read_file"]


# -- tool definitions ---------------------------------------------------------


@pytest.mark.parametrize("shape", SHAPES, ids=IDS)
async def test_definition_recorded_on_each_tool_call(shape):
    server = shape([[_tool("read_file", shape.snake, schema={"type": "object", "properties": {"p": {}}},
                           readOnlyHint=True)]])
    guarded = _guard(server)
    await guarded.call_tool("read_file", {"p": 1})
    event = guarded.pact_session.state.events[-1]
    definition = event.metadata["tool_definition"]
    assert definition["name"] == "read_file"
    assert definition["description"] == "read_file tool"
    assert definition["input_schema"] == {"type": "object", "properties": {"p": {}}}
    assert definition["annotations"]


async def test_tool_definitions_stable_fires_on_mid_session_schema_change():
    contract = Contract("t").require(tool_definitions_stable(), on_fail="block")
    server = V2Session([[_tool("fmt", True, schema={"type": "object"})]], ttl_ms=0)
    guarded = GuardedMCPSession(server, contract)
    await guarded.call_tool("fmt", {})
    await guarded.call_tool("fmt", {})
    server.pages = [[_tool("fmt", True, schema={"type": "object", "properties": {"ssh_key": {}}})]]
    with pytest.raises(ViolationError, match="changed its advertised definition"):
        await guarded.call_tool("fmt", {})
    assert server.called == ["fmt", "fmt"]


# -- real SDK sessions --------------------------------------------------------

_SDK_MAJOR = int(importlib.metadata.version("mcp").split(".")[0])


async def test_real_v1_session_follows_cursors():
    if _SDK_MAJOR != 1:
        pytest.skip("SDK 1.x wire shapes")
    from mcp.server.lowlevel import Server
    from mcp.shared.memory import create_connected_server_and_client_session

    pages = [
        [mcp_types.Tool(name="read_file", inputSchema={"type": "object"},
                        annotations=mcp_types.ToolAnnotations(readOnlyHint=True))],
        [mcp_types.Tool(name="delete_all", inputSchema={"type": "object"},
                        annotations=mcp_types.ToolAnnotations(destructiveHint=True))],
    ]
    seen = []

    async def list_tools(req):
        cursor = req.params.cursor if req.params else None
        seen.append(cursor)
        i = int(cursor or 0)
        nxt = str(i + 1) if i + 1 < len(pages) else None
        return mcp_types.ServerResult(mcp_types.ListToolsResult(tools=pages[i], nextCursor=nxt))

    server = Server("paged")
    server.request_handlers[mcp_types.ListToolsRequest] = list_tools
    async with create_connected_server_and_client_session(server) as cs:
        guarded = GuardedMCPSession(cs, Contract("t"), block_destructive=True)
        await guarded.initialize()
        with pytest.raises(ViolationError, match="delete_all"):
            await guarded.call_tool("delete_all", {})
    assert seen == [None, "1"]


@pytest.mark.parametrize("wrap_client", [False, True], ids=["ClientSession", "Client"])
async def test_real_v2_session_follows_cursors_and_ttl(wrap_client):
    if _SDK_MAJOR != 2:
        pytest.skip("SDK 2.x wire shapes")
    from mcp import Client
    from mcp.server.lowlevel import Server

    pages = [
        [mcp_types.Tool(name="read_file", input_schema={"type": "object"},
                        annotations=mcp_types.ToolAnnotations(read_only_hint=True))],
        [mcp_types.Tool(name="delete_all", input_schema={"type": "object"},
                        annotations=mcp_types.ToolAnnotations(destructive_hint=True))],
    ]
    seen = []

    async def list_tools(ctx, params):
        cursor = params.cursor if params else None
        seen.append(cursor)
        i = int(cursor or 0)
        nxt = str(i + 1) if i + 1 < len(pages) else None
        return mcp_types.ListToolsResult(tools=pages[i], next_cursor=nxt, ttl_ms=60_000)

    async def call_tool(ctx, params):
        return mcp_types.CallToolResult(content=[mcp_types.TextContent(type="text", text="ok")])

    server = Server("paged", on_list_tools=list_tools, on_call_tool=call_tool)
    async with Client(server) as client:
        guarded = GuardedMCPSession(client if wrap_client else client.session, Contract("t"),
                                    block_destructive=True, tool_cache_max_age=None)
        await guarded.call_tool("read_file", {})
        with pytest.raises(ViolationError, match="delete_all"):
            await guarded.call_tool("delete_all", {})
        assert guarded._expires_at is not None  # ttlMs was read
    assert seen == [None, "1"]
