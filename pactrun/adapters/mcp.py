"""MCP adapter — apply a pactrun Contract to an MCP ClientSession's tool calls.

Wrap a Model Context Protocol ``ClientSession`` so every ``call_tool`` runs
through a pactrun Contract: the full tool-predicate suite (``must_not_call``,
``tools_allowed``, ``max_tool_calls``, ``tool_order``, ``no_loops`` ...) applies
to MCP tools for free, plus an optional ``block_destructive`` policy that uses
the server's own ``destructiveHint`` / ``readOnlyHint`` annotations.

    from pactrun import Contract, must_not_call
    from pactrun.adapters import GuardedMCPSession

    contract = Contract("mcp_agent").forbid(must_not_call("delete_file"))
    guarded = GuardedMCPSession(client_session, contract, block_destructive=True)
    await guarded.initialize()
    await guarded.call_tool("read_file", {"path": "a.txt"})    # ok
    await guarded.call_tool("delete_file", {"path": "a.txt"})  # raises ViolationError

The adapter keeps its own copy of the server's tool listing. It follows
``tools/list`` pagination to the end (``max_tool_pages`` caps it; a listing
that does not end in time counts as failed). It lists again when the listing
expires — after the server's ``ttlMs`` or ``tool_cache_max_age`` seconds,
whichever is shorter — when ``invalidate_tools()`` is called (wire it to
``notifications/tools/list_changed`` in your message handler; neither SDK line
offers a hook that does not take over that handler), and once whenever a
called tool is missing from the copy. Each re-list is one or more extra
round-trips before the call, and a ``ttlMs`` of 0 (the v2 SDK server default)
means one before every call. Under ``block_destructive`` a tool that is still
not listed, or any tool while the listing fails, is blocked.

Each TOOL_CALL event carries the definition the server advertised at call time
(name, description, input schema, annotations) in
``metadata["tool_definition"]``, so ``tool_definitions_stable()`` catches a
server that changes a tool mid-session.

Honesty: ``destructiveHint`` is an ADVISORY hint from a possibly-untrusted
server, so ``block_destructive`` is defense-in-depth, not a guarantee — pair it
with an explicit ``tools_allowed`` for high assurance. A server can also change
a tool between the listing and the call. MCP ``CallToolResult`` carries no
token/usage field, so MCP cost control is count-based (``max_tool_calls``),
never per-tool dollar metering.
"""

from __future__ import annotations

import inspect
import logging
import time
from typing import Any

from pactrun.core.enums import ClauseKind, OnFail, Severity
from pactrun.core.errors import ViolationError
from pactrun.core.models import Violation

try:
    from mcp.types import CallToolResult, PaginatedRequestParams, TextContent
except ImportError as exc:  # pragma: no cover - only without the extra
    raise ImportError(
        "The 'mcp' package is required for the MCP adapter. "
        "Install it with: pip install 'pactrun[mcp]'"
    ) from exc

logger = logging.getLogger("pactrun")

_clock = time.monotonic


def _hint(obj: Any, camel: str, snake: str) -> Any:
    """Read an MCP model field across SDK spellings.

    The MCP Python SDK renamed every model field from camelCase to snake_case
    in v2 (the 2026-07-28 spec line), so ``ToolAnnotations.destructiveHint``
    became ``destructive_hint`` and ``nextCursor`` became ``next_cursor``.
    Reading only the camelCase name silently yields ``None`` on v2 — which
    makes ``destructive_policy="hint"`` fail **open**, allowing every
    destructive tool while still reporting that blocking is enabled. Accept
    both spellings.
    """
    value = getattr(obj, camel, None)
    if value is None:
        value = getattr(obj, snake, None)
    return value


def _ttl_seconds(page: Any) -> float | None:
    """The listing's ``ttlMs`` in seconds, or None when the server sent none."""
    fields_set = getattr(page, "model_fields_set", None)
    # SDK v2 fills ttl_ms=0 when an older server omits it, so only trust a sent value.
    if isinstance(fields_set, (set, frozenset)) and not fields_set & {"ttl_ms", "ttlMs"}:
        return None
    ttl = _hint(page, "ttlMs", "ttl_ms")
    if isinstance(ttl, bool) or not isinstance(ttl, (int, float)) or ttl < 0:
        return None
    return ttl / 1000.0


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool, list, dict)):
        return value
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump(by_alias=True, exclude_none=True, mode="json")
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return str(value)


def _definition(tool: Any) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": getattr(tool, "description", None),
        "input_schema": _plain(_hint(tool, "inputSchema", "input_schema")),
        "annotations": _plain(getattr(tool, "annotations", None)),
    }


class GuardedMCPSession:
    """Wraps an MCP ClientSession so tool contracts apply to its tool calls."""

    def __init__(
        self,
        mcp_session: Any,
        contract: Any,
        *,
        block_destructive: bool = False,
        destructive_policy: str = "hint",
        on_violation: str = "block",
        tool_cache_max_age: float | None = 60.0,
        max_tool_pages: int = 100,
    ) -> None:
        self._mcp = mcp_session
        self._contract = contract
        self._pact = contract.session()
        self._block_destructive = block_destructive
        self._destructive_policy = destructive_policy
        self._on_violation = OnFail(on_violation)
        self._max_age = tool_cache_max_age
        self._max_pages = max_tool_pages
        self._destructive: set[str] = set()
        self._readonly: set[str] = set()
        self._definitions: dict[str, dict[str, Any]] = {}
        self._listed = False
        self._expires_at: float | None = None
        self._list_error: str | None = None
        self._list_params: set[str] | None = None

    def __getattr__(self, name: str) -> Any:
        mcp_session = self.__dict__.get("_mcp")
        if mcp_session is None:
            raise AttributeError(name)
        return getattr(mcp_session, name)

    @property
    def pact_session(self):
        """The underlying pactrun Session (violations, summary, etc.)."""
        return self._pact

    async def initialize(self, *args: Any, **kwargs: Any) -> Any:
        result = await self._mcp.initialize(*args, **kwargs)
        await self._refresh_tools()
        return result

    def invalidate_tools(self) -> None:
        """Mark the tool listing stale so the next ``call_tool`` lists again.

        Call it from your message handler on ``notifications/tools/list_changed``.
        """
        self._listed = False

    async def call_tool(self, name: str, arguments: dict | None = None, *args: Any, **kwargs: Any) -> Any:
        refreshed = False
        if self._is_stale():
            await self._refresh_tools()
            refreshed = True
        if name not in self._definitions and not refreshed:
            await self._refresh_tools()

        reason = self._blocked_reason(name)
        if reason is not None:
            violation = _make_violation(name, reason)
            self._pact._violations.append(violation)
            return self._handle_block(violation)

        definition = self._definitions.get(name)
        metadata = {"tool_definition": definition} if definition is not None else None
        try:
            self._pact.emit_tool_call(name, args=arguments or {}, metadata=metadata)
        except ViolationError as exc:
            return self._handle_block(exc.violation)

        return await self._mcp.call_tool(name, arguments, *args, **kwargs)

    # -- internal ----------------------------------------------------------

    def _is_stale(self) -> bool:
        if not self._listed or self._list_error is not None:
            return True
        return self._expires_at is not None and _clock() >= self._expires_at

    async def _refresh_tools(self) -> None:
        try:
            tools, ttl = await self._list_all_tools()
        except Exception as exc:
            # A silent failure here is indistinguishable from "this server
            # declares no destructive tools". Forget the old listing so every
            # tool is unknown (blocked under block_destructive) and say so.
            self._destructive, self._readonly, self._definitions = set(), set(), {}
            self._list_error = str(exc) or type(exc).__name__
            self._listed = True
            self._expires_at = None
            logger.warning(
                "pactrun: could not load MCP tool annotations (%s); %s",
                self._list_error,
                "every call is blocked until a listing succeeds"
                if self._block_destructive
                else f"destructive_policy={self._destructive_policy!r} has no annotations to act on",
            )
            return

        destructive: set[str] = set()
        maybe_readonly: set[str] = set()
        not_readonly: set[str] = set()
        definitions: dict[str, dict[str, Any]] = {}
        for tool in tools:
            definitions.setdefault(tool.name, _definition(tool))
            ann = getattr(tool, "annotations", None)
            # A name listed twice is destructive if any entry says so, read-only only if all do.
            if ann is not None and _hint(ann, "destructiveHint", "destructive_hint") is True:
                destructive.add(tool.name)
            if ann is not None and _hint(ann, "readOnlyHint", "read_only_hint") is True:
                maybe_readonly.add(tool.name)
            else:
                not_readonly.add(tool.name)

        self._destructive = destructive
        self._readonly = maybe_readonly - not_readonly
        self._definitions = definitions
        self._list_error = None
        self._listed = True
        ages = [age for age in (ttl, self._max_age) if age is not None]
        self._expires_at = _clock() + min(ages) if ages else None

    async def _list_all_tools(self) -> tuple[list[Any], float | None]:
        tools: list[Any] = []
        ttls: list[float] = []
        cursor: str | None = None
        for _ in range(self._max_pages):
            page = await self._list_page(cursor)
            tools.extend(getattr(page, "tools", None) or [])
            ttl = _ttl_seconds(page)
            if ttl is not None:
                ttls.append(ttl)
            cursor = _hint(page, "nextCursor", "next_cursor")
            if cursor is None:
                return tools, min(ttls) if ttls else None
        raise RuntimeError(f"tools/list did not end within {self._max_pages} pages")

    async def _list_page(self, cursor: str | None) -> Any:
        # SDK 1.10 takes list_tools(cursor), 1.2x and 2.x ClientSession take
        # params=, and the 2.x Client takes cursor= plus a response cache.
        if self._list_params is None:
            try:
                self._list_params = set(inspect.signature(self._mcp.list_tools).parameters)
            except (TypeError, ValueError):
                self._list_params = set()
        kwargs: dict[str, Any] = {}
        if "cache_mode" in self._list_params:
            kwargs["cache_mode"] = "refresh"
        if cursor is not None:
            if "params" in self._list_params:
                kwargs["params"] = PaginatedRequestParams(cursor=cursor)
            else:
                kwargs["cursor"] = cursor
        return await self._mcp.list_tools(**kwargs)

    def _blocked_reason(self, name: str) -> str | None:
        if not self._block_destructive:
            return None
        policy = self._destructive_policy
        if name not in self._definitions:
            why = (
                f"the tool listing failed ({self._list_error})"
                if self._list_error is not None
                else "the server does not list it"
            )
            return f"MCP tool '{name}' blocked by destructive policy '{policy}': {why}, so it cannot be checked"
        if policy == "strict":
            # Unannotated tools are presumed destructive (the MCP spec warns
            # that annotations from untrusted servers should not be relied on).
            blocked = name not in self._readonly
        else:
            # "hint": block only tools the server explicitly marks destructive.
            blocked = name in self._destructive
        return f"MCP tool '{name}' blocked by destructive policy '{policy}'" if blocked else None

    def _is_blocked_destructive(self, name: str) -> bool:
        return self._blocked_reason(name) is not None

    def _handle_block(self, violation: Violation) -> Any:
        if self._on_violation == OnFail.BLOCK:
            raise ViolationError(violation)
        return CallToolResult(
            content=[TextContent(type="text", text=f"pactrun blocked: {violation.message}")],
            isError=True,
        )


def _make_violation(name: str, message: str) -> Violation:
    return Violation(
        clause_description="mcp destructive-tool gate",
        kind=ClauseKind.FORBID,
        severity=Severity.CRITICAL,
        on_fail=OnFail.BLOCK,
        message=message,
        expected=f"'{name}' not destructive",
        actual=f"'{name}' blocked",
    )
