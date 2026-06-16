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

Honesty: ``destructiveHint`` is an ADVISORY hint from a possibly-untrusted
server, so ``block_destructive`` is defense-in-depth, not a guarantee — pair it
with an explicit ``tools_allowed`` for high assurance. MCP ``CallToolResult``
carries no token/usage field, so MCP cost control is count-based
(``max_tool_calls``), never per-tool dollar metering.
"""

from __future__ import annotations

from typing import Any

from pactrun.core.enums import ClauseKind, OnFail, Severity
from pactrun.core.errors import ViolationError
from pactrun.core.models import Violation

try:
    from mcp.types import CallToolResult, TextContent
except ImportError as exc:  # pragma: no cover - only without the extra
    raise ImportError(
        "The 'mcp' package is required for the MCP adapter. "
        "Install it with: pip install 'pactrun[mcp]'"
    ) from exc


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
    ) -> None:
        self._mcp = mcp_session
        self._contract = contract
        self._pact = contract.session()
        self._block_destructive = block_destructive
        self._destructive_policy = destructive_policy
        self._on_violation = OnFail(on_violation)
        self._destructive: set[str] = set()
        self._readonly: set[str] = set()
        self._annotations_loaded = False

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
        await self._load_annotations()
        return result

    async def call_tool(self, name: str, arguments: dict | None = None, *args: Any, **kwargs: Any) -> Any:
        await self._load_annotations()

        if self._is_blocked_destructive(name):
            violation = _make_violation(
                name, f"MCP tool '{name}' blocked by destructive policy '{self._destructive_policy}'"
            )
            self._pact._violations.append(violation)
            return self._handle_block(violation)

        try:
            self._pact.emit_tool_call(name, args=arguments or {})
        except ViolationError as exc:
            return self._handle_block(exc.violation)

        return await self._mcp.call_tool(name, arguments, *args, **kwargs)

    # -- internal ----------------------------------------------------------

    async def _load_annotations(self) -> None:
        if self._annotations_loaded:
            return
        self._annotations_loaded = True
        try:
            listed = await self._mcp.list_tools()
            for tool in getattr(listed, "tools", None) or []:
                ann = getattr(tool, "annotations", None)
                if ann is None:
                    continue
                if getattr(ann, "destructiveHint", None) is True:
                    self._destructive.add(tool.name)
                if getattr(ann, "readOnlyHint", None) is True:
                    self._readonly.add(tool.name)
        except Exception:
            pass

    def _is_blocked_destructive(self, name: str) -> bool:
        if not self._block_destructive:
            return False
        if self._destructive_policy == "strict":
            # Unannotated tools are presumed destructive (the MCP spec warns
            # that annotations from untrusted servers should not be relied on).
            return name not in self._readonly
        # "hint": block only tools the server explicitly marks destructive.
        return name in self._destructive

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
