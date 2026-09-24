"""Network-egress predicates — constrain the hosts a tool's arguments reach."""

from __future__ import annotations

from pactrun.core.enums import EventKind
from pactrun.core.models import Event, PredicateResult, SessionState
from pactrun.predicates._neturl import _candidate_hosts, _host_matches, _is_private_host, _url_like
from pactrun.predicates.base import predicate


@predicate("tool_host_within", owasp=("ASI02", "ASI07",))
def tool_host_within(
    allow: list[str] | None = None,
    deny: list[str] | None = None,
    block_private: bool = False,
    tool: str | None = None,
    arg: str | None = None,
    arg_keys: list[str] | None = None,
):
    """Network-egress guard: URL/host-shaped tool args must target allowed hosts.

    The egress sibling of :func:`tool_path_within`. For each URL/host-looking
    string argument, the host is extracted and checked:

    - ``deny`` wins — host matching any deny pattern fails;
    - ``allow`` is implicit-deny-by-default — if ``allow`` is given and the host
      matches none of it, it fails;
    - ``block_private`` — a private / loopback / link-local IP **literal** or
      ``localhost`` fails, including the cloud-metadata address
      ``169.254.169.254`` (a common SSRF target).

    Patterns are host globs (``fnmatch``, lowercased: ``"*.corp.com"``) or
    IP/CIDR (``"10.0.0.0/8"``). **No DNS resolution** is performed — this is
    deterministic and TOCTOU-free, but blocks literal hosts only; pair with
    network-level egress control for full assurance. Pass at least one of
    ``allow`` / ``deny`` / ``block_private``. By default every string arg that
    looks like a URL/host is checked; narrow with ``arg`` or ``arg_keys``.
    """
    if not allow and not deny and not block_private:
        raise ValueError("tool_host_within: pass allow, deny, and/or block_private")
    keys = set(arg_keys) if arg_keys else ({arg} if arg else None)

    def _evaluate(host: str):
        if deny and _host_matches(host, deny):
            return f"host '{host}' matches deny list"
        if allow is not None and not _host_matches(host, allow):
            return f"host '{host}' is not in the allow list"
        if block_private and _is_private_host(host):
            return f"host '{host}' is a private/loopback/link-local address"
        return None

    def check(event: Event, state: SessionState) -> PredicateResult:
        if event.kind != EventKind.TOOL_CALL:
            return PredicateResult(passed=True)
        if tool is not None and event.tool_name != tool:
            return PredicateResult(passed=True)
        for key, value in (event.tool_args or {}).items():
            if keys is not None and key not in keys:
                continue
            if not isinstance(value, str):
                continue
            if keys is None and not _url_like(value):
                continue
            # Check every host a client might actually connect to (legacy
            # numeric IPv4 forms, RFC-3986 vs WHATWG backslash handling); the
            # value is blocked if ANY interpretation is disallowed.
            hosts = _candidate_hosts(value)
            if not hosts:
                continue
            reason = next((r for r in map(_evaluate, hosts) if r), None)
            if reason:
                return PredicateResult(
                    passed=False,
                    expected="tool reaches only allowed hosts",
                    actual=value,
                    message=f"Tool '{event.tool_name}' arg '{key}': {reason}",
                )
        return PredicateResult(passed=True)

    check.predicate_name = "tool_host_within"  # type: ignore[attr-defined]
    return check
