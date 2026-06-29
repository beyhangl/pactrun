"""Tool predicates — control which tools agents can/must call."""

from __future__ import annotations

from pactrun.core.enums import EventKind
from pactrun.core.models import Event, PredicateResult, SessionState
from pactrun.predicates.base import predicate


@predicate("must_call")
def must_call(tool: str):
    """Agent must call this tool by session end."""
    def check(event: Event, state: SessionState) -> PredicateResult:
        return PredicateResult(
            passed=tool in state.tool_call_history,
            expected=f"'{tool}' in tool history",
            actual=str(state.tool_call_history),
            message=f"Tool '{tool}' was never called",
        )
    check.predicate_name = "must_call"  # type: ignore[attr-defined]
    check._check_on = "session_end"  # type: ignore[attr-defined]
    return check


@predicate("must_not_call")
def must_not_call(tool: str):
    """Agent must never call this tool."""
    def check(event: Event, state: SessionState) -> PredicateResult:
        if event.kind == EventKind.TOOL_CALL and event.tool_name == tool:
            return PredicateResult(
                passed=False,
                expected=f"'{tool}' never called",
                actual=f"'{tool}' was called",
                message=f"Forbidden tool '{tool}' was called",
            )
        return PredicateResult(passed=True)
    check.predicate_name = "must_not_call"  # type: ignore[attr-defined]
    return check


@predicate("tool_order")
def tool_order(expected: list[str], strict: bool = False):
    """Tools must be called in this order (checked at session end)."""
    def check(event: Event, state: SessionState) -> PredicateResult:
        history = state.tool_call_history
        if strict:
            passed = history == expected
        else:
            it = iter(history)
            passed = all(t in it for t in expected)
        return PredicateResult(
            passed=passed,
            expected=str(expected),
            actual=str(history),
            message=f"Tool order mismatch: expected {expected}, got {history}",
        )
    check.predicate_name = "tool_order"  # type: ignore[attr-defined]
    check._check_on = "session_end"  # type: ignore[attr-defined]
    return check


@predicate("tools_allowed")
def tools_allowed(whitelist: list[str]):
    """Only these tools may be called."""
    def check(event: Event, state: SessionState) -> PredicateResult:
        if event.kind == EventKind.TOOL_CALL and event.tool_name:
            if event.tool_name not in whitelist:
                return PredicateResult(
                    passed=False,
                    expected=f"tool in {whitelist}",
                    actual=event.tool_name,
                    message=f"Tool '{event.tool_name}' not in allowed list: {whitelist}",
                )
        return PredicateResult(passed=True)
    check.predicate_name = "tools_allowed"  # type: ignore[attr-defined]
    return check


@predicate("max_tool_calls")
def max_tool_calls(limit: int):
    """Total tool calls must not exceed limit."""
    def check(event: Event, state: SessionState) -> PredicateResult:
        return PredicateResult(
            passed=state.total_tool_calls <= limit,
            expected=f"<= {limit} tool calls",
            actual=f"{state.total_tool_calls} tool calls",
            message=f"Tool call count {state.total_tool_calls} exceeds limit {limit}",
        )
    check.predicate_name = "max_tool_calls"  # type: ignore[attr-defined]
    return check


@predicate("tool_args_match")
def tool_args_match(tool: str | None, schema: dict):
    """A tool call's arguments must validate against a JSON Schema.

    Validates ``event.tool_args`` for the given tool (or every tool when
    ``tool`` is None) against ``schema``. Requires the ``jsonschema`` extra.
    """
    def check(event: Event, state: SessionState) -> PredicateResult:
        if event.kind != EventKind.TOOL_CALL:
            return PredicateResult(passed=True)
        if tool is not None and event.tool_name != tool:
            return PredicateResult(passed=True)
        try:
            from jsonschema import Draft202012Validator
        except ImportError as exc:
            raise ImportError(
                "tool_args_match requires the 'jsonschema' package. "
                "Install it with: pip install 'pactrun[jsonschema]'"
            ) from exc
        errors = sorted(
            Draft202012Validator(schema).iter_errors(event.tool_args or {}),
            key=lambda e: list(e.path),
        )
        if errors:
            return PredicateResult(
                passed=False,
                expected="tool args match schema",
                actual=errors[0].message,
                message=f"Tool '{event.tool_name}' args invalid: {errors[0].message}",
            )
        return PredicateResult(passed=True)
    check.predicate_name = "tool_args_match"  # type: ignore[attr-defined]
    return check


# Best-effort denylist of dangerous argument values (defense-in-depth, NOT a
# guarantee — obfuscation/encoding can evade). Patterns are matched
# case-insensitively against the serialized tool arguments.
_DESTRUCTIVE_PATTERNS: list[tuple[str, str]] = [
    (r"rm\s+-[a-z]*r[a-z]*f|rm\s+-[a-z]*f[a-z]*r", "rm -rf"),
    (r"\bDROP\s+(TABLE|DATABASE|SCHEMA)\b", "DROP TABLE/DATABASE"),
    (r"\bTRUNCATE\b", "TRUNCATE"),
    (r"chmod\s+-?[a-z]*\s*0?777", "chmod 777"),
    (r":\s*\(\s*\)\s*\{", "fork bomb"),
    (r"git\s+push\b[^\"']*--force", "git push --force"),
    (r"\bmkfs\b", "mkfs"),
    (r"of=/dev/(disk|sd|hd|nvme)", "dd to raw device"),
]


@predicate("no_destructive_args")
def no_destructive_args(tool: str | None = None, extra: list[str] | None = None):
    """A tool call's arguments must not contain dangerous values.

    Best-effort denylist (defense-in-depth, not a guarantee): scans the
    serialized argument values for destructive patterns (``rm -rf``,
    ``DROP TABLE``, fork bombs, ...). ``extra`` adds caller regex patterns.
    Distinct from the MCP adapter's ``block_destructive`` (which reads server
    *annotations*); this inspects the actual argument values. Pair with
    ``tools_allowed`` for high assurance.
    """
    import re

    patterns = [(re.compile(p, re.IGNORECASE), label) for p, label in _DESTRUCTIVE_PATTERNS]
    patterns += [(re.compile(p, re.IGNORECASE), p) for p in (extra or [])]

    def check(event: Event, state: SessionState) -> PredicateResult:
        if event.kind != EventKind.TOOL_CALL:
            return PredicateResult(passed=True)
        if tool is not None and event.tool_name != tool:
            return PredicateResult(passed=True)
        blob = _args_blob(event.tool_args)
        for rx, label in patterns:
            if rx.search(blob):
                return PredicateResult(
                    passed=False,
                    expected="no destructive tool arguments",
                    actual=f"matched '{label}'",
                    message=f"Tool '{event.tool_name}' argument contains a destructive pattern: {label}",
                )
        return PredicateResult(passed=True)
    check.predicate_name = "no_destructive_args"  # type: ignore[attr-defined]
    return check


@predicate("tool_path_within")
def tool_path_within(root: str, tool: str | None = None, arg_keys: list[str] | None = None):
    """Every path-like tool argument must resolve inside an allowed root.

    Resolves each candidate path with ``realpath`` (after ``~`` expansion) and
    requires it to equal ``root`` or sit under ``root`` + separator — guarding
    path traversal (``../../etc/passwd``) and sibling-prefix tricks
    (``/root-evil`` vs ``/root``). With ``arg_keys`` only those argument keys
    are checked; otherwise every string value that looks path-like is checked.
    """
    import os

    root_real = os.path.realpath(os.path.expanduser(root))

    def _within(value: str) -> bool:
        rp = os.path.realpath(os.path.expanduser(value))
        return rp == root_real or rp.startswith(root_real + os.sep)

    def check(event: Event, state: SessionState) -> PredicateResult:
        if event.kind != EventKind.TOOL_CALL:
            return PredicateResult(passed=True)
        if tool is not None and event.tool_name != tool:
            return PredicateResult(passed=True)
        for key, value in (event.tool_args or {}).items():
            if arg_keys is not None and key not in arg_keys:
                continue
            if not isinstance(value, str):
                continue
            if arg_keys is None and not _looks_like_path(value):
                continue
            if not _within(value):
                return PredicateResult(
                    passed=False,
                    expected=f"path within {root_real}",
                    actual=value,
                    message=f"Tool '{event.tool_name}' arg '{key}' resolves outside {root_real}: {value}",
                )
        return PredicateResult(passed=True)
    check.predicate_name = "tool_path_within"  # type: ignore[attr-defined]
    return check


@predicate("tool_arg_value_guard")
def tool_arg_value_guard(
    tool: str | None,
    field: str,
    deny=None,
    allow=None,
    match: str = "exact",
    normalize=None,
    dedupe_within_session: bool = False,
):
    """Allow/deny a specific tool-argument field by value, with optional dedupe.

    Reads the value at ``field`` (a dotted path: ``"recipient.email"``,
    ``"items.0.name"``) of the named tool's arguments and enforces:

    - ``deny`` — fail when the value matches any denylist entry (pass when the
      field is absent);
    - ``allow`` — fail when the value is NOT in the allowlist (**fail-closed**
      when the field is absent).

    Pass ``deny`` OR ``allow`` (not both), and/or ``dedupe_within_session``.
    ``match`` chooses comparison: ``"exact"`` | ``"ci"`` (case-insensitive) |
    ``"glob"`` (fnmatch) | ``"regex"`` (``re.search``). ``deny`` / ``allow`` may
    be a list/set/tuple **or a zero-arg callable** returning one — the callable
    is re-evaluated every event, so a suppression list loaded from a mutating
    file is honored live. ``normalize`` (``Callable[[str], str]``) is applied to
    both the live value and each list entry before comparison.

    With ``dedupe_within_session=True`` the same normalized value may be
    targeted at most once per session: a later tool call carrying the same key
    fails. Dedupe scans recorded events only (no closure-mutable state), so it
    is replay-deterministic.
    """
    if deny is not None and allow is not None:
        raise ValueError("tool_arg_value_guard: pass deny OR allow, not both")
    if deny is None and allow is None and not dedupe_within_session:
        raise ValueError(
            "tool_arg_value_guard: pass deny, allow, or dedupe_within_session=True"
        )
    if match not in ("exact", "ci", "glob", "regex"):
        raise ValueError(f"tool_arg_value_guard: unknown match {match!r}")

    def _norm(v) -> str:
        s = v if isinstance(v, str) else str(v)
        return normalize(s) if normalize else s

    def _resolve_set(spec):
        raw = spec() if callable(spec) else spec
        return {_norm(x) for x in (raw or [])}

    def check(event: Event, state: SessionState) -> PredicateResult:
        if event.kind != EventKind.TOOL_CALL:
            return PredicateResult(passed=True)
        if tool is not None and event.tool_name != tool:
            return PredicateResult(passed=True)

        found, value = _resolve_path(event.tool_args or {}, field)
        norm_value = _norm(value) if found else None

        if allow is not None:
            if not found:
                return PredicateResult(
                    passed=False,
                    expected=f"'{field}' present and in allowlist",
                    actual="field absent",
                    message=f"Tool '{event.tool_name}' arg '{field}' missing (allowlist is fail-closed)",
                )
            if not _value_in(norm_value, _resolve_set(allow), match):
                return PredicateResult(
                    passed=False,
                    expected=f"'{field}' in allowlist",
                    actual=str(value),
                    message=f"Tool '{event.tool_name}' arg '{field}'={value!r} is not in the allowlist",
                )

        if deny is not None and found:
            if _value_in(norm_value, _resolve_set(deny), match):
                return PredicateResult(
                    passed=False,
                    expected=f"'{field}' not in denylist",
                    actual=str(value),
                    message=f"Tool '{event.tool_name}' arg '{field}'={value!r} is on the denylist",
                )

        if dedupe_within_session and found:
            for e in state.events:
                if e.id == event.id or e.kind != EventKind.TOOL_CALL:
                    continue
                if tool is not None and e.tool_name != tool:
                    continue
                pf, pv = _resolve_path(e.tool_args or {}, field)
                if pf and _norm(pv) == norm_value:
                    return PredicateResult(
                        passed=False,
                        expected=f"'{field}' targeted at most once per session",
                        actual=str(value),
                        message=f"Tool '{event.tool_name}' already targeted '{field}'={value!r} this session",
                    )

        return PredicateResult(passed=True)

    check.predicate_name = "tool_arg_value_guard"  # type: ignore[attr-defined]
    return check


@predicate("required_disclosure")
def required_disclosure(
    tool: str | None,
    arg: str,
    must_contain,
    match: str = "all",
    pattern: bool = False,
    case_sensitive: bool = False,
):
    """A tool-call argument must contain required disclosure phrase(s).

    **Fail-closed**: if the named tool's ``arg`` is missing, ``None``, or
    non-string, the check fails (the disclosure cannot be present). Use it to
    require, e.g., that an outreach message states it is automated and on whose
    behalf, *before* the send tool fires.

    ``must_contain`` is a phrase or list of phrases. ``match="all"`` requires
    every phrase; ``match="any"`` requires at least one. With ``pattern=True``
    each phrase is a regular expression (``re.search``). Matching is
    case-insensitive unless ``case_sensitive=True``.
    """
    needles = [must_contain] if isinstance(must_contain, str) else list(must_contain)
    if match not in ("all", "any"):
        raise ValueError(f"required_disclosure: match must be 'all' or 'any', got {match!r}")
    reducer = all if match == "all" else any

    def _present(text: str, needle: str) -> bool:
        if pattern:
            import re

            flags = 0 if case_sensitive else re.IGNORECASE
            return re.search(needle, text, flags) is not None
        if case_sensitive:
            return needle in text
        return needle.lower() in text.lower()

    def check(event: Event, state: SessionState) -> PredicateResult:
        if event.kind != EventKind.TOOL_CALL:
            return PredicateResult(passed=True)
        if tool is not None and event.tool_name != tool:
            return PredicateResult(passed=True)
        val = (event.tool_args or {}).get(arg)
        if not isinstance(val, str):
            return PredicateResult(
                passed=False,
                expected=f"'{arg}' contains required disclosure",
                actual=f"{arg}={val!r}",
                message=f"Tool '{event.tool_name}' arg '{arg}' is missing or not text — disclosure absent",
            )
        if not reducer(_present(val, n) for n in needles):
            missing = [n for n in needles if not _present(val, n)]
            return PredicateResult(
                passed=False,
                expected=f"{match} of {needles}",
                actual=val,
                message=f"Tool '{event.tool_name}' arg '{arg}' missing required disclosure: {missing}",
            )
        return PredicateResult(passed=True)

    check.predicate_name = "required_disclosure"  # type: ignore[attr-defined]
    return check


@predicate("tool_host_within")
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
            host = _extract_host(value)
            if host is None:
                continue
            reason = _evaluate(host)
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


@predicate("consent_token_required")
def consent_token_required(
    tools,
    *,
    token_key: str = "user_consent",
    bind_args: list[str] | None = None,
    max_age_s: float | None = 300,
    secret=None,
):
    """Gate side-effecting tools on a fresh, action-bound consent token.

    Raises the bar from "the model self-authorized" to "the host carried a
    consent token scoped to THIS exact action into the turn". For each call to
    a tool in ``tools``, a token is read from ``event.metadata[token_key]``
    (per-turn), falling back to ``state.metadata[token_key]``. The token is a
    dict ``{"action", "sig", "issued_at"}`` (use :func:`mint_consent_token` to
    produce one). The call passes only if:

    - the token is present;
    - its ``sig`` matches a signature recomputed from the live ``tool_name`` and
      the values of ``bind_args`` — so a token issued for a *different* action
      or different arguments is rejected (no replay);
    - it is fresh: ``time.time() - issued_at <= max_age_s`` (skipped when
      ``max_age_s is None``);
    - if ``secret`` is given, the signature is an HMAC verified with
      ``hmac.compare_digest`` (constant-time).

    Pair with ``on_fail="approve"`` to route a tokenless call to a human, or
    ``on_fail="block"`` to refuse outright. Honest bound: this validates that a
    matching, unexpired, action-bound token was presented; it cannot attest the
    token's origin beyond the shared-secret HMAC the host signs with.
    """
    tools_set = {tools} if isinstance(tools, str) else set(tools)

    def check(event: Event, state: SessionState) -> PredicateResult:
        if event.kind != EventKind.TOOL_CALL or event.tool_name not in tools_set:
            return PredicateResult(passed=True)
        token = (event.metadata or {}).get(token_key)
        if token is None:
            token = (state.metadata or {}).get(token_key)
        if not isinstance(token, dict):
            return PredicateResult(
                passed=False,
                expected=f"consent token for '{event.tool_name}'",
                actual="no token",
                message=f"Tool '{event.tool_name}' requires a consent token (none presented)",
            )

        import hmac

        expected = _action_sig(event.tool_name, event.tool_args, bind_args, secret)
        if not hmac.compare_digest(str(token.get("sig", "")), expected):
            return PredicateResult(
                passed=False,
                expected="consent token bound to this action",
                actual="signature mismatch",
                message=f"Consent token does not match this '{event.tool_name}' call (wrong action/args or bad secret)",
            )

        if max_age_s is not None:
            import time

            issued = token.get("issued_at")
            if not isinstance(issued, (int, float)) or (time.time() - issued) > max_age_s:
                return PredicateResult(
                    passed=False,
                    expected=f"token issued within {max_age_s:.0f}s",
                    actual=f"issued_at={issued!r}",
                    message=f"Consent token for '{event.tool_name}' is expired or undated",
                )
        return PredicateResult(passed=True)

    check.predicate_name = "consent_token_required"  # type: ignore[attr-defined]
    return check


def _name_matches(name, patterns) -> bool:
    """True if a tool name matches any glob pattern (``fetch_*``, ``send_email``)."""
    if not name:
        return False
    from fnmatch import fnmatch

    return any(fnmatch(name, p) for p in patterns)


@predicate("no_exfiltration_after_untrusted")
def no_exfiltration_after_untrusted(
    untrusted_tools=("web_fetch", "read_email", "search", "browse"),
    exfil_tools=("send_email", "http_post", "slack_post", "webhook"),
    untrusted_tag: str = "source",
    untrusted_tag_values=None,
    encoded_arg_min_len: int = 64,
):
    """Block an outbound call that fires *after* untrusted content entered the run.

    The classic injection→exfiltration chain: the agent reads attacker-controlled
    content (a fetched page, an email) and is then steered into sending data out.
    Either capability alone is fine; the **sequence** is the risk. This fails an
    exfil-tool call (name in ``exfil_tools``, or *any* tool whose arguments carry
    a long encoded blob of ``encoded_arg_min_len`` chars — an out-of-band channel
    even off the exfil list) when a prior event in the run was an untrusted
    ingest: a tool name in ``untrusted_tools``, or an event the host tagged
    ``metadata[untrusted_tag]`` (optionally restricted to ``untrusted_tag_values``).
    """
    untrusted_tools = tuple(untrusted_tools)
    exfil_tools = tuple(exfil_tools)
    tag_values = set(untrusted_tag_values) if untrusted_tag_values else None

    def _is_exfil(event: Event) -> bool:
        if _name_matches(event.tool_name, exfil_tools):
            return True
        if encoded_arg_min_len:
            import re

            blob = _args_blob(event.tool_args)
            if re.search(r"[A-Za-z0-9+/=_\-]{%d,}" % int(encoded_arg_min_len), blob):
                return True
        return False

    def _prior_untrusted(event: Event, state: SessionState) -> bool:
        for e in state.events:
            if e.id == event.id:
                continue
            if e.kind == EventKind.TOOL_CALL and _name_matches(e.tool_name, untrusted_tools):
                return True
            val = (e.metadata or {}).get(untrusted_tag)
            if val and (tag_values is None or val in tag_values):
                return True
        return False

    def check(event: Event, state: SessionState) -> PredicateResult:
        if event.kind != EventKind.TOOL_CALL or not _is_exfil(event):
            return PredicateResult(passed=True)
        if _prior_untrusted(event, state):
            return PredicateResult(
                passed=False,
                expected="no outbound call after untrusted ingest",
                actual=f"'{event.tool_name}' fired after untrusted content entered the run",
                message=f"Possible injection→exfil: '{event.tool_name}' sends out after an untrusted ingest",
            )
        return PredicateResult(passed=True)

    check.predicate_name = "no_exfiltration_after_untrusted"  # type: ignore[attr-defined]
    return check


@predicate("lethal_trifecta_guard")
def lethal_trifecta_guard(
    untrusted_sources,
    private_data_tools,
    egress_tools,
    *,
    taint_key: str = "untrusted",
    mode: str = "diagnostic",
):
    """Fail a run that combines all three legs of the "lethal trifecta".

    A run that has access to **untrusted input**, **private data**, and a way to
    **send data out** can be turned into an exfiltration tool by a prompt
    injection — any two legs alone are far safer. This fails when one run touches
    all three classes (tool names matched as globs; untrusted is also satisfied
    by an event tagged ``metadata[taint_key]``).

    ``mode="diagnostic"`` checks at session end; ``mode="incremental"`` fails the
    moment the third leg is touched. Each class must be non-empty.
    """
    untrusted_sources = tuple(untrusted_sources)
    private_data_tools = tuple(private_data_tools)
    egress_tools = tuple(egress_tools)
    if not untrusted_sources or not private_data_tools or not egress_tools:
        raise ValueError("lethal_trifecta_guard: all three tool classes must be non-empty")
    if mode not in ("diagnostic", "incremental"):
        raise ValueError(f"lethal_trifecta_guard: mode must be diagnostic/incremental, got {mode!r}")

    def check(event: Event, state: SessionState) -> PredicateResult:
        untrusted = private = egress = False
        for e in state.events:
            if e.kind == EventKind.TOOL_CALL:
                if _name_matches(e.tool_name, untrusted_sources):
                    untrusted = True
                if _name_matches(e.tool_name, private_data_tools):
                    private = True
                if _name_matches(e.tool_name, egress_tools):
                    egress = True
            if (e.metadata or {}).get(taint_key):
                untrusted = True
        present = [n for n, f in (("untrusted", untrusted), ("private-data", private), ("egress", egress)) if f]
        return PredicateResult(
            passed=not (untrusted and private and egress),
            expected="at most two of {untrusted, private-data, egress} in one run",
            actual=f"touched: {present}",
            message="Lethal trifecta: this run combined untrusted input, private-data access, and external egress",
        )

    check.predicate_name = "lethal_trifecta_guard"  # type: ignore[attr-defined]
    check._check_on = "session_end" if mode == "diagnostic" else "every_event"  # type: ignore[attr-defined]
    return check


@predicate("multi_party_approval_required")
def multi_party_approval_required(
    tools,
    n_required: int = 2,
    approvers=None,
    *,
    bind_args: list[str] | None = None,
    token_key: str = "approvals",
    max_age_s: float | None = 600,
    secret=None,
):
    """Dual-control: a high-risk tool needs a quorum of distinct signed approvals.

    A call to a tool in ``tools`` passes only when at least ``n_required`` valid,
    unexpired, action-bound approval tokens from **distinct** approver identities
    are presented at ``event.metadata[token_key]`` (or ``state.metadata`` as a
    fallback) — the classic two-person rule for irreversible actions (wire
    transfers, prod deploys). Two tokens from the same approver count once.

    Each token is ``{"approver", "action", "sig", "issued_at"}`` from
    :func:`mint_approval_token`. The signature covers the approver id and the
    bound argument values, so a token can't be re-pointed to another approver or
    a different call. ``approvers`` (if given) restricts who may sign; ``secret``
    upgrades the signature to an HMAC.
    """
    tools_set = {tools} if isinstance(tools, str) else set(tools)
    approvers_set = set(approvers) if approvers else None

    def check(event: Event, state: SessionState) -> PredicateResult:
        if event.kind != EventKind.TOOL_CALL or event.tool_name not in tools_set:
            return PredicateResult(passed=True)
        raw = (event.metadata or {}).get(token_key)
        if raw is None:
            raw = (state.metadata or {}).get(token_key)
        if isinstance(raw, dict):
            tokens = [raw]
        elif isinstance(raw, (list, tuple)):
            tokens = list(raw)
        else:
            tokens = []

        import hmac
        import time

        approved: set = set()
        for tok in tokens:
            if not isinstance(tok, dict):
                continue
            approver = tok.get("approver")
            if not approver:
                continue
            if approvers_set is not None and approver not in approvers_set:
                continue
            expected = _approval_sig(approver, event.tool_name, event.tool_args, bind_args, secret)
            if not hmac.compare_digest(str(tok.get("sig", "")), expected):
                continue
            if max_age_s is not None:
                issued = tok.get("issued_at")
                if not isinstance(issued, (int, float)) or (time.time() - issued) > max_age_s:
                    continue
            approved.add(approver)

        n = len(approved)
        return PredicateResult(
            passed=n >= n_required,
            expected=f">= {n_required} distinct approvals for '{event.tool_name}'",
            actual=f"{n} valid distinct approval(s)",
            message=f"Tool '{event.tool_name}' needs {n_required} distinct approvals; got {n}",
        )

    check.predicate_name = "multi_party_approval_required"  # type: ignore[attr-defined]
    return check


def _args_blob(tool_args) -> str:
    import json

    if not tool_args:
        return ""
    try:
        return json.dumps(tool_args, default=str)
    except (TypeError, ValueError):
        return str(tool_args)


def _looks_like_path(value: str) -> bool:
    return ("/" in value) or ("\\" in value) or value.startswith("~")


def _resolve_path(obj, path: str):
    """Walk a dotted path (dict keys + int list indices). Returns (found, value).

    ``"recipient.email"`` descends dict keys; numeric segments index lists or
    tuples (negative indices allowed). Returns ``(False, None)`` if any segment
    is absent or the container type doesn't match.
    """
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict):
            if part not in cur:
                return False, None
            cur = cur[part]
        elif isinstance(cur, (list, tuple)):
            try:
                idx = int(part)
            except ValueError:
                return False, None
            if -len(cur) <= idx < len(cur):
                cur = cur[idx]
            else:
                return False, None
        else:
            return False, None
    return True, cur


def _value_in(value: str, entries: set, match: str) -> bool:
    """True if ``value`` matches any of ``entries`` under the given match mode."""
    if match == "exact":
        return value in entries
    if match == "ci":
        v = value.casefold()
        return any(v == e.casefold() for e in entries)
    if match == "glob":
        from fnmatch import fnmatch

        return any(fnmatch(value, e) for e in entries)
    if match == "regex":
        import re

        return any(re.search(e, value) for e in entries)
    return False


def _as_ip(host: str):
    import ipaddress

    try:
        return ipaddress.ip_address(host)
    except ValueError:
        return None


def _extract_host(value: str):
    """Pull the host out of a URL or bare host[:port][/path]; lowercased, no brackets."""
    from urllib.parse import urlsplit

    v = value.strip()
    if not v:
        return None
    try:
        netloc_form = v if ("://" in v or v.startswith("//")) else "//" + v
        host = urlsplit(netloc_form).hostname
    except ValueError:
        return None
    return host.lower() if host else None


def _url_like(value: str) -> bool:
    """Heuristic: is this string worth treating as a URL/host for egress checks?"""
    v = value.strip()
    if not v or any(ch.isspace() for ch in v):
        return False
    if "://" in v:
        return True
    head = v.split("/")[0]
    hostpart = head.rsplit(":", 1)[0].strip("[]")
    if hostpart == "localhost":
        return True
    if _as_ip(hostpart) is not None:
        return True
    return ("." in hostpart) and all(ch.isalnum() or ch in ".-" for ch in hostpart)


def _host_matches(host: str, patterns: list[str]) -> bool:
    """True if host matches any glob host pattern or IP/CIDR in ``patterns``."""
    import ipaddress
    from fnmatch import fnmatch

    host_ip = _as_ip(host)
    for p in patterns:
        pl = p.lower()
        if host_ip is not None:
            try:
                net = ipaddress.ip_network(p, strict=False)
            except ValueError:
                net = None
            if net is not None and host_ip.version == net.version and host_ip in net:
                return True
        if fnmatch(host, pl):
            return True
    return False


def _is_private_host(host: str) -> bool:
    """True for localhost or a private/loopback/link-local/reserved IP literal."""
    if host == "localhost":
        return True
    ip = _as_ip(host)
    if ip is None:
        return False
    return ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved


def _canonical_action(action: str, tool_args, bind_args) -> str:
    """Stable canonical string of (action, bound-arg values) for signing."""
    import json

    payload = {"action": action}
    if bind_args:
        payload["args"] = {k: _resolve_path(tool_args or {}, k)[1] for k in bind_args}
    return json.dumps(payload, sort_keys=True, default=str)


def _action_sig(action: str, tool_args, bind_args, secret) -> str:
    """Signature binding a consent token to an action: HMAC if secret, else sha256."""
    import hashlib
    import hmac

    msg = _canonical_action(action, tool_args, bind_args).encode("utf-8")
    if secret:
        key = secret if isinstance(secret, (bytes, bytearray)) else str(secret).encode("utf-8")
        return hmac.new(key, msg, hashlib.sha256).hexdigest()
    return hashlib.sha256(msg).hexdigest()


def mint_consent_token(
    action: str,
    *,
    args: dict | None = None,
    bind_args: list[str] | None = None,
    secret=None,
    issued_at: float | None = None,
) -> dict:
    """Produce a consent token for :func:`consent_token_required` (host-side).

    Sign the exact ``action`` (tool name) and, if ``bind_args`` is given, the
    values of those argument paths in ``args`` — so the token only validates a
    call carrying the same values. ``issued_at`` defaults to ``time.time()``.
    """
    import time

    sig = _action_sig(action, args or {}, bind_args, secret)
    return {
        "action": action,
        "sig": sig,
        "issued_at": time.time() if issued_at is None else issued_at,
    }


def _approval_sig(approver: str, action: str, tool_args, bind_args, secret) -> str:
    """Signature for an approval token, binding the approver id + action + args."""
    import hashlib
    import hmac
    import json

    payload = {"approver": approver, "action": action}
    if bind_args:
        payload["args"] = {k: _resolve_path(tool_args or {}, k)[1] for k in bind_args}
    msg = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    if secret:
        key = secret if isinstance(secret, (bytes, bytearray)) else str(secret).encode("utf-8")
        return hmac.new(key, msg, hashlib.sha256).hexdigest()
    return hashlib.sha256(msg).hexdigest()


def mint_approval_token(
    approver: str,
    *,
    tool: str,
    args: dict | None = None,
    bind_args: list[str] | None = None,
    secret=None,
    issued_at: float | None = None,
) -> dict:
    """Produce one approver's token for :func:`multi_party_approval_required`."""
    import time

    sig = _approval_sig(approver, tool, args or {}, bind_args, secret)
    return {
        "approver": approver,
        "action": tool,
        "sig": sig,
        "issued_at": time.time() if issued_at is None else issued_at,
    }
