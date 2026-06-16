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
