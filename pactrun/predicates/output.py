"""Output predicates — validate agent outputs."""

from __future__ import annotations

import json
import re

from pactrun.core.enums import EventKind
from pactrun.core.models import Event, PredicateResult, SessionState
from pactrun.predicates.base import predicate


_PII_PATTERNS = [
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "email"),
    (r"\b\d{3}[-.]?\d{2}[-.]?\d{4}\b", "SSN"),
    (r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "phone"),
    (r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", "credit card"),
]


@predicate("no_pii")
def no_pii():
    """Output must not contain PII (email, SSN, phone, credit card)."""
    def check(event: Event, state: SessionState) -> PredicateResult:
        output = str(event.output or "")
        if not output:
            return PredicateResult(passed=True)
        for pattern, pii_type in _PII_PATTERNS:
            match = re.search(pattern, output)
            if match:
                return PredicateResult(
                    passed=False,
                    expected="no PII in output",
                    actual=f"Found {pii_type}: {match.group()[:20]}...",
                    message=f"Output contains {pii_type}",
                )
        return PredicateResult(passed=True)
    check.predicate_name = "no_pii"  # type: ignore[attr-defined]
    return check


@predicate("output_contains")
def output_contains(substring: str, case_sensitive: bool = True):
    """Output must contain this substring."""
    def check(event: Event, state: SessionState) -> PredicateResult:
        if not state.output_history:
            return PredicateResult(passed=False, message="No output to check")
        last_output = state.output_history[-1]
        if case_sensitive:
            passed = substring in last_output
        else:
            passed = substring.lower() in last_output.lower()
        return PredicateResult(
            passed=passed,
            expected=f"contains '{substring}'",
            actual=last_output[:100],
            message=f"Output does not contain '{substring}'",
        )
    check.predicate_name = "output_contains"  # type: ignore[attr-defined]
    check._check_on = "session_end"  # type: ignore[attr-defined]
    return check


@predicate("output_matches")
def output_matches(pattern: str):
    """Output must match regex pattern."""
    def check(event: Event, state: SessionState) -> PredicateResult:
        if not state.output_history:
            return PredicateResult(passed=False, message="No output to check")
        last_output = state.output_history[-1]
        passed = bool(re.search(pattern, last_output))
        return PredicateResult(
            passed=passed,
            expected=f"matches '{pattern}'",
            actual=last_output[:100],
            message=f"Output does not match pattern '{pattern}'",
        )
    check.predicate_name = "output_matches"  # type: ignore[attr-defined]
    check._check_on = "session_end"  # type: ignore[attr-defined]
    return check


@predicate("max_output_length")
def max_output_length(max_chars: int):
    """Output must not exceed character limit."""
    def check(event: Event, state: SessionState) -> PredicateResult:
        output = str(event.output or "")
        return PredicateResult(
            passed=len(output) <= max_chars,
            expected=f"<= {max_chars} chars",
            actual=f"{len(output)} chars",
            message=f"Output length {len(output)} exceeds limit {max_chars}",
        )
    check.predicate_name = "max_output_length"  # type: ignore[attr-defined]
    return check


@predicate("output_must_not_contain")
def output_must_not_contain(pattern: str):
    """Output must not match this regex pattern."""
    def check(event: Event, state: SessionState) -> PredicateResult:
        output = str(event.output or "")
        if not output:
            return PredicateResult(passed=True)
        match = re.search(pattern, output)
        if match:
            return PredicateResult(
                passed=False,
                expected=f"does not match '{pattern}'",
                actual=f"matched: {match.group()[:50]}",
                message=f"Output contains forbidden pattern '{pattern}'",
            )
        return PredicateResult(passed=True)
    check.predicate_name = "output_must_not_contain"  # type: ignore[attr-defined]
    return check


@predicate("valid_json")
def valid_json():
    """The final output must parse as JSON (checked at session end)."""
    def check(event: Event, state: SessionState) -> PredicateResult:
        if not state.output_history:
            return PredicateResult(passed=False, message="No output to check")
        try:
            json.loads(state.output_history[-1])
        except (ValueError, TypeError) as exc:
            return PredicateResult(
                passed=False, expected="valid JSON output",
                actual=str(exc), message=f"Output is not valid JSON: {exc}",
            )
        return PredicateResult(passed=True)
    check.predicate_name = "valid_json"  # type: ignore[attr-defined]
    check._check_on = "session_end"  # type: ignore[attr-defined]
    return check


@predicate("json_schema_valid")
def json_schema_valid(schema: dict):
    """Final output must parse as JSON and validate against a JSON Schema.

    Requires the ``jsonschema`` extra: ``pip install 'pactrun[jsonschema]'``.
    """
    def check(event: Event, state: SessionState) -> PredicateResult:
        if not state.output_history:
            return PredicateResult(passed=False, message="No output to check")
        try:
            from jsonschema import Draft202012Validator
        except ImportError as exc:
            raise ImportError(
                "json_schema_valid requires the 'jsonschema' package. "
                "Install it with: pip install 'pactrun[jsonschema]'"
            ) from exc
        try:
            data = json.loads(state.output_history[-1])
        except (ValueError, TypeError) as exc:
            return PredicateResult(
                passed=False, expected="JSON matching schema",
                actual=f"not JSON: {exc}", message=f"Output is not valid JSON: {exc}",
            )
        errors = sorted(Draft202012Validator(schema).iter_errors(data), key=lambda e: list(e.path))
        if errors:
            return PredicateResult(
                passed=False, expected="JSON matching schema",
                actual=errors[0].message,
                message=f"Output JSON does not match schema: {errors[0].message}",
            )
        return PredicateResult(passed=True)
    check.predicate_name = "json_schema_valid"  # type: ignore[attr-defined]
    check._check_on = "session_end"  # type: ignore[attr-defined]
    return check


# Best-effort credential patterns (regex, label). NON-EXHAUSTIVE starter set —
# regex detection is best-effort, never a guarantee.
_SECRET_PATTERNS: list[tuple[str, str]] = [
    (r"AKIA[0-9A-Z]{16}", "AWS access key id"),
    (r"ghp_[A-Za-z0-9]{36}", "GitHub token"),
    (r"sk-[A-Za-z0-9]{20,}T3BlbkFJ[A-Za-z0-9]{20,}", "provider API key"),
    (r"AIza[0-9A-Za-z_\-]{35}", "Google API key"),
    (r"xox[bpas]-[0-9A-Za-z\-]{10,}", "Slack token"),
    (r"sk_live_[0-9A-Za-z]{20,}", "payment live key"),
    (r"eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+", "JWT"),
    (r"-----BEGIN [A-Z ]*PRIVATE KEY-----", "private key"),
]


@predicate("no_secrets")
def no_secrets(scan_tool_args: bool = False):
    """Output (and optionally tool args) must not contain leaked credentials.

    Best-effort regex scan for API keys / tokens / private keys; the violation
    message redacts the matched value so it does not re-leak the secret. The
    pattern bank is non-exhaustive and not a guarantee.
    """
    patterns = [(re.compile(p), label) for p, label in _SECRET_PATTERNS]

    def check(event: Event, state: SessionState) -> PredicateResult:
        blobs = [str(event.output or "")]
        if scan_tool_args and event.tool_args:
            blobs.append(json.dumps(event.tool_args, default=str))
        for blob in blobs:
            if not blob:
                continue
            for rx, label in patterns:
                match = rx.search(blob)
                if match:
                    return PredicateResult(
                        passed=False, expected="no leaked credentials",
                        actual=f"Found {label}: {match.group()[:8]}...redacted",
                        message=f"Output contains a leaked credential ({label})",
                    )
        return PredicateResult(passed=True)
    check.predicate_name = "no_secrets"  # type: ignore[attr-defined]
    return check


@predicate("tenant_response_isolation")
def tenant_response_isolation(
    tenant_key="tenant",
    *,
    response_tag_key: str = "tenant",
    known_tenants: list[str] | None = None,
):
    """Fail closed if a response carries a tenant tag other than the run's tenant.

    A cross-customer-bleed guard, keyed on *provenance* rather than content
    (the sibling of :func:`no_pii` / :func:`no_secrets`, which scan content).
    The run's active tenant comes from ``state.metadata[tenant_key]`` — set it
    with ``Session(metadata={"tenant": "acme"})`` — or from a
    ``Callable[[SessionState], str]`` passed as ``tenant_key``. Each event's
    tenant tag comes from ``event.metadata[response_tag_key]``.

    - If the run has no bound tenant, the check **fails closed** (you asked for
      isolation but didn't say whose data this is).
    - If an event is tagged with a tenant different from the run's, it fails.
    - With ``known_tenants``, the response text is also scanned for any *other*
      tenant's identifier leaking into this run's output.
    """
    def _run_tenant(state: SessionState):
        if callable(tenant_key):
            return tenant_key(state)
        return (state.metadata or {}).get(tenant_key)

    def check(event: Event, state: SessionState) -> PredicateResult:
        active = _run_tenant(state)
        if not active:
            return PredicateResult(
                passed=False,
                expected="a bound run tenant",
                actual="unbound",
                message="tenant_response_isolation: no active tenant on the run (fail-closed)",
            )

        tag = (event.metadata or {}).get(response_tag_key)
        if tag is not None and tag != active:
            return PredicateResult(
                passed=False,
                expected=f"tenant == {active!r}",
                actual=f"tenant == {tag!r}",
                message=f"Response tagged for tenant {tag!r} surfaced in a {active!r} run",
            )

        if known_tenants:
            text = str(event.output or "")
            if text:
                for other in known_tenants:
                    if other != active and other in text:
                        return PredicateResult(
                            passed=False,
                            expected=f"only {active!r} data in output",
                            actual=f"found {other!r}",
                            message=f"Output in a {active!r} run references another tenant {other!r}",
                        )
        return PredicateResult(passed=True)

    check.predicate_name = "tenant_response_isolation"  # type: ignore[attr-defined]
    return check
