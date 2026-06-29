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


# Invisible / smuggled-text codepoint classes (used by no_invisible_text).
_ZERO_WIDTH_CPS = {0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF, 0x180E, 0x00AD}
_BIDI_CPS = {0x202A, 0x202B, 0x202C, 0x202D, 0x202E, 0x2066, 0x2067, 0x2068, 0x2069}
_VALID_INVIS_DETECT = {"zero_width", "tags_block", "bidi", "homoglyph"}
_VALID_INVIS_SCAN = {"input", "output", "tool_result"}


def _homoglyph_hits(text: str):
    """Tokens mixing ASCII letters with Cyrillic/Greek confusables (opt-in, noisy)."""
    hits = []
    for token in text.split():
        has_latin = any("a" <= c.lower() <= "z" for c in token)
        confusable = next(
            (ord(c) for c in token if 0x0400 <= ord(c) <= 0x04FF or 0x0370 <= ord(c) <= 0x03FF),
            None,
        )
        if has_latin and confusable is not None:
            hits.append((confusable, "homoglyph"))
    return hits


@predicate("no_invisible_text")
def no_invisible_text(
    scan=("input", "output", "tool_result"),
    detect=("zero_width", "tags_block", "bidi"),
    max_occurrences: int = 0,
):
    """Flag hidden / smuggled-instruction codepoints in agent text surfaces.

    LLM prompt injections often hide instructions in characters a human never
    sees: zero-width spaces, the Unicode **Tags block** (``U+E0000``–``U+E007F``,
    used to smuggle ASCII), and **bidi overrides** that reorder displayed text.
    This classifies codepoints (it does not regex visible text) across the
    chosen surfaces — by default ``input`` (the prompt), ``output``, and
    ``tool_result`` (where injected content arrives).

    ``detect`` selects classes: ``"zero_width"``, ``"tags_block"``, ``"bidi"``,
    and the opt-in ``"homoglyph"`` (ASCII mixed with Cyrillic/Greek look-alikes
    — noisier, off by default). Fails when more than ``max_occurrences`` hidden
    codepoints are found. The message names the codepoints (``U+200B``) and
    never echoes the characters themselves.
    """
    detect = tuple(detect)
    scan = tuple(scan)
    bad_detect = set(detect) - _VALID_INVIS_DETECT
    if bad_detect:
        raise ValueError(f"no_invisible_text: unknown detect {sorted(bad_detect)}")
    bad_scan = set(scan) - _VALID_INVIS_SCAN
    if bad_scan:
        raise ValueError(f"no_invisible_text: unknown scan {sorted(bad_scan)}")

    def _scan_text(text: str):
        hits = []
        for ch in text:
            cp = ord(ch)
            if "zero_width" in detect and cp in _ZERO_WIDTH_CPS:
                hits.append((cp, "zero_width"))
            elif "tags_block" in detect and 0xE0000 <= cp <= 0xE007F:
                hits.append((cp, "tags_block"))
            elif "bidi" in detect and cp in _BIDI_CPS:
                hits.append((cp, "bidi"))
        if "homoglyph" in detect:
            hits.extend(_homoglyph_hits(text))
        return hits

    def check(event: Event, state: SessionState) -> PredicateResult:
        surfaces = []
        if "input" in scan:
            surfaces.append(str(event.input or ""))
        if "output" in scan:
            surfaces.append(str(event.output or ""))
        if "tool_result" in scan:
            surfaces.append(str(event.tool_result or ""))
        hits = []
        for text in surfaces:
            if text:
                hits.extend(_scan_text(text))
        if len(hits) > max_occurrences:
            cats = sorted({c for _, c in hits})
            cps = sorted({cp for cp, _ in hits})
            sample = ", ".join(f"U+{cp:04X}" for cp in cps[:5])
            return PredicateResult(
                passed=False,
                expected=f"<= {max_occurrences} hidden codepoint(s)",
                actual=f"{len(hits)} hidden codepoint(s) [{', '.join(cats)}]: {sample}",
                message=f"Hidden text detected ({', '.join(cats)}): {sample}",
            )
        return PredicateResult(passed=True)

    check.predicate_name = "no_invisible_text"  # type: ignore[attr-defined]
    return check


_VALID_EXFIL_FORMS = {"markdown_image", "markdown_link", "html_image", "html_link"}
_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(\s*<?([^)>\s]+)")
_MD_LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\(\s*<?([^)>\s]+)")
_HTML_IMG_RE = re.compile(r"<img\b[^>]*?\bsrc\s*=\s*[\"']?([^\"'>\s]+)", re.IGNORECASE)
_HTML_A_RE = re.compile(r"<a\b[^>]*?\bhref\s*=\s*[\"']?([^\"'>\s]+)", re.IGNORECASE)


def _extract_exfil_links(text: str, forms):
    """Yield (url, is_image, form) from markdown/HTML image & link constructs."""
    out = []
    if "markdown_image" in forms:
        out += [(m.group(1), True, "markdown_image") for m in _MD_IMAGE_RE.finditer(text)]
    if "markdown_link" in forms:
        out += [(m.group(1), False, "markdown_link") for m in _MD_LINK_RE.finditer(text)]
    if "html_image" in forms:
        out += [(m.group(1), True, "html_image") for m in _HTML_IMG_RE.finditer(text)]
    if "html_link" in forms:
        out += [(m.group(1), False, "html_link") for m in _HTML_A_RE.finditer(text)]
    return out


def _has_encoded_query(url: str, leak_param_names) -> bool:
    from urllib.parse import parse_qsl, urlsplit

    try:
        query = urlsplit(url if "://" in url else "//" + url).query
    except ValueError:
        return False
    for key, value in parse_qsl(query):
        if leak_param_names and key in leak_param_names:
            return True
        if len(value) >= 32 and re.fullmatch(r"[A-Za-z0-9+/=_-]+", value):
            return True
    return False


@predicate("no_exfil_links")
def no_exfil_links(
    allow_hosts: list | None = None,
    forms=("markdown_image", "markdown_link", "html_image", "html_link"),
    block_images: bool = True,
    flag_encoded_query: bool = False,
    block_private: bool = True,
    deny_hosts: list | None = None,
    leak_param_names: list | None = None,
):
    """Catch data exfiltration through links/images rendered from agent output.

    A prompt-injected agent can leak data by emitting a markdown image whose URL
    encodes the data — the victim's renderer fetches it automatically (a
    zero-click channel). This extracts URLs from markdown/HTML image and link
    constructs in ``event.output`` and checks each host (reusing the same glob/
    CIDR matcher as :func:`tool_host_within`):

    - ``deny_hosts`` match, or a private/loopback host (``block_private``), fails;
    - with ``block_images`` (default), an image to a non-``allow_hosts`` host
      fails — the zero-click case;
    - with ``allow_hosts`` set, every link host must match it;
    - with ``flag_encoded_query``, a URL whose query carries a long base64-ish
      value (or a ``leak_param_names`` key) fails.

    Relative paths, anchors, ``mailto:``/``tel:``/``data:`` are ignored.
    """
    from pactrun.predicates.tools import _extract_host, _host_matches, _is_private_host

    forms = tuple(forms)
    bad = set(forms) - _VALID_EXFIL_FORMS
    if bad:
        raise ValueError(f"no_exfil_links: unknown forms {sorted(bad)}")

    def _reason(url: str, is_image: bool):
        low = url.strip().lower()
        if not low or low.startswith(("#", "./", "../", "data:", "mailto:", "tel:")):
            return None
        if low.startswith("/") and not low.startswith("//"):
            return None  # absolute same-origin path, no host
        host = _extract_host(url)
        if host is None:
            return None
        if deny_hosts and _host_matches(host, deny_hosts):
            return f"host '{host}' is on the deny list"
        if block_private and _is_private_host(host):
            return f"host '{host}' is private/loopback"
        if is_image and block_images:
            if not allow_hosts or not _host_matches(host, allow_hosts):
                return f"image points to non-allowlisted host '{host}' (zero-click exfil)"
        elif allow_hosts is not None and not _host_matches(host, allow_hosts):
            return f"host '{host}' is not in the allow list"
        if flag_encoded_query and _has_encoded_query(url, leak_param_names):
            return "URL query carries an encoded payload"
        return None

    def check(event: Event, state: SessionState) -> PredicateResult:
        text = str(event.output or "")
        if not text:
            return PredicateResult(passed=True)
        for url, is_image, form in _extract_exfil_links(text, forms):
            reason = _reason(url, is_image)
            if reason:
                shown = url if len(url) <= 120 else url[:117] + "..."
                return PredicateResult(
                    passed=False,
                    expected="output links/images reach only allowed hosts",
                    actual=f"{form}: {shown}",
                    message=f"Possible data exfiltration via {form}: {reason} ({shown})",
                )
        return PredicateResult(passed=True)

    check.predicate_name = "no_exfil_links"  # type: ignore[attr-defined]
    return check
