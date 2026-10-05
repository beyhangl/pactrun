"""Content-security predicates — inbound injection detection + prompt-leak canary.

``no_injection_phrases`` scans untrusted inbound text (tool results / input) for
known prompt-injection signatures; ``canary_not_leaked`` detects a planted
system-prompt canary token appearing in the output. Both are heuristic
tripwires, not guarantees — pair with consent gating and sandboxing.
"""

from __future__ import annotations

from pactrun.core.models import Event, PredicateResult, SessionState
from pactrun.predicates.base import predicate

# Curated, whitespace/case-tolerant injection signatures, tiered by confidence.
_INJECTION_SIGNATURES: list[tuple[str, str]] = [
    (r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions|prompts?|context|messages?)", "high"),
    (r"disregard\s+(all\s+)?(previous|prior|the\s+above|your)\b", "high"),
    (r"forget\s+(everything|all|what|your\s+(instructions|rules|prompt))", "high"),
    (r"(reveal|print|repeat|show|output|display)\s+(your|the)\s+(system\s+prompt|initial\s+prompt|instructions)", "high"),
    (r"override\s+(your\s+)?(safety|guidelines|rules|instructions)", "high"),
    (r"you\s+are\s+now\s+(a|an|in|no\s+longer)\b", "medium"),
    (r"new\s+(instructions?|task|role|system\s+prompt)\s*[:=]", "medium"),
    (r"do\s+anything\s+now|\bDAN\s+mode\b", "medium"),
    (r"act\s+as\s+(if\s+)?(a|an|though)\b", "medium"),
    (r"(developer|system|admin|god)\s+mode", "low"),
    (r"</?\s*(system|instructions?|prompt)\s*>", "low"),
    (r"end\s+of\s+(prompt|instructions?)\b", "low"),
]

_CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}
_VALID_SCAN = {"input", "output", "tool_result"}


def _decoded_variants(text: str, decode) -> list[str]:
    """Original text plus URL- and base64-decoded views (best effort)."""
    variants = [text]
    if "url" in decode:
        from urllib.parse import unquote

        variants.append(unquote(text))
    if "base64" in decode:
        import base64
        import re

        # A payload is rarely a clean token: it sits after "payload=", "id_" or
        # a URL, so the matched run can start mid-group. Padding only ends a
        # payload, and the standard and URL-safe alphabets disagree on "+/"
        # versus "-_". Try each "="-free segment in both alphabets at every
        # alignment so a prefix cannot throw the decode off.
        for m in re.finditer(r"[A-Za-z0-9+/=_-]{16,}", text):
            for segment in re.split(r"=+", m.group()):
                if len(segment) < 16:
                    continue
                for alphabet in (b"+/", b"-_"):
                    for offset in range(4):
                        piece = segment[offset:]
                        padded = piece + "=" * (-len(piece) % 4)
                        try:
                            raw = base64.b64decode(padded, altchars=alphabet, validate=False)
                        except Exception:  # noqa: BLE001 - malformed base64 is simply skipped
                            continue
                        variants.append(raw.decode("utf-8", "ignore"))
    return variants


@predicate("no_injection_phrases", owasp=("ASI01", "ASI06",))
def no_injection_phrases(
    scan=("tool_result",),
    decode=(),
    extra_patterns=(),
    min_confidence: str = "medium",
    aggregate: bool = False,
):
    """Scan untrusted inbound text for known prompt-injection signatures.

    Checks the enabled ``scan`` surfaces — ``"tool_result"`` (default),
    ``"input"``, ``"output"`` — against a curated bank of instruction-override /
    role-reset / prompt-leak phrases (case- and whitespace-tolerant).
    ``min_confidence`` (``"low"`` | ``"medium"`` | ``"high"``) sets the bank
    tier; ``extra_patterns`` adds caller regexes (always checked). With
    ``decode`` (``"base64"`` / ``"url"``) each surface is also scanned after
    decoding, catching obfuscated payloads. ``aggregate`` reports every match
    instead of failing on the first.

    Heuristic tripwire, not a guarantee — a determined attacker can paraphrase
    around a static bank. Pair with ``consent_token_required`` / sandboxing.
    """
    import re

    scan = tuple(scan)
    bad = set(scan) - _VALID_SCAN
    if bad:
        raise ValueError(f"no_injection_phrases: unknown scan {sorted(bad)}")
    floor = _CONFIDENCE_RANK.get(min_confidence)
    if floor is None:
        raise ValueError(f"no_injection_phrases: min_confidence must be low/medium/high, got {min_confidence!r}")

    banks = [
        (re.compile(p, re.IGNORECASE), conf)
        for p, conf in _INJECTION_SIGNATURES
        if _CONFIDENCE_RANK[conf] >= floor
    ]
    banks += [(re.compile(p, re.IGNORECASE), "custom") for p in extra_patterns]

    def _surfaces(event: Event):
        out = []
        if "input" in scan:
            out.append(str(event.input or ""))
        if "output" in scan:
            out.append(str(event.output or ""))
        if "tool_result" in scan:
            out.append(str(event.tool_result or ""))
        return out

    def check(event: Event, state: SessionState) -> PredicateResult:
        hits = []
        for surface in _surfaces(event):
            if not surface:
                continue
            for variant in _decoded_variants(surface, decode):
                normalized = re.sub(r"\s+", " ", variant)
                for rx, conf in banks:
                    if rx.search(normalized):
                        label = rx.pattern[:48]
                        hits.append(f"{conf}:{label}")
                        if not aggregate:
                            return PredicateResult(
                                passed=False,
                                expected="no injection signatures in untrusted text",
                                actual=f"matched {conf} signature",
                                message=f"Possible prompt injection in inbound text ({conf} confidence)",
                            )
        if hits:
            return PredicateResult(
                passed=False,
                expected="no injection signatures in untrusted text",
                actual=f"{len(hits)} signature(s)",
                message=f"Possible prompt injection: {len(hits)} signature(s) matched",
            )
        return PredicateResult(passed=True)

    check.predicate_name = "no_injection_phrases"  # type: ignore[attr-defined]
    return check


def mint_canary(prefix: str = "pactrun") -> str:
    """Return a unique canary token to embed in a system prompt for leak detection."""
    import secrets

    return f"{prefix}-canary-{secrets.token_hex(8)}"


@predicate("canary_not_leaked", owasp=("ASI01",))
def canary_not_leaked(token: str, transforms=("verbatim", "base64", "reversed")):
    """Fail if a planted system-prompt canary token appears in the output.

    Embed a :func:`mint_canary` token in the system prompt; if it surfaces in an
    output, the model leaked its instructions. Detects the token under the
    enabled ``transforms``: ``"verbatim"``, ``"base64"`` (standard + urlsafe,
    padded/unpadded), ``"reversed"``, and the opt-in ``"spaced"``. Only catches
    *this* token under *these* transforms — paraphrase or novel encodings evade.
    """
    valid = {"verbatim", "base64", "reversed", "spaced"}
    bad = set(transforms) - valid
    if bad:
        raise ValueError(f"canary_not_leaked: unknown transforms {sorted(bad)}")

    reps: list[tuple[str, str]] = []
    if "verbatim" in transforms:
        reps.append(("verbatim", token))
    if "reversed" in transforms:
        reps.append(("reversed", token[::-1]))
    if "base64" in transforms:
        import base64

        raw = token.encode("utf-8")
        for enc in (base64.b64encode, base64.urlsafe_b64encode):
            s = enc(raw).decode("ascii")
            reps.append(("base64", s))
            reps.append(("base64", s.rstrip("=")))
    if "spaced" in transforms:
        reps.append(("spaced", " ".join(token)))

    def check(event: Event, state: SessionState) -> PredicateResult:
        text = str(event.output or "")
        if not text:
            return PredicateResult(passed=True)
        for kind, rep in reps:
            if rep and rep in text:
                return PredicateResult(
                    passed=False,
                    expected="system-prompt canary never appears in output",
                    actual=f"canary leaked ({kind})",
                    message=f"System-prompt leak: canary token surfaced in output ({kind} form)",
                )
        return PredicateResult(passed=True)

    check.predicate_name = "canary_not_leaked"  # type: ignore[attr-defined]
    return check
