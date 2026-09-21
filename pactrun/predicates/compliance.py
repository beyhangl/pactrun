"""Compliance / transparency predicates.

Runtime-checkable governance obligations — e.g. disclosing that a user is
interacting with an AI system (EU AI Act Art. 50 transparency).
"""

from __future__ import annotations

from pactrun.core.enums import EventKind
from pactrun.core.models import Event, PredicateResult, SessionState
from pactrun.predicates.base import predicate


@predicate("ai_disclosure_in_output", owasp=("ASI09",))
def ai_disclosure_in_output(
    must_contain=("automated assistant", "responses are AI-generated"),
    match: str = "any",
    first_only: bool = True,
    pattern: bool = False,
    case_sensitive: bool = False,
    on_behalf_of: str | None = None,
):
    """Require an AI-disclosure phrase in the first user-facing reply.

    Transparency obligations often require telling users, "at the latest at the
    first interaction," that they're dealing with an AI. This checks the first
    reply (LLM/output event) for a disclosure phrase and, with
    ``first_only=True`` (default), latches — later replies need not repeat it.
    With ``first_only=False`` every reply must carry it.

    ``must_contain`` is a phrase or list; ``match="any"`` needs one, ``"all"``
    needs every. ``pattern=True`` treats phrases as regexes. **Fail-closed**: a
    first reply that is missing, empty, or non-text fails (no disclosure present).

    ``on_behalf_of`` enforces the *second* half of the obligation. The Art. 50
    guidance expects an agent to disclose both that it is artificial **and the
    person or entity it is acting for**; a reply that only says "I am an AI"
    satisfies half of it. Set this to the principal's name and it must appear
    in the same reply.
    """
    needles = [must_contain] if isinstance(must_contain, str) else list(must_contain)
    if match not in ("all", "any"):
        raise ValueError(f"ai_disclosure_in_output: match must be 'all' or 'any', got {match!r}")
    reducer = all if match == "all" else any

    def _present(text: str, needle: str) -> bool:
        if pattern:
            import re

            flags = 0 if case_sensitive else re.IGNORECASE
            return re.search(needle, text, flags) is not None
        if case_sensitive:
            return needle in text
        return needle.lower() in text.lower()

    def _reply_ok(text) -> bool:
        if not isinstance(text, str) or not text:
            return False
        if not reducer(_present(text, n) for n in needles):
            return False
        # Art. 50 expects the principal to be named too, not just AI-ness.
        return on_behalf_of is None or _present(text, on_behalf_of)

    def check(event: Event, state: SessionState) -> PredicateResult:
        replies = [e for e in state.events if e.kind in (EventKind.LLM_CALL, EventKind.OUTPUT)]
        if not replies:
            return PredicateResult(passed=True)
        targets = replies[:1] if first_only else replies
        for reply in targets:
            if not _reply_ok(reply.output):
                which = "first reply" if first_only else "a reply"
                return PredicateResult(
                    passed=False,
                    expected=f"{match} of {needles} in {which}",
                    actual=repr(reply.output)[:80],
                    message=f"Missing AI disclosure in {which}: expected {match} of {needles}",
                )
        return PredicateResult(passed=True)

    check.predicate_name = "ai_disclosure_in_output"  # type: ignore[attr-defined]
    return check
