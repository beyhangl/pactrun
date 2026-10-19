"""Host-issued approval tokens bound to an action, time, nonce and context."""

from __future__ import annotations

import hmac
import json
import math
import time
import uuid
from collections.abc import Callable

from pactrun.approval_store import ApprovalReplayStore
from pactrun.core.enums import EventKind
from pactrun.core.models import Event, PredicateResult, SessionState
from pactrun.predicates._signing import TOKEN_VERSION, _canonical_json, _secret_bytes, _token_sig
from pactrun.predicates.base import predicate

ApprovalContext = dict | Callable[[Event, SessionState], dict] | None


def _validate_options(max_age_s, secret, replay_store) -> None:
    _secret_bytes(secret)
    if max_age_s is not None and (
        isinstance(max_age_s, bool)
        or not isinstance(max_age_s, (int, float))
        or not math.isfinite(max_age_s)
        or max_age_s < 0
    ):
        raise ValueError("max_age_s must be finite and nonnegative, or None")
    if replay_store is not None and secret is None:
        raise ValueError("Replay-protected approval requires an HMAC secret")


def _expected_context(context: ApprovalContext, event: Event, state: SessionState) -> dict:
    value = context(event, state) if callable(context) else {} if context is None else context
    if not isinstance(value, dict):
        raise ValueError("Approval context must be a host-owned dict")
    _canonical_json(value)
    return value


def _valid_token(token, *, kind, event, bind_args, secret, max_age_s, context, now) -> bool:
    if not isinstance(token, dict):
        return False
    keys = {"version", "kind", "action", "issued_at", "nonce", "context", "sig"}
    if kind == "approval":
        keys.add("approver")
    if set(token) != keys or type(token.get("version")) is not int or token["version"] != TOKEN_VERSION:
        return False
    if token.get("kind") != kind or token.get("action") != event.tool_name:
        return False
    issued = token.get("issued_at")
    if isinstance(issued, bool) or not isinstance(issued, (int, float)):
        return False
    if not math.isfinite(issued) or issued > now:
        return False
    if max_age_s is not None and now - issued > max_age_s:
        return False
    nonce, signature = token.get("nonce"), token.get("sig")
    if not isinstance(nonce, str) or not nonce or not isinstance(signature, str):
        return False
    approver = token.get("approver") if kind == "approval" else None
    if kind == "approval" and (not isinstance(approver, str) or not approver):
        return False
    try:
        if _canonical_json(token.get("context")) != _canonical_json(context):
            return False
        expected = _token_sig(
            kind=kind, action=event.tool_name, tool_args=event.tool_args,
            bind_args=bind_args, secret=secret, issued_at=issued, nonce=nonce,
            context=context, approver=approver,
        )
        return hmac.compare_digest(signature, expected)
    except (TypeError, ValueError, OverflowError):
        return False


def _consume(replay_store: ApprovalReplayStore, nonces: list[str], state: SessionState) -> bool:
    if state.evaluation_mode != "monitor":
        return replay_store.consume_many(nonces)
    # Shadow evaluation is isolated from durable grants. Simulate duplicate use
    # within this session only; historical store consumption is not reconstructed.
    if len(set(nonces)) != len(nonces) or state.approval_nonces.intersection(nonces):
        return False
    state.approval_nonces.update(nonces)
    return True


def _shadow_restore(check, replay_store):
    """Session restore hook: rebuild the monitor-mode shadow nonces only.

    In enforce mode the durable store already holds what the earlier run
    consumed, so restoring does nothing and never consumes a grant again.
    """
    def restore(event: Event, state: SessionState) -> None:
        if replay_store is not None and state.evaluation_mode == "monitor":
            check(event, state)
    return restore


def _now(event: Event, state: SessionState) -> float:
    historical = state.evaluation_mode == "monitor" and state.evaluation_clock == "event"
    value = event.timestamp if historical else time.time()
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Approval evaluation clock must be finite")
    return value


@predicate("consent_token_required", owasp=("ASI03", "ASI09"))
def consent_token_required(
    tools,
    *,
    token_key: str = "user_consent",
    bind_args: list[str] | None = None,
    max_age_s: float | None = 300,
    secret=None,
    context: ApprovalContext = None,
    replay_store: ApprovalReplayStore | None = None,
):
    """Validate a versioned host-issued token before a covered tool call.

    The signature covers action, selected arguments, timestamp, random nonce,
    and exact ``context``. Supply context as host-owned data or a callback that
    reads authoritative tenant/principal/policy/business state. Do not return
    model-supplied assertions as authoritative context. With no ``bind_args``,
    arguments are not bound; list every argument that changes the approved effect.

    ``replay_store`` optionally consumes a valid nonce atomically before returning
    success, even if another clause later denies the call or execution fails.
    Use ``on_fail="block"`` for enforcement; monitor/log modes do not prevent tools.
    Without a store, tokens are reusable. Monitor/replay never consumes the store:
    it simulates use within the trace, without knowing prior durable consumption.
    Replay uses event timestamps; live sessions use the wall clock.
    ``max_age_s=None`` disables expiration, but timestamps must still be finite
    and nonfuture. Old unsigned-timestamp formats are rejected; reissue tokens.

    A missing secret retains an unkeyed digest for compatibility/testing only;
    anyone can mint it. Store mode requires an HMAC secret. Shared-secret HMAC
    authenticates the signing host, not the identity of a named human approver.
    Neither validation nor nonce consumption proves external tool success or
    exactly-once execution; use authoritative backend idempotency and reconciliation.
    """
    _validate_options(max_age_s, secret, replay_store)
    tools_set = {tools} if isinstance(tools, str) else set(tools)

    def check(event: Event, state: SessionState) -> PredicateResult:
        if event.kind != EventKind.TOOL_CALL or event.tool_name not in tools_set:
            return PredicateResult(passed=True)
        token = (event.metadata or {}).get(token_key)
        if token is None:
            token = (state.metadata or {}).get(token_key)
        try:
            expected_context = _expected_context(context, event, state)
            valid = _valid_token(
                token, kind="consent", event=event, bind_args=bind_args,
                secret=secret, max_age_s=max_age_s, context=expected_context, now=_now(event, state),
            )
            if valid and replay_store is not None and isinstance(token, dict):
                valid = _consume(replay_store, [token["nonce"]], state)
        except Exception:
            # A failed host lookup or durable store cannot grant authorization.
            valid = False
        return PredicateResult(
            passed=valid,
            expected=f"valid unused consent token for '{event.tool_name}'" if replay_store else
                     f"valid consent token for '{event.tool_name}'",
            actual="valid" if valid else "missing, invalid, expired, used, or unavailable",
            message=f"Tool '{event.tool_name}' requires a valid host-issued consent token",
        )

    check.predicate_name = "consent_token_required"  # type: ignore[attr-defined]
    check._restore = _shadow_restore(check, replay_store)  # type: ignore[attr-defined]
    return check


@predicate("multi_party_approval_required", owasp=("ASI03", "ASI09"))
def multi_party_approval_required(
    tools,
    n_required: int = 2,
    approvers=None,
    *,
    bind_args: list[str] | None = None,
    token_key: str = "approvals",
    max_age_s: float | None = 600,
    secret=None,
    context: ApprovalContext = None,
    replay_store: ApprovalReplayStore | None = None,
):
    """Require distinct, fresh host-asserted approver IDs for the exact action.

    Context/signing boundaries match :func:`consent_token_required`. An HMAC
    covers the asserted approver ID but does not independently prove its human
    identity. With a store, the first valid quorum in presentation order is
    consumed atomically; shortfalls consume nothing. A previously used token
    in that quorum rejects the call without consuming any other token.
    """
    _validate_options(max_age_s, secret, replay_store)
    if isinstance(n_required, bool) or not isinstance(n_required, int) or n_required < 1:
        raise ValueError("n_required must be a positive integer")
    tools_set = {tools} if isinstance(tools, str) else set(tools)
    approvers_set = set(approvers) if approvers is not None else None

    def check(event: Event, state: SessionState) -> PredicateResult:
        if event.kind != EventKind.TOOL_CALL or event.tool_name not in tools_set:
            return PredicateResult(passed=True)
        raw = (event.metadata or {}).get(token_key)
        if raw is None:
            raw = (state.metadata or {}).get(token_key)
        tokens = [raw] if isinstance(raw, dict) else list(raw) if isinstance(raw, (list, tuple)) else []
        approved: dict[str, str] = {}
        available = True
        try:
            expected_context = _expected_context(context, event, state)
            now = _now(event, state)
            for token in tokens:
                if not isinstance(token, dict):
                    continue
                approver = token.get("approver")
                if not isinstance(approver, str) or (approvers_set is not None and approver not in approvers_set):
                    continue
                if _valid_token(
                    token, kind="approval", event=event, bind_args=bind_args,
                    secret=secret, max_age_s=max_age_s, context=expected_context, now=now,
                ):
                    approved.setdefault(approver, token["nonce"])
            if len(approved) >= n_required and replay_store is not None:
                available = _consume(replay_store, list(approved.values())[:n_required], state)
        except Exception:
            available = False
        n = len(approved)
        return PredicateResult(
            passed=n >= n_required and available,
            expected=f">= {n_required} distinct approvals for '{event.tool_name}'",
            actual=f"{n} valid distinct approval(s); quorum unused and available={available}",
            message=f"Tool '{event.tool_name}' requires {n_required} valid distinct approvals",
        )

    check.predicate_name = "multi_party_approval_required"  # type: ignore[attr-defined]
    check._restore = _shadow_restore(check, replay_store)  # type: ignore[attr-defined]
    return check


def _mint_token(kind, action, *, args, bind_args, secret, issued_at, nonce, context, approver=None):
    if not isinstance(action, str) or not action:
        raise ValueError("Approval action must be a nonempty string")
    issued = time.time() if issued_at is None else issued_at
    if isinstance(issued, bool) or not isinstance(issued, (int, float)) or not math.isfinite(issued):
        raise ValueError("issued_at must be a finite number")
    nonce = uuid.uuid4().hex if nonce is None else nonce
    if not isinstance(nonce, str) or not nonce:
        raise ValueError("Approval nonce must be a nonempty string")
    context = {} if context is None else context
    if not isinstance(context, dict):
        raise ValueError("Approval context must be a dict of finite JSON values")
    # Detach the payload from the host's mutable business-state object.
    context = json.loads(_canonical_json(context))
    token = {
        "version": TOKEN_VERSION, "kind": kind, "action": action,
        "issued_at": issued, "nonce": nonce, "context": context,
    }
    if kind == "approval":
        if not isinstance(approver, str) or not approver:
            raise ValueError("Approver must be a nonempty host-asserted ID")
        token["approver"] = approver
    token["sig"] = _token_sig(
        kind=kind, action=action, tool_args=args, bind_args=bind_args, secret=secret,
        issued_at=issued, nonce=nonce, context=context, approver=approver,
    )
    return token


def mint_consent_token(
    action: str,
    *,
    args: dict | None = None,
    bind_args: list[str] | None = None,
    secret=None,
    issued_at: float | None = None,
    nonce: str | None = None,
    context: dict | None = None,
) -> dict:
    """Mint a versioned host-side token; the signing host must verify authority.

    Bind selected effect arguments and authoritative context. The nonce defaults
    to a random UUID; timestamp/context/nonce are all authenticated with a secret.
    No secret produces an unauthenticated compatibility/testing digest.
    """
    return _mint_token(
        "consent", action, args=args, bind_args=bind_args, secret=secret,
        issued_at=issued_at, nonce=nonce, context=context,
    )


def mint_approval_token(
    approver: str,
    *,
    tool: str,
    args: dict | None = None,
    bind_args: list[str] | None = None,
    secret=None,
    issued_at: float | None = None,
    nonce: str | None = None,
    context: dict | None = None,
) -> dict:
    """Mint a token covering the host-asserted approver ID and exact context."""
    return _mint_token(
        "approval", tool, args=args, bind_args=bind_args, secret=secret,
        issued_at=issued_at, nonce=nonce, context=context, approver=approver,
    )
