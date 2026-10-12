"""Canonical, versioned signing for consent and multi-party approval tokens."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from typing import Any

from pactrun.predicates._argpath import _resolve_path

TOKEN_VERSION = 2


def _canonical_json(value: Any) -> str:
    """Encode JSON values without lossy string coercion or nonfinite numbers."""
    def validate(item: Any) -> None:
        if isinstance(item, dict):
            if not all(isinstance(key, str) for key in item):
                raise ValueError("Approval context and argument keys must be strings")
            for child in item.values():
                validate(child)
        elif isinstance(item, list):
            for child in item:
                validate(child)
        elif item is None or isinstance(item, (str, bool, int)):
            return
        elif isinstance(item, float) and math.isfinite(item):
            return
        else:
            raise ValueError("Approval context and arguments must contain finite JSON values")

    validate(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _secret_bytes(secret: str | bytes | bytearray | None) -> bytes | None:
    if secret is None:
        return None
    if not isinstance(secret, (str, bytes, bytearray)) or not secret:
        raise ValueError("Approval secret must be a nonempty string or bytes")
    return secret.encode("utf-8") if isinstance(secret, str) else bytes(secret)


def _token_sig(
    *,
    kind: str,
    action: str,
    tool_args: dict | None,
    bind_args: list[str] | None,
    secret: str | bytes | bytearray | None,
    issued_at: int | float,
    nonce: str,
    context: dict,
    approver: str | None = None,
) -> str:
    payload: dict[str, Any] = {
        "version": TOKEN_VERSION,
        "kind": kind,
        "action": action,
        "issued_at": issued_at,
        "nonce": nonce,
        "context": context,
    }
    if bind_args:
        bound = {}
        for path in bind_args:
            found, value = _resolve_path(tool_args or {}, path)
            if not found:
                raise ValueError(f"Approval argument path is missing: {path}")
            bound[path] = value
        payload["args"] = bound
    if approver is not None:
        payload["approver"] = approver
    message = _canonical_json(payload).encode("utf-8")
    key = _secret_bytes(secret)
    if key is not None:
        return hmac.new(key, message, hashlib.sha256).hexdigest()
    # Compatibility mode only: an unkeyed digest cannot authenticate approval.
    return hashlib.sha256(message).hexdigest()
