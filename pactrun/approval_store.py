"""Atomic, durable consumption of approval nonces.

This records authorization use, not tool success. Once consumed, a nonce is
never released automatically: a failed or uncertain tool attempt requires
reconciliation and a new host-issued approval. Backend idempotency remains
the host's responsibility.
"""

from __future__ import annotations

import math
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol


class ApprovalReplayStore(Protocol):
    """Store interface; a quorum must be consumed atomically, all or none."""

    def consume_many(self, nonces: Sequence[str]) -> bool:
        """Return False without consuming any nonce if one was already used."""
        ...


class SQLiteApprovalStore:
    """Persist consumed nonces in a host-owned SQLite file.

    Instances and processes sharing the same file serialize consumption with
    SQLite transactions. Keep the file across sessions/restarts; do not place
    it in model-accessible storage. Deleting it also deletes replay protection.
    The table retains nonces indefinitely because token lifetime may be unlimited.
    """

    def __init__(self, path: str | Path, *, timeout_s: float = 5.0) -> None:
        self.path = str(path)
        if self.path == ":memory:" or not self.path:
            raise ValueError("SQLiteApprovalStore requires a persistent file path")
        if isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")
        self.timeout_s = timeout_s
        connection = sqlite3.connect(self.path, timeout=self.timeout_s)
        try:
            with connection:
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS pactrun_used_approvals "
                    "(nonce TEXT PRIMARY KEY NOT NULL, consumed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
                )
        finally:
            connection.close()

    def consume_many(self, nonces: Sequence[str]) -> bool:
        """Consume a nonempty set atomically; database failures propagate closed."""
        values = list(nonces)
        if not values or any(not isinstance(nonce, str) or not nonce for nonce in values):
            raise ValueError("Approval nonces must be nonempty strings")
        if len(set(values)) != len(values):
            return False
        connection = sqlite3.connect(self.path, timeout=self.timeout_s)
        try:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.executemany(
                    "INSERT INTO pactrun_used_approvals (nonce) VALUES (?)",
                    [(nonce,) for nonce in values],
                )
            except sqlite3.IntegrityError:
                connection.rollback()
                return False
            connection.commit()
            return True
        finally:
            connection.close()
