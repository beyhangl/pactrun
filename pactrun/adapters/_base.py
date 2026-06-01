"""Shared adapter utilities."""

from __future__ import annotations

from pactrun.session import get_active_session, Session


def get_session() -> Session | None:
    """Get the active enforcement session, if any.

    Adapters call this to find the session to emit events to.
    Returns None if no session is active (graceful no-op).
    """
    return get_active_session()
