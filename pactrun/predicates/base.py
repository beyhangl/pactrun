"""Predicate registration system.

Predicates are factory functions that return a checker function.
The checker takes (Event, SessionState) and returns PredicateResult.

Usage::

    @predicate("cost_under")
    def cost_under(max_usd: float):
        def check(event, state):
            return PredicateResult(
                passed=state.total_cost_usd <= max_usd,
                expected=f"<= ${max_usd:.4f}",
                actual=f"${state.total_cost_usd:.4f}",
            )
        return check
"""

from __future__ import annotations

from typing import Any, Callable

from pactrun.core.models import Event, PredicateResult, SessionState


# Global registry: name → factory function
_PREDICATE_REGISTRY: dict[str, Callable[..., Callable[[Event, SessionState], PredicateResult]]] = {}


def predicate(name: str) -> Callable:
    """Decorator to register a predicate factory function.

    The decorated function should accept configuration args and return
    a checker function ``(Event, SessionState) -> PredicateResult``.
    """
    def decorator(fn: Callable) -> Callable:
        _PREDICATE_REGISTRY[name] = fn
        fn.predicate_name = name  # type: ignore[attr-defined]
        return fn
    return decorator


def get_predicate(name: str) -> Callable:
    """Look up a predicate factory by name.

    Raises KeyError if the predicate is not registered.
    """
    if name not in _PREDICATE_REGISTRY:
        raise KeyError(
            f"Unknown predicate: {name!r}. "
            f"Available: {sorted(_PREDICATE_REGISTRY.keys())}"
        )
    return _PREDICATE_REGISTRY[name]


def list_predicates() -> list[str]:
    """Return sorted list of all registered predicate names."""
    return sorted(_PREDICATE_REGISTRY.keys())
