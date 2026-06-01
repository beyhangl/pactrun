"""pactrun.wrap() — a one-line pre-call enforcement gate.

Wrap an LLM client and every call is checked **before** it bills: pactrun
estimates the worst-case cost of the next call (your prompt + the maximum
output you allow) and refuses it if that would push your run over budget — and
it enforces tool / loop / turn limits on the response before your code proceeds.

    import openai, pactrun

    client = pactrun.wrap(
        openai.OpenAI(),
        max_cost="$0.50",          # whole-run budget — refused before the call that would cross it
        no_loops=True,             # stop repeating tool loops
        forbid_tools=["delete_account"],
    )
    client.chat.completions.create(model="gpt-4.1", messages=[...])

Honesty note: the pre-call cost check is a **worst-case bound** — you cannot know
the real completion-token count before a call, and reasoning models (o3/o4) can
spend hidden tokens beyond the visible estimate. The post-call check (on the real
usage reported by the provider) is the exact backstop. ``wrap()`` supports the
OpenAI and Anthropic client SDKs today; for other providers use the framework
adapters with a ``Contract`` directly.
"""

from __future__ import annotations

from typing import Any

from pactrun.contract import Contract
from pactrun.core.enums import ClauseKind, OnFail, Severity
from pactrun.core.errors import ViolationError
from pactrun.core.models import Violation
from pactrun.predicates import (
    cost_under,
    drift_bounds,
    max_turns as max_turns_predicate,
    must_not_call,
    no_loops as no_loops_predicate,
    tools_allowed as tools_allowed_predicate,
)
from pactrun.recovery.engine import apply_recovery


def _parse_cost(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value).replace("$", "").strip())


def wrap(
    client: Any,
    *,
    max_cost: Any = None,
    max_turns: int | None = None,
    no_loops: bool = False,
    max_drift: float | None = None,
    forbid_tools: list[str] | None = None,
    tools_allowed: list[str] | None = None,
    on_violation: str = "block",
    default_max_tokens: int = 4096,
    escalation_handler: Any = None,
) -> "GuardedClient":
    """Wrap an OpenAI/Anthropic client with a pre-call enforcement gate.

    Returns a drop-in client proxy: call it exactly as you would the original
    (``client.chat.completions.create`` / ``client.messages.create``). On each
    call the worst-case cost is checked *before* the request, and tool/loop/turn
    limits are checked on the response before it is handed back.
    """
    budget = _parse_cost(max_cost)

    contract = Contract("pactrun.wrap").on_violation(on_violation)
    if budget is not None:
        contract.require(cost_under(budget))
    if isinstance(max_turns, int):
        contract.require(max_turns_predicate(max_turns))
    if no_loops:
        contract.require(no_loops_predicate())
    if max_drift is not None:
        contract.require(drift_bounds(cost_pct=max_drift))
    for tool in forbid_tools or []:
        contract.forbid(must_not_call(tool))
    if tools_allowed:
        contract.require(tools_allowed_predicate(list(tools_allowed)))

    return GuardedClient(
        client,
        contract,
        budget=budget,
        on_fail=OnFail(on_violation),
        default_max_tokens=default_max_tokens,
        escalation_handler=escalation_handler,
    )


class _Namespace:
    """A tiny attribute holder used to mirror nested SDK call paths."""

    def __init__(self, **attrs: Any) -> None:
        self.__dict__.update(attrs)


class GuardedClient:
    """A client proxy that enforces a contract around every LLM call."""

    def __init__(
        self,
        client: Any,
        contract: Contract,
        *,
        budget: float | None,
        on_fail: OnFail,
        default_max_tokens: int,
        escalation_handler: Any = None,
    ) -> None:
        self._client = client
        self._contract = contract
        self._session = contract.session()
        self._budget = budget
        self._on_fail = on_fail
        self._max_tokens = default_max_tokens
        self._escalation_handler = escalation_handler
        self._kind = _detect_kind(client)

        if self._kind == "openai":
            real = client.chat.completions.create
            self.chat = _Namespace(completions=_Namespace(create=self._guard(real)))
        elif self._kind == "anthropic":
            real = client.messages.create
            self.messages = _Namespace(create=self._guard(real))
        else:
            raise ValueError(
                "pactrun.wrap() supports OpenAI and Anthropic client SDKs today. "
                "For other providers, use the framework adapters with a Contract directly."
            )

    def __getattr__(self, name: str) -> Any:
        client = self.__dict__.get("_client")
        if client is None:
            raise AttributeError(name)
        return getattr(client, name)

    @property
    def session(self):
        """The underlying enforcement Session (for summaries / violations)."""
        return self._session

    def __enter__(self) -> "GuardedClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    # -- the gate ----------------------------------------------------------

    def _guard(self, real_create):
        def guarded(*args: Any, **kwargs: Any):
            self._precall_cost_gate(kwargs)
            response = real_create(*args, **kwargs)
            self._record(kwargs, response)  # post-call: tokens, tools, loops, turns
            return response

        return guarded

    def _precall_cost_gate(self, kwargs: dict) -> None:
        if self._budget is None:
            return
        worst = _estimate_worstcase_cost(kwargs, self._max_tokens)
        projected = self._session.state.total_cost_usd + worst
        if projected <= self._budget:
            return
        violation = Violation(
            clause_description="pre-call cost gate",
            kind=ClauseKind.REQUIRE,
            severity=Severity.ERROR,
            on_fail=self._on_fail,
            message=(
                f"pre-call refusal: worst-case cost ~${worst:.4f} would push the run to "
                f"~${projected:.4f}, over the ${self._budget:.4f} budget"
            ),
            expected=f"<= ${self._budget:.4f}",
            actual=f"~${projected:.4f} projected",
        )
        self._session._violations.append(violation)
        apply_recovery(violation, escalation_handler=self._escalation_handler)

    def _record(self, kwargs: dict, response: Any) -> None:
        if self._kind == "openai":
            _record_openai(self._session, kwargs, response)
        elif self._kind == "anthropic":
            _record_anthropic(self._session, kwargs, response)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def _detect_kind(client: Any) -> str:
    chat = getattr(client, "chat", None)
    if chat is not None and getattr(getattr(chat, "completions", None), "create", None):
        return "openai"
    messages = getattr(client, "messages", None)
    if messages is not None and getattr(messages, "create", None):
        return "anthropic"
    return "unknown"


# ---------------------------------------------------------------------------
# Worst-case pre-call cost estimate
# ---------------------------------------------------------------------------

# (input, output) USD per 1M tokens. Unknown models fall back to a conservative
# default so the budget gate still errs toward refusing.
_PRICING: dict[str, tuple[float, float]] = {
    "gpt-5.4": (2.50, 15.00),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4o": (2.50, 10.00),
    "o3": (2.00, 8.00),
    "o4-mini": (0.55, 2.20),
    "claude-opus-4": (15.00, 75.00),
    "claude-sonnet-4": (3.00, 15.00),
    "claude-haiku-4": (0.80, 4.00),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.5-flash": (0.30, 2.50),
}
_DEFAULT_PRICE = (2.50, 10.00)


def _price(model: str) -> tuple[float, float]:
    if model in _PRICING:
        return _PRICING[model]
    for key, prices in _PRICING.items():
        if model.startswith(key):
            return prices
    return _DEFAULT_PRICE


def _messages_text(kwargs: dict) -> str:
    parts: list[str] = []
    system = kwargs.get("system")
    if isinstance(system, str):
        parts.append(system)
    for message in kwargs.get("messages", []) or []:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    parts.append(block["text"])
    return " ".join(parts)


def _count_tokens(text: str) -> int:
    # Deliberately a cheap heuristic (~4 chars/token); the output term dominates
    # the worst case anyway. No tokenizer dependency.
    return max(1, len(text) // 4)


def _estimate_worstcase_cost(kwargs: dict, default_max_tokens: int) -> float:
    model = kwargs.get("model", "") or ""
    input_tokens = _count_tokens(_messages_text(kwargs))
    max_output = (
        kwargs.get("max_tokens")
        or kwargs.get("max_completion_tokens")
        or default_max_tokens
    )
    in_price, out_price = _price(model)
    return (input_tokens * in_price + int(max_output) * out_price) / 1_000_000


# ---------------------------------------------------------------------------
# Post-call recording (mirrors the OpenAI / Anthropic adapters)
# ---------------------------------------------------------------------------

def _record_openai(session: Any, kwargs: dict, response: Any) -> None:
    model = getattr(response, "model", None) or kwargs.get("model", "unknown")
    prompt_tokens = 0
    completion_tokens = 0
    usage = getattr(response, "usage", None)
    if usage is not None:
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0

    output = ""
    tool_names: list[tuple[str, dict]] = []
    try:
        message = response.choices[0].message
        output = getattr(message, "content", None) or ""
        for call in getattr(message, "tool_calls", None) or []:
            fn = getattr(call, "function", None)
            name = getattr(fn, "name", None)
            if name:
                tool_names.append((name, {}))
    except (AttributeError, IndexError, TypeError):
        pass

    cost = _actual_cost(model, prompt_tokens, completion_tokens)
    session.emit_llm_response(
        model=model, output=output, prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens, cost=cost,
    )
    for name, args in tool_names:
        session.emit_tool_call(name, args=args)


def _record_anthropic(session: Any, kwargs: dict, response: Any) -> None:
    model = getattr(response, "model", None) or kwargs.get("model", "unknown")
    prompt_tokens = 0
    completion_tokens = 0
    usage = getattr(response, "usage", None)
    if usage is not None:
        prompt_tokens = getattr(usage, "input_tokens", 0) or 0
        completion_tokens = getattr(usage, "output_tokens", 0) or 0

    output = ""
    tool_names: list[str] = []
    try:
        for block in response.content or []:
            if hasattr(block, "text"):
                output += block.text
            elif getattr(block, "name", None):
                tool_names.append(block.name)
    except (AttributeError, TypeError):
        pass

    cost = _actual_cost(model, prompt_tokens, completion_tokens)
    session.emit_llm_response(
        model=model, output=output, prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens, cost=cost,
    )
    for name in tool_names:
        session.emit_tool_call(name, args={})


def _actual_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    in_price, out_price = _price(model)
    return (prompt_tokens * in_price + completion_tokens * out_price) / 1_000_000
