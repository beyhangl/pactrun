"""Demo: recovery actions — retry, fallback, and escalate.

When a contract is breached, pactrun does more than just block: it can retry the
call, fall back to a safe agent, or escalate to a human/webhook.

    python examples/recovery.py
"""

from pactrun import Contract, EscalationError, cost_under, get_active_session

# --- retry: re-run until the agent stays under budget ----------------------
attempts = {"n": 0}
retrying = Contract("retry_demo").require(cost_under(0.05), on_fail="retry").with_retries(3)


@retrying.enforce
def flaky_agent():
    attempts["n"] += 1
    cost = 0.09 if attempts["n"] == 1 else 0.02   # overspends once, then behaves
    get_active_session().emit_llm_response(model="gpt-4.1", output="...", cost=cost)
    return f"succeeded on attempt {attempts['n']}"


print("retry    ->", flaky_agent())


# --- fallback: switch to a cheap safe agent when the primary breaks the rules
def safe_agent(*args, **kwargs):
    return "served by the safe fallback agent"


guarded = Contract("fallback_demo").require(cost_under(0.05), on_fail="fallback").fallback(safe_agent)


@guarded.enforce
def expensive_agent():
    get_active_session().emit_llm_response(model="gpt-4.1", output="...", cost=0.20)
    return "primary (never returned)"


print("fallback ->", expensive_agent())


# --- escalate: notify a human / webhook, then halt -------------------------
def page_oncall(violation):
    print(f"escalate -> paging on-call: {violation.message}")


escalating = Contract("escalate_demo").require(cost_under(0.05), on_fail="escalate").on_escalate(page_oncall)

try:
    with escalating.session() as session:
        session.emit_llm_response(model="gpt-4.1", output="...", cost=0.20)
except EscalationError as exc:
    print(f"escalate -> halted after escalation: {exc.violation.message}")
