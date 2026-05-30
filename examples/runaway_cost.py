"""Demo: a whole-run cost cap that stops an agent cold when it overspends.

This is the core pactrun value in ~6 lines: a budget over the *entire* session,
enforced the instant it is crossed.

    python examples/runaway_cost.py
"""

from pactrun import Contract, cost_under, max_turns, ViolationError

contract = (
    Contract("budget_demo")
    .require(cost_under(0.05))      # 5-cent ceiling for the whole run
    .require(max_turns(50))
    .on_violation("block")
)

print("Running an agent that just keeps spending...\n")
session = None
try:
    with contract.session() as session:
        for turn in range(1, 100):
            session.emit_llm_response(model="gpt-4.1", output=f"step {turn}", cost=0.012)
            print(f"  turn {turn}: total ${session.state.total_cost_usd:.3f}")
except ViolationError as exc:
    print(f"\n  BLOCKED: {exc.violation.message}")
    print(
        f"  stopped after {session.state.total_llm_calls} calls, "
        f"${session.state.total_cost_usd:.3f} spent (cap $0.05)"
    )
