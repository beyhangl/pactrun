"""Demo: session-end tool guarantees via .require().

Shows must_call / tool_order working through the fluent .require() path: these
predicates only make sense once the whole run is done, so pactrun defers them to
session end instead of failing on the first event (before the tool could run).

    python examples/tool_guardrails.py
"""

from pactrun import Contract, must_call, must_not_call, tool_order

contract = (
    Contract("refund_agent")
    .require(must_call("verify_identity"))                       # must happen by the end
    .require(tool_order(["verify_identity", "issue_refund"]))    # and in this order
    .forbid(must_not_call("delete_account"))                    # never, ever
    .on_violation("log")
)

with contract.session() as session:
    # A "thinking" step happens BEFORE any tool is called. Under the old buggy
    # code, must_call fired here and failed. Now it correctly waits for the end.
    session.emit_llm_response(model="gpt-4.1", output="Let me verify the customer first.", cost=0.002)
    session.emit_tool_call("verify_identity")
    session.emit_tool_call("issue_refund")

summary = session.summary()
print(f"compliant     : {summary.is_compliant}")
print(f"tools called  : {summary.tool_call_history}")
print(f"violations    : {summary.violation_count}")
