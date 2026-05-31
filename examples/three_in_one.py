"""One contract, three catastrophes averted — cost, forbidden tool, drift.

A single ``pactrun.wrap()`` line refuses a call that would blow the run budget
(before it bills), blocks a forbidden tool the model tries to use, and flags
cost drift. No provider key needed — the client here is a local stand-in.

    python examples/three_in_one.py
"""

import pactrun
from pactrun import ViolationError


# --- a local stand-in for an OpenAI-style client (no network/key) ----------
class _Usage:
    def __init__(self, p, c):
        self.prompt_tokens = p
        self.completion_tokens = c


class _Fn:
    def __init__(self, name):
        self.name = name
        self.arguments = "{}"


class _ToolCall:
    def __init__(self, name):
        self.function = _Fn(name)


class _Message:
    def __init__(self, content, tools):
        self.content = content
        self.tool_calls = [_ToolCall(t) for t in tools]


class _Choice:
    def __init__(self, message):
        self.message = message


class Response:
    def __init__(self, text="ok", tools=(), prompt_tokens=1000, completion_tokens=1000, model="gpt-4.1"):
        self.model = model
        self.usage = _Usage(prompt_tokens, completion_tokens)
        self.choices = [_Choice(_Message(text, tools))]


class _Completions:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        return self._responses[min(self.calls - 1, len(self._responses) - 1)]


class _Chat:
    def __init__(self, completions):
        self.completions = completions


class FakeOpenAI:
    def __init__(self, responses):
        self.completions = _Completions(responses)
        self.chat = _Chat(self.completions)


def banner(title):
    print("\n" + "=" * 64 + "\n" + title + "\n" + "=" * 64)


# 1) COST — refuse the runaway call BEFORE it bills -------------------------
banner("1) COST — refuse the runaway call BEFORE it bills")
fake = FakeOpenAI([Response()])
client = pactrun.wrap(fake, max_cost="$0.50")
try:
    client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": "loop forever"}],
        max_tokens=200_000,  # worst-case cost blows the $0.50 cap
    )
except ViolationError as exc:
    print("  refused:", exc.violation.message)
    print(f"  calls that actually hit the API: {fake.completions.calls}  (stopped before billing)")


# 2) TOOL — block a forbidden tool the model tries to call ------------------
banner("2) TOOL — block a forbidden tool the model tries to call")
fake = FakeOpenAI([Response(text=None, tools=["delete_account"])])
client = pactrun.wrap(fake, max_cost="$1.00", forbid_tools=["delete_account"], default_max_tokens=50)
try:
    client.chat.completions.create(
        model="gpt-4.1", messages=[{"role": "user", "content": "clean up my account"}], max_tokens=50
    )
except ViolationError as exc:
    print("  blocked:", exc.violation.message)


# 3) DRIFT — flag cost-per-turn creep across the run ------------------------
banner("3) DRIFT — flag cost-per-turn creep across the run")
turns = [
    Response(prompt_tokens=100, completion_tokens=100),
    Response(prompt_tokens=100, completion_tokens=120),
    Response(prompt_tokens=100, completion_tokens=150),
    Response(prompt_tokens=200, completion_tokens=2000),  # a spike
]
fake = FakeOpenAI(turns)
client = pactrun.wrap(fake, max_cost="$5.00", max_drift=0.60, on_violation="log", default_max_tokens=50)
for i in range(len(turns)):
    client.chat.completions.create(
        model="gpt-4.1", messages=[{"role": "user", "content": f"step {i + 1}"}], max_tokens=50
    )
session = client.session
print("  compliant:", session.is_compliant)
for violation in session.violations:
    print("  flagged:", violation.message)

print("\nAll three — cost, forbidden tool, drift — enforced by one wrap() line, on a plain client.")
