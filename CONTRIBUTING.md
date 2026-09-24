# Contributing to pactrun

Thanks for helping. pactrun is a guardrail, so correctness matters more than
features: a check that silently stops checking is worse than a missing one.

## Reporting security issues

**Don't open a public issue.** Follow [SECURITY.md](SECURITY.md).

## Development setup

```bash
git clone https://github.com/beyhangl/pactrun && cd pactrun
pip install -e ".[dev]"        # or: uv pip install -e ".[dev]"
```

Before opening a pull request, run the same checks CI runs:

```bash
pytest
ruff check pactrun/ tests/
mypy pactrun/
```

## Testing policy

- **Every bug fix needs a regression test.**
- **Every security fix needs a test that fails without the fix.** Before
  submitting, revert your change to the source (keep the test) and confirm the
  test fails. A test that passes against the vulnerable code proves nothing.
- Tests that depend on an optional SDK (OpenAI, Anthropic, `jsonschema`,
  `httpx`, …) must skip cleanly when it isn't installed — use
  `pytest.importorskip(...)` or a `skipif`. The core test matrix installs only
  `.[dev]`.

## Adding a predicate

A predicate is a factory that validates its configuration and returns a pure
checker `(Event, SessionState) -> PredicateResult`.

```python
from pactrun.core.amounts import require_limit
from pactrun.predicates.base import predicate

@predicate("my_check", owasp=("ASI02",))   # tag the OWASP Agentic risks it mitigates
def my_check(limit: float):
    require_limit("my_check(limit)", limit)  # reject nonsense config up front

    def check(event, state):
        ...
        return PredicateResult(passed=..., expected=..., actual=..., message=...)

    check.predicate_name = "my_check"
    return check
```

Guidelines:

- **Fail closed.** No input should make a check pass that ought to fail. If a
  value can't be interpreted, treat it as a failure, not a pass.
- **Validate configuration in the factory**, so a bad contract fails when it is
  built rather than on the first event.
- **Be honest about limits.** If the check is a heuristic (a regex bank, a
  denylist), say so in its docstring.
- Checks that can only be judged at the end of a run set
  `check._check_on = "session_end"`.
- Register the predicate in `pactrun/predicates/__init__.py` and
  `pactrun/__init__.py`, add tests, and add a row to the README's predicate
  table. Only tag an OWASP risk you genuinely mitigate at runtime.

## Pull requests

- Keep changes focused; one concern per pull request.
- Write the title so it could go straight into the changelog.
- Add an entry to the `[Unreleased]` section of [CHANGELOG.md](CHANGELOG.md).
  Mark breaking changes **BREAKING:** and security fixes under **Security**.
- If you used an AI assistant, say so in the pull request, and make sure you
  have run and understood every test you submit.
