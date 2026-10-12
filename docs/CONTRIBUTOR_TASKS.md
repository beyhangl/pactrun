# Contributor tasks for support actions

Maintainer: [beyhangl](https://github.com/beyhangl) — beyhangl@gmail.com.

The runnable entry point is [the synthetic refund example](SUPPORT_REFUND.md).
Use fake identities, payments and model replies. An initial contribution should
add one independently useful case or binding without requiring a hosted service.

| Task | Expected review evidence |
| --- | --- |
| Add an account/subscription-change case | Authoritative before/after state, ownership rules, allowed and blocked controls |
| Add a provider fault | Deterministic reproduction, unknown outcome handling, no blind resend |
| Bind one requested agent framework | Show refusal before the callback and observe the actual result |
| Extend cassette support | Preserve IDs, tokens, provenance and timing; refuse lossy conversions |
| Add an approval authority integration | Explicit host identity source and signed context; do not equate HMAC with human identity |
| Coordinate ACS/Probity interoperability | Reuse existing implementations and upstream scope; avoid another standard or duplicate Python Guardian |

Each case must state initial state, proposed action, fault, expected decision,
expected state and the observations available. Include an allowed control case
to detect over-blocking. Tests must run without real model calls or payment APIs.
Never edit a recorded cassette or golden baseline to make a test pass.

For local review run `pytest`, `ruff check pactrun tests` and `mypy pactrun`.
The refund example and bridge tests additionally need Evalcraft; SDK playback
uses a mocked optional OpenAI SDK. Keep CI/deployment files and shell scripts for
user-designated DEVOPS agents.

An existing ecosystem integration is useful even if its users do not adopt both
projects. Measure repeat outside contributions and independent use rather than
promising a star count or contributor growth rate.
