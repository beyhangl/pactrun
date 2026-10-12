# Client wrapper coverage and budget accounting

`pactrun.wrap()` checks supported SDK calls before dispatch and records their
responses. Use one returned `GuardedClient` for calls that share a budget.

| Path | Sync | Async | Streaming |
| --- | --- | --- | --- |
| OpenAI `chat.completions.create()` | Supported | Supported | `stream=True` |
| Anthropic `messages.create()` | Supported | Supported | `stream=True` |
| OpenAI `responses` | Explicitly refused | Explicitly refused | Explicitly refused |
| Other endpoints, helper methods, cloned clients and providers | Outside coverage | Outside coverage | Outside coverage |

Accessing `guarded.responses` raises `NotImplementedError` before an underlying
Responses request can run. It never silently returns the raw Responses client.
Other attributes are delegated for SDK compatibility; this does not instrument
methods such as SDK-specific parsing/streaming helpers or clients returned by
`with_options()`. Do not use delegated request methods as enforcement boundaries.
Only call the supported `create()` paths through the returned wrapper. Calls on
the original client remain outside coverage.

The wrapper can inspect a returned tool proposal, but it does not own execution
of your application's tools. A proposal passing these checks is not proof of
authorized execution. Use a tool adapter or an application control at the actual
action boundary as well.

## Concurrent calls

Admission atomically reserves each call's pre-call cost estimate. Projection is
recorded cost plus all pending reservations plus the next estimate. A thread
lock protects admission and settlement; the provider request runs outside the
lock. This protects concurrent sync threads and async tasks sharing the same
`GuardedClient`. It does not share budgets across wrapper instances, processes,
workers, machines, or restarted sessions. Direct changes through `session` are
not part of the wrapper's concurrency boundary.

A completed response replaces its reservation only when both required provider
token totals are present, normalized counts are valid, and the computed cost is
finite and non-negative. Explicitly reported zero totals are valid zero usage.
The estimate and actual amount are never both added. A post-call violation can
raise after the provider has already billed; the recorded cost is retained.

`block` refuses an over-budget estimate before dispatch. `log` and `warn` still
allow the call and reserve its estimate. `approve` allows it only when the
configured handler approves. These modes do not promise the same budget limit.

## Streams, errors and cancellation

A stream keeps its reservation until final usage or explicit cleanup. Final
usage replaces the reservation with usage-based cost. Closing, exiting a context,
a stream error, and cancellation settle once. Cleanup also closes the upstream
stream, including async SDK streams exposing `close()` rather than `aclose()`.

If a request fails or is cancelled without a response, the provider may already
have accepted or billed it. The wrapper conservatively records its reserved
estimate and preserves the original request error. Missing or incomplete
response/stream usage also retains that estimate. Invalid counts or priced
amounts retain the estimate and trigger the configured recovery action;
diagnostics preserve the reported invalid values. Such events include
`cost_basis: pre-call-estimate`
and an unavailable-usage reason in their metadata. They are accounting records,
not proof that the provider charged this amount. A definitively unbilled provider
failure can therefore overstate spend. Use provider invoices or authoritative
usage records to reconcile uncertain charges.

When you stop reading a stream early, call `close()`/`aclose()` or use its context
manager. An abandoned stream without cleanup keeps its reservation; garbage
collection is not a reliable settlement mechanism.

## Estimation limits

Tool names, descriptions and input schemas are included in pre-call input counts,
along with Anthropic system text blocks, previous assistant tool arguments and
text in tool results.
OpenAI uses a tokenizer on text and serialized definitions when available;
non-OpenAI estimates pass definitions to LiteLLM's token counter. Fallback counts
include serialized definitions too. JSON serialization is an estimate of the
provider's representation, not an exact wire token count.

Set `max_tokens` or `max_completion_tokens` for a provider output limit. If you omit
both, `default_max_tokens` is used only for estimation; the wrapper does not send
that limit to the provider. Provider framing, media inputs, hidden reasoning,
provider-specific billing, and stale pricing can exceed an estimate even with an
explicit output limit. Cache-write premiums are estimated; cache-read discounts
are not assumed before a call. Reported usage is priced after the call, and
pricing lookup can itself fall back to a static table.

These checks provide estimated budget admission for covered calls, not a
universal hard billing cap or a process/network sandbox. For a strict shared
financial limit, use an authoritative reservation service at a boundary that
controls every relevant request and handles provider billing uncertainty.
