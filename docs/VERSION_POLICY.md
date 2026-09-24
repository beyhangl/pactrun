# Versioning and API stability

pactrun follows [Semantic Versioning](https://semver.org/). It is currently
`0.x`, which changes what a version bump promises.

## While pactrun is 0.x

- **Patch releases** (`0.4.1` → `0.4.2`) contain only fixes and never break the
  public API — with one exception, below.
- **Minor releases** (`0.4` → `0.5`) may contain breaking changes. Every one is
  listed in [CHANGELOG.md](../CHANGELOG.md) with a **BREAKING:** prefix and a
  note on how to update.
- After `1.0`, breaking changes will only land in major releases.

## Security fixes can change behavior in any release

pactrun is a guardrail. If a check fails open — letting through something its
documented guarantee says it blocks — the fix makes it fail closed, and that
fix ships in the next release **even if it is a patch** and even if it turns
something that used to pass into a violation. Keeping a bypass open for
compatibility's sake is never the right trade. Such changes are listed under
**Security** in the changelog.

## What counts as public API

- Names exported from `pactrun` (listed in `pactrun.__all__`).
- The documented subpackages `pactrun.adapters`, `pactrun.recovery`, and
  `pactrun.observability`, and the names they export.
- The YAML contract format.
- The CLI commands and their options.

**Not** public, and free to change without notice:

- Any module or name starting with an underscore — for example
  `pactrun.predicates._neturl` or `pactrun.predicates._argpath`.
- The exact wording of violation messages. Match on the structured fields
  (`predicate_name`, `expected`, `actual`, `enforced`), not on message text.
- Telemetry attribute names. They track the OpenTelemetry GenAI semantic
  conventions, which are themselves not yet stable; changes are listed in the
  changelog but may land in any minor release.

## Not considered breaking

- New predicates, new recovery actions, new adapters.
- New optional parameters with defaults that preserve existing behavior.
- New fields on result objects such as `Violation` or `SessionSummary`.

## Deprecations

A public name or parameter scheduled for removal first emits a
`DeprecationWarning` for at least one minor release, and the deprecation is
noted in the changelog.

## Python versions

pactrun supports every CPython version that has not reached its end of life.
Support for a version is dropped in the first minor release after its
end-of-life date.
