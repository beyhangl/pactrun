"""pactrun command-line interface.

    pactrun init        scaffold a starter contract YAML
    pactrun validate    load and validate contract YAML file(s)
    pactrun show        pretty-print a contract's clauses
    pactrun predicates  list the built-in predicates
    pactrun replay      replay a recorded JSONL trace against a contract
    pactrun test        run the policy tests in a contract's tests: block
"""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from pactrun import Contract, __version__
from pactrun.core.errors import ContractLoadError
from pactrun.predicates.base import list_predicates

console = Console()

# Placeholder is replaced rather than str.format()'d so the YAML's flow-style
# braces ({ max_usd: 0.50 }) survive untouched.
_STARTER = """\
name: __NAME__
version: "1.0"
description: A starter pactrun contract. Edit me.
on_fail: block

clauses:
  # Whole-run budget
  - require: cost_under
    args: { max_usd: 0.50 }
  - require: max_turns
    args: { n: 20 }
  # Catch infinite tool loops
  - require: no_loops
  # Never call a dangerous tool
  - forbid: must_not_call
    args: { tool: delete_account }
  # Warn (don't block) if PII leaks into the output
  - require: no_pii
    severity: warning
    on_fail: warn
"""


@click.group()
@click.version_option(__version__, prog_name="pactrun")
def cli() -> None:
    """pactrun — behavioral contracts for AI agents."""


@cli.command()
@click.option("--name", default="agent", help="Contract name (and file stem).")
@click.option(
    "--output", "-o", default="contracts",
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory to write the contract into.",
)
@click.option("--force", is_flag=True, help="Overwrite the file if it already exists.")
def init(name: str, output: Path, force: bool) -> None:
    """Scaffold a starter contract YAML."""
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{name}.yaml"
    if path.exists() and not force:
        console.print(f"[red]✗[/red] {path} already exists (use --force to overwrite).")
        raise SystemExit(1)
    path.write_text(_STARTER.replace("__NAME__", name))
    console.print(f"[green]✓[/green] wrote {path}")
    console.print(f"  validate it with: [bold]pactrun validate {path}[/bold]")


def _yaml_files(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(path.glob("*.yaml")) + sorted(path.glob("*.yml"))
    return [path]


@cli.command()
@click.argument("path", type=click.Path(exists=True, path_type=Path))
def validate(path: Path) -> None:
    """Load and validate contract YAML file(s). PATH may be a file or directory."""
    files = _yaml_files(path)
    if not files:
        console.print(f"[yellow]No .yaml/.yml files found in {path}[/yellow]")
        raise SystemExit(1)

    failures = 0
    for file in files:
        try:
            contract = Contract.from_yaml(file)
        except ContractLoadError as exc:
            failures += 1
            console.print(f"[red]✗ {file}[/red]: {exc}")
            continue
        console.print(f"[green]✓ {file}[/green]: '{contract.name}' — {len(contract.clauses)} clause(s)")

    if failures:
        console.print(f"\n[red]{failures} of {len(files)} contract(s) failed validation.[/red]")
        raise SystemExit(1)
    console.print(f"\n[green]All {len(files)} contract(s) valid.[/green]")


@cli.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def show(path: Path) -> None:
    """Pretty-print a contract's clauses."""
    try:
        contract = Contract.from_yaml(path)
    except ContractLoadError as exc:
        console.print(f"[red]✗ {path}[/red]: {exc}")
        raise SystemExit(1) from exc

    console.print(f"[bold]{contract.name}[/bold]  v{contract.version}")
    if contract.description:
        console.print(f"[dim]{contract.description}[/dim]")
    console.print(f"default on_fail: {contract.default_on_fail.value}\n")

    table = Table(show_header=True, header_style="bold")
    for col in ("kind", "predicate", "check_on", "severity", "on_fail"):
        table.add_column(col)
    for clause in contract.clauses:
        table.add_row(
            clause.kind.value,
            clause.predicate_name,
            clause.check_on,
            clause.severity.value,
            clause.on_fail.value,
        )
    console.print(table)


@cli.command()
@click.option("--owasp", is_flag=True, help="Group by OWASP Agentic Top-10 risk instead.")
def predicates(owasp: bool = False) -> None:
    """List the built-in predicates available in contracts."""
    from pactrun.predicates.base import OWASP_AGENTIC_2026, owasp_coverage, predicate_owasp

    names = list_predicates()
    if owasp:
        coverage = owasp_coverage()
        covered = sum(1 for ids in coverage.values() if ids)
        console.print(
            f"[bold]OWASP Top 10 for Agentic Applications (2026) — "
            f"runtime controls for {covered}/10 risks[/bold]\n"
        )
        for rid, title in OWASP_AGENTIC_2026.items():
            preds = coverage[rid]
            if preds:
                console.print(f"[bold]{rid}[/bold] {title} [dim]({len(preds)})[/dim]")
                console.print(f"  [dim]{', '.join(preds)}[/dim]")
            else:
                console.print(f"[bold]{rid}[/bold] {title} [dim]— no runtime control[/dim]")
        console.print(
            "\n[dim]A tag means the predicate provides a partial runtime control, "
            "not certified coverage.[/dim]"
        )
        return

    console.print(f"[bold]{len(names)} built-in predicates:[/bold]")
    for name in names:
        ids = predicate_owasp(name)
        suffix = f"  [dim]{' '.join(ids)}[/dim]" if ids else ""
        console.print(f"  • {name}{suffix}")


@cli.command()
@click.argument("contract_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("trace_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--json", "as_json", is_flag=True, help="Print the result as JSON.")
def replay(contract_path: Path, trace_path: Path, as_json: bool) -> None:
    """Replay a recorded TRACE (JSONL) against a CONTRACT without enforcing.

    Exit code 0 if the run complies, 1 if the contract would have flagged it,
    2 if the contract or trace could not be loaded - so it can gate CI.
    """
    import json as _json

    from pactrun.replay import TraceLoadError, load_trace, replay_trace

    try:
        contract = Contract.from_yaml(contract_path)
        events = load_trace(trace_path)
    except (ContractLoadError, TraceLoadError) as exc:
        console.print(f"[red]✗ {exc}[/red]")
        raise SystemExit(2) from exc

    result = replay_trace(contract, events)

    if as_json:
        click.echo(_json.dumps({
            "contract": result.contract_name,
            "events": len(result.events),
            "compliant": result.compliant,
            "violated": result.violated,
            "violations": [
                {**v.to_dict(), "event_index": result.event_index(v)} for v in result.violations
            ],
        }, indent=2, default=str))
    elif result.compliant:
        console.print(
            f"[green]✓ '{result.contract_name}': {len(result.events)} event(s), no violations.[/green]"
        )
    else:
        table = Table(title=f"'{result.contract_name}' would flag {len(result.violations)} violation(s)")
        table.add_column("event")
        table.add_column("predicate")
        table.add_column("would do")
        table.add_column("message")
        for v in result.violations:
            idx = result.event_index(v)
            table.add_row(
                str(idx) if idx is not None else "end",
                v.predicate_name or v.clause_description,
                v.on_fail.value,
                v.message,
            )
        console.print(table)

    raise SystemExit(0 if result.compliant else 1)


@cli.command(name="test")
@click.argument("contract_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def test_cmd(contract_path: Path) -> None:
    """Run the policy tests in a CONTRACT file's `tests:` block.

    Exit code 0 if every test passes, 1 if any fails, 2 if the contract, its
    tests, or a referenced trace could not be loaded.
    """
    from pactrun.replay import TraceLoadError, run_contract_tests

    try:
        report = run_contract_tests(contract_path)
    except (ContractLoadError, TraceLoadError) as exc:
        console.print(f"[red]✗ {exc}[/red]")
        raise SystemExit(2) from exc

    for r in report.results:
        want = "pass" if not r.expected else f"violated {r.expected}"
        if r.runs > 1:
            want += f", {r.runs_passed}/{r.runs} runs"
        if r.passed:
            console.print(f"[green]✓[/green] {r.name} [dim]({want})[/dim]")
        else:
            console.print(f"[red]✗ {r.name}[/red] — expected {want}; {r.detail}")

    failed = sum(not r.passed for r in report.results)
    total = len(report.results)
    if report.repeated:
        console.print(
            f"\nConsistency: Pass^k {total - failed}/{total} test(s) right on every run, "
            f"Mean@k {report.mean_k:.0%} of runs right"
        )
    if failed:
        console.print(f"\n[red]{failed} of {total} policy test(s) failed.[/red]")
        raise SystemExit(1)
    console.print(f"\n[green]All {total} policy test(s) passed.[/green]")


if __name__ == "__main__":
    cli()
