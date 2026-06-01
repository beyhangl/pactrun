"""pactrun command-line interface.

    pactrun init        scaffold a starter contract YAML
    pactrun validate    load and validate contract YAML file(s)
    pactrun show        pretty-print a contract's clauses
    pactrun predicates  list the built-in predicates
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
        raise SystemExit(1)

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
def predicates() -> None:
    """List the built-in predicates available in contracts."""
    names = list_predicates()
    console.print(f"[bold]{len(names)} built-in predicates:[/bold]")
    for name in names:
        console.print(f"  • {name}")


if __name__ == "__main__":
    cli()
