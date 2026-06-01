"""Tests for the pactrun CLI."""

from click.testing import CliRunner

from pactrun.cli.main import cli


def test_version():
    result = CliRunner().invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_init_then_validate(tmp_path):
    runner = CliRunner()
    out = tmp_path / "contracts"

    r = runner.invoke(cli, ["init", "--name", "demo", "--output", str(out)])
    assert r.exit_code == 0
    written = out / "demo.yaml"
    assert written.exists()

    r2 = runner.invoke(cli, ["validate", str(written)])
    assert r2.exit_code == 0
    assert "valid" in r2.output.lower()


def test_init_refuses_overwrite_then_force(tmp_path):
    runner = CliRunner()
    out = tmp_path / "contracts"
    runner.invoke(cli, ["init", "--name", "demo", "--output", str(out)])

    again = runner.invoke(cli, ["init", "--name", "demo", "--output", str(out)])
    assert again.exit_code == 1

    forced = runner.invoke(cli, ["init", "--name", "demo", "--output", str(out), "--force"])
    assert forced.exit_code == 0


def test_validate_rejects_unknown_predicate(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\nclauses:\n  - require: nonexistent_pred\n    args: {}\n")
    r = CliRunner().invoke(cli, ["validate", str(bad)])
    assert r.exit_code == 1


def test_validate_directory(tmp_path):
    runner = CliRunner()
    out = tmp_path / "contracts"
    runner.invoke(cli, ["init", "--name", "a", "--output", str(out)])
    runner.invoke(cli, ["init", "--name", "b", "--output", str(out)])
    r = runner.invoke(cli, ["validate", str(out)])
    assert r.exit_code == 0
    assert "All 2 contract(s) valid" in r.output


def test_show(tmp_path):
    runner = CliRunner()
    out = tmp_path / "contracts"
    runner.invoke(cli, ["init", "--name", "demo", "--output", str(out)])
    r = runner.invoke(cli, ["show", str(out / "demo.yaml")])
    assert r.exit_code == 0
    assert "cost_under" in r.output


def test_predicates_lists_builtins():
    r = CliRunner().invoke(cli, ["predicates"])
    assert r.exit_code == 0
    assert "cost_under" in r.output
    assert "no_loops" in r.output
