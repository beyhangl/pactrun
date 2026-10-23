"""Exercise the documented offline CLI and its artifact preservation boundary."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("evalcraft.scenarios")

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "refund_scenarios.py"


def _run(*args):
    return subprocess.run(
        [sys.executable, str(EXAMPLE), *map(str, args)],
        capture_output=True, text=True, timeout=30,
    )


def test_cli_full_pack_and_saved_artifacts_never_replace_existing_output(tmp_path):
    output = tmp_path / "run"
    result = _run("--output-dir", output)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "26/26 scenarios passed" in result.stdout
    report = json.loads((output / "report.json").read_text())
    assert report["passed"] and report["total_cases"] == 26
    assert len(list(output.glob("*/cassette.json"))) == 26
    saved = {path: path.read_bytes() for path in output.rglob("*.json")}
    repeated = _run("--output-dir", output)
    assert repeated.returncode == 2
    assert "already exists" in repeated.stderr
    assert all(path.read_bytes() == content for path, content in saved.items())


def test_cli_lists_selects_and_refuses_unknown_cases(tmp_path):
    listed = _run("--list")
    assert listed.returncode == 0
    names = listed.stdout.splitlines()
    assert len(names) == 26 and len(set(names)) == 26
    chosen = _run("--case", names[0], "--case", names[-1], "--output-dir", tmp_path / "selected")
    assert chosen.returncode == 0, chosen.stderr
    assert "2/2 scenarios passed" in chosen.stdout
    unknown = _run("--case", "does-not-exist", "--output-dir", tmp_path / "invalid")
    assert unknown.returncode == 2
    assert "unknown case" in unknown.stderr
    assert not (tmp_path / "invalid").exists()
