"""Tests for the Agent Cost & Safety Index harness (scripts/cost_safety_index.py)."""

import sys
from pathlib import Path

# scripts/ is not an importable package — add it to the path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import cost_safety_index as csi  # noqa: E402


def test_mock_workload_measures():
    row = csi.run_workload(csi.MockClient(), "mock-model")
    assert row.llm_calls == csi.STEPS
    assert row.tool_calls == csi.STEPS        # one 'search' tool per step
    assert row.total_tokens > 0
    assert row.cost_usd > 0
    assert row.cost_per_call > 0
    assert row.loop_flagged is True           # repeated 'search' tool
    assert row.drift_flagged is True          # escalating completion tokens


def test_render_markdown_table():
    row = csi.run_workload(csi.MockClient(), "mock-model")
    row.provider = "mock"
    table = csi.render_markdown([row])
    assert "| Provider |" in table
    assert "mock" in table
    assert "mock-model" in table


def test_main_runs_mock(capsys):
    assert csi.main(["--providers", "mock"]) == 0
    out = capsys.readouterr().out
    assert "Agent Cost & Safety Index" in out
    assert "mock-model" in out
