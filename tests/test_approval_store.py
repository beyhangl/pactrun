"""Durable consumption semantics across atomic batches and a fresh process."""

import subprocess
import sys

import pytest

from pactrun.approval_store import SQLiteApprovalStore


def test_batch_rolls_back_if_any_nonce_was_used(tmp_path):
    store = SQLiteApprovalStore(tmp_path / "approvals.sqlite")
    assert store.consume_many(["old"])
    assert not store.consume_many(["new", "old"])
    assert store.consume_many(["new"])
    assert not store.consume_many(["duplicate", "duplicate"])
    assert store.consume_many(["duplicate"])


def test_consumption_survives_interpreter_restart(tmp_path):
    path = tmp_path / "approvals.sqlite"
    assert SQLiteApprovalStore(path).consume_many(["once"])
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; from pactrun.approval_store import SQLiteApprovalStore; "
         "print(SQLiteApprovalStore(sys.argv[1]).consume_many(['once']))",
         str(path)],
        check=True, capture_output=True, text=True,
    )
    assert result.stdout.strip() == "False"


def test_independent_processes_only_consume_once(tmp_path):
    path = tmp_path / "approvals.sqlite"
    SQLiteApprovalStore(path)
    command = [
        sys.executable, "-c",
        "import sys; from pactrun.approval_store import SQLiteApprovalStore; "
        "print(SQLiteApprovalStore(sys.argv[1]).consume_many(['shared-grant']))",
        str(path),
    ]
    processes = [subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                 for _ in range(2)]
    outputs = [process.communicate(timeout=10) for process in processes]
    assert all(process.returncode == 0 for process in processes), outputs
    assert sorted(stdout.strip() for stdout, stderr in outputs) == ["False", "True"]


@pytest.mark.parametrize("nonces", [[], [""], [None]])
def test_invalid_nonce_batches_rejected(tmp_path, nonces):
    with pytest.raises(ValueError):
        SQLiteApprovalStore(tmp_path / "approvals.sqlite").consume_many(nonces)


@pytest.mark.parametrize("path", ["", ":memory:"])
def test_ephemeral_paths_rejected(path):
    with pytest.raises(ValueError):
        SQLiteApprovalStore(path)
