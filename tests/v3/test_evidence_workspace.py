from __future__ import annotations

import json
import subprocess
import sys

from socialclaw.v3.evidence import EvidenceIndex, EvidenceRole, index_workspace
from socialclaw.v3.tools import EFPSToolExecutor
from socialclaw.v3.workspace import EFPSGameWorkspace
from tycho.workspace.sandbox import PythonSandbox


def _record_small_workspace(root, monkeypatch) -> EFPSGameWorkspace:
    monkeypatch.setenv("SC_V3_RUN_ID", "test_run")
    workspace = EFPSGameWorkspace(
        "demo-1234",
        root=str(root),
        render=False,
        available_actions=["ACTION1"],
    )
    workspace.record(
        [[0, 1], [0, 0]],
        level=0,
        turn_in_level=0,
        available=["ACTION1"],
    )
    workspace.record(
        [[0, 0], [0, 1]],
        level=0,
        turn_in_level=1,
        action="ACTION1",
        available=["ACTION1"],
    )
    workspace.record_terminal(
        0,
        [[0, 0], [1, 0]],
        action="ACTION1",
    )
    return workspace


def test_workspace_seeds_helpers_and_indexes_typed_observations(tmp_path, monkeypatch) -> None:
    workspace = _record_small_workspace(tmp_path, monkeypatch)

    assert (workspace.dir / "efps_runtime.py").is_file()
    assert (workspace.dir / "efps_audit.py").is_file()
    assert "def evidence_refs" in (workspace.dir / "wmlib.py").read_text()
    index = EvidenceIndex.read(workspace.dir / "notes" / "evidence_index.json")
    roles = [item.role for item in index.evidence]
    assert roles == [
        EvidenceRole.LEVEL_INITIALIZATION.value,
        EvidenceRole.COMPLETION_TERMINAL.value,
        EvidenceRole.DECISION.value,
    ]
    assert len(index.ids) == 3

    completed = subprocess.run(
        [sys.executable, "-B", "efps_audit.py"],
        cwd=workspace.dir,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["status"] == "ok"
    manifest = json.loads((workspace.dir / "notes" / "efps_manifest.json").read_text())
    assert manifest["prototypes"] == []
    assert manifest["schemas"] == []


def test_evidence_ids_do_not_depend_on_absolute_workspace_path(tmp_path, monkeypatch) -> None:
    left = _record_small_workspace(tmp_path / "left", monkeypatch)
    right = _record_small_workspace(tmp_path / "right", monkeypatch)

    left_index = index_workspace(left.dir, run_id="test_run", game_id="demo")
    right_index = index_workspace(right.dir, run_id="test_run", game_id="demo")

    assert [item.evidence_id for item in left_index.evidence] == [
        item.evidence_id for item in right_index.evidence
    ]


def test_reset_archives_are_separate_evidence(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SC_V3_RUN_ID", "reset_run")
    workspace = EFPSGameWorkspace(
        "demo-1234",
        root=str(tmp_path),
        render=False,
        available_actions=["ACTION1"],
    )
    workspace.record([[0]], level=0, turn_in_level=0, available=["ACTION1"])
    workspace.record([[1]], level=0, turn_in_level=1, action="ACTION1", available=["ACTION1"])
    workspace.reset_level(0, reason="actor_reset")
    workspace.record([[0]], level=0, turn_in_level=0, action="RESET", available=["ACTION1"])

    index = EvidenceIndex.read(workspace.dir / "notes" / "evidence_index.json")
    reset_refs = [item for item in index.evidence if item.role == EvidenceRole.RESET_ARCHIVE.value]
    initial_refs = [
        item for item in index.evidence
        if item.role == EvidenceRole.LEVEL_INITIALIZATION.value
    ]
    assert len(reset_refs) == 1
    assert len(initial_refs) == 2
    assert {item.attempt for item in initial_refs} == {0, 1}


def test_world_model_edit_runs_tycho_verifier_and_efps_audit(tmp_path, monkeypatch) -> None:
    workspace = _record_small_workspace(tmp_path, monkeypatch)
    executor = EFPSToolExecutor(
        workspace,
        python_sandbox=PythonSandbox(runtime="host"),
    )
    source = (workspace.dir / "world_model.py").read_text()
    source += '\nEFPS_APPLICABILITY = "partial"\n'

    feedback = executor.execute(
        "write_file",
        {"path": "world_model.py", "content": source},
    )

    assert "[auto world-model feedback]" in feedback
    assert "[auto EFPS audit]" in feedback
    assert '"status": "ok"' in feedback
