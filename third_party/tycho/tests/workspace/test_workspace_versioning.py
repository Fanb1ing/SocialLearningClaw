from __future__ import annotations

from pathlib import Path

import numpy as np

from tycho.workspace.version_store import (
    WorkspaceVersionStore,
    materialize_workspace_snapshot,
)
from tycho.workspace.workspace import GameWorkspace


def test_snapshot_keeps_exact_long_text_and_binary_history(tmp_path: Path) -> None:
    workspace = GameWorkspace("zz99-test", root=str(tmp_path), render=False)
    long_source = "x = 1\n" + "# exact history\n" * 1_000
    (workspace.dir / "world_model.py").write_text(long_source)
    array = np.arange(64, dtype=np.int16).reshape(8, 8)
    np.save(workspace.dir / "world_map.npy", array)
    level_dir = workspace.dir / "level_0"
    level_dir.mkdir()
    (level_dir / "turn_000.txt").write_text("harness evidence")
    (level_dir / "agent_helper.json").write_text('{"known": true}')
    archived = workspace.dir / "attempts" / "level_0_attempt_000" / "level_0"
    archived.mkdir(parents=True)
    (archived / "turn_000.txt").write_text("prior attempt evidence")

    snapshot = workspace.snapshot(level=0, turn_in_level=0)

    assert snapshot["snapshot_schema"] == 2
    assert snapshot["contents"]["world_model.py"] == long_source
    assert len(snapshot["contents"]["world_model.py"]) > 12_000
    assert "world_map.npy" in snapshot["file_versions"]
    assert "world_map.npy" not in snapshot["contents"]
    assert "level_0/turn_000.txt" not in snapshot["file_versions"]
    assert "attempts/level_0_attempt_000/level_0/turn_000.txt" not in snapshot["file_versions"]
    assert snapshot["contents"]["level_0/agent_helper.json"] == '{"known": true}'

    descriptor = snapshot["file_versions"]["world_map.npy"]
    assert descriptor["kind"] == "binary"
    blob = tmp_path / ".workspace_blobs" / descriptor["sha256"]
    assert blob.read_bytes() == (workspace.dir / "world_map.npy").read_bytes()


def test_snapshot_tracks_binary_changes_and_deletion(tmp_path: Path) -> None:
    workspace = GameWorkspace("zz99-test", root=str(tmp_path), render=False)
    state_path = workspace.dir / "state.bin"
    state_path.write_bytes(b"first")
    first = workspace.snapshot(0, 0)
    state_path.write_bytes(b"second")
    second = workspace.snapshot(0, 1)
    state_path.unlink()
    third = workspace.snapshot(0, 2)

    first_sha = first["file_versions"]["state.bin"]["sha256"]
    second_sha = second["file_versions"]["state.bin"]["sha256"]
    assert first_sha != second_sha
    assert (tmp_path / ".workspace_blobs" / first_sha).read_bytes() == b"first"
    assert (tmp_path / ".workspace_blobs" / second_sha).read_bytes() == b"second"
    assert "state.bin" not in third["file_versions"]


def test_large_file_omission_is_explicit(tmp_path: Path) -> None:
    workspace = tmp_path / "game"
    workspace.mkdir()
    (workspace / "large.bin").write_bytes(b"12345")
    store = WorkspaceVersionStore(workspace, max_blob_bytes=4)

    versions, contents, warnings = store.capture()

    assert contents == {}
    assert versions["large.bin"]["stored"] is False
    assert versions["large.bin"]["status"] == "omitted_large"
    assert versions["large.bin"]["size"] == 5
    assert warnings and "large.bin" in warnings[0]


def test_materializer_restores_manifest_and_preserves_harness_evidence(tmp_path: Path) -> None:
    source = tmp_path / "source" / "game"
    source.mkdir(parents=True)
    (source / "world_model.py").write_text("VALUE = 7\n")
    (source / "state.bin").write_bytes(b"latent")
    store = WorkspaceVersionStore(source)
    versions, _, _ = store.capture()

    destination = tmp_path / "restored"
    (destination / "level_0").mkdir(parents=True)
    (destination / "level_0" / "turn_000.txt").write_text("observation")
    (destination / "stale.py").write_text("remove me")

    materialize_workspace_snapshot(
        destination,
        versions,
        blob_dir=source.parent / ".workspace_blobs",
    )

    assert (destination / "world_model.py").read_text() == "VALUE = 7\n"
    assert (destination / "state.bin").read_bytes() == b"latent"
    assert not (destination / "stale.py").exists()
    assert (destination / "level_0" / "turn_000.txt").read_text() == "observation"
