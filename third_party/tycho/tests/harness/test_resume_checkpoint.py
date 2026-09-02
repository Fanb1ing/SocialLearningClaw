from __future__ import annotations

import json
from pathlib import Path

import pytest

from tycho.harness import resume
from tycho.harness.resume import GameJournal, ResumeCheckpointError


def _journal(tmp_path: Path) -> tuple[GameJournal, Path]:
    journal = GameJournal(tmp_path / "resume" / "game01")
    journal.init_meta(game_id="game01-test", fingerprint="fingerprint", context_mode="tail_evict")
    workspace = tmp_path / "ws" / "game01"
    workspace.mkdir(parents=True)
    return journal, workspace


def _record(journal: GameJournal, workspace: Path, k: int, snapshot: object) -> None:
    journal.record_step(
        k,
        action="ACTION1",
        x=None,
        y=None,
        frame_hash=100 + k,
        level=0,
        state="GameState.NOT_FINISHED",
        agent_snapshot=snapshot,
        workspace_dir=workspace,
    )


def test_exact_checkpoint_restores_files_deletions_and_actor_state(tmp_path: Path) -> None:
    journal, workspace = _journal(tmp_path)
    (workspace / "world_model.py").write_text("MODEL = 1\n")
    (workspace / "notes").mkdir()
    (workspace / "notes" / "belief.md").write_text("known\n")
    (workspace / "latent.bin").write_bytes(b"\x00\xffstate")
    _record(journal, workspace, 0, {"history": ["turn 0"], "counter": 1})

    # Simulate an interrupted next turn that edited, deleted, and added visible files.
    (workspace / "world_model.py").write_text("FUTURE = True\n")
    (workspace / "notes" / "belief.md").unlink()
    (workspace / "future.txt").write_text("must disappear")

    checkpoint = journal.load_checkpoint()
    journal.restore_workspace(workspace, checkpoint)

    assert (workspace / "world_model.py").read_text() == "MODEL = 1\n"
    assert (workspace / "notes" / "belief.md").read_text() == "known\n"
    assert (workspace / "latent.bin").read_bytes() == b"\x00\xffstate"
    assert not (workspace / "future.txt").exists()
    assert journal.load_actor_state(checkpoint) == {"history": ["turn 0"], "counter": 1}


def test_checkpoint_retains_only_head_and_previous_generations(tmp_path: Path) -> None:
    journal, workspace = _journal(tmp_path)
    for k in range(8):
        (workspace / "world_model.py").write_text(f"VERSION = {k}\n")
        _record(journal, workspace, k, {"history": ["x" * 100_000] * (k + 1)})

    generations = list(journal.generation_dir.iterdir())
    blobs = list(journal.blob_dir.iterdir())
    assert len(generations) == 2
    # Two actor states and two workspace versions. Repeated bodies would reduce this further.
    assert len(blobs) <= 4
    assert len(journal.steps()) == 8


def test_uncommitted_action_tail_is_ignored_and_replaced(tmp_path: Path) -> None:
    journal, workspace = _journal(tmp_path)
    (workspace / "state.txt").write_text("committed")
    _record(journal, workspace, 0, {"turn": 1})
    with journal.actions_path.open("a") as handle:
        handle.write(json.dumps({"k": 1, "action": "ACTION7"}) + "\n")

    assert [step["action"] for step in journal.steps()] == ["ACTION1"]
    (workspace / "state.txt").write_text("next")
    _record(journal, workspace, 1, {"turn": 2})

    assert [step["action"] for step in journal.steps()] == ["ACTION1", "ACTION1"]


def test_missing_workspace_blob_blocks_resume_without_mutating_workspace(tmp_path: Path) -> None:
    journal, workspace = _journal(tmp_path)
    (workspace / "state.txt").write_text("committed")
    _record(journal, workspace, 0, {"turn": 1})
    checkpoint = journal.load_checkpoint()
    descriptor = checkpoint["workspace"]["files"]["state.txt"]
    (journal.blob_dir / descriptor["sha256"]).unlink()
    (workspace / "state.txt").write_text("live")

    with pytest.raises(ResumeCheckpointError, match="missing blob"):
        journal.restore_workspace(workspace, checkpoint)
    assert (workspace / "state.txt").read_text() == "live"


def test_corrupt_actor_blob_blocks_resume(tmp_path: Path) -> None:
    journal, workspace = _journal(tmp_path)
    (workspace / "state.txt").write_text("committed")
    _record(journal, workspace, 0, {"turn": 1})
    checkpoint = journal.load_checkpoint()
    actor_path = journal.blob_dir / checkpoint["actor"]["sha256"]
    actor_path.write_bytes(b"corrupt")

    with pytest.raises(ResumeCheckpointError, match="corrupt blob"):
        journal.load_actor_state(checkpoint)


def test_workspace_symlink_fails_closed(tmp_path: Path) -> None:
    journal, workspace = _journal(tmp_path)
    target = workspace / "target.txt"
    target.write_text("target")
    (workspace / "link.txt").symlink_to(target)

    with pytest.raises(ResumeCheckpointError, match="symlink"):
        _record(journal, workspace, 0, {"turn": 1})


def test_crash_after_action_append_before_head_leaves_no_committed_step(
    tmp_path: Path, monkeypatch
) -> None:
    journal, workspace = _journal(tmp_path)
    (workspace / "state.txt").write_text("candidate")
    original = resume._atomic_write_json

    def fail_head(path, value):
        if path.name == "HEAD.json":
            raise OSError("injected crash")
        return original(path, value)

    monkeypatch.setattr(resume, "_atomic_write_json", fail_head)
    with pytest.raises(OSError, match="injected crash"):
        _record(journal, workspace, 0, {"turn": 1})
    assert journal.steps() == []

    monkeypatch.setattr(resume, "_atomic_write_json", original)
    (workspace / "state.txt").write_text("retry")
    _record(journal, workspace, 0, {"turn": 1})
    assert len(journal.steps()) == 1
    assert len(journal.actions_path.read_text().splitlines()) == 1
