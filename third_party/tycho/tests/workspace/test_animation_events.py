from __future__ import annotations

from pathlib import Path

import numpy as np

from tycho.workspace.workspace import GameWorkspace


def _frames(offset: int = 0, n: int = 3) -> list[list[list[int]]]:
    frames = []
    for i in range(n):
        g = np.zeros((4, 4), dtype=np.int16)
        g[i % 4, :] = offset + i + 1
        frames.append(g.tolist())
    return frames


def test_animation_event_retention_uses_numeric_turn_order(tmp_path: Path) -> None:
    ws = GameWorkspace(
        "anim-retention",
        root=str(tmp_path),
        render=False,
        available_actions=["ACTION1"],
    )

    for turn in (998, 999, 1000):
        ws.record_animation_event(
            level=0,
            turn_in_level=turn,
            action="ACTION1",
            frames=_frames(turn),
            selected_indices=[0, 2],
            keep_per_level=2,
        )

    level = ws.dir / "level_0"
    assert not (level / "animation_998_ACTION1").exists()
    assert (level / "animation_999_ACTION1").exists()
    assert (level / "animation_1000_ACTION1").exists()


def test_animation_event_persists_all_frames_and_selected_keyframes(tmp_path: Path) -> None:
    ws = GameWorkspace(
        "anim-full-frames",
        root=str(tmp_path),
        render=False,
        available_actions=["ACTION1"],
    )

    event = ws.record_animation_event(
        level=0,
        turn_in_level=1,
        action="ACTION1",
        frames=_frames(n=5),
        selected_indices=[0, 4],
        keep_per_level=5,
    )

    assert event is not None
    assert len(event["all_frame_files"]) == 5
    assert [f["index"] for f in event["selected_frame_files"]] == [0, 4]

    import importlib.util

    spec = importlib.util.spec_from_file_location("wmlib_under_test", ws.dir / "wmlib.py")
    wmlib = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(wmlib)

    events = wmlib.animation_index(str(ws.dir))
    assert len(events) == 1
    assert events[0]["directory"] == "level_0/animation_001_ACTION1"
    assert "features" not in events[0]
    assert "frame_paths" not in events[0]

    detailed = wmlib.animation_index(str(ws.dir), level=0, turn=1, details=True)
    assert detailed[0]["frame_paths"] == [
        f"level_0/animation_001_ACTION1/frame_{i:03d}.txt"
        for i in range(5)
    ]
    assert detailed[0]["keyframe_paths"] == [
        "level_0/animation_001_ACTION1/frame_000.txt",
        "level_0/animation_001_ACTION1/frame_004.txt",
    ]
    assert "frames" not in events[0]
    assert "keyframes" not in events[0]
    assert "all_frame_files" not in events[0]
    assert "selected_frame_files" not in events[0]

    all_grids = wmlib.animation_grids(events[0], root=str(ws.dir))
    keyframes = wmlib.animation_grids(events[0], root=str(ws.dir), keyframes=True)
    indexed = wmlib.animation_grids(events[0]["directory"], root=str(ws.dir), indices=[1, 3])
    assert len(all_grids) == 5
    assert [int(g.max()) for g in keyframes] == [1, 5]
    assert [int(g.max()) for g in indexed] == [2, 4]


def test_animation_index_filters_before_returning_recent_events(tmp_path: Path) -> None:
    ws = GameWorkspace("anim-index", root=str(tmp_path), render=False, available_actions=["ACTION1"])
    for level, turn in ((0, 1), (0, 2), (1, 1)):
        ws.record_animation_event(
            level=level,
            turn_in_level=turn,
            action="ACTION1",
            frames=_frames(n=3),
            selected_indices=[0, 2],
            keep_per_level=5,
        )

    import importlib.util

    spec = importlib.util.spec_from_file_location("wmlib_filter_test", ws.dir / "wmlib.py")
    wmlib = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(wmlib)

    assert [e["turn"] for e in wmlib.animation_index(str(ws.dir), level=0)] == [1, 2]
    assert [e["level"] for e in wmlib.animation_index(str(ws.dir), turn=1)] == [0, 1]
    assert [(e["level"], e["turn"]) for e in wmlib.animation_index(str(ws.dir), last=2)] == [
        (0, 2),
        (1, 1),
    ]


def test_reset_archives_animation_evidence_with_its_attempt(tmp_path: Path) -> None:
    ws = GameWorkspace(
        "anim-attempts",
        root=str(tmp_path),
        render=False,
        available_actions=["ACTION1"],
    )
    ws.record([[0]], level=0, turn_in_level=0, available=["ACTION1"])
    ws.record_animation_event(
        level=0,
        turn_in_level=99,
        action="ACTION1",
        frames=_frames(n=3),
        selected_indices=[0, 2],
        keep_per_level=2,
    )
    ws.write_file(
        "notes/animation_evidence.md",
        "frames: level_0/animation_099_ACTION1\n",
    )

    ws.reset_level(0, reason="actor_reset")

    import importlib.util

    spec = importlib.util.spec_from_file_location("wmlib_attempt_test", ws.dir / "wmlib.py")
    wmlib = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(wmlib)

    assert wmlib.animation_index(str(ws.dir)) == []
    attempt = wmlib.attempts(str(ws.dir))[0]
    archived = wmlib.animation_index(attempt["root"])
    assert len(archived) == 1
    assert len(wmlib.animation_grids(archived[0], root=attempt["root"])) == 3
    note = ws.read_file("notes/animation_evidence.md")
    assert "frames: attempts/level_0_attempt_000/level_0/animation_099_ACTION1" in note

    # A new low-turn event is retained independently of the prior attempt's high turn number.
    ws.record([[0]], level=0, turn_in_level=0, available=["ACTION1"])
    ws.record_animation_event(
        level=0,
        turn_in_level=1,
        action="ACTION1",
        frames=_frames(offset=10, n=3),
        selected_indices=[0, 2],
        keep_per_level=2,
    )
    live = wmlib.animation_index(str(ws.dir))
    assert [(event["turn"], event["directory"]) for event in live] == [
        (1, "level_0/animation_001_ACTION1")
    ]
