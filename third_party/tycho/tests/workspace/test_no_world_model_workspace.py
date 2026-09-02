from __future__ import annotations

from pathlib import Path

from tycho.workspace.workspace import GameWorkspace


def _tool_description(name: str, *, world_model_enabled: bool) -> str:
    from tycho.workspace.agent_tools import tool_specs

    for spec in tool_specs(world_model_enabled=world_model_enabled):
        if spec["name"] == name:
            return spec["description"]
    raise AssertionError(f"missing tool spec {name}")


def test_no_world_model_workspace_seeds_observation_helpers_only(tmp_path: Path) -> None:
    ws = GameWorkspace(
        "dummy-game",
        root=str(tmp_path),
        render=False,
        available_actions=["ACTION1"],
        seed_world_model=False,
    )

    assert (ws.dir / "wmlib.py").exists()
    assert not (ws.dir / "world_model.py").exists()
    assert not (ws.dir / "verify.py").exists()
    assert not (ws.dir / "plan.py").exists()


def test_no_world_model_run_python_description_keeps_observation_helpers_without_hidden_files() -> None:
    desc = _tool_description("run_python", world_model_enabled=False)

    assert "grid" in desc
    assert "wmlib.frames()" in desc
    assert "dict {(level, turn): grid}" in desc
    assert "{level, turn, prev, next, action, row, col}" in desc
    assert "wmlib.death_events()" in desc
    assert "not np.loadtxt-compatible" in desc
    assert "use `wmlib.frames()` for prior grids" in desc
    assert "world_model.py" not in desc
    assert "verify.py" not in desc
    assert "plan.py" not in desc
    assert "no-world-model" not in desc


def test_current_grid_uses_explicit_current_marker_over_stale_future_levels(tmp_path: Path) -> None:
    from tycho.workspace import wmlib_template as wmlib

    ws = GameWorkspace(
        "game01-current-marker",
        root=str(tmp_path),
        render=False,
        available_actions=["ACTION1"],
        seed_world_model=False,
    )
    ws.record([[6]], level=6, turn_in_level=63, state="NOT_FINISHED", available=["ACTION1"])
    ws.record([[2]], level=2, turn_in_level=0, state="NOT_FINISHED", available=["ACTION1"])

    assert wmlib.current_frame_key(root=str(ws.dir)) == (2, 0)
    assert int(wmlib.current_grid(root=str(ws.dir))[0, 0]) == 2


def test_fresh_workspace_reset_removes_prior_game_notes(tmp_path: Path) -> None:
    first = GameWorkspace(
        "game01-old",
        root=str(tmp_path),
        render=False,
        available_actions=["ACTION1"],
    )
    stale = first.dir / "notes" / "level_6_insights.md"
    stale.write_text("prior run note")

    fresh = GameWorkspace(
        "game01-new",
        root=str(tmp_path),
        render=False,
        available_actions=["ACTION1"],
        resume=False,
    )

    assert fresh.dir == first.dir
    assert not stale.exists()
    assert (fresh.dir / "notes").is_dir()
    assert (fresh.dir / "world_model.py").exists()


def test_resume_workspace_preserves_prior_game_notes(tmp_path: Path) -> None:
    first = GameWorkspace(
        "game01-old",
        root=str(tmp_path),
        render=False,
        available_actions=["ACTION1"],
    )
    note = first.dir / "notes" / "level_0_insights.md"
    note.write_text("resume note")
    model = first.dir / "world_model.py"
    model.write_text("RESUME_MODEL = True\n")

    resumed = GameWorkspace(
        "game01-new",
        root=str(tmp_path),
        render=False,
        available_actions=["ACTION1"],
        resume=True,
    )

    assert resumed.dir == first.dir
    assert note.read_text() == "resume note"
    assert model.read_text() == "RESUME_MODEL = True\n"
