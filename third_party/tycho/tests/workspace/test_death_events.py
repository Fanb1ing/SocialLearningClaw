"""GAME_OVER side-channel evidence.

Death/reset events are not ordinary transitions: the engine returns GAME_OVER and then RESET starts
the same level again. The workspace keeps fatal evidence separately so the verifier can learn
hazards without stitching a fatal action to the reset frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from arcengine import GameAction

from tycho.workspace import wmlib_template as wmlib
from tycho.workspace.workspace import GameWorkspace


@dataclass
class HiddenDeathState:
    steps: int
    dead: bool = False


class HiddenDeathModel:
    @staticmethod
    def init_state(grid0, level):
        return HiddenDeathState(steps=0)

    @staticmethod
    def transition(state, action):
        steps = state.steps + 1
        return HiddenDeathState(steps=steps, dead=state.dead or steps >= 2)

    @staticmethod
    def render(state):
        return np.asarray([[state.steps]], dtype=np.int16)

    @staticmethod
    def outcome(state):
        return "game_over" if state.dead else "ongoing"


def test_death_outcome_replays_failed_attempt_prefix(tmp_path: Path) -> None:
    ws = GameWorkspace(
        "death-prefix",
        root=str(tmp_path),
        render=False,
        available_actions=[GameAction.ACTION1, GameAction.ACTION2],
    )
    ws.record([[0]], level=0, turn_in_level=0, available=["ACTION1", "ACTION2"])
    ws.record([[1]], level=0, turn_in_level=1, action="ACTION1", available=["ACTION1", "ACTION2"])
    event = ws.record_game_over(
        level=0,
        turn_in_level=2,
        action="ACTION2",
        prev_grid=[[1]],
        game_over_grid=[[9]],
    )

    assert event is not None
    assert event["attempt_start"] == [[0]]
    assert event["attempt_actions"] == [{"action": "ACTION1", "row": None, "col": None}]

    out = wmlib.verify_outcome(HiddenDeathModel, root=str(ws.dir))
    assert out["game_over_on_death"] == 1
    assert out["game_over_ok"] is True


def test_same_turn_deaths_are_not_overwritten(tmp_path: Path) -> None:
    ws = GameWorkspace(
        "death-collisions",
        root=str(tmp_path),
        render=False,
        available_actions=[GameAction.ACTION1],
    )
    ws.record([[0]], level=0, turn_in_level=0, available=["ACTION1"])
    first = ws.record_game_over(
        level=0,
        turn_in_level=1,
        action="ACTION1",
        prev_grid=[[0]],
        game_over_grid=[[9]],
    )
    ws.reset_level(0)
    ws.record([[0]], level=0, turn_in_level=0, available=["ACTION1"])
    second = ws.record_game_over(
        level=0,
        turn_in_level=1,
        action="ACTION1",
        prev_grid=[[0]],
        game_over_grid=[[8]],
    )

    assert first is not None and first["stem"] == "death_001"
    assert second is not None and second["stem"] == "death_001_002"
    assert (ws.dir / "level_0" / "death_001.json").exists()
    assert (ws.dir / "level_0" / "death_001_002.json").exists()
    assert len(wmlib.death_events(root=str(ws.dir))) == 2


def test_reset_archives_prior_attempt_without_stitching_current_frames(tmp_path: Path) -> None:
    ws = GameWorkspace(
        "attempt-history",
        root=str(tmp_path),
        render=False,
        available_actions=[GameAction.ACTION1],
    )
    ws.record([[0]], level=0, turn_in_level=0, available=["ACTION1"])
    ws.record([[1]], level=0, turn_in_level=1, action="ACTION1", available=["ACTION1"])

    ws.reset_level(0, reason="actor_reset")

    assert wmlib.frames(root=str(ws.dir)) == {}
    archived = wmlib.attempts(root=str(ws.dir))
    assert len(archived) == 1
    assert archived[0]["reason"] == "actor_reset"
    assert archived[0]["n_frames"] == 2
    prior_frames = wmlib.frames(root=archived[0]["root"])
    assert sorted(prior_frames) == [(0, 0), (0, 1)]
    prior_transitions = wmlib.transitions(root=archived[0]["root"])
    assert len(prior_transitions) == 1
    assert prior_transitions[0]["action"] == "ACTION1"

    ws.record([[0]], level=0, turn_in_level=0, available=["ACTION1"])
    assert sorted(wmlib.frames(root=str(ws.dir))) == [(0, 0)]
    assert wmlib.transitions(root=str(ws.dir)) == []

    ws.reset_level(0, reason="game_over")
    archived = wmlib.attempts(root=str(ws.dir))
    assert [(a["attempt"], a["reason"]) for a in archived] == [
        (0, "actor_reset"),
        (1, "game_over"),
    ]
