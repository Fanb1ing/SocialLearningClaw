from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tycho.workspace import wmlib_template as wmlib


def _grid(path: Path, value: int) -> None:
    path.write_text(f"{value:x}\n")


@dataclass
class State:
    level: int
    step: int


class WrongInitialModel:
    @staticmethod
    def init_state(grid0, level):
        return State(level=int(level), step=0)

    @staticmethod
    def transition(state, action):
        return State(level=state.level, step=1)

    @staticmethod
    def render(state):
        if state.step:
            return np.asarray([[1]], dtype=np.int16)
        return np.asarray([[9]], dtype=np.int16)

    @staticmethod
    def outcome(state):
        return "ongoing"


def test_initial_render_is_independent_of_transition_accuracy(tmp_path: Path) -> None:
    level0 = tmp_path / "level_0"
    level0.mkdir()
    _grid(level0 / "turn_000.txt", 0)
    _grid(level0 / "turn_001.txt", 1)
    (level0 / "turn_001.json").write_text(json.dumps({"action": "ACTION1"}))
    level1 = tmp_path / "level_1"
    level1.mkdir()
    _grid(level1 / "turn_000.txt", 0)

    result = wmlib.verify(WrongInitialModel, root=str(tmp_path))

    assert result["simulation_accuracy"] == 1
    assert result["initial_render_ok"] is False
    assert result["initial_render_accuracy"] == 0
    assert result["n_initial_render"] == 2
    assert result["n_matched"] == 1
    assert result["n_changing"] == 1
    assert result["n_total"] == 1
    assert result["n_observed_changed"] == 1
    assert result["n_observed_noop"] == 0
    assert result["n_joint_noop"] == 0
    assert result["n_unpredicted_noop"] == 0
    assert result["n_false_change_on_noop"] == 0
    assert result["archived_attempts"] == 0
    assert result["archived_transitions"] == 0
    assert set(result["per_level"]) == {0, 1}
    assert result["per_level"][1]["simulation_accuracy"] is None
    assert result["per_level"][1]["initial_render_first_mismatch"]


def test_archived_attempts_are_inventoried_but_not_replay_scored(tmp_path: Path) -> None:
    current = tmp_path / "level_0"
    current.mkdir()
    _grid(current / "turn_000.txt", 0)
    _grid(current / "turn_001.txt", 1)
    (current / "turn_001.json").write_text(json.dumps({"action": "ACTION1"}))

    archived = tmp_path / "attempts" / "level_0_attempt_000"
    archived_level = archived / "level_0"
    archived_level.mkdir(parents=True)
    (archived / "attempt.json").write_text(json.dumps({"level": 0, "attempt": 0}))
    _grid(archived_level / "turn_000.txt", 0)
    _grid(archived_level / "turn_001.txt", 1)
    (archived_level / "turn_001.json").write_text(json.dumps({"action": "ACTION1"}))

    result = wmlib.verify(WrongInitialModel, root=str(tmp_path))

    # simulation_accuracy remains the current attempt's metric. Archived RESET evidence is
    # separately visible so reports cannot silently present it as replay-verified.
    assert result["simulation_accuracy"] == 1
    assert result["n_total"] == 1
    assert result["archived_attempts"] == 1
    assert result["archived_transitions"] == 1
    assert result["archived_observed_changed"] == 1
    assert result["archived_by_level"][0] == {
        "attempts": 1,
        "transitions": 1,
        "observed_changed": 1,
    }
