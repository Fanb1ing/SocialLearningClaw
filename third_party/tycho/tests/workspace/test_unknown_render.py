"""UNKNOWN=-1 render semantics.

UNKNOWN is an abstention over cells the model cannot claim yet. It should not be counted as a
known-cell mismatch, but strict/full-grid diagnostics and coverage must expose the abstention.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tycho.workspace import wmlib_template as wmlib


def _write_grid(path: Path, grid: list[list[int]]) -> None:
    path.write_text("\n".join(" ".join(format(v, "x") for v in row) for row in grid) + "\n")


def _make_workspace(root: Path) -> None:
    ld = root / "level_0"
    ld.mkdir()
    _write_grid(ld / "turn_000.txt", [[1, 1, 1, 1]])
    _write_grid(ld / "turn_001.txt", [[1, 1, 0, 0]])
    (ld / "turn_001.json").write_text(json.dumps({"action": "ACTION1", "row": None, "col": None}))
    (ld / "terminal.json").write_text(json.dumps({
        "level": 0,
        "pre_turn": 1,
        "action": {"action": "ACTION1", "row": None, "col": None},
        "terminal_grid": [[1, 0, 0, 0]],
        "outcome": "SOLVED",
    }))


@dataclass
class State:
    steps: int


class UnknownModel:
    @staticmethod
    def init_state(grid0, level):
        return State(steps=0)

    @staticmethod
    def transition(state, action):
        return State(steps=state.steps + 1)

    @staticmethod
    def render(state):
        if state.steps == 0:
            return np.asarray([[1, 1, 1, 1]], dtype=np.int16)
        if state.steps == 1:
            return np.asarray([[1, 1, wmlib.UNKNOWN, 0]], dtype=np.int16)
        return np.asarray([[1, wmlib.UNKNOWN, 0, 0]], dtype=np.int16)

    @staticmethod
    def outcome(state):
        return "level_complete" if state.steps >= 2 else "ongoing"


class WrongKnownCellModel(UnknownModel):
    @staticmethod
    def render(state):
        if state.steps == 0:
            return np.asarray([[1, 1, 1, 1]], dtype=np.int16)
        if state.steps == 1:
            return np.asarray([[1, 0, wmlib.UNKNOWN, 0]], dtype=np.int16)
        return np.asarray([[1, wmlib.UNKNOWN, 0, 0]], dtype=np.int16)


class AllUnknownModel(UnknownModel):
    @staticmethod
    def render(state):
        return np.asarray([[wmlib.UNKNOWN, wmlib.UNKNOWN, wmlib.UNKNOWN, wmlib.UNKNOWN]], dtype=np.int16)


def test_unknown_cells_are_abstentions_with_coverage(tmp_path: Path) -> None:
    _make_workspace(tmp_path)

    v = wmlib.verify(UnknownModel, root=str(tmp_path))
    assert v["simulation_accuracy"] == 1
    assert v["strict_simulation_accuracy"] == 0
    assert v["known_cell_accuracy"] == 1
    assert v["prediction_coverage"] == 0.75
    assert v["prediction_coverage_status"] == "partial"
    assert v["unknown_used"] == 1
    assert v["unknown_acceptance_rate"] == 1
    assert v["cell_accuracy"] == 0.75
    assert v["first_divergence"] is None

    g = wmlib.verify_outcome(UnknownModel, root=str(tmp_path))
    assert g["terminal_render_ok"] is True
    assert g["terminal_render_exact"] == 1
    assert g["terminal_render_strict_exact"] == 0
    assert g["terminal_render_known_cell"] == 1
    assert g["terminal_render_prediction_coverage"] == 0.75
    assert g["terminal_render_coverage_status"] == "partial"
    assert g["terminal_render_unknown_used"] == 1


def test_wrong_claimed_cells_still_fail(tmp_path: Path) -> None:
    _make_workspace(tmp_path)

    v = wmlib.verify(WrongKnownCellModel, root=str(tmp_path))
    assert v["simulation_accuracy"] == 0
    assert v["known_cell_accuracy"] == 2 / 3
    assert v["prediction_coverage"] == 0.75
    assert v["first_divergence"] is not None
    assert "claimed cells differ" in v["first_divergence"]["diff"]


def test_all_unknown_render_is_vacuous_not_correct(tmp_path: Path) -> None:
    _make_workspace(tmp_path)

    v = wmlib.verify(AllUnknownModel, root=str(tmp_path))
    assert v["simulation_accuracy"] == 0
    assert v["strict_simulation_accuracy"] == 0
    assert v["prediction_coverage"] == 0
    assert v["prediction_coverage_status"] == "vacuous"
    assert v["ok"] is False
    assert v["first_divergence"] is not None

    g = wmlib.verify_outcome(AllUnknownModel, root=str(tmp_path))
    assert g["terminal_render_ok"] is False
    assert g["terminal_render_exact"] == 0
    assert g["terminal_render_prediction_coverage"] == 0
    assert g["terminal_render_coverage_status"] == "vacuous"
