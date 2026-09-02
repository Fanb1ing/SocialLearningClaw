"""Observation-variant verifier smoke tests.

The canonical render remains deterministic. Bounded observation_variants can accept small display
quantization differences without making trigger mode treat the world model as dynamically wrong.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

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


class NoVariantsModel:
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
            return np.asarray([[1, 1, 1, 0]], dtype=np.int16)  # one-pixel HUD quantization miss
        return np.asarray([[1, 1, 0, 0]], dtype=np.int16)      # terminal HUD miss

    @staticmethod
    def outcome(state):
        return "level_complete" if state.steps >= 2 else "ongoing"


class VariantModel(NoVariantsModel):
    @staticmethod
    def observation_variants(state):
        base = VariantModel.render(state).copy()
        if state.steps == 1:
            alt = base.copy()
            alt[0, 2] = 0
            return [alt]
        if state.steps >= 2:
            alt = base.copy()
            alt[0, 1] = 0
            return [alt]
        return []


class BadRenderShapeVariantModel(NoVariantsModel):
    @staticmethod
    def render(state):
        return np.asarray([1, 1, 1, 0], dtype=np.int16)

    @staticmethod
    def observation_variants(state):
        if state.steps == 1:
            return [np.asarray([[1, 1, 0, 0]], dtype=np.int16)]
        if state.steps >= 2:
            return [np.asarray([[1, 0, 0, 0]], dtype=np.int16)]
        return []


class UnknownMaskPreservingVariantModel(NoVariantsModel):
    @staticmethod
    def render(state):
        if state.steps == 0:
            return np.asarray([[1, 1, 1, 1]], dtype=np.int16)
        if state.steps == 1:
            return np.asarray([[1, 1, wmlib.UNKNOWN, 1]], dtype=np.int16)
        return np.asarray([[1, wmlib.UNKNOWN, 0, 1]], dtype=np.int16)

    @staticmethod
    def observation_variants(state):
        base = UnknownMaskPreservingVariantModel.render(state).copy()
        if state.steps == 1:
            alt = base.copy()
            alt[0, 3] = 0
            return [alt]
        if state.steps >= 2:
            alt = base.copy()
            alt[0, 3] = 0
            return [alt]
        return []


class UnknownMaskChangingVariantModel(NoVariantsModel):
    @staticmethod
    def observation_variants(state):
        base = UnknownMaskChangingVariantModel.render(state).copy()
        if state.steps == 1:
            alt = base.copy()
            alt[0, 2] = wmlib.UNKNOWN
            return [alt]
        return []


def _check_without_variants(root: Path) -> dict[str, bool]:
    v = wmlib.verify(NoVariantsModel, root=str(root))
    g = wmlib.verify_outcome(NoVariantsModel, root=str(root))
    return {
        "strict_and_accepted_dynamics_fail": (
            v["simulation_accuracy"] == 0
            and v["strict_simulation_accuracy"] == 0
            and v["first_divergence"] is not None
        ),
        "terminal_render_fails": (
            g["outcome_ok"] is True
            and g["terminal_render_ok"] is False
            and g["terminal_render_exact"] == 0
            and g["terminal_render_strict_exact"] == 0
        ),
        "no_variant_noise": not v.get("variant_used") and not g.get("terminal_render_variant_used"),
    }


def _check_bad_render_shape(root: Path) -> dict[str, bool]:
    v = wmlib.verify(BadRenderShapeVariantModel, root=str(root))
    g = wmlib.verify_outcome(BadRenderShapeVariantModel, root=str(root))
    return {
        "variants_do_not_replace_render_shape": (
            v["simulation_accuracy"] == 0
            and v["strict_simulation_accuracy"] == 0
            and v["variant_used"] == 0
            and v["first_divergence"] is not None
            and "render shape" in v["first_divergence"]["diff"]
        ),
        "terminal_variants_do_not_replace_render_shape": (
            g["terminal_render_ok"] is False
            and g["terminal_render_variant_used"] == 0
        ),
    }


def _check_with_variants(root: Path) -> dict[str, bool]:
    v = wmlib.verify(VariantModel, root=str(root))
    g = wmlib.verify_outcome(VariantModel, root=str(root))
    return {
        "accepted_dynamics_pass": (
            v["simulation_accuracy"] == 1
            and v["strict_simulation_accuracy"] == 0
            and v["variant_used"] == 1
            and v["first_divergence"] is None
        ),
        "canonical_cell_diagnostic_remains": v["cell_accuracy"] == 0.75,
        "terminal_render_accepted": (
            g["ok"] is True
            and g["terminal_render_ok"] is True
            and g["terminal_render_exact"] == 1
            and g["terminal_render_strict_exact"] == 0
            and g["terminal_render_variant_used"] == 1
        ),
    }


def _check_unknown_mask_variants(root: Path) -> dict[str, bool]:
    keep_v = wmlib.verify(UnknownMaskPreservingVariantModel, root=str(root))
    keep_g = wmlib.verify_outcome(UnknownMaskPreservingVariantModel, root=str(root))
    change_v = wmlib.verify(UnknownMaskChangingVariantModel, root=str(root))
    return {
        "variants_may_preserve_unknown_mask": (
            keep_v["simulation_accuracy"] == 1
            and keep_v["strict_simulation_accuracy"] == 0
            and keep_v["variant_used"] == 1
            and keep_v["prediction_coverage_status"] == "partial"
            and keep_g["terminal_render_ok"] is True
            and keep_g["terminal_render_variant_used"] == 1
            and keep_g["terminal_render_coverage_status"] == "partial"
        ),
        "variants_may_not_change_unknown_mask": (
            change_v["simulation_accuracy"] == 0
            and change_v["variant_used"] == 0
            and any("changes the UNKNOWN mask" in e for e in change_v.get("variant_errors") or [])
        ),
    }


def main() -> int:
    ok = True
    with TemporaryDirectory() as td:
        root = Path(td)
        _make_workspace(root)
        for group, checks in (
            ("without_variants", _check_without_variants(root)),
            ("bad_render_shape", _check_bad_render_shape(root)),
            ("with_variants", _check_with_variants(root)),
            ("unknown_mask_variants", _check_unknown_mask_variants(root)),
        ):
            print(f"=== {group} ===")
            for name, passed in checks.items():
                print(f"  {name}: {passed}")
                ok = ok and passed
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def test_observation_variants_contracts(tmp_path: Path) -> None:
    _make_workspace(tmp_path)
    groups = {
        "without_variants": _check_without_variants(tmp_path),
        "bad_render_shape": _check_bad_render_shape(tmp_path),
        "with_variants": _check_with_variants(tmp_path),
        "unknown_mask_variants": _check_unknown_mask_variants(tmp_path),
    }
    failed = [f"{group}.{name}" for group, checks in groups.items()
              for name, passed in checks.items() if not passed]
    assert not failed


if __name__ == "__main__":
    raise SystemExit(main())
