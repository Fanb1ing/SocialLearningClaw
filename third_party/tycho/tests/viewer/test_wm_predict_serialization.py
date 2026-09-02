"""World-model metric serialization for viewer overlays."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from tycho.harness.run_parallel import _bake_wm_predictions
from tycho.viewer.viz import _compute_wm_predictions
from tycho.viewer.wm_predict import predict_for_workspace


def _write_grid(path: Path, grid: list[list[int]]) -> None:
    path.write_text("\n".join(" ".join(format(v, "x") for v in row) for row in grid) + "\n")


def _write_world_model(path: Path) -> None:
    path.write_text("""
from dataclasses import dataclass
import numpy as np
import wmlib


@dataclass
class State:
    steps: int


def init_state(grid0, level):
    return State(steps=0)


def transition(state, action):
    return State(steps=state.steps + 1)


def render(state):
    if state.steps == 0:
        return np.asarray([[1, 1, 1, 1]], dtype=np.int16)
    if state.steps == 1:
        return np.asarray([[1, 1, 1, 0]], dtype=np.int16)
    return np.asarray([[1, 1, 0, 0]], dtype=np.int16)


def observation_variants(state):
    base = render(state).copy()
    if state.steps == 1:
        alt = base.copy()
        alt[0, 2] = 0
        return [alt]
    if state.steps >= 2:
        alt = base.copy()
        alt[0, 1] = 0
        return [alt]
    return []


def outcome(state):
    return "level_complete" if state.steps >= 2 else "ongoing"


def actions(state):
    return [{"action": "ACTION1", "row": None, "col": None}]


def heuristic(state):
    return max(0, 2 - state.steps)
""".lstrip())


def _make_workspace(root: Path) -> None:
    from tycho.workspace import wmlib_template

    (root / "wmlib.py").write_text(Path(wmlib_template.__file__).read_text())
    _write_world_model(root / "world_model.py")
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


def _check_predict_for_workspace(root: Path) -> dict[str, bool]:
    res = predict_for_workspace(str(root), timeout=10)
    verify = res.get("verify") or {}
    outcome = res.get("outcome") or {}
    preds = res.get("predictions") or {}
    return {
        "worker_no_error": not res.get("error"),
        "variant_metrics_serialized": (
            verify.get("simulation_accuracy") == 1
            and verify.get("strict_simulation_accuracy") == 0
            and verify.get("variant_used") == 1
            and verify.get("prediction_coverage_status") == "complete"
        ),
        "terminal_variant_metrics_serialized": (
            outcome.get("terminal_render_exact") == 1
            and outcome.get("terminal_render_strict_exact") == 0
            and outcome.get("terminal_render_variant_used") == 1
            and outcome.get("terminal_render_coverage_status") == "complete"
        ),
        "predictions_present": "0_0" in preds and "0_1" in preds,
    }


def _check_bake_predictions(root: Path) -> dict[str, bool]:
    parent = root.parent
    rec = {
        "trace": [
            {"frame": [[1, 1, 1, 1]], "turn": 0, "reasoning": {"level": 0, "turn_in_level": 0}},
            {"frame": [[1, 1, 0, 0]], "turn": 1, "reasoning": {"level": 0, "turn_in_level": 1}},
        ]
    }
    _bake_wm_predictions(rec, root.name, ws_dir=parent)
    r0 = rec["trace"][0]["reasoning"]
    return {
        "verify_baked": (
            r0.get("verify", {}).get("simulation_accuracy") == 1
            and r0.get("verify", {}).get("strict_simulation_accuracy") == 0
            and r0.get("verify", {}).get("prediction_coverage_status") == "complete"
            and r0.get("simulation_accuracy") == 1
        ),
        "outcome_baked": (
            r0.get("outcome", {}).get("terminal_render_variant_used") == 1
        ),
        "wm_pred_baked": bool(r0.get("wm_pred", {}).get("plan")),
    }


def main() -> int:
    ok = True
    with TemporaryDirectory() as td:
        root = Path(td) / "tinygame"
        root.mkdir()
        _make_workspace(root)
        for group, checks in (
            ("predict_for_workspace", _check_predict_for_workspace(root)),
            ("bake_predictions", _check_bake_predictions(root)),
        ):
            print(f"=== {group} ===")
            for name, passed in checks.items():
                print(f"  {name}: {passed}")
                ok = ok and passed
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def test_wm_predict_serialization(monkeypatch) -> None:
    monkeypatch.setenv("TYCHO_SANDBOX_RUNTIME", "host")
    with TemporaryDirectory() as td:
        root = Path(td) / "tinygame"
        root.mkdir()
        _make_workspace(root)
        groups = {
            "predict_for_workspace": _check_predict_for_workspace(root),
            "bake_predictions": _check_bake_predictions(root),
        }
    failed = [f"{group}.{name}" for group, checks in groups.items()
              for name, passed in checks.items() if not passed]
    assert not failed


def test_static_viewer_skips_wm_predict_for_no_world_model_runs(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps({"mode": "no_world_model"}))

    assert _compute_wm_predictions(run_dir, tmp_root=str(tmp_path)) == {}


def test_viewer_surfaces_unknown_coverage_status() -> None:
    src = Path(_compute_wm_predictions.__code__.co_filename).read_text()
    assert "prediction_coverage_status" in src
    assert "terminal_render_coverage_status" in src
    assert "vacuous means render() claimed no cells" in src


if __name__ == "__main__":
    raise SystemExit(main())
