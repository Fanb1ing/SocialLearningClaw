"""Export and replay Tycho action traces for ARC-AGI-3 scorecards.

The expensive part of a Tycho run is the agent deciding actions. This module separates that from
the official ARC scorecard creation step:

1. export a canonical action trace from a completed Tycho results directory;
2. replay that trace through the ARC engine/API, optionally in competition mode.

Replay is intentionally fail-closed. If an exported trace includes frame hashes, replay validates
state, levels-completed, and rendered-grid hashes after every action. A mismatch means the engine or
game version is not reproducing the trajectory that produced the trace, so the replay stops before
submitting more actions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from arcengine import GameAction, GameState

from tycho.harness.inference_budget import cap_trace_by_inference_cost
from tycho.serving.pricing import PRICE_SCHEDULE

TRACE_SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_DIR = REPO_ROOT / "environment_files"

# Loaded lazily so `python -m tycho.harness.submission_replay --help` does not import arc_agi's
# rendering stack. Tests can monkeypatch Arcade directly before replay_trace() calls this loader.
Arcade = None
OperationMode = None


def _load_arc_types():
    global Arcade, OperationMode
    if Arcade is None or OperationMode is None:
        from arc_agi import Arcade as _Arcade, OperationMode as _OperationMode
        if Arcade is None:
            Arcade = _Arcade
        if OperationMode is None:
            OperationMode = _OperationMode
    return Arcade, OperationMode


def grid_sha256(grid: Any) -> str | None:
    """Stable hash for a 64x64 integer grid. Returns None if no grid is available."""

    if grid is None:
        return None
    rows = [[int(v) for v in row] for row in grid]
    payload = json.dumps(rows, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _source_run_id(directory: Path, manifest: dict) -> str:
    existing = manifest.get("run_spec_fingerprint") or manifest.get("run_id")
    if existing:
        return str(existing)
    digest = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    )
    for path in sorted(directory.glob("game_*.json")):
        digest.update(path.name.encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _grid_to_list(frame) -> list | None:
    if frame is None or not len(frame):
        return None
    grid = frame[-1]
    return [list(map(int, row)) for row in grid]


def _grid_to_first(frame) -> list | None:
    if frame is None or not len(frame):
        return None
    grid = frame[0]
    return [list(map(int, row)) for row in grid]


def _observed_grid(frame) -> list | None:
    """Grid convention matching tycho.harness.harness.TraceStep.frame."""

    if frame is None:
        return None
    if getattr(frame, "state", None) == GameState.GAME_OVER:
        return _grid_to_first(frame.frame) or _grid_to_list(frame.frame)
    return _grid_to_list(frame.frame)


def _frame_grid_hashes(frame) -> list[str]:
    """Hashes for every grid in an ARC frame sequence."""

    raw = getattr(frame, "frame", None)
    if raw is None:
        return []
    hashes: list[str] = []
    for grid in raw:
        h = grid_sha256(grid)
        if h is not None:
            hashes.append(h)
    return hashes


def _frame_state_name(frame) -> str:
    state = getattr(frame, "state", "")
    return str(state)


def _scorecard_url(card_id: str | None) -> str | None:
    if not card_id:
        return None
    return f"https://arcprize.org/scorecards/{card_id}"


@dataclass
class ReplayAction:
    turn: int
    action: str
    x: int | None = None
    y: int | None = None
    expected_state: str | None = None
    expected_levels_completed: int | None = None
    expected_frame_sha256: str | None = None

    @classmethod
    def from_trace_step(cls, step: dict) -> "ReplayAction":
        return cls(
            turn=int(step.get("turn") or 0),
            action=str(step["action"]),
            x=step.get("x"),
            y=step.get("y"),
            expected_state=step.get("state"),
            expected_levels_completed=step.get("levels_completed"),
            expected_frame_sha256=grid_sha256(step.get("frame")) if step.get("frame") is not None else None,
        )

    def to_json(self) -> dict:
        return {
            "turn": self.turn,
            "action": self.action,
            "x": self.x,
            "y": self.y,
            "expected_state": self.expected_state,
            "expected_levels_completed": self.expected_levels_completed,
            "expected_frame_sha256": self.expected_frame_sha256,
        }


@dataclass
class GameTrace:
    game_id: str
    short_id: str
    n_levels: int
    baselines: list[int]
    levels_completed: int
    final_state: str
    env_score: float
    total_actions: int
    stop_reason: str | None
    error: str | None
    actions: list[ReplayAction] = field(default_factory=list)
    inference_cost_usd: float | None = None
    original_inference_cost_usd: float | None = None
    inference_budget: dict | None = None

    @classmethod
    def from_record(cls, rec: dict) -> "GameTrace":
        game_id = str(rec["game_id"])
        return cls(
            game_id=game_id,
            short_id=game_id.split("-", 1)[0],
            n_levels=int(rec.get("n_levels") or 0),
            baselines=[int(x) for x in (rec.get("baselines") or [])],
            levels_completed=int(rec.get("levels_completed") or 0),
            final_state=str(rec.get("final_state") or ""),
            env_score=float(rec.get("env_score") or 0.0),
            total_actions=int(rec.get("total_actions") or 0),
            stop_reason=rec.get("stop_reason"),
            error=rec.get("error"),
            actions=[ReplayAction.from_trace_step(s) for s in (rec.get("trace") or [])],
        )

    def to_json(self) -> dict:
        return {
            "game_id": self.game_id,
            "short_id": self.short_id,
            "n_levels": self.n_levels,
            "baselines": self.baselines,
            "levels_completed": self.levels_completed,
            "final_state": self.final_state,
            "env_score": self.env_score,
            "total_actions": self.total_actions,
            "stop_reason": self.stop_reason,
            "error": self.error,
            "actions": [a.to_json() for a in self.actions],
            "inference_cost_usd": self.inference_cost_usd,
            "original_inference_cost_usd": self.original_inference_cost_usd,
            "inference_budget": self.inference_budget,
        }

    @classmethod
    def from_json(cls, data: dict) -> "GameTrace":
        return cls(
            game_id=str(data["game_id"]),
            short_id=str(data.get("short_id") or str(data["game_id"]).split("-", 1)[0]),
            n_levels=int(data.get("n_levels") or 0),
            baselines=[int(x) for x in (data.get("baselines") or [])],
            levels_completed=int(data.get("levels_completed") or 0),
            final_state=str(data.get("final_state") or ""),
            env_score=float(data.get("env_score") or 0.0),
            total_actions=int(data.get("total_actions") or 0),
            stop_reason=data.get("stop_reason"),
            error=data.get("error"),
            actions=[ReplayAction(**a) for a in (data.get("actions") or [])],
            inference_cost_usd=data.get("inference_cost_usd"),
            original_inference_cost_usd=data.get("original_inference_cost_usd"),
            inference_budget=data.get("inference_budget"),
        )


def _game_trace_with_budget(
    rec: dict,
    *,
    max_cost_per_game_usd: float,
    max_cost_per_level_usd: float,
    default_model: str,
) -> GameTrace:
    result = cap_trace_by_inference_cost(
        rec.get("trace") or [],
        max_cost_per_game_usd=max_cost_per_game_usd,
        max_cost_per_level_usd=max_cost_per_level_usd,
        default_model=default_model,
    )
    capped = dict(rec)
    capped["trace"] = result.trace
    level_results: list[dict] = []
    completed_before = 0
    actions_this_level = 0
    baselines = [int(value) for value in (rec.get("baselines") or [])]
    for step in result.trace:
        actions_this_level += 1
        completed_after = int(step.get("levels_completed") or 0)
        if completed_after > completed_before:
            level_index = completed_before + 1
            level_results.append({
                "level_index": level_index,
                "completed": True,
                "actions_taken": actions_this_level,
                "baseline_actions": baselines[level_index - 1],
            })
            completed_before = completed_after
            actions_this_level = 0
    if result.trace:
        last = result.trace[-1]
        capped["levels_completed"] = int(last.get("levels_completed") or 0)
        capped["final_state"] = str(last.get("state") or "")
    else:
        capped["levels_completed"] = 0
        capped["final_state"] = ""
    capped["level_results"] = level_results
    capped["total_actions"] = sum(level["actions_taken"] for level in level_results)
    capped["env_score"] = _rhae(level_results, int(rec.get("n_levels") or 0))
    if result.cap_triggered:
        capped["stop_reason"] = result.stop_reason
    game = GameTrace.from_record(capped)
    game.inference_cost_usd = result.cost_usd
    game.original_inference_cost_usd = result.original_cost_usd
    game.inference_budget = {
        "price_schedule": PRICE_SCHEDULE,
        "policy": "post_action_soft_cap",
        "max_cost_per_game_usd": max_cost_per_game_usd,
        "max_cost_per_level_usd": max_cost_per_level_usd,
        "cap_triggered": result.cap_triggered,
        "stop_reason": result.stop_reason,
        "source_actions": len(rec.get("trace") or []),
        "retained_actions": len(result.trace),
    }
    return game


def _rhae(level_results: list[dict], n_levels: int) -> float:
    denominator = sum(range(1, n_levels + 1))
    if not denominator:
        return 0.0
    weighted = 0.0
    completed_weight = 0
    for level in level_results:
        index = int(level["level_index"])
        actions = int(level["actions_taken"])
        baseline = int(level["baseline_actions"])
        weighted += index * min((baseline / actions) ** 2 * 100.0, 115.0)
        completed_weight += index
    return min(weighted / denominator, 100.0 * completed_weight / denominator)


def _materialized_budget_record(
    rec: dict,
    *,
    max_cost_per_game_usd: float,
    max_cost_per_level_usd: float,
    default_model: str,
) -> tuple[dict, dict]:
    """Return a self-consistent results record truncated at the inference-budget boundary.

    Action counts follow official scorecard semantics: every committed in-play control, including
    RESET, is charged. The source record remains untouched. Wall time is deliberately unset because
    a trace contains model latency but not enough information to reconstruct capped end-to-end time.
    """

    result = cap_trace_by_inference_cost(
        rec.get("trace") or [],
        max_cost_per_game_usd=max_cost_per_game_usd,
        max_cost_per_level_usd=max_cost_per_level_usd,
        default_model=default_model,
    )
    trace = result.trace
    baselines = [int(value) for value in (rec.get("baselines") or [])]
    level_results: list[dict] = []
    completed_before = 0
    level_start = 0
    actions_before_first_level = None
    for index, step in enumerate(trace):
        completed_after = int(step.get("levels_completed") or 0)
        if completed_after <= completed_before:
            continue
        for level_index in range(completed_before + 1, completed_after + 1):
            actions_taken = index + 1 - level_start
            level_results.append({
                "level_index": level_index,
                "completed": True,
                "actions_taken": actions_taken,
                "baseline_actions": baselines[level_index - 1],
            })
            if actions_before_first_level is None:
                actions_before_first_level = index + 1
            level_start = index + 1
        completed_before = completed_after

    levels_completed = int(trace[-1].get("levels_completed") or 0) if trace else 0
    completed_actions = level_start
    unfinished_actions = len(trace) - completed_actions
    n_levels = int(rec.get("n_levels") or 0)
    budget = {
        "price_schedule": PRICE_SCHEDULE,
        "policy": "post_action_soft_cap",
        "max_cost_per_game_usd": max_cost_per_game_usd,
        "max_cost_per_level_usd": max_cost_per_level_usd,
        "cap_triggered": result.cap_triggered,
        "stop_reason": result.stop_reason,
        "source_actions": len(rec.get("trace") or []),
        "retained_actions": len(trace),
        "inference_cost_usd": result.cost_usd,
    }
    seen_frame_keys: set[Any] = set()
    revisits = 0
    for step in trace:
        frame_key = step.get("frame_key")
        if bool(step.get("frame_changed")) and frame_key in seen_frame_keys:
            revisits += 1
        seen_frame_keys.add(frame_key)

    capped = dict(rec)
    capped.update({
        "levels_completed": levels_completed,
        "total_actions": completed_actions,
        "resets": sum(step.get("action") == "RESET" for step in trace),
        "final_state": str(trace[-1].get("state") or "") if trace else "",
        "env_score": _rhae(level_results, n_levels),
        "level_results": level_results,
        "wall_clock_s": None,
        "source_wall_clock_s": rec.get("wall_clock_s"),
        "truncated_levels": [],
        "completed_level_actions": completed_actions,
        "unfinished_level_index": (
            levels_completed + 1 if levels_completed < n_levels and unfinished_actions else None
        ),
        "unfinished_level_actions": unfinished_actions,
        "total_actions_including_unfinished": len(trace),
        "noop_actions": sum(
            step.get("action") != "RESET" and not bool(step.get("frame_changed"))
            for step in trace
        ),
        "distinct_frames": len(seen_frame_keys),
        "revisits": revisits,
        "actions_before_first_level": actions_before_first_level,
        "error": None,
        "stop_reason": result.stop_reason if result.cap_triggered else rec.get("stop_reason"),
        "builder_invocations": sum(
            len(((step.get("reasoning") or {}).get("builder_runs") or [])) for step in trace
        ) if rec.get("builder_invocations") is not None else None,
        "trace": trace,
        "partial": False,
        "inference_budget": budget,
        "action_accounting": "all_in_play_controls_including_reset",
    })
    return capped, budget


def _trace_usage(trace: list[dict]) -> dict[str, int]:
    totals = {
        "tokens_in": 0,
        "tokens_out": 0,
        "cache_read": 0,
        "cache_write": 0,
        "latency_ms": 0,
        "model_calls": 0,
    }
    for step in trace:
        for call in ((step.get("reasoning") or {}).get("llm_calls") or []):
            totals["model_calls"] += 1
            for key in ("tokens_in", "tokens_out", "cache_read", "cache_write", "latency_ms"):
                totals[key] += int(call.get(key) or 0)
    return totals


def materialize_budgeted_run(
    run_dir: str | Path,
    out_dir: str | Path,
    *,
    max_inference_cost_per_game_usd: float = 0.0,
    max_inference_cost_per_level_usd: float = 0.0,
    price_model: str = "",
) -> dict:
    """Create a normal-looking results directory whose traces stop at the budget boundary.

    Workspaces are intentionally not copied. Legacy slim records omit exact binary history; newer
    schema-2 records retain exact causal manifests and blobs, but this trace materializer does not
    yet prune harness-owned observations to the budget boundary. Copying a terminal source
    workspace would therefore leak post-cap evidence. Trace-only analyses remain supported by the
    materialized game files.
    """

    source = Path(run_dir).resolve()
    target = Path(out_dir).resolve()
    if source == target:
        raise ValueError("budgeted run must be materialized into a separate directory")
    if target.exists() and any(target.iterdir()):
        raise ValueError(f"output directory is not empty: {target}")
    target.mkdir(parents=True, exist_ok=True)
    source_manifest = json.loads((source / "manifest.json").read_text())
    default_model = price_model or str(source_manifest.get("model") or "")

    per_game: dict[str, dict] = {}
    budget_games: dict[str, dict] = {}
    usage = {key: 0 for key in _trace_usage([])}
    source_per_game = source_manifest.get("per_game") or {}
    paths = sorted(source.glob("game_*.json"))
    for path in paths:
        rec = json.loads(path.read_text())
        capped, budget = _materialized_budget_record(
            rec,
            max_cost_per_game_usd=max_inference_cost_per_game_usd,
            max_cost_per_level_usd=max_inference_cost_per_level_usd,
            default_model=default_model,
        )
        short = str(rec["game_id"]).split("-", 1)[0]
        output_path = target / path.name
        output_path.write_text(json.dumps(capped, separators=(",", ":")))
        retained_trace = capped.get("trace") or []
        for key, value in _trace_usage(retained_trace).items():
            usage[key] += value
        budget_games[short] = budget
        item = {
            "rhae": capped["env_score"],
            "levels": capped["levels_completed"],
            "mode": "BUDGETED" if budget["cap_triggered"] else "COMPLETE",
            "wall_clock_s": None,
            "stop_reason": capped.get("stop_reason"),
            "total_actions": capped["total_actions"],
            "completed_level_actions": capped["completed_level_actions"],
            "unfinished_level_index": capped.get("unfinished_level_index"),
            "unfinished_level_actions": capped["unfinished_level_actions"],
            "total_actions_including_unfinished": len(retained_trace),
        }
        source_activity = (source_per_game.get(short) or {}).get("wm_activity")
        if source_activity and not budget["cap_triggered"]:
            item["wm_activity"] = source_activity
        per_game[short] = item

    scores = [float(item["rhae"]) for item in per_game.values()]
    manifest = dict(source_manifest)
    manifest.update({
        "source_run_id": _source_run_id(source, source_manifest),
        "materialized_at": datetime.now(timezone.utc).isoformat(),
        "inference_budget": {
            "price_schedule": PRICE_SCHEDULE,
            "policy": "post_action_soft_cap",
            "max_cost_per_game_usd": max_inference_cost_per_game_usd,
            "max_cost_per_level_usd": max_inference_cost_per_level_usd,
            "price_model_override": price_model or None,
            "games": budget_games,
        },
        "action_accounting": "all_in_play_controls_including_reset",
        "workspace": {
            "status": "not_materialized",
            "reason": (
                "legacy traces may lack exact causal file history and source terminal workspaces "
                "may contain post-budget observation evidence"
            ),
        },
        "n_requested": len(paths),
        "n_finished": len(paths),
        "n_pending": 0,
        "pending_games": [],
        "wall_clock_s": None,
        "compute_s": None,
        "parallelism_x": None,
        "throughput": usage,
        "wm_plan_coverage": None,
        "mean_rhae": round(sum(scores) / len(scores), 3) if scores else None,
        "mean_rhae_clean": round(sum(scores) / len(scores), 3) if scores else None,
        "n_reportable": len(scores),
        "n_partial": 0,
        "games": list(per_game),
        "requested_games": list(per_game),
        "per_game": per_game,
    })
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _level_regressions(game: GameTrace) -> list[str]:
    out: list[str] = []
    prev: int | None = None
    prev_turn: int | None = None
    for action in game.actions:
        cur = action.expected_levels_completed
        if cur is None:
            continue
        if prev is not None and int(cur) < int(prev):
            out.append(
                f"{game.short_id}: levels_completed decreases at turn {action.turn} "
                f"({prev_turn}->{action.turn}: {prev}->{cur})"
            )
        prev = int(cur)
        prev_turn = int(action.turn)
    return out


def _validate_no_level_regressions(games: list[GameTrace]) -> None:
    errors = [msg for game in games for msg in _level_regressions(game)]
    if errors:
        raise ValueError(
            "submission trace contains level regressions, usually from full-game RESET semantics; "
            "rerun those games with ONLY_RESET_LEVELS=true:\n" + "\n".join(errors)
        )


def export_trace(
    run_dir: str | Path | list[str | Path],
    out_path: str | Path,
    *,
    require_clean: bool = True,
    max_inference_cost_per_game_usd: float = 0.0,
    max_inference_cost_per_level_usd: float = 0.0,
    price_model: str = "",
) -> dict:
    """Export canonical replay JSON from one or more Tycho results directories."""

    run_dirs = [Path(p) for p in (run_dir if isinstance(run_dir, list) else [run_dir])]
    out_path = Path(out_path)
    manifests = [
        json.loads((d / "manifest.json").read_text()) if (d / "manifest.json").exists() else {}
        for d in run_dirs
    ]
    games: list[GameTrace] = []
    errors: list[str] = []
    seen: dict[str, Path] = {}
    for d in run_dirs:
        for rec_path in sorted(d.glob("game_*.json")):
            rec = json.loads(rec_path.read_text())
            default_model = price_model or str((manifests[run_dirs.index(d)] or {}).get("model") or "")
            if max_inference_cost_per_game_usd > 0 or max_inference_cost_per_level_usd > 0:
                trace = _game_trace_with_budget(
                    rec,
                    max_cost_per_game_usd=max_inference_cost_per_game_usd,
                    max_cost_per_level_usd=max_inference_cost_per_level_usd,
                    default_model=default_model,
                )
            else:
                trace = GameTrace.from_record(rec)
            if trace.short_id in seen:
                errors.append(f"{trace.short_id}: duplicate record in {seen[trace.short_id]} and {rec_path}")
                continue
            seen[trace.short_id] = rec_path
            if require_clean and trace.error:
                errors.append(f"{trace.short_id}: record has error {trace.error}")
            if require_clean and not trace.actions:
                errors.append(f"{trace.short_id}: record has no trace actions")
            if require_clean:
                errors.extend(_level_regressions(trace))
            games.append(trace)
    if require_clean and errors:
        raise ValueError("cannot export submission trace:\n" + "\n".join(errors))
    doc = {
        "schema": "tycho.arc_agi_3.submission_trace",
        "schema_version": TRACE_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_run_ids": [
            _source_run_id(directory, manifest)
            for directory, manifest in zip(run_dirs, manifests)
        ],
        "source_manifests": [
            {
                k: manifest.get(k)
                for k in (
                    "approach", "model", "seed", "effort", "mode", "backend", "git_version",
                    "mean_rhae_clean", "mean_rhae", "n_reportable", "n_finished", "requested_games",
                )
                if k in manifest
            }
            for manifest in manifests
        ],
        "seed": _common_seed(manifests),
        "inference_budget": {
            "price_schedule": PRICE_SCHEDULE,
            "policy": "post_action_soft_cap",
            "max_cost_per_game_usd": max_inference_cost_per_game_usd,
            "max_cost_per_level_usd": max_inference_cost_per_level_usd,
            "price_model_override": price_model or None,
        } if max_inference_cost_per_game_usd > 0 or max_inference_cost_per_level_usd > 0 else None,
        "games": [g.to_json() for g in games],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2, sort_keys=True))
    return doc


def _common_seed(manifests: list[dict]) -> int:
    seeds = {int(m.get("seed") or 0) for m in manifests if m}
    if len(seeds) > 1:
        raise ValueError(f"cannot combine traces with different seeds: {sorted(seeds)}")
    return next(iter(seeds), 0)


def load_trace(path: str | Path) -> tuple[dict, list[GameTrace]]:
    doc = json.loads(Path(path).read_text())
    if doc.get("schema") != "tycho.arc_agi_3.submission_trace":
        raise ValueError(f"not a Tycho submission trace: {path}")
    if int(doc.get("schema_version") or 0) != TRACE_SCHEMA_VERSION:
        raise ValueError(f"unsupported submission trace schema_version={doc.get('schema_version')}")
    return doc, [GameTrace.from_json(g) for g in doc.get("games", [])]


def _arcade_mode(name: str, competition: bool) -> OperationMode:
    _, operation_mode = _load_arc_types()
    if competition:
        return operation_mode.COMPETITION
    return {
        "offline": operation_mode.OFFLINE,
        "normal": operation_mode.NORMAL,
        "online": operation_mode.ONLINE,
        "competition": operation_mode.COMPETITION,
    }[name]


def _initial_frame(env):
    # Local and remote wrappers reset during construction. Prefer the already-open frame so
    # competition replay does not spend an accidental extra RESET.
    frame = getattr(env, "observation_space", None)
    if frame is not None:
        return frame
    return env.reset()


def _game_action(action: ReplayAction) -> tuple[GameAction, dict | None]:
    ga = GameAction[action.action]
    data = None
    if ga == GameAction.ACTION6:
        if action.x is None or action.y is None:
            raise ValueError(f"ACTION6 at turn {action.turn} is missing x/y")
        data = {"x": int(action.x), "y": int(action.y)}
    return ga, data


def _validate_step(game: GameTrace, action: ReplayAction, frame, *, validate_frames: bool) -> None:
    state = _frame_state_name(frame)
    if action.expected_state is not None and state != action.expected_state:
        raise RuntimeError(
            f"{game.short_id} turn {action.turn}: state mismatch {state} != {action.expected_state}"
        )
    got_levels = int(getattr(frame, "levels_completed", -1))
    if action.expected_levels_completed is not None and got_levels != action.expected_levels_completed:
        raise RuntimeError(
            f"{game.short_id} turn {action.turn}: levels_completed mismatch "
            f"{got_levels} != {action.expected_levels_completed}"
        )
    if validate_frames and action.expected_frame_sha256:
        got_hash = grid_sha256(_observed_grid(frame))
        # GAME_OVER responses may contain a multi-frame terminal/animation sequence. Older trace
        # records stored the last grid while newer harness code uses the first terminal-evidence
        # grid for the side channel. Both are valid if the recorded grid appears in the deterministic
        # response sequence; state and levels_completed are still checked above.
        if (
            got_hash != action.expected_frame_sha256
            and not (state == "GameState.GAME_OVER" and action.expected_frame_sha256 in _frame_grid_hashes(frame))
        ):
            raise RuntimeError(
                f"{game.short_id} turn {action.turn}: frame hash mismatch "
                f"{got_hash} != {action.expected_frame_sha256}"
            )


def replay_trace(
    trace_path: str | Path,
    out_dir: str | Path,
    *,
    api_key: str = "",
    mode: str = "offline",
    competition: bool = False,
    close_scorecard: bool = True,
    validate_frames: bool = True,
    source_url: str | None = None,
    tags: list[str] | None = None,
) -> dict:
    """Replay an exported trace through ARC and write a replay manifest."""

    trace_doc, games = load_trace(trace_path)
    _validate_no_level_regressions(games)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "replay_manifest.json"
    replay_manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    operation_mode = _arcade_mode(mode, competition)
    if operation_mode.value in ("online", "competition") and not api_key:
        raise ValueError("ARC API key is required for online/competition trace replay")
    arcade_cls, _ = _load_arc_types()
    arc = arcade_cls(
        arc_api_key=api_key,
        operation_mode=operation_mode,
        environments_dir=str(ENV_DIR),
        recordings_dir=str(out_dir / "recordings"),
    )

    scorecard_id = replay_manifest.get("scorecard_id")
    if not scorecard_id:
        trace_sha256 = hashlib.sha256(Path(trace_path).read_bytes()).hexdigest()
        opaque = {
            "tycho_submission_trace_sha256": trace_sha256,
            "source_manifests": trace_doc.get("source_manifests")
            or ([trace_doc["source_manifest"]] if trace_doc.get("source_manifest") else []),
        }
        scorecard_id = arc.create_scorecard(
            source_url=source_url,
            tags=tags or ["tycho", "trace-replay"],
            opaque=opaque,
        )
        replay_manifest = {
            "schema": "tycho.arc_agi_3.trace_replay_manifest",
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "trace_sha256": trace_sha256,
            "operation_mode": operation_mode.value,
            "competition": bool(competition),
            "scorecard_id": scorecard_id,
            "scorecard_url": _scorecard_url(scorecard_id) if competition else None,
            "completed_games": [],
            "failed": None,
            "closed": False,
        }
        manifest_path.write_text(json.dumps(replay_manifest, indent=2, sort_keys=True))

    completed = set(replay_manifest.get("completed_games") or [])
    t0 = time.time()
    for i, game in enumerate(games, 1):
        if game.short_id in completed or game.game_id in completed:
            print(f"[{i}/{len(games)}] {game.short_id}: already replayed; skipping", flush=True)
            continue
        print(f"[{i}/{len(games)}] {game.short_id}: replaying {len(game.actions)} actions", flush=True)
        try:
            env = arc.make(game.game_id, seed=int(trace_doc.get("seed") or 0), scorecard_id=scorecard_id)
            if env is None:
                raise RuntimeError(f"arc.make returned None for {game.game_id}")
            frame = _initial_frame(env)
            if frame is None:
                raise RuntimeError(f"initial frame unavailable for {game.game_id}")
            for action in game.actions:
                ga, data = _game_action(action)
                frame = env.step(
                    ga,
                    data=data,
                    reasoning={"source": "tycho_trace_replay", "trace_turn": action.turn},
                )
                if frame is None:
                    raise RuntimeError(f"env.step returned None after {action.action} turn {action.turn}")
                _validate_step(game, action, frame, validate_frames=validate_frames)
        except Exception as e:  # noqa: BLE001
            replay_manifest["failed"] = {
                "game": game.short_id,
                "game_id": game.game_id,
                "error": type(e).__name__,
                "elapsed_s": round(time.time() - t0, 2),
            }
            manifest_path.write_text(json.dumps(replay_manifest, indent=2, sort_keys=True))
            raise
        completed.add(game.short_id)
        replay_manifest["completed_games"] = sorted(completed)
        replay_manifest["last_update"] = datetime.now(timezone.utc).isoformat()
        replay_manifest["elapsed_s"] = round(time.time() - t0, 2)
        manifest_path.write_text(json.dumps(replay_manifest, indent=2, sort_keys=True))

    if close_scorecard and not replay_manifest.get("closed"):
        scorecard = arc.close_scorecard(scorecard_id)
        replay_manifest["closed"] = True
        replay_manifest["closed_at"] = datetime.now(timezone.utc).isoformat()
        replay_manifest["scorecard"] = (
            scorecard.model_dump(mode="json") if hasattr(scorecard, "model_dump") else str(scorecard)
        )
        replay_manifest["scorecard_score"] = getattr(scorecard, "score", None)
        replay_manifest["scorecard_url"] = _scorecard_url(scorecard_id) if competition else None
        manifest_path.write_text(json.dumps(replay_manifest, indent=2, sort_keys=True))

    return replay_manifest


def _main_export(args) -> int:
    doc = export_trace(
        args.run_dir,
        args.out,
        require_clean=not args.allow_errors,
        max_inference_cost_per_game_usd=args.max_inference_cost_per_game,
        max_inference_cost_per_level_usd=args.max_inference_cost_per_level,
        price_model=args.price_model,
    )
    print(json.dumps({
        "out": str(args.out),
        "games": len(doc["games"]),
        "source_run_ids": doc["source_run_ids"],
    }, indent=2))
    return 0


def _main_replay(args) -> int:
    tags = [t for t in (args.tag or []) if t]
    manifest = replay_trace(
        args.trace,
        args.out_dir,
        api_key=args.api_key or os.environ.get("ARC_API_KEY", ""),
        mode=args.mode,
        competition=args.competition,
        close_scorecard=not args.no_close,
        validate_frames=not args.no_frame_validation,
        source_url=args.source_url,
        tags=tags or None,
    )
    print(json.dumps({
        "out_dir": str(args.out_dir),
        "scorecard_id": manifest.get("scorecard_id"),
        "scorecard_url": manifest.get("scorecard_url"),
        "closed": manifest.get("closed"),
        "scorecard_score": manifest.get("scorecard_score"),
        "completed_games": len(manifest.get("completed_games") or []),
    }, indent=2))
    return 0


def _main_materialize(args) -> int:
    manifest = materialize_budgeted_run(
        args.run_dir,
        args.out_dir,
        max_inference_cost_per_game_usd=args.max_inference_cost_per_game,
        max_inference_cost_per_level_usd=args.max_inference_cost_per_level,
        price_model=args.price_model,
    )
    print(json.dumps({
        "out_dir": str(args.out_dir),
        "games": manifest["n_finished"],
        "mean_rhae": manifest["mean_rhae"],
        "capped_games": [
            game for game, budget in manifest["inference_budget"]["games"].items()
            if budget["cap_triggered"]
        ],
        "workspace_status": manifest["workspace"]["status"],
    }, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Export and replay Tycho ARC-AGI-3 submission traces.")
    sub = p.add_subparsers(dest="cmd", required=True)

    ex = sub.add_parser("export", help="export canonical replay trace from a results directory")
    ex.add_argument("--run-dir", required=True, type=Path, action="append",
                    help="results directory to export; repeat to combine disjoint game sets")
    ex.add_argument("--out", required=True, type=Path)
    ex.add_argument("--allow-errors", action="store_true", help="allow records with error/no trace")
    ex.add_argument("--max-inference-cost-per-game", type=float, default=0.0,
                    help="post-action public-list USD-equivalent cap per game (0 = off)")
    ex.add_argument("--max-inference-cost-per-level", type=float, default=0.0,
                    help="post-action public-list USD-equivalent cap per level (0 = off)")
    ex.add_argument("--price-model", default="",
                    help="model used when a recorded call lacks its own model id")
    ex.set_defaults(func=_main_export)

    mat = sub.add_parser(
        "materialize", help="create a capped results directory for trace-based analysis"
    )
    mat.add_argument("--run-dir", required=True, type=Path)
    mat.add_argument("--out-dir", required=True, type=Path)
    mat.add_argument("--max-inference-cost-per-game", type=float, default=0.0)
    mat.add_argument("--max-inference-cost-per-level", type=float, default=0.0)
    mat.add_argument("--price-model", default="")
    mat.set_defaults(func=_main_materialize)

    rp = sub.add_parser("replay", help="replay a canonical trace through ARC")
    rp.add_argument("--trace", required=True, type=Path)
    rp.add_argument("--out-dir", required=True, type=Path)
    rp.add_argument("--api-key", default="", help="ARC_API_KEY; falls back to environment")
    rp.add_argument("--mode", choices=["offline", "normal", "online", "competition"], default="offline")
    rp.add_argument("--competition", action="store_true", help="use OperationMode.COMPETITION")
    rp.add_argument("--source-url", default=None)
    rp.add_argument("--tag", action="append", default=[])
    rp.add_argument("--no-close", action="store_true", help="leave the scorecard open")
    rp.add_argument("--no-frame-validation", action="store_true", help="validate state/levels only")
    rp.set_defaults(func=_main_replay)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
