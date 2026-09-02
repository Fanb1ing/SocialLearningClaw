"""Parallel game runner for (slow) LLM archetypes.

Games are fully independent.  The coordinator supervises one subprocess per game, so a provider
failure or code exception cannot damage another game's actor state.  A small thread per game waits
for its worker and optional delayed retries; a semaphore limits concurrent model workloads.

Per-game results are saved incrementally as each game finishes (a crash/kill never
loses completed games). With --viz, full LLM I/O is captured per step (the client
recorder is thread-local, so concurrent games' logs don't interleave) — written to
ONE per-game JSON each for post-run analysis.

Usage (background it, no shell timeout):
  LLM_BACKEND=anthropic LLM_MODEL=<model> LLM_EFFORT=high \\
    .venv/bin/python -m tycho.harness.run_parallel --approach tycho \\
      --games <id1>,<id2>,<id3> --out-dir results/A2v3 --viz
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import inspect
import json
import os
import subprocess
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

warnings.filterwarnings("ignore")

from arc_agi import Arcade, OperationMode
from tycho.harness.harness import run_env, ENV_DIR
from tycho.harness.diagnostics import diagnose_env
from tycho.harness.run import _agent_factory, resolve_games
from tycho.harness.record_slim import slim_record
from tycho.harness.run_hooks import worker_environment

REPO = Path(__file__).resolve().parent.parent.parent  # harness/ -> tycho/ -> repo root


def _error_label(error: BaseException) -> str:
    """Stable error value suitable for result and status artifacts."""
    return type(error).__name__


def _stable_error_value(value) -> str | None:
    """Reduce a detailed exception string to its stable category."""
    if value is None:
        return None
    label = str(value).partition(":")[0].strip()
    return label or "Error"


def _diagnostic_path(out_dir: Path, *parts: str) -> Path | None:
    """Return an optional operator-local path for verbose diagnostics.

    Diagnostics are deliberately separate from the result directory because they may contain
    host paths or provider responses. The normal run artifacts retain only stable status labels.
    """
    configured = os.environ.get("TYCHO_DIAGNOSTICS_DIR", "").strip()
    if not configured:
        return None
    root = Path(configured).expanduser().resolve()
    result_root = out_dir.resolve()
    try:
        root.relative_to(result_root)
    except ValueError:
        pass
    else:
        raise ValueError("TYCHO_DIAGNOSTICS_DIR must be outside the result directory")
    path = root / out_dir.name
    for part in parts:
        path /= part
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _recorded_run_config(file_applied: dict[str, str] | None = None) -> tuple[dict, dict]:
    from tycho.config import recorded_config
    from tycho.serving.llm_client import LLMConfig, public_identity

    cfg = LLMConfig.from_env()
    identity = public_identity(cfg)
    snapshot = recorded_config(
        file_applied=file_applied,
        api_protocol=identity["api_protocol"],
        model=identity["model"],
    )
    return snapshot, identity


def _world_model_enabled() -> bool:
    try:
        from tycho.agent.modes import resolve_mode
        return resolve_mode(os.environ.get("TYCHO_MODE", "single").lower()).wm_variant != "none"
    except Exception:
        return True


def _effective_mode_name() -> str:
    """Canonical orchestration label, including direct TYCHO_AGENT_GRAPH builder wiring.

    Config-file graph runs set TYCHO_MODE=trigger before launch. This helper also covers the direct
    env-var path used in tests/manual experiments, where a builder graph implies trigger semantics
    even if TYCHO_MODE was omitted.
    """
    from tycho.agent.modes import resolve_agent_graph, resolve_mode
    spec = resolve_mode(os.environ.get("TYCHO_MODE", "single").lower())
    graph = os.environ.get("TYCHO_AGENT_GRAPH", "")
    if graph and spec.wm_variant != "none" and not spec.needs_builder:
        try:
            decls = resolve_agent_graph(json.loads(graph))
            if decls and any(d.type == "builder" for d in decls):
                return "trigger"
        except Exception:  # noqa: BLE001
            pass
    return spec.name


def _git_version() -> str:
    """Short SHA + a '-dirty' suffix if the working tree has uncommitted changes — stamped
    into the manifest so a multi-hour run is traceable to the exact harness state it used
    (a bare SHA is not enough when the tree is dirty, which it usually is mid-development)."""
    import subprocess
    try:
        sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"],
                              capture_output=True, text=True, timeout=5).stdout.strip()
        if dirty:
            diff = subprocess.run(["git", "-C", str(REPO), "diff", "--no-ext-diff", "--binary", "HEAD", "--"],
                                  capture_output=True, text=True, timeout=10).stdout
            import hashlib
            dh = hashlib.sha256((dirty + "\n" + diff).encode()).hexdigest()[:12]
            return f"{sha}-dirty-{dh}" if sha else f"unknown-dirty-{dh}"
        return sha or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def _run_spec_extra_sources(games: dict) -> dict[str, str]:
    """Score-affecting callable/data identities supplied by an optional extension."""
    sources = {"run_parallel._run_one": inspect.getsource(_run_one)}
    try:
        from tycho.harness import _run_extension
    except ImportError:
        return sources
    hook = getattr(_run_extension, "execution_extra_sources", None)
    if callable(hook):
        sources.update({str(key): str(value) for key, value in hook(games).items()})
    return sources


def _operation_mode_from_name(name: str) -> OperationMode:
    value = (name or "normal").lower().strip()
    if value == "offline":
        return OperationMode.OFFLINE
    if value == "normal":
        return OperationMode.NORMAL
    if value == "online":
        return OperationMode.ONLINE
    raise SystemExit(f"unsupported operation mode {name!r}; expected normal, offline, or online")


def _new_arcade(operation_mode: OperationMode) -> Arcade:
    """Construct Arcade consistently in the coordinator and isolated workers."""
    return Arcade(
        arc_api_key=os.environ.get("ARC_API_KEY", ""),
        operation_mode=operation_mode,
        environments_dir=str(ENV_DIR),
    )


def _select_resume_games(games: dict, selector: str) -> dict:
    selected = {item.strip() for item in selector.split(",") if item.strip()}
    if not selected:
        return dict(games)
    available = set(games) | {gid.split("-")[0] for gid in games}
    unknown = selected - available
    if unknown:
        raise ValueError(f"unknown game IDs: {', '.join(sorted(unknown))}")
    return {
        gid: baselines for gid, baselines in games.items()
        if gid in selected or gid.split("-")[0] in selected
    }


def _bake_wm_predictions(rec_d: dict, short: str, ws_dir=None) -> None:
    """Attach the world-model planned-path overlay to each step's reasoning for post-run analysis.
    Computed here while the
    workspace is on disk. Plan-prioritized (no ACTION6 per-action enumeration — 4096 targets).
    Best-effort: any failure/timeout leaves the record untouched. Keyed by (level,turn).

    ws_dir: the durable workspace (results/<run>/ws/<game>) — passed explicitly because runs now use
    a DURABLE workspace, not the ephemeral $TMPDIR/arcws_* that find_workspaces globs."""
    # TYCHO_SKIP_WM_BAKE=1 skips this analysis-only overlay and world-model activity metrics. The bake
    # runs after scoring and never affects RHAE or gameplay. Skipping can make recovery of a long run
    # substantially faster when planning through the final model is expensive.
    if os.environ.get("TYCHO_SKIP_WM_BAKE", "").lower() in ("1", "on", "true"):
        return
    if not _world_model_enabled():
        return
    try:
        from tycho.harness.model_replay import find_workspaces, predict_for_workspace
        wpath = None
        if ws_dir is not None and Path(ws_dir, short, "world_model.py").exists():
            wpath = str(Path(ws_dir, short))
        else:
            ws = find_workspaces([short])
            wpath = ws.get(short)
        if not wpath:
            return
        res = predict_for_workspace(wpath, timeout=60)
        if res.get("error"):
            print(f"  wm-bake[{short}]: prediction worker error: {res['error']}")
        pmap = res.get("predictions") or {}
        # run-level model-quality channels (describe the FINAL model) — overlaid on every step so the
        # col-1 world-model + outcome panels have data. verify = the simulation channel
        # (simulation_accuracy headline); outcome = the terminal-status channel.
        verify = res.get("verify") or {}
        outcome = res.get("outcome") or {}
        if not pmap and not verify and not outcome:
            print(f"  wm-bake[{short}]: no predictions/verify/outcome (model absent or unbuildable)")
            return
        # Frame-bearing trace steps are keyed by level and turn within level.
        for t in rec_d.get("trace", []):
            if t.get("frame") is None:
                continue
            r = t.get("reasoning")
            if not isinstance(r, dict):
                continue
            # per-step prediction overlay, keyed by (level, turn_in_level) — the predictor builds its
            # keys from frames() = level_<L>/turn_<NNN>.txt, where NNN is the PER-LEVEL turn, NOT the
            # global turn. Keying on the global turn matched only on level 0 and dropped every later
            # level's plans. Fall back to the global turn only if turn_in_level is absent (legacy).
            lvl = (r.get("level") if isinstance(r.get("level"), int) else t.get("levels_completed", 0))
            til = r.get("turn_in_level")
            key = f"{lvl}_{til}" if til is not None else f"{lvl}_{t.get('turn')}"
            bundle = pmap.get(key)
            if bundle:
                wm_pred = {}
                if "plan" in bundle:           # the planned path (the decided scope)
                    wm_pred["plan"] = bundle["plan"]
                if "per_action" in bundle:     # per-action predictions (shown when no usable plan)
                    wm_pred["per_action"] = bundle["per_action"]
                if "no_plan_reason" in bundle:  # WHY there's no plan (no-outcome / outcome-wrong / no-path)
                    wm_pred["no_plan_reason"] = bundle["no_plan_reason"]
                if wm_pred:
                    r["wm_pred"] = wm_pred
            # run-level metrics on every step (the FINAL model's quality + terminal-status correctness)
            if verify and not verify.get("error"):
                r["verify"] = verify
                if verify.get("simulation_accuracy") is not None:
                    r["simulation_accuracy"] = verify["simulation_accuracy"]
            if outcome and not outcome.get("error") and outcome.get("outcome_observable"):
                r["outcome"] = outcome
        # CURRENT-MODEL channel: also predict with the model AS IT EXISTED at each turn (the per-turn
        # world_model.py snapshot), so analysis can compare current and final models. Bounded:
        # batched by DISTINCT snapshot (the model changes only when edited), not per turn.
        _bake_current_model_predictions(rec_d, wpath)
    except Exception as e:  # noqa: BLE001 — overlay is cosmetic; never break the run, but log it
        print(f"  wm-bake[{short}]: skipped ({type(e).__name__}: {e})")


def _bake_current_model_predictions(rec_d: dict, wpath: str) -> None:
    """Attach predictions from the complete causal workspace as it existed at each turn.

    Schema-2 records restore every captured causal file from immutable blobs. Legacy records fall
    back to swapping only their embedded ``world_model.py`` source. Distinct versions are evaluated
    once per game; failures remain cosmetic and never affect play.
    """
    import tempfile
    from pathlib import Path as _Path

    from tycho.harness.model_replay import predict_for_workspace, predict_with_model_src
    from tycho.workspace.version_store import (
        SnapshotMaterializationError,
        is_causal_workspace_path,
        materialize_workspace_snapshot,
    )

    steps = [t for t in rec_d.get("trace", []) if t.get("frame") is not None and isinstance(t.get("reasoning"), dict)]

    def _resolve_source(i):
        seen = 0
        while seen < 5000:
            wm = ((steps[i].get("reasoning") or {}).get("workspace") or {}).get("contents", {}).get("world_model.py")
            if not isinstance(wm, str):
                return None
            if wm.startswith("\x00="):
                i = int(wm[2:]); seen += 1; continue
            return wm
        return None

    def _resolve_versions(i):
        seen = 0
        while seen < 5000:
            value = ((steps[i].get("reasoning") or {}).get("workspace") or {}).get("file_versions")
            if isinstance(value, str) and value.startswith("\x00="):
                i = int(value[2:])
                seen += 1
                continue
            return value if isinstance(value, dict) else None
        return None

    def _attach(t, res):
        pmap = (res or {}).get("predictions") or {}
        r = t["reasoning"]
        lvl = (r.get("level") if isinstance(r.get("level"), int) else t.get("levels_completed", 0))
        til = r.get("turn_in_level")
        key = f"{lvl}_{til}" if til is not None else f"{lvl}_{t.get('turn')}"
        bundle = pmap.get(key)
        cur = {}
        if bundle:
            for fld in ("plan", "per_action", "no_plan_reason"):
                if fld in bundle:
                    cur[fld] = bundle[fld]
        # also carry the current model's verify headline so the panel can show its quality
        if (res or {}).get("verify") and not res["verify"].get("error"):
            cur["sim_acc"] = res["verify"].get("simulation_accuracy")
        if cur:
            r.setdefault("wm_pred", {})["current"] = cur

    def _schema_is_exact(t) -> bool:
        try:
            workspace = ((t.get("reasoning") or {}).get("workspace") or {})
            return int(workspace.get("snapshot_schema") or 1) >= 2
        except (TypeError, ValueError):
            return False

    has_exact_versions = any(
        _schema_is_exact(t) and _resolve_versions(idx) is not None
        for idx, t in enumerate(steps)
    )
    cache: dict = {}
    if not has_exact_versions:
        for idx, t in enumerate(steps):
            src = _resolve_source(idx)
            if not src or len(src) < 50 or "<move something in s>" in src:
                continue
            if src not in cache:
                try:
                    cache[src] = predict_with_model_src(wpath, src, timeout=45)
                except Exception:  # noqa: BLE001
                    cache[src] = {"error": "current-model predict failed"}
            _attach(t, cache[src])
        return

    source_workspace = _Path(wpath)
    blob_dir = source_workspace.parent / ".workspace_blobs"
    with tempfile.TemporaryDirectory(prefix="tycho_wm_snapshot_") as tmp:
        snapshot_workspace = _Path(tmp) / source_workspace.name
        snapshot_workspace.mkdir()
        # Observation/event evidence is immutable and harness-owned, so share it read-only while
        # causal root files are replaced from each exact manifest.
        for child in source_workspace.rglob("*"):
            if not child.is_file():
                continue
            rel = child.relative_to(source_workspace).as_posix()
            if is_causal_workspace_path(rel):
                continue
            target = snapshot_workspace / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(child.resolve())
        for idx, t in enumerate(steps):
            src = _resolve_source(idx)
            versions = _resolve_versions(idx)
            if not src or not versions or len(src) < 50 or "<move something in s>" in src:
                continue
            key = tuple(
                (path, descriptor.get("sha256"), descriptor.get("status"))
                for path, descriptor in sorted(versions.items())
                if isinstance(descriptor, dict)
            )
            if key not in cache:
                try:
                    materialize_workspace_snapshot(
                        snapshot_workspace,
                        versions,
                        blob_dir=blob_dir,
                    )
                    cache[key] = predict_for_workspace(str(snapshot_workspace), timeout=45)
                except (OSError, SnapshotMaterializationError, ValueError) as exc:
                    cache[key] = {"error": f"current workspace restore failed: {exc}"}
                except Exception:  # noqa: BLE001
                    cache[key] = {"error": "current-model predict failed"}
            _attach(t, cache[key])


def _wm_activity(rec_d: dict) -> dict:
    """Per-game world-model + planning ACTIVITY summary (observability — answers 'how often was the
    builder fired, with what final simulation accuracy, was the terminal outcome verified, and was a plan found').
    Derived from the already-recorded trace: builder_runs + the baked verify (simulation accuracy) +
    outcome channel + wm_pred plan overlay. Cheap (no re-run).

    Adds two per-level coverage metrics over the levels the agent actually COMPLETED — they isolate
    'how much did world-modelling / planning help' from the (final) model's quality:
      wm_correct_levels  : completed levels where the FINAL model's accepted dynamics pass and
                           prediction_coverage is not degenerate. Denominator = completed_levels.
      wm_strict_exact_levels : completed levels where canonical render reproduces every cell exactly.
      plan_levels        : completed levels on which the planner produced a reachable path on ≥1 turn.
    These aggregate the FINAL model (the on-disk workspace at game end) for a consistent
    retrospective metric. Schema-2 records separately support exact per-turn causal workspace
    diagnostics. This is a lower bound on per-level correctness when the model was REWRITTEN late
    for a hard level (an early level's dynamics may no longer reproduce under the final model), and
    an over-count is impossible."""
    builds = 0
    plan_steps = 0
    plan_levels: set = set()
    for t in rec_d.get("trace", []):
        r = t.get("reasoning") or {}
        builds += len(r.get("builder_runs") or [])
        wp = r.get("wm_pred") or {}
        if wp.get("plan"):
            plan_steps += 1
            lvl = r.get("level")
            if isinstance(lvl, int):
                plan_levels.add(lvl)
    # final-model quality: the baked verify is overlaid on every step (describes the FINAL model),
    # so read it off the last step that carries it.
    sim = outcome_ok = None
    per_level: dict = {}
    for t in reversed(rec_d.get("trace", [])):
        v = (t.get("reasoning") or {}).get("verify") or {}
        if v and not v.get("error"):
            sim = v.get("simulation_accuracy")
            per_level = v.get("per_level") or {}    # keys are strings (JSON round-trip from the worker)
            o = (t.get("reasoning") or {}).get("outcome") or {}
            outcome_ok = o.get("ok") if o else None
            break
    # per-level coverage over COMPLETED levels (0..levels_completed-1). A level "has a correct WM" iff
    # the final model's per_level simulation_accuracy is 1.0 and coverage is above the trigger guard
    # threshold (None/<1 → not correct; broad UNKNOWN abstentions don't count as correct).
    completed = rec_d.get("levels_completed", 0) or 0
    min_cov = float(os.environ.get("TYCHO_WM_MIN_COVERAGE", "0.75"))
    wm_correct = 0
    wm_strict_exact = 0
    for L in range(completed):
        pl = per_level.get(str(L))
        cov = (pl or {}).get("prediction_coverage")
        cov_status = (pl or {}).get("prediction_coverage_status")
        if pl and pl.get("strict_simulation_accuracy") == 1.0:
            wm_strict_exact += 1
        if (pl and pl.get("simulation_accuracy") == 1.0
                and cov_status != "vacuous"
                and (cov is None or cov >= min_cov)):
            wm_correct += 1
    plan_cov = sum(1 for L in range(completed) if L in plan_levels)
    return {"builder_fires": builds,             # how many times the builder ran this game
            "final_simulation_accuracy": sim,    # final model's dynamics accuracy by simulation (or None)
            "outcome_verified": outcome_ok,      # did outcome() pass terminal-status verification
            "steps_with_plan": plan_steps,       # turns where the planner produced a path
            "completed_levels": completed,       # denominator for the two coverage metrics below
            "wm_correct_levels": wm_correct,     # completed levels with a final-model sim_acc==1.0
            "wm_strict_exact_levels": wm_strict_exact,
            "plan_levels": plan_cov}             # completed levels with ≥1 planned path


def _coverage_agg(results: dict) -> dict | None:
    """Aggregate the per-game wm_activity coverage counts into run-level shares: of all levels
    completed across games, the fraction with a correct final-model WM and the fraction the planner
    found a path for. None until ≥1 level has been completed. These describe the model AVAILABLE at
    game end (final model replayed over history), i.e. 'was a correct/plannable model available for
    this level', NOT a live causal attribution of the solve to planning."""
    acts = [v["wm_activity"] for v in results.values() if v.get("wm_activity")]
    tot = sum(a.get("completed_levels", 0) for a in acts)
    if not tot:
        return None
    wm_ok = sum(a.get("wm_correct_levels", 0) for a in acts)
    pl_ok = sum(a.get("plan_levels", 0) for a in acts)
    return {"completed_levels": tot,
            "wm_correct_levels": wm_ok, "wm_correct_frac": round(wm_ok / tot, 3),
            "plan_levels": pl_ok, "plan_frac": round(pl_ok / tot, 3)}


def _atomic_result(path: Path, value: dict, *, indent: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, indent=indent))
    tmp.replace(path)


_MANIFEST_RESULT_KEYS = (
    "rhae", "levels", "mode", "wall_clock_s", "stop_reason", "total_actions",
    "completed_level_actions", "unfinished_level_index", "unfinished_level_actions",
    "total_actions_including_unfinished", "inference_budget",
)


def _manifest_result(result: dict) -> dict:
    item = {key: result.get(key) for key in _MANIFEST_RESULT_KEYS}
    if result.get("wm_activity"):
        item["wm_activity"] = result["wm_activity"]
    return item


def _authoritative_manifest_results(
    out_dir: Path,
    requested_games: list[str],
    local_results: dict[str, dict],
) -> dict[str, dict]:
    """Merge coordinator-local results with each worker's latest durable result.

    Multiple coordinators may coexist when an operator selectively resumes one failed game while
    the original benchmark is still running. The per-game worker lease serializes that game, and
    its status record wins over any stale result retained by either coordinator.
    """
    from tycho.harness.run_status import GameStatusStore

    merged = {game: dict(value) for game, value in local_results.items()}
    for game in requested_games:
        result = GameStatusStore(out_dir / "status" / game).read().get("result")
        if isinstance(result, dict) and result.get("game") == game:
            merged[game] = _manifest_result(result)
    return merged


def _run_one(
    build,
    game_id,
    baselines,
    seed,
    viz,
    out_dir,
    *,
    operation_mode,
    resume: bool = False,
    status_store=None,
    max_actions_per_level: int | None = None,
    stop_after_levels: int | None = None,
):
    # Each worker process makes its own Arcade and fresh agent.
    # ISOLATED: one game's provider or code exception must not
    # kill the batch — an unattended overnight run should keep every game it can.
    short = game_id.split("-")[0]
    wrote_current_record = False
    final_path = out_dir / f"game_{short}.json"
    partial_path = out_dir / f".partial_{short}.json"
    try:
        import shutil
        from tycho.harness.resume import GameJournal
        arc = _new_arcade(operation_mode)
        # Durable per-game workspace + resume journal under the out-dir, so a mid-game kill (LLM
        # outage / cred lapse) can be replayed instead of restarted. ws/<game> holds the agent's
        # evolving files; resume/<game> holds the action log and bounded exact checkpoint.
        ws_root_path = out_dir / "ws"
        ws_root = str(ws_root_path)
        journal_dir = out_dir / "resume" / short
        if not resume:
            # A non-resume run in an existing out-dir must be a fresh trial. Clear only this game's
            # durable state; otherwise old journals/workspace files can leak actions, beliefs, or traces.
            shutil.rmtree(ws_root_path / short, ignore_errors=True)
            shutil.rmtree(journal_dir, ignore_errors=True)
            final_path.unlink(missing_ok=True)
            partial_path.unlink(missing_ok=True)
            (out_dir / f"game_{short}.ERROR.txt").unlink(missing_ok=True)
        journal = GameJournal(journal_dir)

        # LEVEL-GRANULAR OBSERVABILITY: on each level completion, write a partial sidecar that
        # monitors can read before the whole game finishes. Sidecar (.partial.json), not the real
        # game_<short>.json, so it can never corrupt
        # the final record, confuse --resume, or pollute analysis. Only meaningful with frames (viz).
        # NOTE: name it ".partial_<short>.json" (NOT "game_*.json") so it does NOT match the
        # game_*.json glob used by analysis and manifests; partials are surfaced explicitly instead,
        # avoiding a finished game showing twice (final + stale partial).
        def _on_level(prec, _pp=partial_path, _short=short):
            try:
                pd = slim_record(asdict(prec)) if viz else asdict(prec)
                pd["error"] = _stable_error_value(pd.get("error"))
                if viz and _world_model_enabled():
                    _bake_wm_predictions(pd, _short, ws_dir=ws_root)
                tmp = _pp.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(pd))
                tmp.replace(_pp)          # atomic: readers never see a half-written partial
            except Exception:  # noqa: BLE001 — live-view convenience must never break the run
                pass

        prior_source = None
        prior_wall_s = 0.0
        if resume:
            for cand in (final_path, partial_path):
                if not cand.exists():
                    continue
                try:
                    prev = json.loads(cand.read_text())
                except Exception:  # noqa: BLE001
                    continue
                if prev.get("trace"):
                    prior_source = str(cand)
                prior_wall_s = max(prior_wall_s, float(prev.get("wall_clock_s") or 0.0))
                if prior_source:
                    break

        rec = run_env(build(), arc, game_id, baselines, seed=seed, keep_frames=viz,
                      journal=journal, ws_root=ws_root,
                      status_store=status_store,
                      on_level_complete=_on_level if viz else None,
                      # on resume, harvest replayed turns' reasoning from the prior partial sidecar so
                      # the trace record retains replayed-turn reasoning after resume.
                      reasoning_source=prior_source if viz else None,
                      max_actions_per_level=max_actions_per_level,
                      stop_after_levels=stop_after_levels)
        if resume and prior_wall_s:
            rec.wall_clock_s = round(float(rec.wall_clock_s or 0.0) + prior_wall_s, 2)
        # Write the ALREADY-SLIM, deduped record (lossless): drops accumulating per-frame files
        # from each workspace snapshot + versions causal text/manifests across steps. This makes
        # results/<run>/game_*.json directly analysis-ready and removes O(turns) repeated content.
        rec_d = slim_record(asdict(rec)) if viz else asdict(rec)
        rec_d["error"] = _stable_error_value(rec_d.get("error"))
        if viz and _world_model_enabled():
            _bake_wm_predictions(rec_d, short, ws_dir=ws_root)   # overlay + col-1 audit (best-effort)
        _atomic_result(final_path, rec_d)  # save immediately
        wrote_current_record = True
        partial_path.unlink(missing_ok=True)  # final record supersedes the live partial sidecar
        d = diagnose_env(rec)
        # rec.error is set if the game ended on a mid-run exception (partial record,
        # but the trace up to the failure is preserved and saved above).
        mode = f"PARTIAL:{rec.error.split(':')[0]}" if rec.error else d.failure_mode
        return {"game": short, "rhae": rec.env_score, "levels": rec.levels_completed,
                "mode": mode, "wall_clock_s": rec.wall_clock_s,
                "stop_reason": rec.stop_reason,
                "total_actions": rec.total_actions,
                "completed_level_actions": rec.completed_level_actions,
                "unfinished_level_index": rec.unfinished_level_index,
                "unfinished_level_actions": rec.unfinished_level_actions,
                "total_actions_including_unfinished": rec.total_actions_including_unfinished,
                "inference_budget": rec.inference_budget,
                "wm_activity": _wm_activity(rec_d) if (viz and _world_model_enabled()) else None}
    except Exception as e:  # noqa: BLE001 — deliberately broad; isolate the failure
        import traceback
        diagnostic = _diagnostic_path(out_dir, f"game_{short}.ERROR.txt")
        if diagnostic is not None:
            diagnostic.write_text(traceback.format_exc())
        error_label = _error_label(e)
        try:
            from tycho.harness.resume import ResumeError
            prefix = "BLOCKED" if isinstance(e, ResumeError) else "ERROR"
        except Exception:  # noqa: BLE001
            prefix = "ERROR"
        # An exact-resume failure is about continuation integrity, not loss of the already committed
        # result. Preserve the prior partial/final JSON for audit and monitoring instead of replacing
        # it with a synthetic zero-score error record.
        prior = None
        if prefix == "BLOCKED" and resume and final_path.exists():
            try:
                prior = json.loads(final_path.read_text())
            except (OSError, json.JSONDecodeError):
                prior = None
        if not wrote_current_record and prior is None:
            err_d = {
                "game_id": game_id, "n_levels": len(baselines), "baselines": baselines,
                "levels_completed": 0, "total_actions": 0, "resets": 0,
                "stop_reason": "exception",
                "completed_level_actions": 0, "unfinished_level_index": None,
                "unfinished_level_actions": 0, "total_actions_including_unfinished": 0,
                "final_state": "ERROR", "env_score": 0.0, "level_results": [],
                "wall_clock_s": None, "truncated_levels": [], "error": error_label,
                "trace": [], "partial": False,
            }
            _atomic_result(final_path, err_d)
        return {"game": short,
                "rhae": float((prior or {}).get("env_score") or 0.0),
                "levels": int((prior or {}).get("levels_completed") or 0),
                "mode": f"{prefix}:{type(e).__name__}",
                "wall_clock_s": None, "stop_reason": "exception",
                "error": error_label,
                "total_actions": int((prior or {}).get("total_actions") or 0),
                "completed_level_actions": int((prior or {}).get("completed_level_actions") or 0),
                "unfinished_level_index": (prior or {}).get("unfinished_level_index"),
                "unfinished_level_actions": int((prior or {}).get("unfinished_level_actions") or 0),
                "total_actions_including_unfinished": int(
                    (prior or {}).get("total_actions_including_unfinished") or 0
                )}

def _worker_main(args, build, games, operation_mode, out_dir: Path) -> int:
    """Run exactly one game inside an isolated child process."""
    from tycho.config import resolved_config
    from tycho.harness.run_spec import RunSpecError, build_run_spec, ensure_run_spec
    from tycho.harness.run_status import GameStatusStore, WorkerLeaseError, utc_now

    result_path = Path(args.worker_result)
    provisional_short = args.worker_game.split("-")[0]
    try:
        saved_spec = json.loads((out_dir / "run_spec.json").read_text())
        all_games = saved_spec["policy"]["games"]
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise SystemExit(f"worker cannot read immutable run_spec.json: {exc}") from exc
    recorded, _identity = _recorded_run_config()
    candidate_spec = build_run_spec(
        repo=REPO,
        approach=args.approach,
        games=all_games,
        seed=args.seed,
        viz=args.viz,
        operation_mode=operation_mode.value,
        resolved_config=resolved_config(),
        recorded_config=recorded,
        git_version=_git_version(),
        extra_sources=_run_spec_extra_sources(all_games),
        experiment_limits=_experiment_limits(args),
    )
    try:
        ensure_run_spec(out_dir / "run_spec.json", candidate_spec, resume=True)
    except RunSpecError as exc:
        error_label = _error_label(exc)
        store = GameStatusStore(out_dir / "status" / provisional_short)
        result = {
            "game": provisional_short, "rhae": 0.0, "levels": 0,
            "mode": "BLOCKED:RunSpecError", "wall_clock_s": None,
            "stop_reason": "run_spec_mismatch", "error": error_label,
            "total_actions": 0, "completed_level_actions": 0,
            "unfinished_level_index": None, "unfinished_level_actions": 0,
            "total_actions_including_unfinished": 0,
        }
        store.event(
            "worker_blocked",
            error=error_label,
            status_fields={
                "state": "blocked", "pid": None, "error": error_label, "result": result,
            },
        )
        _atomic_result(result_path, result)
        return 78

    matches = [(gid, baselines) for gid, baselines in games.items()
               if gid == args.worker_game or gid.split("-")[0] == args.worker_game]
    if len(matches) != 1:
        raise SystemExit(f"worker game {args.worker_game!r} resolved to {len(matches)} games")
    game_id, baselines = matches[0]
    short = game_id.split("-")[0]
    store = GameStatusStore(out_dir / "status" / short)
    os.environ["TYCHO_GAME_STATUS_DIR"] = str(store.dir)
    try:
        store.claim(game=short, attempt=args.worker_attempt, resume=args.resume)
    except WorkerLeaseError as exc:
        error_label = _error_label(exc)
        result = {
            "game": short, "rhae": 0.0, "levels": 0,
            "mode": "ACTIVE_WORKER", "wall_clock_s": None,
            "stop_reason": "active_worker", "error": error_label,
            "total_actions": 0, "completed_level_actions": 0,
            "unfinished_level_index": None, "unfinished_level_actions": 0,
            "total_actions_including_unfinished": 0,
        }
        _atomic_result(result_path, result)
        return 75

    stop_heartbeat = threading.Event()

    def _heartbeat() -> None:
        while not stop_heartbeat.wait(30):
            try:
                store.heartbeat()
            except Exception:  # noqa: BLE001 - telemetry must not interrupt a game
                pass

    heartbeat = threading.Thread(target=_heartbeat, name=f"heartbeat-{short}", daemon=True)
    heartbeat.start()
    try:
        result = _run_one(
            build, game_id, baselines, args.seed, args.viz, out_dir,
            operation_mode=operation_mode, resume=args.resume, status_store=store,
            max_actions_per_level=args.max_actions_per_level,
            stop_after_levels=args.stop_after_levels,
        )
        mode = str(result.get("mode") or "")
        if mode.startswith("BLOCKED"):
            state = "blocked"
        elif mode.startswith(("ERROR", "PARTIAL")):
            state = "error"
        else:
            state = "finished"
        _atomic_result(result_path, result)
        try:
            store.event(
                "worker_finished",
                mode=mode,
                status_fields={
                    "state": state,
                    "pid": None,
                    "finished_at": utc_now(),
                    "current_rhae": result.get("rhae"),
                    "levels_completed": result.get("levels"),
                    "error": result.get("error") if state != "finished" else None,
                    "result": result,
                },
            )
        except Exception:  # noqa: BLE001 - result is authoritative; telemetry is best-effort
            pass
        return 0 if state == "finished" else 2
    finally:
        stop_heartbeat.set()
        heartbeat.join(timeout=2)


def _supervise_one(
    *,
    args,
    game_id: str,
    out_dir: Path,
    slot: threading.Semaphore,
    initial_resume: bool,
    all_shorts: tuple[str, ...] = (),
) -> dict:
    """Launch/relaunch one game without occupying a model slot during backoff."""
    from tycho.harness.run_status import GameStatusStore, utc_now

    short = game_id.split("-")[0]
    # Optional process-placement hooks are operational only. Rank is computed over the immutable
    # full game set, so selective resume retains the same placement as the initial launch.
    ranked_games = tuple(sorted(all_shorts or (short,)))
    rank = ranked_games.index(short)
    worker_env, _worker_meta = worker_environment(short, rank, len(ranked_games))
    status_dir = out_dir / "status" / short
    if not initial_resume:
        # Reusing an output directory without --resume is an explicit fresh trial. Operational
        # history is no more part of that trial than the workspace/checkpoint cleared by _run_one.
        import shutil
        shutil.rmtree(status_dir, ignore_errors=True)
    store = GameStatusStore(status_dir)
    prior = store.read()
    attempt = int(prior.get("attempt") or 0) + 1
    resume = bool(initial_resume)
    retries = 0
    while True:
        attempt_dir = store.dir / "attempts"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        result_path = attempt_dir / f"attempt_{attempt:04d}_result.json"
        log_path = attempt_dir / f"attempt_{attempt:04d}.log"
        result_path.unlink(missing_ok=True)
        # Do not overwrite a possible live lease with "queued". The child claims atomically; an
        # empty status is already rendered as queued by the monitor.
        store.update(next_resume_at=None, scheduled_attempt=attempt)
        command = [
            sys.executable, "-m", "tycho.harness.run_parallel",
            "--approach", args.approach,
            "--games", short,
            "--out-dir", str(out_dir),
            "--seed", str(args.seed),
            "--max-workers", "1",
            "--operation-mode", args.operation_mode,
            "--worker-game", game_id,
            "--worker-result", str(result_path),
            "--worker-attempt", str(attempt),
            "--viz" if args.viz else "--no-viz",
        ]
        if args.max_actions_per_level is not None:
            command.extend(["--max-actions-per-level", str(args.max_actions_per_level)])
        if args.stop_after_levels is not None:
            command.extend(["--stop-after-levels", str(args.stop_after_levels)])
        if resume:
            command.append("--resume")
        child_env = os.environ.copy()
        child_env.update(worker_env)
        diagnostic_path = _diagnostic_path(
            out_dir, "workers", short, f"attempt_{attempt:04d}.log"
        )
        with log_path.open("a", encoding="utf-8") as log_handle:
            log_handle.write(f"[{utc_now()}] attempt {attempt} started\n")
        with slot:
            if diagnostic_path is None:
                proc = subprocess.run(
                    command,
                    cwd=REPO,
                    env=child_env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            else:
                with diagnostic_path.open("a", encoding="utf-8") as diagnostic_handle:
                    proc = subprocess.run(
                        command,
                        cwd=REPO,
                        env=child_env,
                        stdout=diagnostic_handle,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
        with log_path.open("a", encoding="utf-8") as log_handle:
            log_handle.write(
                f"[{utc_now()}] attempt {attempt} exited with code {proc.returncode}\n"
            )
        try:
            result = json.loads(result_path.read_text())
        except (OSError, json.JSONDecodeError):
            result = {
                "game": short, "rhae": 0.0, "levels": 0,
                "mode": f"ERROR:WorkerExit{proc.returncode}", "wall_clock_s": None,
                "stop_reason": "worker_exit", "total_actions": 0,
                "completed_level_actions": 0, "unfinished_level_index": None,
                "unfinished_level_actions": 0, "total_actions_including_unfinished": 0,
            }
            store.event(
                "worker_process_failed",
                returncode=proc.returncode,
                status_fields={"state": "error", "pid": None, "error": result["mode"]},
            )
        mode = str(result.get("mode") or "")
        retryable = mode.startswith(("ERROR", "PARTIAL"))
        if not args.auto_resume or not retryable or retries >= args.max_auto_resumes:
            return result
        retries += 1
        attempt += 1
        resume = True
        next_at = time.time() + max(0.0, args.resume_delay_s)
        store.event(
            "resume_scheduled",
            attempt=attempt,
            delay_s=args.resume_delay_s,
            status_fields={
                "state": "backoff",
                "next_resume_at": datetime.fromtimestamp(
                    next_at, timezone.utc
                ).isoformat(timespec="seconds"),
            },
        )
        if args.resume_delay_s > 0:
            time.sleep(args.resume_delay_s)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _experiment_limits(args) -> dict[str, int]:
    limits = {}
    if args.max_actions_per_level is not None:
        limits["max_actions_per_level"] = int(args.max_actions_per_level)
    if args.stop_after_levels is not None:
        limits["stop_after_levels"] = int(args.stop_after_levels)
    return limits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--approach", required=True)
    ap.add_argument("--games", default="tr87,vc33,r11l,bp35,ft09")
    ap.add_argument("--out-dir", required=True, help="dir for per-game JSON records")
    ap.add_argument("--seed", type=int, default=0)
    # --viz is ON BY DEFAULT: it captures frames + full LLM I/O per step, which makes a run
    # auditable. It is near-free (frame capture
    # is a 64x64 nested-list copy, no extra LLM calls, no prompt change → arm-neutral; --viz records
    # are actually ~same size since they also trigger slim_record's dedup). Use --no-viz only for a
    # deliberately cheap headline-RHAE-only run. (Frames are also recoverable post-hoc via
    # tycho.harness.backfill_frames, but capturing live is simpler and gives the verbatim transcript.)
    ap.add_argument("--viz", action=argparse.BooleanOptionalAction, default=True,
                    help="capture frames + full LLM I/O per step for analysis (default ON)")
    ap.add_argument("--max-workers", type=int, default=0, help="0 = all games at once")
    ap.add_argument(
        "--max-actions-per-level",
        type=_positive_int,
        default=None,
        help="hard experimental cap on committed actions in each level",
    )
    ap.add_argument(
        "--stop-after-levels",
        type=_positive_int,
        default=None,
        help="stop cleanly after accounting for this many levels",
    )
    ap.add_argument("--resume", action="store_true",
                    help="skip games already finished CLEANLY in --out-dir (no error/PARTIAL); "
                         "re-run missing, errored, or PARTIAL (infra-killed) games. Lets a "
                         "multi-hour run that lost games to a provider outage be re-launched "
                         "without replaying the games that completed.")
    ap.add_argument(
        "--resume-games",
        default="",
        help="comma-separated game IDs to resume; leaves every other unfinished game untouched",
    )
    ap.add_argument(
        "--auto-resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="automatically retry an errored/partial game from its exact checkpoint",
    )
    ap.add_argument("--resume-delay-s", type=float, default=3600.0,
                    help="delay before an automatic retry (default: 3600)")
    ap.add_argument("--max-auto-resumes", type=int, default=3,
                    help="maximum automatic retries per game")
    ap.add_argument("--config", default=None,
                    help="run-config file (YAML/JSON) — sets any config key the ENV has NOT already "
                         "set (env wins). Stable policy fields and provenance are recorded in the "
                         "manifest. See tycho/config.")
    ap.add_argument("--operation-mode", choices=["normal", "offline", "online"],
                    default=os.environ.get("TYCHO_ARCADE_OPERATION_MODE", "normal"),
                    help="Arcade operation mode. Use offline with pre-mounted environment files.")
    # Internal worker protocol. The coordinator is the only supported caller.
    ap.add_argument("--worker-game", default="", help=argparse.SUPPRESS)
    ap.add_argument("--worker-result", default="", help=argparse.SUPPRESS)
    ap.add_argument("--worker-attempt", type=int, default=0, help=argparse.SUPPRESS)
    args = ap.parse_args()
    import time
    _t_start = time.time()                          # for TRUE elapsed wall-clock in the manifest
    # Apply the run-config FILE before anything reads os.environ — it fills only keys the operator
    # did not set on the command line / in the launch script (env precedence). _file_applied records
    # exactly what the file contributed, for provenance in the manifest snapshot.
    from tycho.config import apply_config_file, resolve_orchestration
    _file_applied = apply_config_file(args.config) if args.config else {}
    if _file_applied:
        print(f"[tycho config] {args.config} set {len(_file_applied)} key(s): "
              + ", ".join(sorted(_file_applied)), flush=True)
    # Phase-2 DECLARATIVE GRAPH: if the config file's orchestration block declares an `agents:` list
    # (a multi-agent topology env vars can't express), hand it to the agent via TYCHO_AGENT_GRAPH (JSON
    # — it can't round-trip as scalar env vars). Validate it HERE so a bad graph fails at launch, not
    # mid-run; the agent re-validates on construction. Absent → agents fall back to the MODES preset.
    if args.config:
        import json as _json
        from tycho.agent.modes import resolve_agent_graph
        _orch = resolve_orchestration(args.config)
        _decls = resolve_agent_graph(_orch)
        if _decls is not None:   # validates; raises on a bad graph
            _has_builder = any(d.type == "builder" for d in _decls)
            if _has_builder:
                _raw_mode = os.environ.get("TYCHO_MODE")
                if _raw_mode is None:
                    os.environ["TYCHO_MODE"] = "trigger"
                elif _raw_mode.lower() != "trigger":
                    raise SystemExit(
                        "orchestration.agents declares a builder, which is harness-fired trigger "
                        f"wiring, but TYCHO_MODE={_raw_mode!r}. Use mode: trigger (or omit mode) "
                        "with a builder graph; trigger+subagent/orchestrator hybrids are not wired.")
            os.environ.setdefault("TYCHO_AGENT_GRAPH", _json.dumps(_orch))
            print(f"[tycho config] orchestration.agents graph: "
                  f"{len(_orch['agents'])} agent(s)", flush=True)
    _GIT_VERSION = _git_version()  # capture once at start (tree shouldn't change mid-run)
    print(f"harness git version: {_GIT_VERSION}", flush=True)

    if args.viz:
        os.environ["LLM_LOG"] = "1"  # agents read this to enable I/O capture

    from tycho.workspace.sandbox import PythonSandbox
    _sandbox = PythonSandbox()
    _sandbox.check(require_isolation=True)
    print(f"workspace python runtime: {_sandbox.runtime}", flush=True)

    build = _agent_factory(args.approach)
    operation_mode = _operation_mode_from_name(args.operation_mode)
    arc = _new_arcade(operation_mode)
    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    game_selectors = args.games.split(",")
    # A continuation inherits the original benchmark's complete game set. This makes
    # `--resume --resume-games <id>` sufficient; the operator need not repeat a 25-game list, and
    # the selective scheduling filter cannot accidentally redefine the experiment.
    if args.resume and (out_dir / "run_spec.json").exists():
        try:
            saved = json.loads((out_dir / "run_spec.json").read_text())
            game_selectors = list(saved["policy"]["games"])
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            raise SystemExit(f"cannot recover original game set from run_spec.json: {exc}") from exc
    games = resolve_games(arc, game_selectors)  # {full_id: baselines}
    requested_shorts = sorted(gid.split("-")[0] for gid in games)
    # The immutable full game set lets operational hooks assign stable worker placement. It is
    # computed before selective resume, so a resumed subset keeps its original rank.
    all_shorts = tuple(sorted({str(sel).split("-")[0] for sel in game_selectors}))
    if args.worker_game:
        if not args.worker_result or args.worker_attempt <= 0:
            raise SystemExit("internal worker requires --worker-result and --worker-attempt")
        return _worker_main(args, build, games, operation_mode, out_dir)
    _prev_manifest = {}
    _manifest_path = out_dir / "manifest.json"
    if _manifest_path.exists():
        try:
            _prev_manifest = json.loads(_manifest_path.read_text())
        except Exception:  # noqa: BLE001
            _prev_manifest = {}
    _resume_elapsed_base = float(_prev_manifest.get("wall_clock_s") or 0) if args.resume else 0.0

    def _elapsed_s() -> float:
        return _resume_elapsed_base + (time.time() - _t_start)

    # Immutable execution identity. Monitoring/supervisor edits and transport timeout values are
    # operational; prompts, model policy, action logic, and world-model tooling are score-affecting.
    # The latter fail closed on resume instead of silently creating a mixed benchmark.
    from tycho.config import resolved_config as _resolved_config
    from tycho.harness.run_spec import build_run_spec, ensure_run_spec
    _resolved = _resolved_config(config_file=args.config, file_applied=_file_applied)
    _recorded, _llm_identity = _recorded_run_config(_file_applied)
    _candidate_run_spec = build_run_spec(
        repo=REPO,
        approach=args.approach,
        games=games,
        seed=args.seed,
        viz=args.viz,
        operation_mode=operation_mode.value,
        resolved_config=_resolved,
        recorded_config=_recorded,
        git_version=_GIT_VERSION,
        extra_sources=_run_spec_extra_sources(games),
        experiment_limits=_experiment_limits(args),
    )
    _run_spec = ensure_run_spec(
        out_dir / "run_spec.json", _candidate_run_spec, resume=args.resume
    )

    # --resume: keep games that finished CLEANLY; re-run missing / errored / PARTIAL.
    # A clean record = a game_<id>.json whose record has no `error` (PARTIAL games set it).
    results = {}
    if args.resume:
        skip = {}
        for gid in list(games):
            short = gid.split("-")[0]
            rec_path = out_dir / f"game_{short}.json"
            if not rec_path.exists():
                continue
            try:
                rec = json.loads(rec_path.read_text())
            except Exception:  # noqa: BLE001
                continue
            if not rec.get("error"):  # finished cleanly — keep it, don't re-run
                item = {"rhae": rec.get("env_score", 0.0), "levels": rec.get("levels_completed", 0),
                        "mode": "RESUMED(clean)", "wall_clock_s": rec.get("wall_clock_s"),
                        "stop_reason": rec.get("stop_reason"),
                        "total_actions": rec.get("total_actions", 0),
                        "completed_level_actions": rec.get("completed_level_actions", 0),
                        "unfinished_level_index": rec.get("unfinished_level_index"),
                        "unfinished_level_actions": rec.get("unfinished_level_actions", 0),
                        "total_actions_including_unfinished": rec.get(
                            "total_actions_including_unfinished", rec.get("total_actions", 0))}
                wm = _wm_activity(rec) if (rec.get("trace") and _world_model_enabled()) else None
                if wm:
                    item["wm_activity"] = wm
                skip[gid] = item
        for gid, v in skip.items():
            results[gid.split("-")[0]] = v
            games.pop(gid, None)
        print(f"resume: skipping {len(skip)} cleanly-finished game(s); re-running {len(games)}", flush=True)

    if args.resume_games:
        if not args.resume:
            raise SystemExit("--resume-games requires --resume")
        try:
            games = _select_resume_games(games, args.resume_games)
        except ValueError as exc:
            raise SystemExit(f"--resume-games {exc}") from exc
        print(f"resume: selectively scheduling {', '.join(sorted(g.split('-')[0] for g in games))}",
              flush=True)

    workers = args.max_workers or max(len(games), 1)

    def _agg_throughput(result_index=None):
        """Aggregate per-call token counts from the saved game_*.json traces, so the manifest
        carries runtime telemetry for comparing serving configurations:
        total tokens in/out, decode tok/s (aggregate = concurrent rate, and per-stream), the
        prefill:decode ratio (this workload is prefill-dominated), and mean tokens/latency per call.
        Tokens live either as exact `llm_calls` entries (when LLM_LOG/viz is on) or as per-action
        `token_delta` counters. Older records only have cumulative `token_stats`, so we delta those
        per game instead of summing cumulative values repeatedly."""
        tin = tout = lat = calls = cache_read = cache_write = length_stops = max_tokens_out = 0
        # per-game peak single-prompt size (the context-overflow early-warning: a game whose peak
        # nears --max-model-len risks a late ctx-400, as one game did at 240,315 tok). Captured from each
        # call's per-request input size when available (llm_calls[].tokens_in is the single-call input).
        peak_prompt_per_game: dict = {}

        def _add_stats(stats: dict, *, n_calls: int = 0):
            nonlocal tin, tout, lat, calls, cache_read, cache_write, length_stops, max_tokens_out
            tin += int(stats.get("tokens_in") or 0)
            tout += int(stats.get("tokens_out") or 0)
            lat += int(stats.get("latency_ms") or 0)
            cache_read += int(stats.get("cache_read") or 0)
            cache_write += int(stats.get("cache_write") or 0)
            length_stops += int(stats.get("length_stops") or 0)
            max_tokens_out = max(max_tokens_out, int(stats.get("max_tokens_out") or 0))
            calls += n_calls

        indexed_results = results if result_index is None else result_index
        wanted = set(indexed_results)
        for gj in out_dir.glob("game_*.json"):
            game = gj.stem.replace("game_", "")
            if wanted and game not in wanted:
                continue
            try:
                trace = json.loads(gj.read_text()).get("trace", [])
            except Exception:
                continue
            # Peak single-call prompt occupancy includes cached buckets too. Hosted APIs
            # `tokens_in` is only FRESH input; context pressure is fresh + cache_read + cache_write.
            # Exact per-call llm_calls entries are preferred; newer cumulative token_stats also carry
            # max_prompt_tokens as a fallback.
            for step in trace:
                r = step.get("reasoning") or {}
                if not isinstance(r, dict):
                    continue
                for c in (r.get("llm_calls") or []):
                    if isinstance(c, dict) and (
                        c.get("tokens_in") or c.get("cache_read") or c.get("cache_write")
                    ):
                        peak_prompt_per_game[game] = max(peak_prompt_per_game.get(game, 0),
                                                         int(c.get("tokens_in") or 0)
                                                         + int(c.get("cache_read") or 0)
                                                         + int(c.get("cache_write") or 0))
                ts = r.get("token_stats") or {}
                if isinstance(ts, dict) and ts.get("max_prompt_tokens"):
                    peak_prompt_per_game[game] = max(peak_prompt_per_game.get(game, 0),
                                                     int(ts.get("max_prompt_tokens") or 0))
            prev_stats = {}
            prev_calls = 0
            for step in trace:
                r = step.get("reasoning") or {}
                if not isinstance(r, dict):
                    continue
                llm_calls = r.get("llm_calls") or []
                if isinstance(llm_calls, list) and llm_calls:
                    for c in llm_calls:
                        if isinstance(c, dict):
                            _add_stats(c, n_calls=1)
                    if isinstance(r.get("token_stats"), dict):
                        prev_stats = r["token_stats"]
                    prev_calls = int(r.get("calls") or prev_calls)
                    continue

                cur_calls = int(r.get("calls") or prev_calls)
                n_calls = max(0, cur_calls - prev_calls)
                delta = r.get("token_delta")
                if isinstance(delta, dict):
                    _add_stats(delta, n_calls=n_calls)
                    if isinstance(r.get("token_stats"), dict):
                        prev_stats = r["token_stats"]
                else:
                    stats = r.get("token_stats")
                    if isinstance(stats, dict):
                        _add_stats({k: int(stats.get(k) or 0) - int(prev_stats.get(k) or 0)
                                    for k in stats}, n_calls=n_calls)
                        prev_stats = stats
                prev_calls = cur_calls
        wall = max(1e-9, _elapsed_s())
        comp = sum((v.get("wall_clock_s") or 0) for v in indexed_results.values()) or 1e-9
        return {
            "tokens_in": tin, "tokens_out": tout, "model_calls": calls,
            "cache_read": cache_read, "cache_write": cache_write,
            # client-side prefix-cache reuse: cache_read / total input. NOTE this is the AGENT's view
            # (what the provider reported as cached); the vLLM server log's "Prefix cache hit rate" is
            # the authoritative serving-side number; capture it separately.
            "prefix_cache_hit_rate": (
                round(cache_read / (tin + cache_read + cache_write), 3)
                if (tin + cache_read + cache_write) else None
            ),
            "length_stops": length_stops, "max_tokens_out": max_tokens_out,
            "peak_prompt_tokens_per_game": peak_prompt_per_game,           # ctx-overflow early-warning
            "peak_prompt_tokens_max": max(peak_prompt_per_game.values()) if peak_prompt_per_game else 0,
            "decode_tok_s_aggregate": round(tout / wall, 1),   # concurrent rate (out / true wall)
            "decode_tok_s_per_stream": round(tout / comp, 1),  # ~single-stream (out / compute-sum)
            "prefill_decode_ratio": round(tin / tout, 1) if tout else None,  # >>1 = prefill-bound
            "mean_out_tokens_per_call": round(tout / calls) if calls else 0,
            "mean_call_latency_s": round(lat / calls / 1000, 1) if calls else 0,
        }

    def _is_reportable(v):
        """A game is REPORTABLE iff it finished cleanly — not ERROR (crashed) and not PARTIAL
        (infra-killed mid-run: timeout, cred lapse, kill). A PARTIAL game's env_score is a partial,
        truncated number; averaging it into the experiment mean understates RHAE and isn't comparable.
        The clean mean is the headline; the raw mean is kept for completeness."""
        return not str(v.get("mode", "")).startswith(("ERROR", "PARTIAL"))

    def _write_manifest_unlocked():
        manifest_results = _authoritative_manifest_results(
            out_dir, requested_shorts, results
        )
        mean = sum(v["rhae"] for v in manifest_results.values()) / max(len(manifest_results), 1)
        _clean = [v["rhae"] for v in manifest_results.values() if _is_reportable(v)]
        mean_clean = (sum(_clean) / len(_clean)) if _clean else None
        n_partial = sum(
            1 for v in manifest_results.values()
            if str(v.get("mode", "")).startswith("PARTIAL")
        )
        # compute_s = SUM of per-game wall (serial-equivalent); elapsed_s = TRUE wall-clock of
        # the run at this parallelism. Both are needed for latency analysis: elapsed at a known
        # worker count is the deployable figure; compute/elapsed reveals achieved parallel speedup.
        compute_s = sum((v.get("wall_clock_s") or 0) for v in manifest_results.values())
        pending_games = [g for g in requested_shorts if g not in manifest_results]
        # Canonical orchestration mode: single | orchestrator | trigger |
        # trigger+subagent. Resolve through the MODES table so the recorded label is the CANONICAL
        # spec name (validated — a typo'd TYCHO_MODE raises here exactly as it does at agent
        # construction, so the manifest label can never diverge from the behaviour that ran).
        from tycho.agent.context_config import resolve_context_config
        from tycho.agent.vision import vision_profile
        _mode = _effective_mode_name()
        # Stamp the fully-resolved context/caching config so a run's caching behaviour is recoverable
        # from its result (not just the launch log) — the anti-silent-revert guardrail (a no-cache run
        # is now visible as "prompt_caching": false right here in the manifest).
        _ctx = resolve_context_config().resolved()
        # Stamp the per-model vision profile (render scale + image-fidelity claim) so a wrong render
        # scale (e.g. Qwen rendered at 6px → downscaled, image NOT 1-token/cell) is visible here too.
        _vp = vision_profile(os.environ.get("LLM_MODEL", ""))
        manifest = {"approach": args.approach, "model": _llm_identity["model"],
             "seed": args.seed,                                    # the --seed (was accepted but unstamped)
             "effort": (os.environ.get("TYCHO_EFFORT") or os.environ.get("A5_ORCH_EFFORT")
                        or os.environ.get("LLM_EFFORT", "")),
             "mode": _mode, "backend": _llm_identity["api_protocol"],
             "arcade_operation_mode": operation_mode.value,
             "experiment_limits": _experiment_limits(args),
             "context_config": _ctx,
             "vision": {"render_scale": _vp.render_scale, "lossless_cells": _vp.lossless_cells,
                        "note": _vp.note},
             "run_config": _recorded,
             "run_spec_fingerprint": _run_spec["fingerprint"],
             "git_version": _run_spec.get("initial_git_version", _GIT_VERSION),
             "coordinator_git_version": _GIT_VERSION,
             "workers": workers,
             "n_requested": len(requested_shorts),
             "n_finished": len(manifest_results),
             "n_pending": len(pending_games),
             "pending_games": pending_games,
             "wall_clock_s": round(_elapsed_s(), 1),               # TRUE elapsed (cumulative across resume)
             "compute_s": round(compute_s, 1),                     # sum of per-game wall
             "parallelism_x": round(compute_s / max(1e-9, _elapsed_s()), 1),
             "throughput": _agg_throughput(manifest_results),      # tokens, tok/s, prefill:decode
             "wm_plan_coverage": _coverage_agg(manifest_results),  # WM-correct / planned share of completed levels
             # mean_rhae = raw mean over ALL games (incl. PARTIAL/ERROR, which score low/0). mean_rhae_clean
             # = mean over REPORTABLE games only (cleanly finished) — the headline for an experiment;
             # n_partial games are infra-killed and non-reportable. Both are kept so the gap is visible.
             "mean_rhae": round(mean, 3),
             "mean_rhae_clean": (round(mean_clean, 3) if mean_clean is not None else None),
             "n_reportable": len(_clean), "n_partial": n_partial,
             "games": list(manifest_results), "requested_games": requested_shorts,
             "per_game": manifest_results}
        _atomic_result(out_dir / "manifest.json", manifest, indent=2)

    def _write_manifest():
        # A selective-resume coordinator may finish concurrently with the original coordinator.
        # Serialize the aggregate write and rebuild it from per-game authoritative status while
        # holding the lock, so neither process can publish a stale manifest over the other.
        import fcntl

        lock_path = out_dir / ".manifest.lock"
        lock_path.touch(exist_ok=True)
        with lock_path.open("r+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                _write_manifest_unlocked()
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    print(f"running {len(games)} games on {workers} workers (viz={args.viz})", flush=True)
    if not games:  # --resume with everything already clean: nothing to do but refresh the manifest
        if args.resume and _manifest_path.exists():
            print("resume: all requested games already clean; leaving manifest telemetry unchanged", flush=True)
            return
        _write_manifest()
        return
    # One lightweight supervisor thread per game can sleep during backoff without consuming a
    # model slot. The semaphore alone limits active child processes to --max-workers.
    slot = threading.Semaphore(workers)
    with ThreadPoolExecutor(max_workers=max(len(games), 1)) as ex:
        futs = {
            ex.submit(
                _supervise_one,
                args=args,
                game_id=gid,
                out_dir=out_dir,
                slot=slot,
                initial_resume=args.resume,
                all_shorts=all_shorts,
            ): gid
            for gid in games
        }
        for fut in as_completed(futs):
            gid = futs[fut]
            try:
                r = fut.result()
            except Exception as exc:  # noqa: BLE001 - isolate coordinator-side failures too
                short = gid.split("-")[0]
                r = {
                    "game": short, "rhae": 0.0, "levels": 0,
                    "mode": f"ERROR:Supervisor:{type(exc).__name__}",
                    "wall_clock_s": None, "stop_reason": "supervisor_exception",
                    "total_actions": 0, "completed_level_actions": 0,
                    "unfinished_level_index": None, "unfinished_level_actions": 0,
                    "total_actions_including_unfinished": 0,
                }
            if r.get("mode") == "ACTIVE_WORKER":
                print(f"  active {r['game']:6} already owned by another worker; left untouched",
                      flush=True)
                continue
            results[r["game"]] = {
                k: r[k]
                for k in (
                    "rhae", "levels", "mode", "wall_clock_s", "stop_reason", "total_actions",
                    "completed_level_actions", "unfinished_level_index",
                    "unfinished_level_actions", "total_actions_including_unfinished",
                    "inference_budget",
                )
                if k in r
            }
            if r.get("wm_activity"):   # builder fires / final simulation_accuracy / outcome_verified / plan steps
                results[r["game"]]["wm_activity"] = r["wm_activity"]
            wc = f"{r['wall_clock_s']:.0f}s" if r["wall_clock_s"] is not None else "—"
            print(f"  done {r['game']:6} rhae={r['rhae']:5.2f} levels={r['levels']} "
                  f"{r['mode']:24} {wc}", flush=True)
            # rewrite the manifest after EVERY game, so a crash/kill mid-run still
            # leaves an up-to-date summary of what finished.
            _write_manifest()

    n_err = sum(1 for v in results.values() if str(v["mode"]).startswith("ERROR"))
    n_partial = sum(1 for v in results.values() if str(v["mode"]).startswith("PARTIAL"))
    mean = sum(v["rhae"] for v in results.values()) / max(len(results), 1)
    _clean = [v["rhae"] for v in results.values() if _is_reportable(v)]
    mean_clean = (sum(_clean) / len(_clean)) if _clean else float("nan")
    # The CLEAN mean (reportable games only) is the headline; PARTIAL games are infra-killed and
    # non-reportable (averaging their truncated scores understates RHAE).
    print(f"\nmean RHAE clean: {mean_clean:.2f} over {len(_clean)} reportable "
          f"({n_partial} PARTIAL, {n_err} ERROR excluded) | raw mean (all {len(results)}): {mean:.2f} "
          f"-> {args.out_dir}/manifest.json", flush=True)
    # WM/PLANNING COVERAGE headline: over every level completed across all games, the share with a
    # correct final-model world model and the share the planner found a path for. Tells "how much did
    # world-modelling / planning help" at a glance (per-game breakdown lives in each manifest entry).
    _cov = _coverage_agg(results)
    if _cov:
        _t = _cov["completed_levels"]
        print(f"WM/plan coverage: correct WM on {_cov['wm_correct_levels']}/{_t} completed levels "
              f"({100*_cov['wm_correct_frac']:.0f}%), planned path on {_cov['plan_levels']}/{_t} "
              f"({100*_cov['plan_frac']:.0f}%)", flush=True)
    # Runtime telemetry headline: hardware, parallelism, and token throughput.
    tp = _agg_throughput()
    _gpu = os.environ.get("RUN_HARDWARE", "GPU?")
    _wall = _elapsed_s()
    print(f"runtime: {_gpu} | {workers} workers, {sum((v.get('wall_clock_s') or 0) for v in results.values())/max(_wall,1e-9):.1f}x parallel | "
          f"wall {_wall/3600:.2f}h | decode {tp['decode_tok_s_aggregate']:.0f} tok/s agg "
          f"({tp['decode_tok_s_per_stream']:.0f}/stream) | prefill:decode {tp['prefill_decode_ratio']}:1 | "
          f"{tp['model_calls']} calls @ {tp['mean_out_tokens_per_call']} out-tok, {tp['mean_call_latency_s']}s/call", flush=True)


if __name__ == "__main__":
    raise SystemExit(main() or 0)
