#!/usr/bin/env python3
"""Live status of a Tycho parallel run (results/<run>/). Pairs with run_parallel.py: that wrote
per-game JSON + manifest + a ws/<game> workspace; this reads them and prints a status table.

Designed to be re-runnable on a live run dir — handles both finished and in-flight games. The
"deepest level / turn / last write" columns come from the workspace (still updating during a run);
the RHAE column comes from the per-game JSON (only set when a game finishes, partial or clean).

Usage:
  python -m tycho.harness.run_progress <run-dir>          # one-shot snapshot
  python -m tycho.harness.run_progress <run-dir> --watch  # refresh every 30s

Example:
  python -m tycho.harness.run_progress results/opus47_single_s0
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

# tag a game's outcome unambiguously, separating RHAE from "did it actually solve any levels".
def _outcome(game_json: dict) -> dict:
    """Return a small dict describing what happened. Fields:
      status         "SOLVED" | "PARTIAL" | "STALLED" | "ERROR"  (STALLED = 0 levels solved)
      lc             levels completed
      nl             total levels in the env
      stall_lvl      1-indexed level the agent gave up on (None if all levels solved)
      stall_actions  actions taken at the stall level (0 if solved everything)
      cap            completion-weight cap on the score (0..100; the shipped scorer caps the
                     weighted average at this value, so when rhae == cap the agent's per-level
                     efficiency was actually high and the cap is binding)
    """
    n_levels = game_json.get("n_levels") or 0
    lr = game_json.get("level_results") or []
    levels_completed = game_json.get("levels_completed", sum(1 for l in lr if l.get("completed")))
    err = game_json.get("error")
    stop_reason = game_json.get("stop_reason")
    stall_lvl = stall_actions = None
    if lr and not lr[-1].get("completed"):
        stall_lvl = lr[-1].get("level_index")
        stall_actions = lr[-1].get("actions_taken", 0)
    unfinished_actions = game_json.get("unfinished_level_actions") or 0
    if unfinished_actions:
        stall_lvl = game_json.get("unfinished_level_index") or (levels_completed + 1)
        stall_actions = unfinished_actions
    if err:
        status = "ERROR"
    elif levels_completed >= n_levels and n_levels > 0:
        status = "SOLVED"
    elif levels_completed > 0:
        status = "PARTIAL"
    else:
        status = "STALLED"
    # completion-weight cap = sum(weights of completed levels) / sum(weights 1..n) * 100
    total_weight = n_levels * (n_levels + 1) // 2
    completed_weight = sum(l["level_index"] for l in lr if l.get("completed"))
    cap = (completed_weight / total_weight * 100.0) if total_weight else 0.0
    return {"status": status, "lc": levels_completed, "nl": n_levels,
            "stall_lvl": stall_lvl, "stall_actions": stall_actions, "cap": cap,
            "stop_reason": stop_reason}


def _last_write(ws_dir: str) -> float | None:
    """Most-recent non-pycache mtime in the workspace, or None if empty/missing."""
    if not os.path.isdir(ws_dir):
        return None
    best = 0.0
    for root, dirs, files in os.walk(ws_dir):
        if "__pycache__" in dirs:
            dirs.remove("__pycache__")
        for f in files:
            t = os.path.getmtime(os.path.join(root, f))
            if t > best:
                best = t
    return best or None


def _deepest_level(ws_dir: str) -> tuple[int, int] | None:
    """(level_index, turns_in_that_level), reading the workspace's level_<N>/turn_*.txt layout.
    None if no levels yet."""
    if not os.path.isdir(ws_dir):
        return None
    levels = []
    for d in os.listdir(ws_dir):
        if d.startswith("level_") and d[6:].isdigit():
            levels.append(int(d[6:]))
    if not levels:
        return None
    L = max(levels)
    turns = len(glob.glob(os.path.join(ws_dir, f"level_{L}", "turn_*.txt")))
    return L, turns


def _scan_log_errors(log_path: str) -> dict:
    """Count interesting error patterns in the run log. Cheap line-pass; empty if log missing."""
    out = {"provider_400": 0, "http_400": 0, "http_429": 0, "http_5xx": 0, "timeouts": 0}
    if not os.path.exists(log_path):
        return out
    try:
        with open(log_path, errors="replace") as f:
            for line in f:
                if "Engine not found" in line:
                    out["provider_400"] += 1
                elif "HTTP Error 400" in line:
                    out["http_400"] += 1
                elif "HTTP Error 429" in line or "Throttling" in line:
                    out["http_429"] += 1
                elif "HTTP Error 5" in line:
                    out["http_5xx"] += 1
                elif "timed out" in line.lower() or "TimeoutError" in line:
                    out["timeouts"] += 1
    except OSError:
        pass
    return out


def _fmt_ago(secs: float | None) -> str:
    """'14m' / '2h' / '—' for None."""
    if secs is None:
        return "—"
    s = max(0, int(secs))
    if s < 60: return f"{s}s"
    if s < 3600: return f"{s//60}m"
    return f"{s//3600}h{(s%3600)//60:02d}"


def _structured_status(run: Path, *, as_json: bool = False) -> bool:
    """Print the schema-v2 supervisor view; return False for legacy run directories."""
    if not (run / "run_spec.json").exists() and not any((run / "status").glob("*/status.json")):
        return False
    from tycho.harness.run_status import collect_run_status

    snapshot = collect_run_status(run)
    if as_json:
        print(json.dumps(snapshot, indent=2, sort_keys=True))
        return True
    finished = snapshot["mean_finished_rhae"]
    finished_s = f"{finished:.2f}" if finished is not None else "—"
    print(f"=== {snapshot['run']} ===")
    print(
        f"  model={snapshot.get('model') or '?'}  mode={snapshot.get('mode') or '?'}  "
        f"finished={snapshot['n_finished']}/{snapshot['n_requested']}"
    )
    print(
        f"  floor_rhae={snapshot['floor_rhae']:.2f}  "
        f"mean_finished_rhae={finished_s}"
    )
    print(
        f"  {'game':6} {'state':9} {'rhae':>6} {'lvls':>6} {'actions':>7} "
        f"{'a/h':>7} {'act ago':>7} {'llm ago':>7} {'ETA cap':>8} {'rs':>3}  last response/error"
    )
    for game in snapshot["games"]:
        levels = f"{game['levels_completed']}/{game['n_levels'] or '?'}"
        eta = _fmt_ago(game["eta_to_level_cap_s"])
        note = game.get("error") or game.get("last_llm_error") or game.get("last_llm_response") or ""
        note = " ".join(str(note).split())[-100:]
        print(
            f"  {game['game']:6} {game['state'][:9]:9} {game['rhae']:6.2f} {levels:>6} "
            f"{game['action_count']:7d} {game['actions_per_hour']:7.1f} "
            f"{_fmt_ago(game['seconds_since_action']):>7} "
            f"{_fmt_ago(game['seconds_since_llm']):>7} {eta:>8} "
            f"{game['actionless_resume_count']:3d}  {note}"
        )
    return True


def status(run_dir: str, *, as_json: bool = False) -> None:
    """One snapshot of `run_dir`. Prints a header + per-game table + log error counts."""
    run = Path(run_dir).resolve()
    if not run.is_dir():
        print(f"run dir not found: {run}", file=sys.stderr); sys.exit(2)
    if _structured_status(run, as_json=as_json):
        return

    mpath = run / "manifest.json"
    m = json.load(open(mpath)) if mpath.exists() else {}
    pg = m.get("per_game") or {}

    # The full set of games we expect this run to cover: union of (manifest per_game keys) + (ws dirs)
    # + (.partial_* files). This catches games that haven't even written a partial yet (queued by
    # the worker pool but not yet picked up).
    games = set(pg.keys())
    ws_root = run / "ws"
    if ws_root.is_dir():
        games |= {d.name for d in ws_root.iterdir() if d.is_dir()}
    for p in run.glob(".partial_*.json"):
        games.add(p.stem.replace(".partial_", ""))

    # Read partials for in-flight games (carries n_levels, level_results so we can show "stalled
    # at L<x> after N actions" while the game is still running).
    partials = {}
    for p in run.glob(".partial_*.json"):
        gid = p.stem.replace(".partial_", "")
        try:
            partials[gid] = json.load(open(p))
        except Exception:  # noqa: BLE001
            pass

    # Header
    print(f"=== {run.name} ===")
    if m:
        model = m.get("model", "?"); mode = m.get("mode") or "(graph)"; backend = m.get("backend") or "?"
        wall_h = (m.get("wall_clock_s") or 0) / 3600
        workers = m.get("workers", "?"); par = m.get("parallelism_x")
        par_s = f"{par:.1f}x" if isinstance(par, (int, float)) else "?"
        print(f"  model={model}  mode={mode}  backend={backend}  workers={workers} ({par_s} parallel)")
        if "mean_rhae" in m:
            # Lead with the CLEAN mean (reportable games only) when the manifest carries it; PARTIAL/
            # ERROR games are infra-killed and their truncated scores understate RHAE. Fall back to the
            # raw mean for pre-clean-metric manifests.
            if m.get("mean_rhae_clean") is not None:
                npart = m.get("n_partial", 0)
                print(f"  mean_rhae_clean={m['mean_rhae_clean']:.2f} over {m.get('n_reportable','?')} reportable "
                      f"(raw={m['mean_rhae']:.2f}, {npart} PARTIAL)  games_done={len(pg)}/{len(games)}  wall={wall_h:.1f}h")
            else:
                print(f"  mean_rhae(so_far)={m['mean_rhae']:.2f}  games_done={len(pg)}/{len(games)}  wall={wall_h:.1f}h")

    # Per-game table.
    now = time.time()
    rows = []
    for gid in sorted(games):
        gjson_path = run / f"game_{gid}.json"
        partial = partials.get(gid)
        gj = json.load(open(gjson_path)) if gjson_path.exists() else (partial or {})

        if gjson_path.exists():
            o = _outcome(gj)
            r = pg.get(gid, {}).get("rhae")
            rhae_s = f"{r:5.2f}" if r is not None else "  —  "
            cap_s = f"{o['cap']:.2f}" if o["nl"] else "—"
            if o["status"] == "PARTIAL" and o["stall_lvl"]:
                extra = f"stalled L{o['stall_lvl']}@{o['stall_actions']}a"
            elif o["status"] == "STALLED" and o["stall_lvl"]:
                extra = f"never solved a level; gave up L{o['stall_lvl']}@{o['stall_actions']}a"
            else:
                extra = (gj.get("error") or "")[:50]
            if o.get("stop_reason") == "llm_call_limit" and o["stall_lvl"]:
                extra = f"llm limit; gave up L{o['stall_lvl']}@{o['stall_actions']}a"
            elif o.get("stop_reason") in (
                "inference_cost_game_limit", "inference_cost_level_limit",
            ) and o["stall_lvl"]:
                scope = "game" if o["stop_reason"] == "inference_cost_game_limit" else "level"
                extra = f"inference {scope} budget; stopped L{o['stall_lvl']}@{o['stall_actions']}a"
            status_str = o["status"]; lc, nl = o["lc"], o["nl"]
        elif partial:
            # in-flight: read level_results from the partial
            o = _outcome(partial)
            status_str = "RUNNING"
            lc, nl = o["lc"], o["nl"]
            rhae_s = "  —  "; cap_s = f"{o['cap']:.2f}" if nl else "—"
            extra = ""
        elif (ws_root / gid).is_dir():
            # ws/ dir exists but no partial yet — game is RUNNING in its first level.
            status_str = "RUNNING"
            lc = 0; nl = 0; rhae_s = "  —  "; cap_s = "—"; extra = ""
        else:
            status_str = "QUEUED"
            lc = 0; nl = 0; rhae_s = "  —  "; cap_s = "—"; extra = ""

        ws = ws_root / gid
        dl = _deepest_level(str(ws))
        cur_level_s = f"L{dl[0]}@{dl[1]}t" if dl else "—"
        last = _last_write(str(ws))
        ago = _fmt_ago(now - last) if last else "—"
        rows.append((gid, status_str, rhae_s, cap_s, lc, nl, cur_level_s, ago, extra))

    if rows:
        # cap = ceiling that the shipped scorer enforces (completion-weight fraction × 100). When
        # rhae == cap, the agent's per-level efficiency was binding. When rhae < cap, the
        # per-level score itself was the limit.
        print(f"  {'game':6} {'status':8} {'rhae':>5} {'cap':>5}  {'lvls':5}  {'cur':>9}  {'last':>5}  notes")
        for g, st, rs, cs, lc, nl, cur, ago, extra in rows:
            lvl_s = f"{lc}/{nl}" if nl else f"{lc}/?"
            print(f"  {g:6} {st:8} {rs} {cs:>5}  {lvl_s:5}  {cur:>9}  {ago:>5}  {extra}")

    # Error rollup from the run log (best-effort find: results/<run>_run.log next to the dir, or
    # inside the dir).
    log_candidates = [
        str(run.parent / f"{run.name}_run.log"),
        str(run / f"{run.name}_run.log"),
    ]
    log = next((p for p in log_candidates if os.path.exists(p)), None)
    if log:
        e = _scan_log_errors(log)
        if any(e.values()):
            parts = [f"{k}={v}" for k, v in e.items() if v]
            print(f"  log errors: {', '.join(parts)}  ({log})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="results/<run> directory (where manifest.json + ws/ live)")
    ap.add_argument("--watch", action="store_true", help="refresh every 30s")
    ap.add_argument("--json", action="store_true", help="emit the structured status snapshot")
    args = ap.parse_args()
    while True:
        status(args.run_dir, as_json=args.json)
        if not args.watch:
            return 0
        time.sleep(30)
        print()  # blank line between refreshes


if __name__ == "__main__":
    sys.exit(main())
