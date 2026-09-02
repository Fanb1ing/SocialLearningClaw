"""Serve Tycho run records through the live replay viewer.

The server scans a results directory, lists completed and in-progress runs, and transforms each
game on request through the shared rendering logic in :mod:`tycho.viewer.viz`. Per-game results are
cached in memory and invalidated when their source record or terminal evidence changes.

Usage:
    python -m tycho.viewer.serve [results_dir] [--port 8900] [--host 127.0.0.1]
    # then open http://localhost:8900; refresh to discover newly written records.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

from tycho.viewer.viz import (HTML, ARC16, build_steps, _env_records, _env_token_cost,
                      _price_for)  # single-sourced render + cost logic

REPO = Path(__file__).resolve().parent.parent.parent  # viewer/ -> tycho/ -> repo root


def _run_dirs(results_dir: Path) -> list[Path]:
    """Every immediate subdir of results/ that is a run — i.e. holds a final game_*.json OR a live
    .partial_*.json (a level-granular sidecar written before the first game finishes). Recognizing
    partials is what makes an in-progress run browsable BEFORE any game completes (otherwise a run
    with completed LEVELS but no completed GAMES is invisible). Excludes the legacy _viewer_live
    mirror and viz_* static builds (not runs)."""
    out = []
    for d in sorted(results_dir.iterdir()):
        if not d.is_dir() or d.name in ("_viewer_live",) or d.name.startswith("viz_"):
            continue
        if (any(d.glob("game_*.json")) or any(d.glob(".partial_*.json"))
                or (d / "run_spec.json").exists() or any((d / "status").glob("*/status.json"))):
            out.append(d)
    return out


def _run_meta(d: Path) -> dict:
    """Run-level header info: model/effort/workers/git/wall-clock from manifest.json if present,
    plus run_time = the run dir's mtime (≈ finish time, drives newest-first sort)."""
    meta = {}
    mp = d / "manifest.json"
    if mp.exists():
        try:
            mj = json.loads(mp.read_text())
            for k in ("model", "effort", "git_version", "hardware", "workers", "wall_clock_s",
                      "mode", "backend", "context_config", "vision", "approach", "compute_s",
                      "parallelism_x"):
                meta[k] = mj.get(k)
            # FULL manifest too (minus the heavy per-game/games lists) so the viewer's config panel
            # can show EVERY recorded knob, not just the curated header fields — the click-to-see-all
            # request. Kept separate so the header logic stays on the named fields above.
            meta["manifest"] = {k: v for k, v in mj.items() if k not in ("per_game", "games")}
        except Exception:  # noqa: BLE001
            pass
    # run_time drives the newest-first dropdown sort + the date label. Prefer the MANIFEST's mtime
    # (written once when the run finishes) over the run DIR's mtime: dir mtime is bumped by any later
    # file op inside it (resume-journal cleanup, rsync copies, a stray .DS_Store), which made every
    # touched run falsely read as "today". manifest.json is not rewritten by those ops, so its mtime
    # is the stable ≈finish time. Fall back to the dir mtime only for runs with no manifest.
    try:
        ts_src = mp if mp.exists() else d
        meta["run_time"] = time.strftime("%Y-%m-%d %H:%M", time.localtime(ts_src.stat().st_mtime))
    except Exception:  # noqa: BLE001
        meta["run_time"] = ""
    return meta


def _run_model(d: Path, meta: dict) -> str:
    """Model id for pricing — manifest first, else the first recorded LLM call in any game."""
    if meta.get("model"):
        return meta["model"]
    for env in _env_records(d):
        for t in env.get("trace", []):
            lc = (t.get("reasoning") or {}).get("llm_calls") or []
            if lc:
                return lc[0].get("model") or ""
    return ""


def _safe_run_dir(results_dir: Path, run_id: str) -> Path | None:
    """Resolve results/<run_id> and reject URL path traversal outside results_dir."""
    try:
        root = results_dir.resolve()
        d = (root / run_id).resolve()
        d.relative_to(root)
    except (ValueError, OSError):
        return None
    return d if d.is_dir() else None


def _trace_step_count(env: dict) -> int:
    return sum(1 for t in env.get("trace", []) if t.get("frame") is not None)


# per-run manifest cache: runId -> (dir_mtime, {games, meta}). A run's manifest is recomputed
# only when its dir mtime changes (a game finished / was re-run), so the shell scan stays fast
# (the naive all-runs scan parsed ~9 GB and took ~20 s). In-progress runs refresh as games land.
_MANIFEST_CACHE: dict = {}


def _partial_games(d: Path) -> dict:
    """{short: path} for live partial records (.partial_<short>.json) that have NO final
    game_<short>.json yet — i.e. games still in flight, viewable at level granularity."""
    out = {}
    for p in d.glob(".partial_*.json"):
        short = p.stem[len(".partial_"):]
        if not (d / f"game_{short}.json").exists():
            out[short] = p
    return out


def _run_manifest(d: Path) -> dict | None:
    """{games:[per-game stats], meta:{...}} for one run dir — cached by dir mtime. Reads each
    game_*.json once (top-level fields + a light token sweep), NOT the full trace render."""
    mtimes = [d.stat().st_mtime_ns]
    for p in [d / "manifest.json", *d.glob("game_*.json"), *d.glob(".partial_*.json")]:
        try:
            mtimes.append(p.stat().st_mtime_ns)
        except FileNotFoundError:
            pass
    mtime = max(mtimes)
    cached = _MANIFEST_CACHE.get(d.name)
    if cached and cached[0] == mtime:
        return cached[1]
    meta = _run_meta(d)
    games = []
    # FAST PATH: run_parallel writes a manifest.json with per-game rhae/levels/wall_clock. Use it
    # to avoid parsing the (huge) game_*.json files on the shell scan — token/cost are filled in
    # lazily when a game is opened (the per-game endpoint has the trace). Falls back to parsing
    # game files only when there's no manifest (legacy/partial runs).
    mp = d / "manifest.json"
    mj = None
    if mp.exists():
        try:
            mj = json.loads(mp.read_text())
        except Exception:  # noqa: BLE001
            mj = None
    if mj and mj.get("per_game"):
        # FAST PATH stats come from the manifest, but token/COST do NOT live there — they must be
        # summed from each game's trace. We do that lightweight sweep here (read each game_*.json
        # once, no full build_steps render) so the run header + per-game rows show the $ list-price.
        # Cheap relative to rendering, and the whole manifest is cached by dir mtime so it runs once
        # per run change. (Previously this hard-coded cost=None — the "filled in lazily when opened"
        # comment was aspirational; nothing back-filled it, so EVERY run_parallel run showed no $.)
        model = _run_model(d, meta)
        stats_by_short = {}
        wanted = set(mj["per_game"])
        # LAZY LISTING (default): the per-game token/cost/step sweep below must read+parse every full
        # game_*.json (traces run to tens of MB). With a large results/ (15GB, 360+ records) that made
        # the index take >60s and time out. The manifest already carries levels/rhae/wall for the row,
        # so token/$ is cosmetic-on-listing — skip the heavy sweep and fill $ on demand when a game is
        # opened. Set TYCHO_VIEWER_LIST_COST=1 to restore the eager sweep (small results/ only).
        if os.environ.get("TYCHO_VIEWER_LIST_COST") == "1":
            for env in _env_records(d):
                short = env["game_id"].split("-")[0]
                if short not in wanted:
                    continue
                ti, to, cost, cost_co = _env_token_cost(env, model)
                cr = cw = 0
                for t in env.get("trace", []):
                    for c in (t.get("reasoning") or {}).get("llm_calls") or []:
                        cr += int(c.get("cache_read") or 0); cw += int(c.get("cache_write") or 0)
                stats_by_short[short] = (_trace_step_count(env), ti, to, cr, cw, cost, cost_co)
        for short, v in mj["per_game"].items():
            n_steps, ti, to, cr, cw, cost, cost_co = stats_by_short.get(short, (None, 0, 0, 0, 0, None, None))
            # When the (heavy) trace sweep is skipped for a fast listing, n_steps is None → don't show
            # "?": the manifest already carries the action count. Prefer the real trace step count when
            # we have it (eager mode), else fall back to the manifest's total_actions.
            if n_steps is None:
                n_steps = v.get("total_actions", v.get("total_actions_including_unfinished"))
            games.append({"id": short, "steps": (n_steps if n_steps is not None else "?"),
                          "levels": v.get("levels", 0),
                          "rhae": round(v.get("rhae", 0.0) or 0.0, 2),
                          "wall_s": round(v.get("wall_clock_s", 0.0) or 0.0, 1),
                          "tok_in": ti, "tok_out": to, "cache_read": cr, "cache_write": cw,
                          "cost": (round(cost, 2) if cost is not None else None),
                          "cost_cacheon": (round(cost_co, 2) if cost_co is not None else None),
                          "wm": v.get("wm_activity")})  # builder fires + per-level WM/plan coverage
    else:
        model = _run_model(d, meta)
        for env in _env_records(d):
            short = env["game_id"].split("-")[0]
            n_steps = sum(1 for t in env.get("trace", []) if t.get("frame") is not None)
            if not n_steps:
                continue
            ti, to, cost, cost_co = _env_token_cost(env, model)
            cr = cw = 0
            for t in env.get("trace", []):
                for c in (t.get("reasoning") or {}).get("llm_calls") or []:
                    cr += int(c.get("cache_read") or 0); cw += int(c.get("cache_write") or 0)
            games.append({"id": short, "steps": n_steps,
                          "levels": env.get("levels_completed", 0),
                          "rhae": round(env.get("env_score", 0.0), 2),
                          "wall_s": round(env.get("wall_clock_s", 0.0) or 0.0, 1),
                          "tok_in": ti, "tok_out": to, "cache_read": cr, "cache_write": cw,
                          "cost": (round(cost, 2) if cost is not None else None),
                          "cost_cacheon": (round(cost_co, 2) if cost_co is not None else None)})
    # LIVE: add in-flight games that only have a partial (level-granular) record so far, so a
    # multi-hour run is browsable as each level lands. Marked live=True; superseded automatically
    # once the final game_<short>.json appears (then _partial_games drops it).
    have = {g["id"] for g in games}
    for short, p in _partial_games(d).items():
        if short in have:
            continue
        try:
            env = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            continue
        n_steps = sum(1 for t in env.get("trace", []) if t.get("frame") is not None)
        games.append({"id": short, "steps": n_steps, "levels": env.get("levels_completed", 0),
                      "rhae": round(env.get("env_score", 0.0), 2),
                      "wall_s": round(env.get("wall_clock_s", 0.0) or 0.0, 1),
                      "tok_in": 0, "tok_out": 0, "cache_read": 0, "cache_write": 0,
                      "cost": None, "live": True})
    result = {"games": games, "meta": meta} if games else None
    _MANIFEST_CACHE[d.name] = (mtime, result)
    return result


def _scan_runs(results_dir: Path) -> dict:
    """Build the RUNS map the viewer shell expects: {runId: {games:[manifest], meta:{...}}}.
    Per-run manifests are cached by dir mtime (see _run_manifest), so a refresh only re-reads
    runs that changed — the shell loads fast even with many runs on disk."""
    runs = {}
    for d in _run_dirs(results_dir):
        m = _run_manifest(d)
        if m:
            runs[d.name] = m
    return runs


# in-memory cache: (runId, gameId) -> built steps JSON string. Cleared if the source file's
# mtime changes (so a re-run / in-progress update is picked up on the next request).
_CACHE: dict = {}


def _inject_boundary_terminal_grids(steps: list, ws: Path | None) -> None:
    """Attach full terminal grids to boundary steps from the durable workspace.

    Harness-owned terminal evidence is intentionally outside causal workspace
    snapshots. The live server reads it from disk so the boundary panel can show
    the real solved-level terminal.
    """
    if ws is None:
        return
    for st in steps:
        r = st.get("reasoning") if isinstance(st, dict) else None
        if not isinstance(r, dict):
            continue
        if st.get("just_completed"):
            completed_level = st.get("level")
        else:
            level = r.get("level")
            if r.get("turn_in_level") != 0 or not isinstance(level, int) or level <= 0:
                continue
            completed_level = level - 1
        if not isinstance(completed_level, int) or completed_level < 0:
            continue
        p = ws / f"level_{completed_level}" / "terminal.json"
        if not p.exists():
            continue
        try:
            grid = json.loads(p.read_text()).get("terminal_grid")
        except Exception:  # noqa: BLE001
            continue
        if isinstance(grid, list) and grid and isinstance(grid[0], list):
            st["boundary_terminal_grid"] = grid


def _terminal_dependency_mtime(ws: Path | None) -> int:
    if ws is None:
        return 0
    mtimes = []
    for pattern in ("level_*/terminal.json", "level_*/terminal.txt"):
        for p in ws.glob(pattern):
            try:
                mtimes.append(p.stat().st_mtime_ns)
            except FileNotFoundError:
                pass
    return max(mtimes) if mtimes else 0


def _game_js(results_dir: Path, run_id: str, game_id: str) -> bytes | None:
    """Return the `window.GAMES[run][game]=<steps>;` script for one game, transforming the slim
    record via build_steps on first request (cached by file mtime)."""
    d = _safe_run_dir(results_dir, run_id)
    if d is None:
        return None
    # find the game_*.json whose short id matches (game_<short>.json); fall back to the live
    # partial sidecar (.partial_<short>.json) for an in-flight game with no final record yet.
    path = None
    for p in d.glob("game_*.json"):
        if p.stem[len("game_"):].split("-")[0] == game_id:
            path = p
            break
    if path is None:
        pp = d / f".partial_{game_id}.json"
        path = pp if pp.exists() else None
    if path is None or not path.exists():
        return None
    key = (run_id, game_id)
    ws = _ws_dir(results_dir, run_id, game_id)
    mtime = (path.stat().st_mtime_ns, _terminal_dependency_mtime(ws))
    cached = _CACHE.get(key)
    if cached and cached[0] == mtime:
        return cached[1]
    env = json.loads(path.read_text())
    steps = build_steps(env)   # slims+dedups legacy raw; idempotent on already-slim records
    _inject_boundary_terminal_grids(steps, ws)
    body = (f"(window.GAMES[{json.dumps(run_id)}]=window.GAMES[{json.dumps(run_id)}]||{{}})"
            f"[{json.dumps(game_id)}]=" + json.dumps(steps) + ";").encode()
    _CACHE[key] = (mtime, body)
    return body


def _ws_dir(results_dir: Path, run_id: str, game_id: str) -> Path | None:
    """The durable on-disk workspace for one game: results/<run>/ws/<short>/. This is the FULL
    tree the agent worked in (notes/, level_<L>/turn_*.txt + diffs, world_model.py …) — the record
    carries causal-file snapshots (per-frame files are pruned to avoid O(turns²) bloat), so
    cross-level browsing reads from here on demand instead. None if the run didn't keep a durable ws."""
    run = _safe_run_dir(results_dir, run_id)
    if run is None:
        return None
    try:
        ws_root = (run / "ws").resolve()
        d = (ws_root / game_id).resolve()
        d.relative_to(ws_root)
    except (ValueError, OSError):
        return None
    return d if d.is_dir() else None


def _ws_tree(ws: Path) -> dict:
    """Names-only nested tree of the durable workspace (no file bodies — those load on demand via
    the file endpoint). {name: {...subdirs}, "__files__": [names]}. __pycache__ is skipped."""
    def walk(p: Path) -> dict:
        node: dict = {}
        files = []
        for x in sorted(p.iterdir()):
            if x.name == "__pycache__":
                continue
            if x.is_dir():
                node[x.name] = walk(x)
            else:
                files.append(x.name)
        if files:
            node["__files__"] = files
        return node
    return walk(ws)


def _ws_file(ws: Path, rel: str) -> str | None:
    """One workspace file's text, JAILED to the ws dir (a traversal like ../../secret resolves
    outside and returns None). Capped so a huge frame dump can't wedge the browser."""
    try:
        target = (ws / rel).resolve()
        target.relative_to(ws.resolve())   # raises if rel escaped the jail
    except (ValueError, OSError):
        return None
    if not target.is_file():
        return None
    try:
        return target.read_text(errors="replace")[:200_000]
    except OSError:
        return None


def _ws_blob(ws: Path, digest: str) -> bytes | None:
    """Read one immutable historical workspace body by SHA-256.

    Blob storage is shared by game workspaces under ``ws/.workspace_blobs`` and is
    deliberately outside the agent's own directory. Only canonical lowercase
    digests are accepted, and resolved paths remain jailed to that store.
    """
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        return None
    try:
        root = (ws.parent / ".workspace_blobs").resolve()
        target = (root / digest).resolve()
        target.relative_to(root)
    except (ValueError, OSError):
        return None
    if not target.is_file():
        return None
    try:
        return target.read_bytes()
    except OSError:
        return None


def _status_snapshots(results_dir: Path) -> list[dict]:
    from tycho.harness.run_status import collect_run_status

    snapshots = []
    for run in reversed(_run_dirs(results_dir)):
        if not (run / "run_spec.json").exists() and not any(
            (run / "status").glob("*/status.json")
        ):
            continue
        try:
            snapshot = collect_run_status(run)
        except Exception:  # noqa: BLE001 - one corrupt run must not hide the dashboard
            continue
        if snapshot.get("n_requested"):
            snapshots.append(snapshot)
    return snapshots


def _status_duration(seconds) -> str:
    if seconds is None:
        return "-"
    value = max(0, int(seconds))
    if value < 60:
        return f"{value}s"
    if value < 3600:
        return f"{value // 60}m"
    return f"{value // 3600}h{(value % 3600) // 60:02d}"


def _status_sort_value(value) -> str:
    return "" if value is None else str(value)


def _status_budget(game: dict) -> str:
    cost = game.get("inference_cost_usd")
    cap = game.get("inference_game_cap_usd")
    pct = game.get("inference_game_cap_pct")
    rate = float(game.get("inference_cost_per_hour") or 0.0)
    if cap is None or cap <= 0:
        return f"${cost:,.2f} (uncapped)" if cost is not None else "-"
    if cost is None:
        return (
            "<div class='budget'><span>not metered / "
            f"${cap:,.0f}</span><small>worker telemetry unavailable</small></div>"
        )
    spent = float(cost or 0.0)
    shown_pct = float(pct or 0.0)
    return (
        "<div class='budget' title='Public-list USD-equivalent inference spend'>"
        f"<span>${spent:,.2f} / ${cap:,.0f}</span>"
        f"<progress value='{min(max(shown_pct, 0.0), 100.0):.3f}' max='100'></progress>"
        f"<small>{shown_pct:.1f}%"
        + (f" · ${rate:,.2f}/h" if rate > 0 else "")
        + "</small></div>"
    )


def _status_page(results_dir: Path) -> bytes:
    snapshots = _status_snapshots(results_dir)
    throughput_data = [
        {
            key: run.get(key)
            for key in (
                "run", "total_actions", "started_at", "elapsed_s",
                "average_actions_per_hour", "action_rate_bin_s", "action_rate_series",
            )
        }
        for run in snapshots
    ]
    throughput_json = json.dumps(throughput_data, separators=(",", ":")).replace("<", "\\u003c")
    throughput_options = "".join(
        f"<option value='{html.escape(run['run'], quote=True)}'>{html.escape(run['run'])}</option>"
        for run in snapshots
    )
    rows = []
    for run in snapshots:
        for game in run["games"]:
            response = game.get("error") or game.get("last_llm_error") or game.get("last_llm_response") or ""
            response = " ".join(str(response).split())[-180:]
            levels = f"{game['levels_completed']}/{game['n_levels'] or '?'}"
            level_pct = game.get("level_completion_pct")
            rows.append(
                "<tr>"
                f"<td class='run-name' data-sort='{html.escape(run['run'])}'>"
                f"{html.escape(run['run'])}</td>"
                f"<td data-sort='{html.escape(game['game'])}'>{html.escape(game['game'])}</td>"
                f"<td class='state' data-sort='{html.escape(game['state'])}'>{html.escape(game['state'])}</td>"
                f"<td data-sort='{game['rhae']}'>{game['rhae']:.2f}</td>"
                f"<td data-sort='{_status_sort_value(level_pct)}'>"
                f"{f'{level_pct:.1f}%' if level_pct is not None else '-'}</td>"
                f"<td data-sort='{game['levels_completed']}'>{levels}</td>"
                f"<td data-sort='{game['action_count']}'>{game['action_count']}</td>"
                f"<td data-sort='{game['actions_per_hour']}'>{game['actions_per_hour']:.2f}</td>"
                f"<td data-sort='{_status_sort_value(game.get('seconds_since_action'))}'>"
                f"{_status_duration(game.get('seconds_since_action'))}</td>"
                f"<td data-sort='{_status_sort_value(game.get('seconds_since_llm'))}'>"
                f"{_status_duration(game.get('seconds_since_llm'))}</td>"
                f"<td data-sort='{_status_sort_value(game.get('seconds_since_heartbeat'))}'>"
                f"{_status_duration(game.get('seconds_since_heartbeat'))}</td>"
                f"<td data-sort='{_status_sort_value(game.get('eta_to_level_cap_s'))}'>"
                f"{_status_duration(game.get('eta_to_level_cap_s'))}</td>"
                f"<td data-sort='{_status_sort_value(game.get('inference_game_cap_pct'))}'>"
                f"{_status_budget(game)}</td>"
                f"<td data-sort='{_status_sort_value(game.get('eta_to_inference_game_cap_s'))}'>"
                f"{_status_duration(game.get('eta_to_inference_game_cap_s'))}</td>"
                f"<td data-sort='{game['actionless_resume_count']}'>{game['actionless_resume_count']}</td>"
                f"<td class='detail'>{html.escape(response)}</td></tr>"
            )
    summaries = "".join(
        f"<div><strong>{html.escape(run['run'])}</strong> "
        f"floor {run['floor_rhae']:.2f} · finished "
        f"{run['n_finished']}/{run['n_requested']} · finished mean "
        f"{run['mean_finished_rhae'] if run['mean_finished_rhae'] is not None else '—'}</div>"
        for run in snapshots
    ) or "<div>No supervised runs found.</div>"
    body = f"""<!doctype html><html><head><meta charset='utf-8'>
<meta http-equiv='refresh' content='30'><title>Tycho run status</title><style>
*{{box-sizing:border-box}}
body{{font:14px system-ui,sans-serif;margin:24px;color:#17202a;background:#f7f8fa;min-width:0}}
header{{display:flex;align-items:baseline;justify-content:space-between;gap:12px;flex-wrap:wrap;margin-bottom:18px}}
h1{{font-size:22px;margin:0}}a{{color:#175a91}}.summary{{display:grid;gap:5px;margin:12px 0 20px;min-width:0}}
.summary div{{min-width:0;overflow-wrap:anywhere}}
div.table-wrap{{width:100%;overflow:auto;border:1px solid #d8dde3;background:white}}
table{{width:100%;min-width:1540px;border-collapse:collapse;background:white}}
th,td{{padding:8px 10px;text-align:left;border-bottom:1px solid #e5e8eb;vertical-align:top}}
th{{font-size:12px;text-transform:uppercase;background:#eef1f4;position:sticky;top:0;z-index:1}}
th[data-sortable]{{cursor:pointer;user-select:none;white-space:nowrap}}th[data-dir='asc']::after{{content:' ^'}}
th[data-dir='desc']::after{{content:' v'}}td.detail{{max-width:520px;white-space:normal;overflow-wrap:anywhere}}
.run-name{{min-width:220px;white-space:normal;overflow-wrap:anywhere}}
.budget{{display:grid;grid-template-columns:auto 96px;gap:2px 8px;align-items:center;white-space:nowrap}}
.budget progress{{width:96px;height:10px;accent-color:#20735a}}.budget small{{grid-column:1 / 3;color:#52606d}}
.throughput{{margin-top:28px;padding-top:20px;border-top:1px solid #d8dde3}}
.throughput-head{{display:flex;gap:16px;align-items:end;justify-content:space-between;flex-wrap:wrap}}
.throughput h2{{font-size:18px;margin:0 0 4px}}.throughput label{{display:grid;gap:4px;font-size:12px;text-transform:uppercase}}
.throughput select{{font:inherit;padding:6px 28px 6px 8px;border:1px solid #aeb7c2;background:white;max-width:min(100%,420px)}}
.throughput-stats{{display:flex;gap:24px;flex-wrap:wrap;margin:16px 0 10px}}
.throughput-stats span{{display:grid;gap:2px}}.throughput-stats strong{{font-size:18px}}
.throughput-stats small,.chart-note{{color:#52606d}}.chart-wrap{{overflow-x:auto;background:white;border:1px solid #d8dde3}}
#throughput-chart{{display:block;width:100%;min-width:760px;height:280px}}#throughput-chart text{{font:12px system-ui,sans-serif;fill:#52606d}}
.state{{font-weight:600}}
@media(max-width:800px){{
 body{{margin:10px}}
 .detail{{display:none}}
 .throughput-head{{align-items:stretch}}
 .throughput label,.throughput select{{width:100%}}
 .throughput-stats{{gap:14px}}
}}
</style></head><body><header><h1>Tycho benchmark status</h1><a href='/'>Replay viewer</a></header>
<div class='summary'>{summaries}</div><div class='table-wrap'><table><thead><tr>
<th data-sortable data-type='text'>Run</th><th data-sortable data-type='text'>Game</th>
<th data-sortable data-type='text'>State</th><th data-sortable data-type='number'>RHAE</th>
<th data-sortable data-type='number'>Level completion</th><th data-sortable data-type='number'>Levels</th>
<th data-sortable data-type='number'>Actions</th><th data-sortable data-type='number' title='Trailing 24-hour committed-action rate'>Actions/hour</th>
<th data-sortable data-type='number'>Action age</th><th data-sortable data-type='number'>LLM age</th>
<th data-sortable data-type='number'>Heartbeat age</th><th data-sortable data-type='number'>ETA action cap</th>
<th data-sortable data-type='number'>Game inference cap</th><th data-sortable data-type='number'>ETA game cap</th>
<th data-sortable data-type='number'>Actionless resumes</th>
<th data-sortable data-type='text' class='detail'>Last response or error</th>
</tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<section class='throughput'><div class='throughput-head'><div><h2>Run action throughput</h2>
<div class='chart-note'>Committed actions over time, normalized to actions/hour within each bin.</div></div>
<label>Run<select id='throughput-run'>{throughput_options}</select></label></div>
<div class='throughput-stats'><span><strong id='throughput-total'>-</strong><small>Total actions</small></span>
<span><strong id='throughput-average'>-</strong><small>Average actions/hour</small></span>
<span><strong id='throughput-elapsed'>-</strong><small>Elapsed</small></span>
<span><strong id='throughput-latest'>-</strong><small>Latest-bin actions/hour</small></span></div>
<div class='chart-wrap'><svg id='throughput-chart' viewBox='0 0 1000 280' role='img' aria-label='Actions per hour over the selected run'></svg></div>
<div id='throughput-note' class='chart-note'></div></section>
<script>
(()=>{{
 const table=document.querySelector('table'), body=table.tBodies[0];
 const heads=[...table.querySelectorAll('th[data-sortable]')];
 const storageKey='tycho-status-sort-v1';
 function sort(index,dir,save=true){{
  const type=heads[index].dataset.type;
  const rows=[...body.rows];
  rows.sort((a,b)=>{{
   const av=a.cells[index].dataset.sort??a.cells[index].textContent.trim();
   const bv=b.cells[index].dataset.sort??b.cells[index].textContent.trim();
   const am=av==='',bm=bv===''; if(am||bm)return am===bm?0:(am?1:-1);
   const cmp=type==='number'?(Number(av)-Number(bv)):av.localeCompare(bv,undefined,{{numeric:true}});
   return dir==='asc'?cmp:-cmp;
  }});
  rows.forEach(row=>body.appendChild(row));
  heads.forEach(h=>{{delete h.dataset.dir;h.setAttribute('aria-sort','none')}});
  heads[index].dataset.dir=dir;heads[index].setAttribute('aria-sort',dir==='asc'?'ascending':'descending');
  if(save)localStorage.setItem(storageKey,JSON.stringify({{index,dir}}));
 }}
 heads.forEach((head,index)=>{{
  head.tabIndex=0;head.setAttribute('role','button');head.setAttribute('aria-sort','none');
  const activate=()=>sort(index,head.dataset.dir==='asc'?'desc':'asc');
  head.addEventListener('click',activate);
  head.addEventListener('keydown',event=>{{if(event.key==='Enter'||event.key===' '){{event.preventDefault();activate()}}}});
 }});
 try{{const saved=JSON.parse(localStorage.getItem(storageKey));if(saved&&heads[saved.index])sort(saved.index,saved.dir,false)}}catch(_err){{}}
}})();
(()=>{{
 const runs={throughput_json}, select=document.getElementById('throughput-run');
 const runStorageKey='tycho-status-throughput-run-v1';
 const svg=document.getElementById('throughput-chart'), ns='http://www.w3.org/2000/svg';
 const duration=value=>{{const s=Math.max(0,Number(value)||0);if(s<3600)return Math.round(s/60)+'m';
  if(s<86400)return Math.floor(s/3600)+'h'+String(Math.floor((s%3600)/60)).padStart(2,'0');
  return Math.floor(s/86400)+'d '+Math.floor((s%86400)/3600)+'h';}};
 const binLabel=value=>value<3600?(value/60)+'m bins':(value/3600)+'h bins';
 function node(name,attrs,text){{const el=document.createElementNS(ns,name);Object.entries(attrs||{{}}).forEach(([k,v])=>el.setAttribute(k,v));if(text!=null)el.textContent=text;return el}}
 function render(){{
  const run=runs.find(item=>item.run===select.value)||runs[0];if(!run)return;
  document.getElementById('throughput-total').textContent=(run.total_actions||0).toLocaleString();
  document.getElementById('throughput-average').textContent=Number(run.average_actions_per_hour||0).toFixed(2);
  document.getElementById('throughput-elapsed').textContent=duration(run.elapsed_s);
  const points=run.action_rate_series||[], latest=points.length?points[points.length-1].actions_per_hour:0;
  document.getElementById('throughput-latest').textContent=Number(latest||0).toFixed(2);
  const recorded=points.reduce((sum,p)=>sum+Number(p.actions||0),0);
  document.getElementById('throughput-note').textContent=binLabel(run.action_rate_bin_s||3600)
   +(recorded<(run.total_actions||0)?' · timeline contains '+recorded.toLocaleString()+' actions recorded by status telemetry':'');
  svg.replaceChildren();const x0=58,y0=18,w=920,h=214,base=y0+h;
  const ymax=Math.max(1,...points.map(p=>Number(p.actions_per_hour)||0));
  [0,.5,1].forEach(frac=>{{const y=base-frac*h;svg.appendChild(node('line',{{x1:x0,y1:y,x2:x0+w,y2:y,stroke:'#dfe4e8','stroke-width':1}}));
   svg.appendChild(node('text',{{x:x0-8,y:y+4,'text-anchor':'end'}},(ymax*frac).toFixed(frac===0?0:1)));}});
  if(points.length){{const bw=Math.max(1,w/points.length-1);points.forEach((p,i)=>{{const value=Number(p.actions_per_hour)||0;
   const bh=value/ymax*h;svg.appendChild(node('rect',{{x:x0+i*w/points.length,y:base-bh,width:bw,height:bh,fill:'#20735a'}}));}});
   const start=new Date(points[0].time*1000),end=new Date((points[points.length-1].time+(run.action_rate_bin_s||0))*1000);
   svg.appendChild(node('text',{{x:x0,y:base+25}},start.toLocaleString()));
   svg.appendChild(node('text',{{x:x0+w,y:base+25,'text-anchor':'end'}},end.toLocaleString()));}}
  svg.appendChild(node('text',{{x:15,y:y0+h/2,transform:'rotate(-90 15 '+(y0+h/2)+')','text-anchor':'middle'}},'Actions/hour'));
 }}
 try{{const savedRun=localStorage.getItem(runStorageKey);if(runs.some(item=>item.run===savedRun))select.value=savedRun}}catch(_err){{}}
 select.addEventListener('change',()=>{{try{{localStorage.setItem(runStorageKey,select.value)}}catch(_err){{}}render()}});render();
}})();
</script></body></html>"""
    return body.encode()


class _Handler(BaseHTTPRequestHandler):
    results_dir: Path = REPO / "results"

    def _send(self, body: bytes, ctype="text/html; charset=utf-8", code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):  # quiet the default per-request stderr spam
        pass

    def do_GET(self):
        path = unquote(self.path.split("?", 1)[0])
        if path in ("/", "/index.html"):
            # viewer shell: same HTML template as the static builder, RUNS injected from a LIVE
            # scan (so newly-finished runs appear on refresh). Game data loads lazily per request.
            runs = _scan_runs(self.results_dir)
            if not runs:
                return self._send(b"<h2>no runs with game_*.json found in results/</h2>")
            page = (HTML.replace("__RUNS__", json.dumps(runs))
                        .replace("__PAL__", json.dumps(ARC16))
                        .replace("__TITLE__", "live"))
            page = page.replace("</body>", "<a href='/status' title='Benchmark status' "
                                "style='position:fixed;right:14px;bottom:14px;z-index:20'>Status</a></body>")
            return self._send(page.encode())
        if path == "/status":
            return self._send(_status_page(self.results_dir))
        if path == "/api/status":
            return self._send(json.dumps(_status_snapshots(self.results_dir)).encode(),
                              ctype="application/json")
        m = re.match(r"^/api/status/(?P<run>[^/]+)$", path)
        if m:
            run = _safe_run_dir(self.results_dir, m["run"])
            if run is None:
                return self._send(b"{}", ctype="application/json", code=404)
            from tycho.harness.run_status import collect_run_status
            return self._send(json.dumps(collect_run_status(run)).encode(),
                              ctype="application/json")
        # per-game data: /<runId>/game_<id>.js  (matches viz.py's ensureLoaded src)
        m = re.match(r"^/(?P<run>[^/]+)/game_(?P<game>[^/]+)\.js$", path)
        if m:
            body = _game_js(self.results_dir, m["run"], m["game"])
            if body is None:
                return self._send(b"// game not found", ctype="application/javascript", code=404)
            return self._send(body, ctype="application/javascript")
        # FULL on-disk workspace tree: /<run>/ws/<game>/tree.json (names only — cross-level browse)
        m = re.match(r"^/(?P<run>[^/]+)/ws/(?P<game>[^/]+)/tree\.json$", path)
        if m:
            ws = _ws_dir(self.results_dir, m["run"], m["game"])
            if ws is None:
                return self._send(b"{}", ctype="application/json", code=404)
            return self._send(json.dumps(_ws_tree(ws)).encode(), ctype="application/json")
        # exact historical causal file body: /<run>/ws/<game>/blob/<sha256>
        m = re.match(r"^/(?P<run>[^/]+)/ws/(?P<game>[^/]+)/blob/(?P<digest>[^/]+)$", path)
        if m:
            ws = _ws_dir(self.results_dir, m["run"], m["game"])
            body = _ws_blob(ws, m["digest"]) if ws is not None else None
            if body is None:
                return self._send(b"blob not found", ctype="text/plain", code=404)
            return self._send(body, ctype="application/octet-stream")
        # one workspace file's content: /<run>/ws/<game>/file?path=<rel>  (jailed to the ws dir)
        m = re.match(r"^/(?P<run>[^/]+)/ws/(?P<game>[^/]+)/file$", path)
        if m:
            from urllib.parse import parse_qs, urlparse
            rel = (parse_qs(urlparse(self.path).query).get("path") or [""])[0]
            ws = _ws_dir(self.results_dir, m["run"], m["game"])
            txt = _ws_file(ws, rel) if ws is not None else None
            if txt is None:
                return self._send(b"(file not found)", ctype="text/plain", code=404)
            return self._send(txt.encode(), ctype="text/plain; charset=utf-8")
        self._send(b"not found", code=404)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir", nargs="?", default=str(REPO / "results"),
                    help="dir holding <run>/game_*.json (default: results/)")
    ap.add_argument("--port", type=int, default=8900)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    _Handler.results_dir = Path(args.results_dir).resolve()
    runs = _run_dirs(_Handler.results_dir)
    print(f"serving {len(runs)} run(s) from {_Handler.results_dir} at "
          f"http://{args.host}:{args.port}  (refresh anytime; no build step)", flush=True)
    ThreadingHTTPServer((args.host, args.port), _Handler).serve_forever()


if __name__ == "__main__":
    main()
