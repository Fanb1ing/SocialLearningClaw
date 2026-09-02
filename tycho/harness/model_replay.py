"""Replay executable world models against recorded workspaces.

For a finished Tycho game that produced a coded world_model.py, recompute — per observed frame —
what the model predicts for post-run analysis:

  - PLAN trajectory (Case A): if plan_astar/plan_subgoals finds a path to outcome()=="level_complete" from the
    current frame's parsed state, return the predicted grid AFTER each planned action (a
    navigable predicted future toward level completion).
  - PER-ACTION prediction (Case B): if no plan is found (model has no working planner, or the
    level_complete target isn't reachable from here), return the predicted next grid for each simple available
    action — transition(state, a) — so the user still sees "what the model thinks each move does".
  - SELF-CHECK: the model's prediction of the action the agent ACTUALLY took this turn, vs the
    real next frame (the verify signal, made visual).

Design notes:
  - We use the game's OWN world_model.py + wmlib.py from its workspace dir (the agent's actual
    code). These are agent-authored and may crash/loop, so each game is processed in a SUBPROCESS
    with a hard timeout; a failure yields no predictions rather than breaking the run.
  - The model is the FINAL version the agent settled on; predictions are "final model applied to
    this frame." ACTION6 (4096 click targets) is NOT enumerated for
    per-action prediction — only simple ACTION1-5/7 are, plus ACTION6 only if the model ignores
    row/col (rare). Grids are returned as plain integer lists for trace analysis.
"""

from __future__ import annotations

import json
from pathlib import Path

from tycho.workspace.sandbox import PythonSandbox

# Per-game worker: imported-and-run INSIDE the workspace dir in a subprocess. Emits a JSON map
# {turn -> prediction-bundle} on stdout. Kept as a string so it runs with the workspace on sys.path
# and cwd (so `import world_model, wmlib` + wmlib's root="." resolution both work).
_WORKER = r'''
import sys, os, json, glob, re
sys.dont_write_bytecode = True
os.chdir(sys.argv[1])
sys.path.insert(0, ".")
MAX_PLAN = 12          # cap recorded predicted-trajectory length
SIMPLE = ["ACTION1","ACTION2","ACTION3","ACTION4","ACTION5","ACTION7"]

def g2l(g):
    import numpy as np
    return [[int(v) for v in row] for row in (g.tolist() if hasattr(g,"tolist") else g)]

try:
    for _pyc in glob.glob("__pycache__/*.pyc"):
        try: os.remove(_pyc)
        except OSError: pass
    import wmlib, world_model
except Exception as e:
    print(json.dumps({"error": f"import: {type(e).__name__}: {e}"})); sys.exit(0)

# Verify once so model quality is stored alongside predictions: DYNAMICS = simulation
# accuracy over observed transitions; OUTCOME channel = does outcome() match terminal evidence.
vinfo = {}
try:
    r = wmlib.verify(world_model)
    # per_level slimmed to {sim_acc, n_changing} (drop first_divergence diff strings) — it rides on
    # EVERY step's baked verify, so the per-level WM-correctness metric (_wm_activity) can read it
    # without re-running verify. Keys are ints here; JSON round-trip stringifies them.
    _per = {lvl: {"simulation_accuracy": d.get("simulation_accuracy"),
                  "strict_simulation_accuracy": d.get("strict_simulation_accuracy"),
                  "known_cell_accuracy": d.get("known_cell_accuracy"),
                  "prediction_coverage": d.get("prediction_coverage"),
                  "prediction_coverage_status": d.get("prediction_coverage_status"),
                  "variant_used": d.get("variant_used"),
                  "unknown_used": d.get("unknown_used"),
                  "n_changing": d.get("n_changing")}
            for lvl, d in (r.get("per_level") or {}).items()}
    vinfo = {"simulation_accuracy": r.get("simulation_accuracy"),
             "strict_simulation_accuracy": r.get("strict_simulation_accuracy"),
             "known_cell_accuracy": r.get("known_cell_accuracy"),
             "prediction_coverage": r.get("prediction_coverage"),
             "prediction_coverage_status": r.get("prediction_coverage_status"),
             "unknown_cell_rate": r.get("unknown_cell_rate"),
             "unknown_used": r.get("unknown_used"),
             "unknown_acceptance_rate": r.get("unknown_acceptance_rate"),
             "variant_used": r.get("variant_used"),
             "variant_acceptance_rate": r.get("variant_acceptance_rate"),
             "n_changing": r.get("n_changing"), "n_total": r.get("n_total"),
             "first_divergence": r.get("first_divergence"), "per_level": _per,
             "errors": (r.get("errors") or [])[:1],
             "variant_errors": (r.get("variant_errors") or [])[:1], "ok": bool(r.get("ok"))}
except Exception as e:
    vinfo = {"error": f"verify: {type(e).__name__}: {e}"}

# OUTCOME channel — does outcome() match solved-level and API GAME_OVER evidence?
oinfo = {}
try:
    g = wmlib.verify_outcome(world_model)
    oinfo = {"level_complete_on_terminal": g.get("level_complete_on_terminal"),
             "level_complete_on_nonterminal": g.get("level_complete_on_nonterminal"),
             "game_over_on_death": g.get("game_over_on_death"),
             "game_over_on_nonterminal": g.get("game_over_on_nonterminal"),
             "ok": g.get("ok"), "outcome_ok": g.get("outcome_ok"),
             "level_complete_ok": g.get("level_complete_ok"),
             "game_over_ok": g.get("game_over_ok"),
             "n_level_complete_terminal": g.get("n_level_complete_terminal"),
             "n_game_over_terminal": g.get("n_game_over_terminal"),
             "n_nonterminal": g.get("n_nonterminal"),
             "outcome_observable": g.get("outcome_observable"),
             "level_complete_observable": g.get("level_complete_observable"),
             "game_over_observable": g.get("game_over_observable"),
             "terminal_render_ok": g.get("terminal_render_ok"),
             "terminal_render_exact": g.get("terminal_render_exact"),
             "terminal_render_strict_exact": g.get("terminal_render_strict_exact"),
             "terminal_render_cell": g.get("terminal_render_cell"),
             "terminal_render_known_cell": g.get("terminal_render_known_cell"),
             "terminal_render_prediction_coverage": g.get("terminal_render_prediction_coverage"),
             "terminal_render_coverage_status": g.get("terminal_render_coverage_status"),
             "terminal_render_unknown_used": g.get("terminal_render_unknown_used"),
             "terminal_render_variant_used": g.get("terminal_render_variant_used"),
             "misfires": (g.get("misfires") or [])[:5]}
except Exception as e:
    oinfo = {"error": f"verify_outcome: {type(e).__name__}: {e}"}

# EARLY EMIT: print verify+outcome NOW (flushed) so even if the planning loop below times out and the
# subprocess gets killed, the parent (predict_for_workspace) can still recover the col-1 chips.
# We emit a "partial" line; the LAST line we print (at the very end) is the full result. The
# parent reads the LAST complete JSON line from stdout — same code path, different completeness.
print(json.dumps({"verify": vinfo, "outcome": oinfo, "predictions": {},
                  "has_outcome": False, "has_subgoals": False, "partial": True}), flush=True)

# frames(): {(level,turn): grid}. We predict per OBSERVED frame.
try:
    frames = wmlib.frames(".")
except Exception as e:
    print(json.dumps({"error": f"frames: {type(e).__name__}: {e}", "verify": vinfo, "outcome": oinfo})); sys.exit(0)

heur = getattr(world_model, "heuristic", None)
has_outcome = hasattr(world_model, "outcome")
has_sub = hasattr(world_model, "subgoals")
outcome_known_wrong = bool(oinfo.get("outcome_observable") and oinfo.get("ok") is False)
target = lambda s: wmlib.is_level_complete(world_model, s)
def _render(s):
    import numpy as np
    return np.asarray(world_model.render(s))
# THREADED per-frame state (hidden-state faithful): frame (L,t)'s state is init_state(first frame of
# L) advanced by transition() over the observed actions up to t — NOT a fresh init_state(grid_t),
# which would lose any hidden substate (a counter, phase, learned palette) the planner depends on.
# This matches runtime planning (wmlib.current_state). Without it, plan_levels/overlays for
# hidden-state games are computed from a wrong belief state and mislead the "did world-modelling
# help" diagnosis. Built once up front by threading each level's transitions.
import wmlib as _wmlib
_trans_by_level = {}
for _t in _wmlib.transitions("."):
    _trans_by_level.setdefault(_t["level"], []).append(_t)
def _threaded_states():
    """{(level,turn): state} by threading transition() within each level from its first frame."""
    st = {}
    by_level = {}
    for (l, t) in frames:
        by_level.setdefault(l, []).append(t)
    for l in sorted(by_level):
        turns = sorted(by_level[l])
        seq = _trans_by_level.get(l, [])
        try:
            cur = world_model.init_state(frames[(l, turns[0])], l)
        except Exception:
            continue
        st[(l, turns[0])] = cur
        # seq[i].turn is the turn the i-th transition PRODUCED; advance in order, keying each result.
        for tr in seq:
            try:
                cur = world_model.transition(cur, {"action": tr["action"], "row": tr["row"], "col": tr["col"]})
            except Exception:
                break
            st[(l, tr["turn"])] = cur
    return st
_STATES = _threaded_states()
out = {}
# bound total work two ways. (1) Per-call: max_nodes=2000 caps a hard plan_astar at ~2s (vs ~22s
# at the wmlib default of 20000). With restricted actions(), most calls still finish in
# milliseconds; the cap only bites on no-path-to-level-complete frames where the planner exhausts before
# giving up. (2) Per-run: a wall-clock budget on the planning loop itself. Once it's elapsed,
# stop scheduling new frames — the worker emits the FULL final JSON with whatever predictions it
# already produced. The parent gets verify+outcome+partial-predictions instead of a timeout-kill.
PLAN_NODES = 2000
PLAN_BUDGET_S = 45.0
import time as _time
_plan_start = _time.time()
keys = sorted(frames.keys())
for idx, k in enumerate(keys):
    if _time.time() - _plan_start > PLAN_BUDGET_S:
        break  # leave time for the final emit below

    grid = frames[k]
    # hidden-state faithful: use the THREADED state at this frame (init_state of the level + observed
    # transitions up to here), falling back to a fresh init_state only if threading didn't reach it.
    state = _STATES.get(k)
    if state is None:
        try:
            state = world_model.init_state(grid, k[0])
        except Exception:
            continue
    bundle = {"level": k[0], "turn": k[1]}
    # --- Case A: plan to level_complete ---
    plan = None
    if has_outcome and not outcome_known_wrong:
        try:
            plan = wmlib.plan_astar(world_model, state, target, heuristic=heur, max_nodes=PLAN_NODES)
        except Exception:
            plan = None
    if not plan and has_sub and not outcome_known_wrong:
        try:
            # subgoals is a FUNCTION returning an ordered list of target-style callables — call it with
            # the state (matches the runtime plan.py convention); plan_subgoals iterates the LIST, so
            # passing the bare function would raise 'not iterable' (silently → plan=None, undercount).
            subs = world_model.subgoals(state)
            plan = wmlib.plan_subgoals(world_model, state, subs, heuristic=heur, max_nodes=PLAN_NODES)
        except Exception:
            plan = None
    if plan:
        traj = []
        s = state
        for step in plan[:MAX_PLAN]:
            try:
                s = world_model.transition(s, step)
                traj.append({"action": step.get("action","?"),
                             "row": step.get("row"), "col": step.get("col"),
                             "grid": g2l(_render(s))})
            except Exception:
                break
        if traj:
            bundle["plan"] = traj
    # --- Case B: per-action prediction (only when no usable plan) ---
    if "plan" not in bundle:
        # Distinguish no-plan causes instead of returning a vague
        # "no working planner"): outcome() undefined, outcome known-wrong (we skip planning then), or a
        # path search that returned nothing within the node/time budget.
        if not has_outcome:
            bundle["no_plan_reason"] = "model defines no outcome() — nothing to plan toward"
        elif outcome_known_wrong:
            # Inline WHAT is wrong (direction + the numbers) instead of an opaque "fails outcome":
            # verify_outcome falsified outcome() against observed level-complete / GAME_OVER evidence.
            _lct = oinfo.get("level_complete_on_terminal")
            _lcnt = oinfo.get("level_complete_on_nonterminal")
            _got = oinfo.get("game_over_on_death")
            _gont = oinfo.get("game_over_on_nonterminal")
            _bits = []
            if _lct is not None and _lct < 0.999:
                _bits.append("misses a solved level terminal")
            if _lcnt and _lcnt > 0:
                _bits.append("marks non-terminal frames as level_complete")
            if _got is not None and _got < 0.999:
                _bits.append("misses an observed API GAME_OVER")
            if _gont and _gont > 0:
                _bits.append("marks non-terminal frames as game_over")
            _why = "; ".join(_bits) or "disagrees with observed terminal evidence"
            _lct_s = "n/a" if _lct is None else f"{_lct:.2f}"
            _lcnt_s = "n/a" if _lcnt is None else f"{_lcnt:.2f}"
            _got_s = "n/a" if _got is None else f"{_got:.2f}"
            _gont_s = "n/a" if _gont is None else f"{_gont:.2f}"
            bundle["no_plan_reason"] = (f"outcome() disagrees with observed terminal evidence — {_why} "
                                        f"(level_complete_on_terminal={_lct_s}, "
                                        f"level_complete_on_nonterminal={_lcnt_s}, "
                                        f"game_over_on_death={_got_s}, "
                                        f"game_over_on_nonterminal={_gont_s}); "
                                        "planning skipped until outcome() is fixed")
        else:
            bundle["no_plan_reason"] = (
                f"no path to outcome() == 'level_complete' found within the search budget ({PLAN_NODES} nodes)"
            )
        preds = []
        try:
            avail = list(world_model.actions(state))
        except Exception:
            avail = SIMPLE
        for a in avail:
            # actions(state) may return bare names ("ACTION1") OR action dicts ({"action","row","col"})
            # for CLICK games — normalize so the SIMPLE filter + transition() see a consistent shape.
            act = a if isinstance(a, dict) else {"action": a, "row": None, "col": None}
            name = act.get("action")
            focused_click = name == "ACTION6" and act.get("row") is not None and act.get("col") is not None
            if name not in SIMPLE and not focused_click:   # skip unfocused ACTION6 (4096 targets) and unknowns
                continue
            try:
                ns = world_model.transition(state, act)
                ng = g2l(_render(ns))
                cg = g2l(grid)
                if ng != cg:         # only show actions the model predicts will DO something
                    preds.append({"action": name, "row": act.get("row"), "col": act.get("col"), "grid": ng})
            except Exception:
                continue
        if preds:
            bundle["per_action"] = preds
    if "plan" in bundle or "per_action" in bundle or "no_plan_reason" in bundle:
        out[f"{k[0]}_{k[1]}"] = bundle

print(json.dumps({"verify": vinfo, "outcome": oinfo, "predictions": out,
                  "has_outcome": has_outcome, "has_subgoals": has_sub}))
'''


def _last_json_line(text: str) -> dict | None:
    """The worker prints exactly one JSON object per line — an early 'partial' (verify+outcome only)
    then a final full one. On timeout, the subprocess gets killed but Python's TimeoutExpired
    preserves stdout that was written + flushed before the kill. Walk lines bottom-up and return
    the first one that parses as JSON."""
    for line in reversed((text or "").strip().splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            return json.loads(line)
        except Exception:  # noqa: BLE001
            continue
    return None


def predict_for_workspace(ws_dir: str, timeout: int = 45) -> dict:
    """Run the prediction worker against one game workspace. Returns the parsed JSON
    {verify, outcome, predictions:{`L_T`->bundle}, ...}. On timeout, returns the PARTIAL emit
    (verify+outcome computed before the planning loop) so diagnostics remain available when planning
    was the slow part."""
    result = PythonSandbox().run_source(
        ws_dir,
        _WORKER,
        timeout=timeout,
        args=(".",),
    )
    if result.timed_out:
        partial = _last_json_line(result.stdout)
        if partial and not partial.get("error"):
            partial["error"] = f"planning timed out ({timeout}s) — verify/outcome recovered from partial"
            return partial
        return {"error": f"prediction timed out ({timeout}s)"}
    if result.returncode != 0:
        return {"error": f"worker exit {result.returncode}: {(result.stderr or '')[-200:]}"}
    parsed = _last_json_line(result.stdout)
    if parsed is None:
        return {"error": "parse: no JSON line found in worker stdout"}
    return parsed


def predict_with_model_src(ws_dir: str, world_model_src: str, timeout: int = 45) -> dict:
    """Like predict_for_workspace but runs a GIVEN world_model.py source (a per-turn SNAPSHOT) instead
    of the on-disk FINAL model. Used to show the model AS IT EXISTED at a turn ('current' model) next
    to the final one. Temporarily swaps world_model.py in the workspace (restoring it after), so the
    same wmlib/frames/transitions resolve. The caller batches by DISTINCT snapshot (typically <10 per
    game — the model changes only when edited), not per turn, so this is bounded, not O(turns)."""
    import os as _os
    wm_path = _os.path.join(ws_dir, "world_model.py")
    saved = None
    try:
        with open(wm_path) as f:
            saved = f.read()
    except Exception:  # noqa: BLE001
        return {"error": "no world_model.py to swap"}
    try:
        with open(wm_path, "w") as f:
            f.write(world_model_src)
        return predict_for_workspace(ws_dir, timeout=timeout)
    finally:
        if saved is not None:   # ALWAYS restore the final model, even on error/timeout
            try:
                with open(wm_path, "w") as f:
                    f.write(saved)
            except Exception:  # noqa: BLE001
                pass


def find_workspaces(games: list[str], tmp_root: str | None = None) -> dict:
    """Best-effort map short_game_id -> workspace dir.

    Prefer explicit/durable roots (`<run>/ws/<game>` or `<root>/<game>`) when supplied; fall back to
    old temp `arcws_*/<game>` discovery for legacy records.
    """
    import glob
    import os
    roots = [tmp_root] if tmp_root else []
    roots += [os.environ.get("TMPDIR", "/tmp"), "/tmp", "/private/tmp"]
    out = {}
    for g in games:
        best, bestn = None, -1
        for root in roots:
            if not root:
                continue
            for c in (Path(root) / g, Path(root) / "ws" / g):
                if c.exists() and Path(c, "world_model.py").exists():
                    n = len(list(c.glob("level_*/turn_*.txt"))) + len(list(c.glob("turn_*.txt")))
                    if n > bestn:
                        bestn, best = n, str(c)
            for c in glob.glob(f"{root.rstrip('/')}/arcws_*/{g}"):
                if not Path(c, "world_model.py").exists():
                    continue
                n = len(glob.glob(f"{c}/level_*/turn_*.txt")) + len(glob.glob(f"{c}/turn_*.txt"))
                if n > bestn:
                    bestn, best = n, c
        if best:
            out[g] = best
    return out
