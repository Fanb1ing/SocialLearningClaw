#!/usr/bin/env python3
"""Plan a sequence of actions to outcome(state) == "level_complete" using world_model.py.
Usage: python plan.py [auto|astar|bfs|subgoals]   (default auto)
Optional: TYCHO_PLAN_MAX_NODES=5000 python plan.py astar"""
import hashlib, importlib.util, json, os, sys, time

import numpy as np
import wmlib
import world_model


def _compact_action(a):
    act = a.get("action", "?")
    if act == "ACTION6" and a.get("row") is not None and a.get("col") is not None:
        return f"ACTION6(row={a.get('row')},col={a.get('col')})"
    return act


def _print_subgoal_diagnostics(diag):
    if not diag:
        return
    status = diag.get("status") or "unknown"
    count = diag.get("subgoal_count")
    labels = diag.get("subgoal_labels") or []
    leg_lengths = diag.get("leg_lengths") or []
    print(f"SUBGOAL_STATUS: {status}")
    if count is not None:
        print(f"SUBGOAL_COUNT: {count}")
    if labels:
        label_text = ", ".join(str(x) for x in labels[:8])
        print(f"SUBGOAL_LABELS: {label_text}{' ...' if len(labels) > 8 else ''}")
    if leg_lengths:
        print(f"SUBGOAL_LEG_LENGTHS: {', '.join(str(x) for x in leg_lengths)}")
    if status == "failed_at_subgoal":
        print(f"SUBGOAL_FAILED_INDEX: {diag.get('failed_ordinal')}/{diag.get('subgoal_count')}")
        if diag.get("failed_label"):
            print(f"SUBGOAL_FAILED_LABEL: {diag.get('failed_label')}")
        print(f"SUBGOAL_COMPLETED_PREFIX_LEN: {diag.get('completed_prefix_len', 0)}")
        print(f"SUBGOAL_FAILED_REASON: {diag.get('failed_reason', 'unknown')}")
        if diag.get("failed_errors"):
            print("SUBGOAL_FAILED_ERRORS:")
            for err in diag.get("failed_errors", [])[:5]:
                print(f"  {err}")
        if diag.get("failed_expanded_nodes") is not None:
            print(f"SUBGOAL_FAILED_EXPANDED_NODES: {diag.get('failed_expanded_nodes')}")
            print(f"SUBGOAL_FAILED_FRONTIER_REMAINING: {diag.get('failed_frontier_remaining')}")


def _custom_plan(start):
    """Load an optional compact solver without making it part of the world-model contract."""
    path = "planner.py"
    if not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location("tycho_game_planner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, "plan", None)
    if not callable(fn):
        raise AttributeError("planner.py must define plan(world_model, start_state)")
    result = fn(world_model, start)
    return result.get("plan") if isinstance(result, dict) else result


def _write_validated_artifact(plan, renders, method, elapsed):
    """Persist a compact, anchored plan handoff for the actor and later diagnostics."""
    key = wmlib.current_frame_key()
    grid = wmlib.current_grid()
    if key is None or grid is None:
        return None
    os.makedirs("notes", exist_ok=True)
    model_bytes = open("world_model.py", "rb").read()
    artifact = {
        "schema_version": 1,
        "status": "validated",
        "method": method,
        "start": {
            "level": int(key[0]),
            "turn": int(key[1]),
            "grid_sha256": hashlib.sha256(np.asarray(grid, dtype=np.int16).tobytes()).hexdigest(),
        },
        "world_model_sha256": hashlib.sha256(model_bytes).hexdigest(),
        "plan_length": len(plan),
        "actions": plan,
        "expected_grid_sha256": [
            hashlib.sha256(np.asarray(render, dtype=np.int16).tobytes()).hexdigest()
            for render in renders
        ],
        "validation": "canonical world_model replay reached level_complete",
        "elapsed_s": round(float(elapsed), 3),
    }
    path = "notes/validated_plan.json"
    with open(path, "w") as f:
        json.dump(artifact, f, separators=(",", ":"))
    return path


method = sys.argv[1] if len(sys.argv) > 1 else "auto"
max_nodes = int(os.environ.get("TYCHO_PLAN_MAX_NODES", "20000"))
started = time.monotonic()
print(f"PLANNER_COMMAND: python plan.py {method} (max_nodes={max_nodes})")
for stale in ("notes/validated_plan.json", "notes/validated_plan_expected.npz"):
    try:
        os.remove(stale)
    except FileNotFoundError:
        pass
start = wmlib.current_state(world_model)   # carries hidden state forward through observed actions
heuristic = getattr(world_model, "heuristic", None)   # optional — uninformed (BFS-like) if absent
target = lambda s: wmlib.is_level_complete(world_model, s)
subgoal_diag = None
plan_diag = None
plan_source = method
if method in ("auto", "model") and os.path.exists("planner.py"):
    plan = _custom_plan(start)
    plan_status = "found" if plan is not None else "no_plan_within_budget"
    plan_source = "planner.py"
elif method == "subgoals":
    subgoals = world_model.subgoals(start) if hasattr(world_model, "subgoals") else [target]
    subgoal_diag = wmlib.plan_subgoals_with_diagnostics(
        world_model, start, subgoals, heuristic=heuristic, max_nodes=max_nodes
    )
    plan = subgoal_diag.get("plan") if subgoal_diag.get("status") == "found" else None
    plan_status = "found" if plan is not None else "no_plan_within_budget"
    if subgoal_diag.get("status") == "failed_at_subgoal":
        plan_status = subgoal_diag.get("failed_reason") or "no_plan_within_budget"
elif method == "bfs":
    plan_diag = wmlib.plan_bfs_with_diagnostics(world_model, start, target, max_nodes=max_nodes)
    plan = plan_diag.get("plan") if plan_diag.get("status") == "found" else None
    plan_status = plan_diag.get("status", "no_plan_within_budget")
else:
    plan_diag = wmlib.plan_astar_with_diagnostics(
        world_model, start, target, heuristic=heuristic, max_nodes=max_nodes
    )
    plan = plan_diag.get("plan") if plan_diag.get("status") == "found" else None
    plan_status = plan_diag.get("status", "no_plan_within_budget")
    plan_source = "astar" if method in ("auto", "model") else method
elapsed = time.monotonic() - started
_print_subgoal_diagnostics(subgoal_diag)
if plan_diag and plan_diag.get("expanded_nodes") is not None:
    print(f"PLANNER_EXPANDED_NODES: {plan_diag.get('expanded_nodes')}")
    print(f"PLANNER_FRONTIER_REMAINING: {plan_diag.get('frontier_remaining')}")
    print(f"PLANNER_SEEN_STATES: {plan_diag.get('seen_states')}")
if plan_diag and plan_diag.get("errors"):
    print("PLANNER_MODEL_ERRORS:")
    for err in plan_diag.get("errors", [])[:5]:
        print(f"  {err}")
if plan is None:
    print(f"PLANNER_STATUS: {plan_status} (elapsed_s={elapsed:.2f})")
    if plan_status == "actions_error":
        print("Interpretation: actions(state) must return full action dicts "
              "{\"action\": ..., \"row\": ..., \"col\": ...}. Fix actions() before trusting planner output.")
    elif plan_status == "no_plan_exhausted":
        print("Interpretation: under the current model/action set, reachable states were exhausted without "
              "level_complete. If the target/route/off-screen state is not fully grounded, explore or encode "
              "that evidence before treating this as impossibility. If the model is trusted, consider "
              "RESET/backtracking; otherwise check outcome() and actions().")
    else:
        print("Interpretation: this does not by itself falsify transition(). Check verify.py, outcome(), "
              "actions(), current-state/game-over constraints, unseen or unencoded target/route/off-screen "
              "state, and whether the search budget is too small.")
else:
    validation = wmlib.validate_plan(
        world_model, start, plan, target=target, capture_renders=True
    )
    if validation.get("status") != "found":
        print(f"PLANNER_STATUS: rejected_by_canonical_replay (elapsed_s={elapsed:.2f})")
        print(f"PLANNER_VALIDATION: {validation.get('status')} after "
              f"{validation.get('steps_validated', 0)}/{len(plan)} step(s)")
        for err in validation.get("errors") or []:
            print(f"  {err}")
    elif not plan:
        print(f"PLANNER_STATUS: already_at_level_complete according to outcome() (elapsed_s={elapsed:.2f})")
    else:
        plan = validation["plan"]
        artifact = _write_validated_artifact(
            plan, validation.get("renders") or [], plan_source, elapsed
        )
        print(f"PLANNER_STATUS: plan_found (elapsed_s={elapsed:.2f}, length={len(plan)})")
        print("PLANNER_VALIDATION: canonical_replay=pass; final_outcome=level_complete")
        if artifact:
            print(f"PLANNER_ARTIFACT: {artifact}")
        first = plan[0]
        print(f"PLANNER_PLAN_LEN: {len(plan)}")
        print(f"PLANNER_FIRST_ACTION: {_compact_action(first)}")
        if first.get("row") is not None:
            print(f"PLANNER_FIRST_ACTION_ROW: {first.get('row')}")
            print(f"PLANNER_FIRST_ACTION_COL: {first.get('col')}")
        for i, a in enumerate(plan):
            print(f"  {i+1}. {_compact_action(a)}")
        print("Validated plan: execute from the anchored current frame only; stop and re-plan if an "
              "observed frame diverges from the model.")
