#!/usr/bin/env python3
"""Check world_model.py against everything observed so far. Usage: python verify.py
Prints the same three channels as auto-feedback: initial render, dynamics, and outcome.
Run it any time to see the full per-level breakdown + first divergence."""
import wmlib, world_model


def _f(x):
    return "n/a" if x is None else f"{x:.2f}"


def _uses_unknown(d):
    return bool(d.get("unknown_used")) or (
        d.get("prediction_coverage") is not None and d.get("prediction_coverage") < 0.999
    )


def _coverage_status_value(cov):
    if cov is None:
        return "unobserved"
    if cov <= 0:
        return "vacuous"
    if cov >= 0.999999:
        return "complete"
    return "partial"


def _coverage_status(d, key="prediction_coverage", status_key=None):
    return d.get(status_key or f"{key}_status") or _coverage_status_value(d.get(key))


def _coverage_desc(cov, status):
    if status == "complete":
        return f"{_f(cov)} complete: render() claimed every cell"
    if status == "partial":
        return f"{_f(cov)} partial: render() used UNKNOWN=-1 for unclaimed cells"
    if status == "vacuous":
        return f"{_f(cov)} vacuous: render() claimed no cells; this is not reliable evidence"
    return f"{_f(cov)} unobserved"


def _level_stats(v, lvl):
    if lvl is None:
        return None
    per = v.get("per_level") or {}
    return per.get(lvl) or per.get(str(lvl))


def _level_errors(errs, lvl):
    if lvl is None:
        return []
    prefix = f"level {lvl}"
    return [e for e in errs if str(e).startswith(prefix)]


def _dynamics_status(stats, errors=()):
    if errors:
        return "error"
    if not stats or not stats.get("n_changing"):
        return "unobserved"
    status = _coverage_status(stats)
    sim = stats.get("simulation_accuracy")
    if status == "vacuous":
        return "fail_vacuous"
    if sim is not None and sim >= 0.999 and status == "partial":
        return "pass_partial"
    if sim is not None and sim >= 0.999:
        return "pass"
    return "fail"


def _first_divergence_label(fd):
    if not fd:
        return None
    return f"L{fd['level']} turn {fd['turn']} action {fd['action']} — {fd['diff']}"


v = wmlib.verify(world_model)
n = v["n_changing"]
errs = v.get("errors") or []
sim = v.get("simulation_accuracy")
coverage_status = _coverage_status(v)
current_key = wmlib.current_frame_key()
current_level = current_key[0] if current_key is not None else None
current_stats = _level_stats(v, current_level)
current_dynamics_status = _dynamics_status(current_stats, _level_errors(errs, current_level))
all_dynamics_status = _dynamics_status({
    "n_changing": n,
    "simulation_accuracy": sim,
    "prediction_coverage": v.get("prediction_coverage"),
    "prediction_coverage_status": v.get("prediction_coverage_status"),
}, errs)
fd = v.get("first_divergence")
all_dynamics_display_status = (
    "fail_prior_level"
    if all_dynamics_status in ("error", "fail", "fail_vacuous")
    and current_dynamics_status in ("pass", "pass_partial")
    and fd and fd.get("level") != current_level
    else all_dynamics_status
)
print(f"INITIAL RENDER over {v.get('n_initial_render', 0)} level start(s):")
print(f"  accepted_level_start_accuracy {_f(v.get('initial_render_accuracy'))}"
      "  <- render(init_state(observed_start, level)), with bounded variants/UNKNOWN semantics")
print(f"  strict_level_start_accuracy {_f(v.get('initial_render_strict_accuracy'))}"
      "  <- canonical render() exact full-grid match rate")
if current_stats and not current_stats.get("initial_render_ok"):
    print(f"  current_level L{current_level} mismatch: "
          f"{current_stats.get('initial_render_first_mismatch')}")
for e in (v.get("initial_render_errors") or [])[:3]:
    print(f"  ⚠ initial render error: {e}")
if not n:
    print("no graded transitions observed yet — every observed action so far left the grid unchanged "
          "and the model predicted no visible change.")
    if v.get("archived_attempts"):
        print(f"archived RESET evidence not replay-scored: {v.get('archived_attempts', 0)} attempt(s), "
              f"{v.get('archived_transitions', 0)} transition(s), "
              f"{v.get('archived_observed_changed', 0)} observed-changing")
else:
    print(f"DYNAMICS: simulation_accuracy={_f(v.get('simulation_accuracy'))}; "
          f"matched={v.get('n_matched', 0)}/evaluated={n}/observed={v['n_total']} "
          "current-attempt transitions")
    if v.get("n_joint_noop"):
        print(f"  jointly-unchanged no-ops omitted={v.get('n_joint_noop', 0)}; "
              f"false model changes on observed no-ops={v.get('n_false_change_on_noop', 0)}")
    if v.get("archived_attempts"):
        print(f"  archived RESET evidence not replay-scored: {v.get('archived_attempts', 0)} attempt(s), "
              f"{v.get('archived_transitions', 0)} transition(s), "
              f"{v.get('archived_observed_changed', 0)} observed-changing")
    if v.get("variant_used") or _uses_unknown(v):
        print(f"  simulation_accuracy {_f(v['simulation_accuracy'])}  <- accepted transition match rate after bounded observation_variants and UNKNOWN masking")
        print(f"  strict_simulation_accuracy {_f(v.get('strict_simulation_accuracy'))}  <- canonical render() exact full-grid match rate")
    else:
        print(f"  simulation_accuracy {_f(v['simulation_accuracy'])}  <- exact full-grid match rate; drive to 1.00")
    if v.get("variant_used"):
        print(f"  observation_variants accepted {v.get('variant_used')} transition(s) "
              f"(limit {v.get('variant_limit')})")
    if _uses_unknown(v):
        print(f"  prediction_coverage {_coverage_desc(v.get('prediction_coverage'), coverage_status)}")
        print(f"  known_cell_accuracy {_f(v.get('known_cell_accuracy'))}  <- per-cell accuracy over claimed cells")
        print(f"  unknown_used {v.get('unknown_used', 0)} transition(s); unknown_cell_rate {_f(v.get('unknown_cell_rate'))}")
    if v.get("cell_accuracy") is not None and ((v.get("simulation_accuracy") or 0) < 0.999
                                               or v.get("variant_used") or _uses_unknown(v)):
        print(f"  cell_accuracy {_f(v['cell_accuracy'])}  <- strict full-grid diagnostic; UNKNOWN counts as non-matching")
    pl = {k: lv["simulation_accuracy"] for k, lv in (v["per_level"] or {}).items()}
    if pl:
        print("  per-level: " + ", ".join(f"L{k}={_f(val)}" for k, val in sorted(pl.items())))
    if current_level is not None:
        if current_stats:
            print(f"  current_level L{current_level}: "
                  f"simulation_accuracy={_f(current_stats.get('simulation_accuracy'))}; "
                  f"matched={current_stats.get('n_matched', 0)}/"
                  f"evaluated={current_stats.get('n_changing', 0)}/"
                  f"observed={current_stats.get('n_total', 0)}")
            cfd = current_stats.get("first_divergence")
            if cfd and current_dynamics_status in ("error", "fail", "fail_vacuous"):
                print(f"  current_level_first_divergence: {_first_divergence_label(cfd)}")
        else:
            print(f"  current_level L{current_level}: no observed transitions yet")
    if fd and all_dynamics_display_status == "fail_prior_level":
        print(f"  prior_level_first_divergence: {_first_divergence_label(fd)}")
        print("    -> current level dynamics pass; keep this prior-level mismatch visible for later repair.")
    elif fd:
        print(f"  first divergence: {_first_divergence_label(fd)}")
        print("    -> fix init_state/transition here.")
    if errs:
        for e in errs[:3]:
            print(f"  ⚠ model raised (accuracy unreliable): {e}")
        if len(errs) > 3:
            print(f"    … +{len(errs)-3} more")
    for e in (v.get("variant_errors") or [])[:3]:
        print(f"  ⚠ observation_variants issue: {e}")

g = wmlib.verify_outcome(world_model)
outcome_status = "unobserved"
terminal_render_status = "unobserved"
if not g["outcome_observable"]:
    lc_false = g.get("level_complete_on_nonterminal") or 0
    go_false = g.get("game_over_on_nonterminal") or 0
    if lc_false or go_false or g.get("outcome_errors"):
        outcome_status = "fail"
        print("OUTCOME: ⚠ no terminal evidence observed yet, but outcome() already misfires or raises.")
    else:
        print("OUTCOME: no level_complete or GAME_OVER terminal observed yet; transition dynamics may verify "
              "before the goal/hazard rule is known. outcome() is checked for false positives on observed "
              "decision frames only.")
else:
    if g.get("outcome_ok"):
        outcome_status = "pass"
        print("OUTCOME: status verified on observed terminal evidence "
              f"({g['n_level_complete_terminal']} level_complete, {g['n_game_over_terminal']} game_over).")
    else:
        outcome_status = "fail"
        diag = []
        if (g.get("level_complete_on_terminal") or 0) < 1 and g.get("level_complete_observable"):
            diag.append("misses observed level_complete terminal(s)")
        if (g.get("level_complete_on_nonterminal") or 0) > 0:
            diag.append("returns level_complete before a level ended")
        if (g.get("game_over_on_death") or 0) < 1 and g.get("game_over_observable"):
            diag.append("misses observed API GAME_OVER terminal(s)")
        if (g.get("game_over_on_nonterminal") or 0) > 0:
            diag.append("returns game_over on ordinary decision frame(s)")
        if g.get("outcome_errors"):
            diag.append("outcome() raised errors")
        print("OUTCOME: ⚠ status wrong — "
              f"level_complete_on_terminal={_f(g.get('level_complete_on_terminal'))}, "
              f"level_complete_on_nonterminal={_f(g.get('level_complete_on_nonterminal'))}, "
              f"game_over_on_death={_f(g.get('game_over_on_death'))}, "
              f"game_over_on_nonterminal={_f(g.get('game_over_on_nonterminal'))} — "
              + "; ".join(diag) + ".")
        for kind, lvl, t in g["misfires"][:5]:
            print(f"  {kind}: L{lvl} turn {t}")
for e in (g.get("outcome_errors") or [])[:3]:
    print(f"OUTCOME: {e}")
# Render bridge: the modeled terminal must also look like the frame that actually won (the observed
# winning frame, recorded as a terminal event). render_exact<1.0 means transition(s_pre, win_action)
# doesn't reproduce the real terminal — fix transition/render so the modeled win matches the engine's.
terminal_coverage_status = _coverage_status(
    g, key="terminal_render_prediction_coverage", status_key="terminal_render_coverage_status")
if g.get("terminal_render_variant_used") or g.get("terminal_render_unknown_used"):
    terminal_render_status = (
        "fail_vacuous" if terminal_coverage_status == "vacuous" else
        ("pass_with_variant" if g.get("terminal_render_variant_used") else "pass_with_unknown")
    )
    bits = []
    if g.get("terminal_render_variant_used"):
        bits.append(f"observation_variants on {g.get('terminal_render_variant_used')} terminal(s)")
    if g.get("terminal_render_unknown_used"):
        bits.append(f"UNKNOWN cells on {g.get('terminal_render_unknown_used')} terminal(s)")
    print(f"OUTCOME terminal render: accepted with {' and '.join(bits)}; "
          f"strict_exact={_f(g.get('terminal_render_strict_exact'))}, "
          f"coverage={_coverage_desc(g.get('terminal_render_prediction_coverage'), terminal_coverage_status)}, "
          f"known_cell={_f(g.get('terminal_render_known_cell'))}, "
          f"canonical_cell={_f(g.get('terminal_render_cell'))}.")
if g.get("terminal_render_exact") is not None and (
    g["terminal_render_exact"] < 1.0 or terminal_coverage_status == "vacuous"
):
    terminal_render_status = "fail"
    print(f"OUTCOME terminal render: exact={_f(g['terminal_render_exact'])} (want 1.00), "
          f"coverage={_coverage_desc(g.get('terminal_render_prediction_coverage'), terminal_coverage_status)}, "
          f"cell={_f(g.get('terminal_render_cell'))} — your modeled terminal differs from the observed "
          "winning frame; reconcile transition()/render() at the winning action.")
elif g.get("terminal_render_exact") is not None and terminal_render_status == "unobserved":
    terminal_render_status = "pass_partial" if terminal_coverage_status == "partial" else "pass"
for e in (g.get("variant_errors") or [])[:3]:
    print(f"OUTCOME terminal render: observation_variants issue: {e}")

if current_dynamics_status in ("error", "fail", "fail_vacuous"):
    repair_focus = "dynamics"
elif outcome_status == "fail":
    repair_focus = "outcome"
elif terminal_render_status == "fail":
    repair_focus = "terminal_render"
elif all_dynamics_status == "unobserved":
    repair_focus = "exploration"
elif current_dynamics_status == "unobserved":
    repair_focus = "explore_current_level"
elif all_dynamics_status in ("error", "fail", "fail_vacuous"):
    repair_focus = "plan_or_act_on_current_level"
else:
    repair_focus = "planner_or_action"
print("VERIFIER_TRIAGE:")
print(f"  CURRENT_LEVEL_DYNAMICS_STATUS: {current_dynamics_status}")
print(f"  ALL_LEVELS_DYNAMICS_STATUS: {all_dynamics_display_status}")
print(f"  OUTCOME_STATUS: {outcome_status}")
print(f"  TERMINAL_RENDER_STATUS: {terminal_render_status}")
if repair_focus in ("planner_or_action", "plan_or_act_on_current_level", "explore_current_level"):
    print(f"  NEXT_STEP_FOCUS: {repair_focus}")
else:
    print(f"  REPAIR_FOCUS: {repair_focus}")

try:
    deaths = wmlib.death_events()
    if deaths:
        print(f"GAME_OVER_EVIDENCE: {len(deaths)} fatal action(s) observed outside normal transitions.")
        for ev in deaths[-5:]:
            rc = "" if ev.get("row") is None else f" at ({ev.get('row')},{ev.get('col')})"
            print(f"  L{ev.get('level')} turn {ev.get('turn')} {ev.get('action')}{rc} -> GAME_OVER")
except Exception as e:
    print(f"GAME_OVER_EVIDENCE: check skipped ({type(e).__name__}: {e})")

try:
    print(wmlib.format_game_over_action_report(wmlib.game_over_action_report(world_model)))
except Exception as e:
    print(f"MODEL_GAME_OVER_ACTIONS: check skipped ({type(e).__name__}: {e})")
