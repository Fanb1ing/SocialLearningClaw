"""Semantic contracts for actor prompts.

Golden snapshots catch byte drift. These checks protect the instructions most likely to affect
RHAE when the prompt is rewritten: action economics, world-model ownership, terminal evidence,
planner workflow, variants, and click-action pruning.
"""

from __future__ import annotations

from tycho.agent import agent as A
from tycho.prompts.render import render_prompt


def _render(mode: str) -> str:
    return render_prompt(
        "actor.system",
        colormap=A._COLORMAP,
        wm_variant=mode,
        show_grid=False,
        show_diff=True,
        vision=True,
        lossless_cells=False,
        meta_reflect=False,
    )


def _check_common() -> dict[str, bool]:
    texts = {mode: _render(mode) for mode in ("single", "orch", "trigger", "none")}
    return {
        "all_modes_state_action_economics": all(
            "Tool calls are free for scoring; take_action commits one environment control" in text
            and "Every take_action call, including RESET, spends one scored environment action" in text
            for text in texts.values()
        ),
        "all_modes_separate_reset_control_from_game_actions": all(
            "RESET is always available to restart the current level/attempt" in text
            and "it is an API recovery control, not a learned game mechanic" in text
            and "Game-specific actions may include ACTION1-ACTION4" in text
            and "The current turn header lists the valid take_action choices: RESET plus the frame-declared game actions" in text
            and "prefer RESET over deliberately wasting actions" in text
            for text in texts.values()
        ),
        "all_modes_include_direction_action_prior": all(
            "ACTION1=up, ACTION2=down, ACTION3=left, ACTION4=right" in text
            for text in texts.values()
        ),
        "all_modes_keep_action7_soft_undo_prior": all(
            "ACTION7, which is commonly intended as undo but may still be game-specific" in text
            and "ACTION7≈undo" not in text
            for text in texts.values()
        ),
        "all_modes_have_exact_grid_tooling": all(
            "run_python" in text and "wmlib.current_grid()" in text and "wmlib.frames()" in text
            for text in texts.values()
        ),
        "all_modes_offer_deterministic_diff_summary": all(
            "diff='on' inlines exact cell changes" in text
            and "'summary' only their deterministic count and bounding box" in text
            for text in texts.values()
        ),
        "all_modes_document_wmlib_schema_and_txt_format": all(
            "wmlib.frames() and wmlib.transitions() return current-attempt observations" in text
            and "wmlib.attempts() lists prior attempts" in text
            and "wmlib.death_events() returns a list of API GAME_OVER events" in text
            and "wmlib.terminal_events() returns a dict {level: solved-terminal event}" in text
            and "wmlib.animation_index(level=..., turn=..., last=N)" in text
            and "Do not use np.loadtxt on turn_*.txt" in text
            and "Use wmlib.frames() for parsed prior grids" in text
            and "labeled grid text with row/column guides" in text
            and "space-separated cell colors" not in text
            for text in texts.values()
        ),
        "builder_modes_use_actor_beliefs": (
            "notes/actor_beliefs.md" in texts["orch"]
            and "notes/actor_beliefs.md" in texts["trigger"]
        ),
        "builder_modes_do_not_claim_actor_owns_world_model": (
            "you own `world_model.py`" not in texts["orch"]
            and "you own `world_model.py`" not in texts["trigger"]
        ),
        "builder_modes_qualify_planner_by_grounded_model": (
            "Planner advice only searches goals/subgoals grounded in the current model" in texts["orch"]
            and "no-plan is not proof that the level is impossible" in texts["orch"]
            and "Planner no-plan means not reachable in the current grounded model, not impossible" in texts["trigger"]
        ),
        "no_world_model_keeps_frame_tooling_but_not_wm_loop": (
            "Solve the game by direct reasoning from observations" in texts["none"]
            and "wmlib.current_grid()" in texts["none"]
            and "persistent simulator or planner" in texts["none"]
            and "world_model.py" not in texts["none"]
            and "verify.py" not in texts["none"]
            and "plan.py" not in texts["none"]
            and "No-world-model" not in texts["none"]
            and "executable world-model loop" not in texts["none"]
            and "Single-mode world modeling" not in texts["none"]
            and "invoke_builder" not in texts["none"]
        ),
    }


def _check_single() -> dict[str, bool]:
    text = _render("single")
    return {
        "single_owns_model_and_notes": (
            "Single-mode world modeling: you own `world_model.py` and `notes/world_model.md`" in text
        ),
        "single_no_actor_beliefs_channel": "notes/actor_beliefs.md" not in text,
        "single_uses_model_as_decision_aid": (
            "Use the world model as a decision aid" in text
            and "If the model and outcome rule are credible" in text
            and "act or probe to distinguish competing hypotheses" in text
        ),
        "single_waits_for_model_feedback_before_acting": (
            "After each `world_model.py` edit, the harness verifies dynamics and outcome separately" in text
            and "Wait for that feedback before acting" in text
        ),
        "single_blocks_same_batch_planner_action": (
            "Do not combine a planner call and `take_action` in the same tool batch" in text
            and "the planner result should be visible before you decide" in text
        ),
        "single_terminal_evidence_explicit": (
            "pre-win decision frame" in text
            and "winning action" in text
            and "terminal.txt" in text
            and "terminal.json" in text
        ),
        "single_variants_optional_and_bounded": (
            "Leave `observation_variants(state)` empty" in text
            and "at most 5 full-grid alternatives" in text
            and "ignored by planning" in text
        ),
        "single_cli_planner_instruction": (
            "`python plan.py astar`" in text
            and "`python plan.py bfs`" in text
            and "`python plan.py subgoals`" in text
        ),
        "single_planner_requires_grounded_goal_or_subgoal": (
            "Planning only searches for goals/subgoals that are grounded" in text
            and "if the target, route, or off-screen terrain is still unseen or unencoded" in text
            and "no-plan means" in text
            and "not reachable in this model" in text
            and "the level is impossible" in text
        ),
        "single_focused_click_actions": (
            "`actions(state)` defines the planner's candidate actions" in text
            and "Do not return all 4096 cells" in text
        ),
        "single_noop_death_reset_evidence": (
            "Treat no-ops and fatal/reset outcomes as evidence" in text
        ),
    }


def _check_scribe() -> dict[str, bool]:
    text = render_prompt(
        "scribe.user",
        level=0,
        actor_writes_wm=False,
        wm_variant="none",
    )
    return {
        "no_world_model_scribe_not_builder_owned": (
            "world_model.py is builder-owned" not in text
            and "world_model.py" not in text
            and "verify.py" not in text
            and "plan.py" not in text
            and "world-model loop" not in text
            and "future direct reasoning" in text
        ),
    }


def _check_builder_kickoff() -> dict[str, bool]:
    from tycho.agent.builder import REPORT_FILE

    text = render_prompt(
        "builder.user",
        level=2,
        hint="[auto-trigger] test",
        available_actions=["ACTION1", "ACTION6"],
        report_file=REPORT_FILE,
    )
    return {
        "builder_kickoff_includes_current_legal_actions": (
            "Current legal actions for the actor are: ACTION1, ACTION6" in text
        ),
    }


def _check_builder_system() -> dict[str, bool]:
    text = render_prompt(
        "builder.system",
        colormap=A._COLORMAP,
        report_file="notes/builder_report.md",
        meta_reflect=False,
    )
    return {
        "builder_report_allows_no_action_when_planning_not_grounded": (
            "use none when the objective/route/off-screen state is not grounded enough to plan" in text
        ),
    }


def _actor_user_for_variant(wm_variant: str) -> str:
    return render_prompt(
        "actor.user",
        header="=== Level 1, turn 0 (state=NOT_FINISHED; available actions=['ACTION1']) ===",
        level=1,
        turn_in_level=0,
        frame_boundary="level_start",
        ld="level_1",
        grid_text="0 0\n0 0",
        wm_variant=wm_variant,
        show_grid=False,
        show_diff=False,
        vision=False,
        diff="",
        no_op=False,
        last_action="ACTION1",
        plan_hint=None,
        turn3="000",
    )


def _actor_user_no_visible_change() -> str:
    return render_prompt(
        "actor.user",
        header="=== Level 1, turn 3 (state=NOT_FINISHED; available actions=['RESET', 'ACTION1']) ===",
        level=1,
        turn_in_level=3,
        frame_boundary=None,
        ld="level_1",
        grid_text="0 0\n0 0",
        wm_variant="single",
        show_grid=False,
        show_diff=True,
        vision=True,
        diff="no cells changed",
        no_op=True,
        last_action="ACTION1",
        plan_hint=None,
        turn3="003",
    )


def _actor_user_with_plan_hint() -> str:
    return render_prompt(
        "actor.user",
        header="=== Level 1, turn 3 (state=NOT_FINISHED; available actions=['ACTION1']) ===",
        level=1,
        turn_in_level=3,
        frame_boundary=None,
        ld="level_1",
        grid_text="0 0\n0 0",
        wm_variant="single",
        show_grid=False,
        show_diff=False,
        vision=False,
        diff="",
        no_op=False,
        last_action="ACTION1",
        plan_hint=(
            "Validated plan continuation: observed trajectory matches 2/5 action(s); "
            "next action is ACTION1. Stop and re-plan on any later divergence."
        ),
        turn3="003",
    )


def _check_actor_user_level_start() -> dict[str, bool]:
    none_text = _actor_user_for_variant("none")
    single_text = _actor_user_for_variant("single")
    no_visible_change_text = _actor_user_no_visible_change()
    return {
        "no_world_model_level_start_omits_world_model_terms": (
            "notes/level_0_insights.md exists" in none_text
            and "world model" not in none_text
            and "world_model.py" not in none_text
            and "init_state()" not in none_text
        ),
        "world_model_level_start_mentions_reliable_model_and_init": (
            "notes/level_0_insights.md exists" in single_text
            and "reliable world model" in single_text
            and "init_state() still holds" in single_text
        ),
        "attempt_restart_names_current_level_without_claiming_game_start": (
            "fresh playable frame for the restarted attempt of level 7" in render_prompt(
                "actor.user",
                header="=== Level 7, turn 0 ===",
                level=7,
                turn_in_level=0,
                frame_boundary="attempt_restart",
                ld="level_7",
                grid_text="0 0\n0 0",
                wm_variant="single",
                show_grid=False,
                show_diff=False,
                vision=False,
                diff="",
                no_op=False,
                last_action=None,
                plan_hint=None,
                turn3="000",
            )
        ),
        "no_visible_change_is_factual_not_semantic_noop": (
            "produced no visible cell changes" in no_visible_change_text
            and "This does not prove the action had no effect" in no_visible_change_text
            and "It may be a no-op" not in no_visible_change_text
        ),
        "validated_plan_hint_is_surfaced": (
            "observed trajectory matches 2/5 action(s); next action is ACTION1"
            in _actor_user_with_plan_hint()
        ),
    }


def test_actor_prompt_contracts() -> None:
    groups = {
        "common": _check_common(),
        "single": _check_single(),
        "scribe": _check_scribe(),
        "builder_kickoff": _check_builder_kickoff(),
        "builder_system": _check_builder_system(),
        "actor_user": _check_actor_user_level_start(),
    }
    failed = [f"{group}.{name}" for group, checks in groups.items()
              for name, passed in checks.items() if not passed]
    assert not failed


def main() -> int:
    ok = True
    for group, checks in (
        ("common", _check_common()),
        ("single", _check_single()),
        ("scribe", _check_scribe()),
        ("builder_kickoff", _check_builder_kickoff()),
        ("builder_system", _check_builder_system()),
        ("actor_user", _check_actor_user_level_start()),
    ):
        print(f"=== {group} ===")
        for name, passed in checks.items():
            print(f"  {name}: {passed}")
            ok = ok and passed
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
