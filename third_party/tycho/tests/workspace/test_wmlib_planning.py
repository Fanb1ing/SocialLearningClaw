"""wmlib planner contract tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tycho.workspace import wmlib_template as wmlib


@dataclass
class MutatingState:
    value: int = 0
    history: list[str] = field(default_factory=list)


class MutatingBranchModel:
    @staticmethod
    def actions(state):
        return [{"action": "bad"}, {"action": "good"}]

    @staticmethod
    def transition(state, action):
        # The prompt-visible contract allows transition() to mutate and return state.
        # Planner sibling branches must therefore receive cloned states.
        state.history.append(action["action"])
        if action["action"] == "good" and state.history == ["good"]:
            state.value = 1
        return state

    @staticmethod
    def outcome(state):
        return "level_complete" if state.value == 1 else "ongoing"


class GameOverReportModel:
    @staticmethod
    def actions(state):
        return (
            [{"action": "ACTION1", "row": None, "col": None}]
            + [{"action": "ACTION6", "row": 2, "col": c} for c in range(3, 6)]
        )

    @staticmethod
    def transition(state, action):
        state.history.append(action["action"])
        if action["action"] == "ACTION6":
            state.value = 2
        return state

    @staticmethod
    def outcome(state):
        if state.value == 1:
            return "level_complete"
        if state.value == 2:
            return "game_over"
        return "ongoing"


@dataclass
class LinearState:
    pos: int = 0


@dataclass
class ArrayState:
    grid: np.ndarray
    pos: int = 0


def reach_one(state):
    return state.pos >= 1


def reach_three(state):
    return state.pos >= 3


def impossible_goal(state):
    return state.pos >= 10


class LinearSubgoalModel:
    @staticmethod
    def actions(state):
        return [{"action": "INC"}]

    @staticmethod
    def transition(state, action):
        return LinearState(pos=state.pos + 1)

    @staticmethod
    def outcome(state):
        return "level_complete" if state.pos >= 3 else "ongoing"


class DeadEndModel:
    @staticmethod
    def actions(state):
        return []

    @staticmethod
    def transition(state, action):
        return state

    @staticmethod
    def outcome(state):
        return "ongoing"


class ArrayStateModel:
    @staticmethod
    def actions(state):
        return [{"action": "INC"}]

    @staticmethod
    def transition(state, action):
        grid = state.grid.copy()
        pos = state.pos + 1
        grid[0, 0] = pos
        return ArrayState(grid=grid, pos=pos)

    @staticmethod
    def render(state):
        return state.grid

    @staticmethod
    def outcome(state):
        return "level_complete" if state.pos >= 2 else "ongoing"


class ProjectedPureModel:
    TRANSITION_RETURNS_NEW_STATE = True
    key_calls = 0
    transition_inputs = []

    @classmethod
    def planner_key(cls, state):
        cls.key_calls += 1
        return state.pos

    @staticmethod
    def actions(state):
        return [{"action": "INC"}]

    @classmethod
    def transition(cls, state, action):
        cls.transition_inputs.append(state)
        grid = state.grid.copy()
        pos = state.pos + 1
        grid[0, 0] = pos
        return ArrayState(grid=grid, pos=pos)

    @staticmethod
    def render(state):
        return state.grid

    @staticmethod
    def outcome(state):
        return "level_complete" if state.pos >= 2 else "ongoing"


class BareStringActionsModel:
    @staticmethod
    def actions(state):
        return ["INC"]

    @staticmethod
    def transition(state, action):
        return LinearState(pos=state.pos + (1 if action["action"] == "INC" else 0))

    @staticmethod
    def outcome(state):
        return "level_complete" if state.pos >= 1 else "ongoing"


def _check_planner_clones_mutating_states() -> dict[str, bool]:
    start = MutatingState()
    bfs = wmlib.plan_bfs(MutatingBranchModel, start, max_nodes=10)
    astar = wmlib.plan_astar(MutatingBranchModel, start, max_nodes=10)
    return {
        "bfs_finds_good_branch": bfs == [{"action": "good"}],
        "astar_finds_good_branch": astar == [{"action": "good"}],
        "start_state_not_mutated": start.value == 0 and start.history == [],
    }


def _check_planner_hashes_numpy_state_fields() -> dict[str, bool]:
    start = ArrayState(grid=np.zeros((2, 2), dtype=np.int16))
    bfs_diag = wmlib.plan_bfs_with_diagnostics(ArrayStateModel, start, max_nodes=10)
    astar_diag = wmlib.plan_astar_with_diagnostics(ArrayStateModel, start, max_nodes=10)
    expected = [{"action": "INC", "row": None, "col": None}] * 2
    return {
        "bfs_accepts_numpy_state": bfs_diag["status"] == "found",
        "astar_accepts_numpy_state": astar_diag["status"] == "found",
        "bfs_plan_correct": bfs_diag["plan"] == expected,
        "astar_seen_states": astar_diag["seen_states"] >= 3,
    }


def test_planner_hashes_numpy_state_fields() -> None:
    checks = _check_planner_hashes_numpy_state_fields()
    assert all(checks.values()), checks


def test_planner_uses_semantic_key_and_pure_transition_contract() -> None:
    start = ArrayState(grid=np.zeros((64, 64), dtype=np.int16))
    ProjectedPureModel.key_calls = 0
    ProjectedPureModel.transition_inputs = []

    diag = wmlib.plan_astar_with_diagnostics(ProjectedPureModel, start, max_nodes=10)

    assert diag["status"] == "found"
    assert diag["plan"] == [{"action": "INC", "row": None, "col": None}] * 2
    assert ProjectedPureModel.key_calls > 0
    assert ProjectedPureModel.transition_inputs[0] is start
    assert start.pos == 0
    assert np.count_nonzero(start.grid) == 0


def test_canonical_plan_validation_accepts_only_reaching_plan() -> None:
    start = ArrayState(grid=np.zeros((2, 2), dtype=np.int16))
    good = [{"action": "INC"}] * 2
    short = [{"action": "INC"}]

    accepted = wmlib.validate_plan(
        ArrayStateModel, start, good, capture_renders=True
    )
    rejected = wmlib.validate_plan(ArrayStateModel, start, short)

    assert accepted["status"] == "found"
    assert accepted["final_outcome"] == "level_complete"
    assert accepted["steps_validated"] == 2
    assert len(accepted["renders"]) == 2
    assert rejected["status"] == "goal_not_reached"
    assert rejected["plan"] is None
    assert start.pos == 0


def _check_planner_reports_bare_string_actions() -> dict[str, bool]:
    diag = wmlib.plan_astar_with_diagnostics(BareStringActionsModel, LinearState(), max_nodes=10)
    line = wmlib.format_game_over_action_report(
        wmlib.game_over_action_report(BareStringActionsModel, state=LinearState())
    )
    return {
        "status_is_actions_error": diag["status"] == "actions_error",
        "error_names_contract": "actions(state) must return full action dicts" in diag["errors"][0],
        "not_silent_exhaustion": diag["status"] != "no_plan_exhausted",
        "game_over_report_mentions_error": "actions() item 0" in line,
    }


def test_planner_reports_bare_string_actions() -> None:
    checks = _check_planner_reports_bare_string_actions()
    assert all(checks.values()), checks


def _check_game_over_action_report_compacts_clicks() -> dict[str, bool]:
    report = wmlib.game_over_action_report(GameOverReportModel, state=MutatingState())
    line = wmlib.format_game_over_action_report(report)
    return {
        "fatal_count_detected": report["checked"] == 4 and report["fatal"] == 3,
        "click_cells_grouped": "ACTION6 3 cell(s): row 2, cols 3-5" in line,
        "not_harness_inferred": "MODEL_GAME_OVER_ACTIONS:" in line,
    }


def _check_subgoal_diagnostics() -> dict[str, bool]:
    success = wmlib.plan_subgoals_with_diagnostics(
        LinearSubgoalModel,
        LinearState(),
        [reach_one, reach_three],
        max_nodes=20,
    )
    failed = wmlib.plan_subgoals_with_diagnostics(
        LinearSubgoalModel,
        LinearState(),
        [reach_one, impossible_goal],
        max_nodes=3,
    )
    expected = [{"action": "INC", "row": None, "col": None}] * 3
    return {
        "success_status": success["status"] == "found",
        "success_plan_preserved": success["plan"] == expected,
        "success_leg_lengths": success["leg_lengths"] == [1, 2],
        "success_labels": success["subgoal_labels"] == ["reach_one", "reach_three"],
        "legacy_api_returns_plan": wmlib.plan_subgoals(
            LinearSubgoalModel, LinearState(), [reach_one, reach_three], max_nodes=20
        ) == expected,
        "failure_status": failed["status"] == "failed_at_subgoal",
        "failure_index_is_second": failed["failed_ordinal"] == 2,
        "failure_prefix_len": failed["completed_prefix_len"] == 1,
        "failure_label": failed["failed_label"] == "impossible_goal",
        "legacy_api_returns_none": wmlib.plan_subgoals(
            LinearSubgoalModel, LinearState(), [reach_one, impossible_goal], max_nodes=3
        ) is None,
    }


def test_subgoal_diagnostics_report_success_and_failure() -> None:
    checks = _check_subgoal_diagnostics()
    assert all(checks.values()), checks


def _check_no_plan_diagnostics() -> dict[str, bool]:
    exhausted_bfs = wmlib.plan_bfs_with_diagnostics(DeadEndModel, LinearState(), max_nodes=20)
    exhausted_astar = wmlib.plan_astar_with_diagnostics(DeadEndModel, LinearState(), max_nodes=20)
    budget_astar = wmlib.plan_astar_with_diagnostics(
        LinearSubgoalModel,
        LinearState(),
        target=impossible_goal,
        max_nodes=3,
    )
    return {
        "bfs_exhausted_status": exhausted_bfs["status"] == "no_plan_exhausted",
        "astar_exhausted_status": exhausted_astar["status"] == "no_plan_exhausted",
        "exhausted_frontier_empty": exhausted_astar["frontier_remaining"] is False,
        "budget_status": budget_astar["status"] == "no_plan_within_budget",
        "budget_frontier_remaining": budget_astar["frontier_remaining"] is True,
        "budget_expanded_nodes": budget_astar["expanded_nodes"] == 3,
        "legacy_bfs_still_returns_none": wmlib.plan_bfs(DeadEndModel, LinearState(), max_nodes=20) is None,
        "legacy_astar_still_returns_none": wmlib.plan_astar(DeadEndModel, LinearState(), max_nodes=20) is None,
    }


def test_no_plan_diagnostics_distinguish_exhausted_from_budget() -> None:
    checks = _check_no_plan_diagnostics()
    assert all(checks.values()), checks


def _check_plan_templates_explain_grounded_no_plan() -> dict[str, bool]:
    root = Path(__file__).resolve().parents[2]
    seed_plan = (root / "tycho/workspace/templates/seed_plan.py.tmpl").read_text()
    feedback_probe = (root / "tycho/workspace/templates/wm_feedback_probe.py.tmpl").read_text()
    return {
        "seed_plan_mentions_unseen_state": (
            "target/route/off-screen state is not fully grounded" in seed_plan
            and "unseen or unencoded target/route/off-screen" in seed_plan
        ),
        "auto_feedback_mentions_unseen_state": (
            "target/route/off-screen state is not fully grounded" in feedback_probe
            and "unseen or unencoded target/route/off-screen" in feedback_probe
        ),
        "seed_plan_canonical_replays_and_writes_artifact": (
            "wmlib.validate_plan(" in seed_plan
            and "notes/validated_plan.json" in seed_plan
            and "canonical_replay=pass" in seed_plan
        ),
        "seed_plan_supports_custom_compact_planner": (
            'os.path.exists("planner.py")' in seed_plan
            and "plan(world_model, start_state)" in seed_plan
        ),
    }


def test_auto_planner_timeout_is_control_flow_not_model_error() -> None:
    root = Path(__file__).resolve().parents[2]
    feedback_probe = (root / "tycho/workspace/templates/wm_feedback_probe.py.tmpl").read_text()
    assert "class PlannerTimeout(BaseException)" in feedback_probe

    class PlannerCancellation(BaseException):
        pass

    class CancelledModel:
        @staticmethod
        def actions(_state):
            raise PlannerCancellation()

    try:
        wmlib.plan_astar_with_diagnostics(CancelledModel, LinearState(), max_nodes=2)
    except PlannerCancellation:
        pass
    else:
        raise AssertionError("planner cancellation was mislabeled as an actions() model error")


def test_plan_templates_explain_grounded_no_plan() -> None:
    checks = _check_plan_templates_explain_grounded_no_plan()
    assert all(checks.values()), checks


def main() -> int:
    ok = True
    for group, checks in (
        ("planner_clones_mutating_states", _check_planner_clones_mutating_states()),
        ("planner_hashes_numpy_state_fields", _check_planner_hashes_numpy_state_fields()),
        ("planner_reports_bare_string_actions", _check_planner_reports_bare_string_actions()),
        ("game_over_action_report", _check_game_over_action_report_compacts_clicks()),
        ("subgoal_diagnostics", _check_subgoal_diagnostics()),
        ("no_plan_diagnostics", _check_no_plan_diagnostics()),
        ("plan_templates_grounded_no_plan", _check_plan_templates_explain_grounded_no_plan()),
    ):
        print(f"=== {group} ===")
        for name, passed in checks.items():
            print(f"  {name}: {passed}")
            ok = ok and passed
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
