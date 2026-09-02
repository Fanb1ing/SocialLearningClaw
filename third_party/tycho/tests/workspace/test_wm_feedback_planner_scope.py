"""Auto-feedback planner should stay usable for the current level.

Prior-level outcome mistakes should remain visible as warnings, but they should not suppress a
current-level planner probe when current-level dynamics are verified and current-level outcome evidence is
not falsified.
"""

from __future__ import annotations

from textwrap import dedent

from tycho.workspace.agent_tools import ToolExecutor
from tycho.workspace.sandbox import PythonSandbox
from tycho.workspace.workspace import GameWorkspace


WORLD_MODEL = dedent(
    """
    from dataclasses import dataclass
    import numpy as np

    @dataclass
    class State:
        level: int
        pos: int

    def init_state(grid, level):
        return State(level=int(level), pos=int(np.asarray(grid)[0, 0]))

    def actions(state):
        return [{"action": "ACTION1", "row": None, "col": None}]

    def transition(state, action):
        return State(level=state.level, pos=state.pos + 1)

    def render(state):
        return np.asarray([[state.pos]], dtype=np.int16)

    def outcome(state):
        if state.level == 0:
            return "ongoing"  # deliberately misses the already observed level-0 terminal
        return "level_complete" if state.pos >= 2 else "ongoing"
    """
).strip()


def test_auto_feedback_planner_runs_when_only_prior_level_outcome_failed(tmp_path, monkeypatch) -> None:
    ws = GameWorkspace(
        "planner-scope",
        root=str(tmp_path),
        render=False,
        available_actions=["ACTION1"],
    )
    ws.record([[0]], level=0, turn_in_level=0, available=["ACTION1"])
    ws.record([[1]], level=0, turn_in_level=1, action="ACTION1", available=["ACTION1"])
    ws.record_terminal(level=0, terminal_grid=[[2]], action="ACTION1")
    ws.record_solved(level=0, action="ACTION1")
    ws.new_level()
    ws.record([[0]], level=1, turn_in_level=0, available=["ACTION1"])
    ws.record([[1]], level=1, turn_in_level=1, action="ACTION1", available=["ACTION1"])

    feedback = ToolExecutor(
        ws,
        python_sandbox=PythonSandbox(runtime="host"),
    ).execute("write_file", {"path": "world_model.py", "content": WORLD_MODEL})

    checks = {
        "global_outcome_failure_visible": "OUTCOME_STATUS: fail" in feedback,
        "current_outcome_not_falsified": "CURRENT_LEVEL_OUTCOME_STATUS: unobserved" in feedback,
        "current_level_focus": "NEXT_STEP_FOCUS: plan_or_act_on_current_level" in feedback,
        "prior_warning_present": "PRIOR_LEVEL_WARNINGS:" in feedback,
        "planner_still_runs": "PLANNER_PROBE:" in feedback and "PLANNER_STATUS: plan_found" in feedback,
        "planner_first_action": "PLANNER_FIRST_ACTION: ACTION1" in feedback,
    }
    assert all(checks.values()), {"checks": checks, "feedback": feedback}
