from __future__ import annotations

import hashlib
import json
from textwrap import dedent

import numpy as np

from tycho.workspace.sandbox import PythonSandbox
from tycho.workspace.workspace import GameWorkspace


WORLD_MODEL = dedent(
    """
    from dataclasses import dataclass
    import numpy as np

    TRANSITION_RETURNS_NEW_STATE = True

    @dataclass
    class State:
        pos: int

    def init_state(grid, level):
        return State(pos=int(np.asarray(grid)[0, 0]))

    def actions(state):
        return [{"action": "ACTION1", "row": None, "col": None}]

    def transition(state, action):
        return State(pos=state.pos + 1)

    def render(state):
        return np.asarray([[state.pos]], dtype=np.int16)

    def outcome(state):
        return "level_complete" if state.pos >= 2 else "ongoing"

    def heuristic(state):
        return max(0, 2 - state.pos)
    """
).strip()


def test_plan_script_writes_start_anchored_canonical_artifact(tmp_path) -> None:
    ws = GameWorkspace(
        "validated-plan",
        root=str(tmp_path),
        render=False,
        available_actions=["ACTION1"],
    )
    ws.record([[0]], level=0, turn_in_level=0, available=["ACTION1"])
    ws.write_file("world_model.py", WORLD_MODEL)

    result = PythonSandbox(runtime="host").run_script(
        ws.dir,
        ws.dir / "plan.py",
        timeout=10,
        args=("auto",),
    )

    assert result.returncode == 0, result.stderr
    assert "PLANNER_VALIDATION: canonical_replay=pass" in result.stdout
    assert "PLANNER_PLAN_LEN: 2" in result.stdout
    artifact = json.loads((ws.dir / "notes" / "validated_plan.json").read_text())
    assert artifact["status"] == "validated"
    assert artifact["method"] == "astar"
    assert artifact["start"]["level"] == 0
    assert artifact["start"]["turn"] == 0
    expected_grid_hash = hashlib.sha256(
        np.asarray([[0]], dtype=np.int16).tobytes()
    ).hexdigest()
    assert artifact["start"]["grid_sha256"] == expected_grid_hash
    assert artifact["plan_length"] == 2
    assert artifact["actions"] == [
        {"action": "ACTION1", "row": None, "col": None},
        {"action": "ACTION1", "row": None, "col": None},
    ]

    assert artifact["expected_grid_sha256"] == [
        hashlib.sha256(np.asarray([[1]], dtype=np.int16).tobytes()).hexdigest(),
        hashlib.sha256(np.asarray([[2]], dtype=np.int16).tobytes()).hexdigest(),
    ]

    start_hint = ws.validated_plan_hint(
        [[0]], level=0, turn=0, available=["ACTION1"]
    )
    assert "matches 0/2 action(s); next action is ACTION1" in start_hint
    continuation = ws.validated_plan_hint(
        [[1]], level=0, turn=1, available=["ACTION1"]
    )
    assert "matches 1/2 action(s); next action is ACTION1" in continuation
    assert "diverged after 1/2" in ws.validated_plan_hint(
        [[9]], level=0, turn=1, available=["ACTION1"]
    )
    assert "matched all 2 action(s)" in ws.validated_plan_hint(
        [[2]], level=0, turn=2, available=["ACTION1"]
    )

    ws.write_file("world_model.py", WORLD_MODEL + "\n# changed after validation\n")
    assert "world_model.py changed" in ws.validated_plan_hint(
        [[1]], level=0, turn=1, available=["ACTION1"]
    )
