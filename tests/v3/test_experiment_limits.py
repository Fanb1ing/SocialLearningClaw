from __future__ import annotations

from types import SimpleNamespace

from arcengine import GameAction, GameState

from tycho.harness.agent import ActionChoice
from tycho.harness.harness import run_env
from tycho.harness.run_spec import build_run_spec

from socialclaw.v3.agent import EFPSTychoAgent
from socialclaw.v3.workspace import EFPSGameWorkspace


def _frame(*, levels: int, value: int, state=GameState.NOT_FINISHED):
    return SimpleNamespace(
        frame=[[[value]]],
        levels_completed=levels,
        state=state,
        available_actions=[GameAction.ACTION1],
    )


class _Agent:
    calls = 0
    max_calls = 100
    builder = None

    def reset(self, game_id, available_actions):
        self.calls = 0

    def is_done(self, frames, latest_frame):
        return False

    def choose_action(self, frames, latest_frame, available_actions):
        self.calls += 1
        return ActionChoice(GameAction.ACTION1)


class _Arcade:
    def __init__(self, env):
        self.env = env

    def make(self, game_id, seed=0, scorecard_id=None):
        return self.env


def test_explicit_action_limit_stops_after_exactly_five_actions():
    class NeverCompletes:
        def __init__(self):
            self.actions = 0

        def reset(self):
            return _frame(levels=0, value=0)

        def step(self, action, data=None, reasoning=None):
            self.actions += 1
            return _frame(levels=0, value=self.actions % 2)

    env = NeverCompletes()
    record = run_env(
        _Agent(), _Arcade(env), "bounded-game", [10],
        max_actions_per_level=5,
    )

    assert env.actions == 5
    assert record.total_actions_including_unfinished == 5
    assert record.stop_reason == "requested_action_limit"
    assert record.truncated_levels == [1]


def test_level_limit_does_not_enter_level_two():
    class CompletesEveryAction:
        def __init__(self):
            self.actions = 0

        def reset(self):
            return _frame(levels=0, value=0)

        def step(self, action, data=None, reasoning=None):
            self.actions += 1
            return _frame(levels=self.actions, value=self.actions)

    env = CompletesEveryAction()
    record = run_env(
        _Agent(), _Arcade(env), "bounded-game", [1, 1, 1],
        stop_after_levels=1,
    )

    assert env.actions == 1
    assert record.levels_completed == 1
    assert record.stop_reason == "requested_level_limit"


def test_experiment_limits_are_part_of_immutable_run_identity(tmp_path):
    common = dict(
        repo=tmp_path,
        approach="tycho_efps",
        games={"cd82-test": [1]},
        seed=0,
        viz=True,
        operation_mode="offline",
        resolved_config={},
        git_version="test",
    )
    five = build_run_spec(
        **common,
        experiment_limits={"max_actions_per_level": 5, "stop_after_levels": 1},
    )
    six = build_run_spec(
        **common,
        experiment_limits={"max_actions_per_level": 6, "stop_after_levels": 1},
    )

    assert five["fingerprint"] != six["fingerprint"]
    assert five["policy"]["experiment_limits"]["max_actions_per_level"] == 5


def test_efps_records_last_nonterminal_outcome_at_action_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("SC_V3_RUN_ID", "bounded-test")
    workspace = EFPSGameWorkspace(
        "bounded-game",
        root=str(tmp_path),
        render=False,
        available_actions=[GameAction.ACTION1],
    )
    agent = object.__new__(EFPSTychoAgent)
    agent.ws = workspace
    agent.level = 0
    agent.turn_in_level = 5
    agent._last_action = "ACTION1"
    agent._last_row = None
    agent._last_col = None

    agent.note_final_observation(
        _frame(levels=0, value=9),
        [GameAction.ACTION1],
    )

    metadata = __import__("json").loads(
        (workspace.dir / "level_0" / "turn_005.json").read_text()
    )
    evidence = __import__("json").loads(
        (workspace.dir / "notes" / "evidence_index.json").read_text()
    )["evidence"]
    assert metadata["action"] == "ACTION1"
    assert any(item["turn"] == 5 and item["action"]["action"] == "ACTION1" for item in evidence)
