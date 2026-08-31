from __future__ import annotations

import unittest
from dataclasses import dataclass
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from socialclaw.arc_runner import (
    _apply_level_feedback,
    _learn_transition,
    _parse_agent_action,
    run_arc_agi3,
)
from socialclaw.agent.base import AgentAttempt, ReasoningTrace, Usage
from socialclaw.memory import MemoryBank, MemoryStore
from socialclaw.schema import SchemaManager


@dataclass
class FakeAction:
    name: str


class ArcSchemaRunnerTests(unittest.TestCase):
    def test_action_parser_keeps_click_coordinates(self) -> None:
        actions = [FakeAction("ACTION1"), FakeAction("ACTION6")]
        chosen, data, trace = _parse_agent_action(
            '{"action":"ACTION6","data":{"x":4,"y":7},'
            '"reasoning":{"schemas_used":["schema_x"],"explanation":"click"}}',
            actions,
        )
        self.assertEqual(chosen.name, "ACTION6")
        self.assertEqual(data, {"x": 4, "y": 7})
        self.assertEqual(trace.concepts, ["schema_x"])

    def test_transition_becomes_memory_grounded_schema(self) -> None:
        manager = SchemaManager(memory=MemoryBank(MemoryStore()))
        memory_id, schema_id = _learn_transition(
            manager=manager,
            game_id="game-1",
            level=1,
            step=1,
            query="blue object and lever",
            observation="lever visible",
            action="ACTION1",
            result="grid_changed=True; object moved up",
            state="PLAYING",
        )
        self.assertIsNotNone(manager.memory.recall(memory_id))
        node = manager.graph.get(schema_id)
        self.assertIsNotNone(node)
        self.assertIn(memory_id, node.memory_index.source)
        self.assertIn(memory_id, node.memory_index.positive)

    def test_level_failure_weakens_used_schema_without_gold(self) -> None:
        manager = SchemaManager(memory=MemoryBank(MemoryStore()))
        _, schema_id = _learn_transition(
            manager=manager,
            game_id="game-1",
            level=1,
            step=1,
            query="lever",
            observation="lever visible",
            action="ACTION1",
            result="grid_changed=True",
            state="PLAYING",
        )
        before = manager.graph.get(schema_id).reliability_weight
        feedback_id = _apply_level_feedback(
            manager,
            schema_ids=[schema_id],
            game_id="game-1",
            level=1,
            outcome="GAME_OVER",
        )
        self.assertIsNotNone(feedback_id)
        self.assertLess(manager.graph.get(schema_id).reliability_weight, before)
        self.assertNotIn("WIN", manager.memory.recall(feedback_id).feedback)

    def test_migrated_runner_writes_layered_schema_artifacts(self) -> None:
        class Observation:
            def __init__(self, state, grid):
                self.state = state
                self.frame = [grid]
                self.win_levels = 1

            def is_empty(self):
                return False

        class FakeEnvironment:
            class GameState:
                WIN = "WIN"
                GAME_OVER = "GAME_OVER"

            def __init__(self, game_id, render_mode=None):
                self.playing = Observation("PLAYING", np.zeros((2, 2), dtype=int))

            def reset(self):
                return self.playing

            def get_available_actions(self, obs):
                return [FakeAction("ACTION1")]

            def step(self, action, data=None):
                return Observation("WIN", np.ones((2, 2), dtype=int))

            def get_scorecard(self):
                return None

            @staticmethod
            def grid_to_image(grid, cell_size=8):
                from PIL import Image

                return Image.new("RGB", (16, 16), "black")

        class FakeAgent:
            def answer(self, *, prompt, meta, response_format=None):
                return AgentAttempt(
                    answer_text=(
                        '{"action":"ACTION1","reasoning":{'
                        '"schemas_used":[],"explanation":"test"}}'
                    ),
                    reasoning_trace=ReasoningTrace([], [], "test"),
                    usage=Usage(1, 1, 2),
                    raw={},
                )

        with tempfile.TemporaryDirectory() as directory:
            with patch("socialclaw.arc_runner.ARCAGI3EnvWrapper", FakeEnvironment):
                run_dir = Path(
                    run_arc_agi3(
                        game_id="fake-game",
                        agent=FakeAgent(),
                        embedder=None,
                        schema_llm=None,
                        max_steps_per_level=1,
                        runs_dir=directory,
                        model="fake-model",
                    )
                )
            self.assertTrue((run_dir / "schema" / "memory.json").exists())
            self.assertTrue((run_dir / "schema" / "schema.json").exists())
            episode_path = next(run_dir.glob("*/episode.json"))
            payload = json.loads(episode_path.read_text(encoding="utf-8"))
            self.assertIn("layered_schema", payload["episode"]["flags"])


if __name__ == "__main__":
    unittest.main()
