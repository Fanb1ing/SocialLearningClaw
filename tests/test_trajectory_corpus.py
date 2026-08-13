from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from socialclaw.dataset.arc_agi3 import ARCAGI3EnvWrapper
from socialclaw.trajectory import (
    ARCRecordingSession,
    Action,
    EvidenceTier,
    arc_corpus_coverage,
    replay_arc_episode,
    validate_corpus,
    write_corpus_metadata,
)
from socialclaw.trajectory.arc_policies import plan_cd82_level


GAME_ID = "cd82-fb555c5d"


class ARCTrajectoryCorpusTests(unittest.TestCase):
    def test_visible_grid_policy_completes_all_six_levels(self) -> None:
        env = ARCAGI3EnvWrapper(GAME_ID)
        observation = env.reset()
        action_count = 0
        while observation.levels_completed < observation.win_levels:
            level = observation.levels_completed + 1
            for action in plan_cd82_level(observation.frame[-1], level=level):
                available = {
                    item.name: item for item in env.get_available_actions(observation)
                }
                data = {
                    key: value
                    for key, value in action.arguments.items()
                    if key != "target_role"
                }
                observation = env.step(available[action.name], data=data)
                action_count += 1
        self.assertEqual(str(observation.state.value), "WIN")
        self.assertEqual(observation.levels_completed, 6)
        self.assertEqual(action_count, 70)

    def test_real_public_transition_is_lossless_and_replayable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = ARCRecordingSession(
                game_id=GAME_ID,
                root=root,
                episode_id="real_blocked_action",
                actor="test_policy",
                evidence_tier=EvidenceTier.SOURCE_GUIDED_NATURAL,
                split="train",
                provenance={"public_interface_only": True, "network_calls": 0},
                metadata={"scenario_category": "test"},
            )
            step = session.execute(
                Action("ACTION1"),
                rationale="At the initial top position this legal direction is blocked.",
            )
            episode = session.finish(status="TIMEOUT", success=False)

            self.assertEqual(step.observation.structured["grid_shape"], [64, 64])
            self.assertTrue(step.result.state_delta["grid_changed"])
            self.assertFalse(step.result.state_delta["task_state_changed"])
            self.assertEqual(replay_arc_episode(root, episode)["status"], "passed")

    def test_manifest_splits_assets_and_coverage_validate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = ARCRecordingSession(
                game_id=GAME_ID,
                root=root,
                episode_id="coverage_case",
                actor="test_policy",
                evidence_tier=EvidenceTier.NATURAL,
                split="train",
                provenance={"public_interface_only": True},
                metadata={"scenario_category": "test", "pair_group": "pair"},
            )
            session.execute(Action("ACTION3"), rationale="One visible movement probe.")
            episode = session.finish(status="SCENARIO_COMPLETE", success=None)
            coverage = arc_corpus_coverage([episode])
            write_corpus_metadata(
                root,
                corpus_id="test_corpus",
                benchmark="arc_agi3",
                game_id=GAME_ID,
                environment_fingerprint="0" * 64,
                collector={"network_calls": 0},
                episodes=[episode],
                splits={"train": [episode.episode_id]},
                coverage=coverage,
            )

            result = validate_corpus(root)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(coverage["levels"], [1])
            self.assertEqual(coverage["evidence_tiers"], {"natural": 1})


if __name__ == "__main__":
    unittest.main()
