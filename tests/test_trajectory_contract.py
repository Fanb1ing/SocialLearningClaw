from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from socialclaw.trajectory import (
    Action,
    Decision,
    EpisodeFinished,
    EpisodeStarted,
    EvidenceTier,
    IterableTrajectorySource,
    Observation,
    StepResult,
    StepObserved,
    TrajectoryEpisode,
    TrajectoryOutcome,
    TrajectoryRecorder,
    TrajectoryStep,
)


def observation(position: int) -> Observation:
    return Observation(
        text=f"object at {position}",
        structured={"position": position},
    )


def step(index: int, pre: Observation, post: Observation) -> TrajectoryStep:
    return TrajectoryStep(
        step_index=index,
        observation=pre,
        available_actions=[Action("LEFT"), Action("RIGHT")],
        action=Action("RIGHT"),
        result=StepResult(
            observation=post,
            environment_status="PLAYING",
            state_delta={"position": [index, index + 1]},
        ),
        decision=Decision(
            response="RIGHT",
            rationale="scripted fixture",
            metadata={"policy": "move_right"},
        ),
    )


def episode(initial: Observation | None = None) -> TrajectoryEpisode:
    return TrajectoryEpisode(
        episode_id="fixture.arc.level-1",
        benchmark="fake_benchmark",
        task_id="fake-task-1",
        split="dev",
        actor="scripted_policy",
        evidence_tier=EvidenceTier.SYNTHETIC,
        initial_observation=initial or observation(0),
        provenance={"scenario_id": "two_moves", "seed": 7},
    )


class TrajectoryContractTests(unittest.TestCase):
    def test_multi_step_round_trip_and_source_are_generic(self) -> None:
        first = observation(0)
        second = observation(1)
        third = observation(2)
        with tempfile.TemporaryDirectory() as directory:
            recorder = TrajectoryRecorder(directory, episode(first))
            recorder.record_step(step(0, first, second))
            recorder.record_step(step(1, second, third))
            final = recorder.finalize(
                TrajectoryOutcome(status="WIN", success=True, reward=1.0)
            )
            loaded = TrajectoryRecorder.load(recorder.path)
            raw = json.loads(recorder.path.read_text(encoding="utf-8"))

        self.assertEqual(raw["format_version"], 1)
        self.assertEqual(len(final.steps), 2)
        self.assertEqual(loaded.to_dict(), final.to_dict())
        self.assertEqual(loaded.terminal_outcome.status, "WIN")
        self.assertEqual(
            [item.episode_id for item in IterableTrajectorySource([loaded]).episodes()],
            [loaded.episode_id],
        )
        events = list(IterableTrajectorySource([loaded]).events())
        self.assertIsInstance(events[0], EpisodeStarted)
        self.assertEqual(sum(isinstance(item, StepObserved) for item in events), 2)
        self.assertIsInstance(events[-1], EpisodeFinished)

    def test_single_step_episode_uses_the_same_contract(self) -> None:
        first = Observation(text="What is 1+1?")
        second = Observation(text="Evaluation complete", structured={"correct": True})
        one_step = TrajectoryEpisode(
            episode_id="fixture.static.sample-1",
            benchmark="fake_static",
            task_id="sample-1",
            split="test",
            actor="llm_agent",
            evidence_tier=EvidenceTier.NATURAL,
            initial_observation=first,
            steps=[
                TrajectoryStep(
                    step_index=0,
                    observation=first,
                    available_actions=[],
                    action=Action("ANSWER", {"text": "2"}),
                    result=StepResult(second, "COMPLETE", {"correct": True}),
                )
            ],
            terminal_outcome=TrajectoryOutcome("COMPLETE", success=True),
        )
        self.assertEqual(len(one_step.steps), 1)
        self.assertEqual(one_step.steps[0].action.name, "ANSWER")

    def test_continuity_error_does_not_change_persisted_episode(self) -> None:
        first = observation(0)
        second = observation(1)
        wrong = observation(99)
        with tempfile.TemporaryDirectory() as directory:
            recorder = TrajectoryRecorder(directory, episode(first))
            recorder.record_step(step(0, first, second))
            with self.assertRaisesRegex(ValueError, "continuity mismatch"):
                recorder.record_step(step(1, wrong, observation(100)))
            loaded = TrajectoryRecorder.load(recorder.path)

        self.assertEqual(len(loaded.steps), 1)
        self.assertEqual(loaded.steps[0].result.observation.structured["position"], 1)

    def test_failed_atomic_replace_keeps_previous_snapshot(self) -> None:
        first = observation(0)
        second = observation(1)
        with tempfile.TemporaryDirectory() as directory:
            recorder = TrajectoryRecorder(directory, episode(first))
            with patch(
                "socialclaw.trajectory.recorder.os.replace",
                side_effect=OSError("disk unavailable"),
            ):
                with self.assertRaisesRegex(OSError, "disk unavailable"):
                    recorder.record_step(step(0, first, second))
            loaded = TrajectoryRecorder.load(recorder.path)
            temporary_files = list(recorder.path.parent.glob("*.tmp"))

        self.assertEqual(recorder.episode.steps, [])
        self.assertEqual(loaded.steps, [])
        self.assertEqual(temporary_files, [])

    def test_resume_rejects_a_different_episode_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = episode()
            TrajectoryRecorder(directory, first)
            different = TrajectoryEpisode(
                episode_id=first.episode_id,
                benchmark=first.benchmark,
                task_id="different-task",
                split=first.split,
                actor=first.actor,
                evidence_tier=first.evidence_tier,
                initial_observation=first.initial_observation,
            )
            with self.assertRaisesRegex(ValueError, "mismatched fields"):
                TrajectoryRecorder(directory, different, resume=True)

    def test_unavailable_action_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "not in available actions"):
            TrajectoryStep(
                step_index=0,
                observation=observation(0),
                available_actions=[Action("LEFT")],
                action=Action("RIGHT"),
                result=StepResult(observation(1), "PLAYING"),
            )


if __name__ == "__main__":
    unittest.main()
