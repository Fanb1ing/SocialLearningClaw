from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from socialclaw.memory import JsonMemoryStore, MemoryKind
from socialclaw.memory.assets import MemoryArtifactRef
from socialclaw.schema.trajectory_pipeline import (
    PrototypeTrajectorySchemaInducer,
    TrajectoryMemoryProjector,
)
from socialclaw.trajectory.models import (
    Action,
    EvidenceTier,
    Observation,
    StepResult,
    TrajectoryEpisode,
    TrajectoryOutcome,
    TrajectoryStep,
)


def observation(grid_id: str, *, level: int) -> Observation:
    ref = MemoryArtifactRef(
        artifact_id=grid_id,
        role="environment_state",
        media_type="application/x.numpy.ndarray",
        relative_path=f"grids/{grid_id}.npy",
        sha256="0" * 64,
        size_bytes=1,
        metadata={"logical_sha256": "1" * 64},
    )
    return Observation(structured={"level": level, "logical_grid_sha256": grid_id}, artifacts=[ref])


def episode(episode_id: str, changed: bool) -> TrajectoryEpisode:
    pre = observation(f"{episode_id}_pre", level=1)
    post = observation(f"{episode_id}_post", level=1)
    step = TrajectoryStep(
        step_index=0,
        observation=pre,
        available_actions=[Action("ACTION1")],
        action=Action("ACTION1"),
        result=StepResult(
            observation=post,
            environment_status="NOT_FINISHED",
            state_delta={
                "task_state_changed": changed,
                "task_changed_cells": 3 if changed else 0,
                "level_delta": 0,
            },
        ),
    )
    return TrajectoryEpisode(
        episode_id=episode_id,
        benchmark="arc_agi3",
        task_id="example-game",
        split="train",
        actor="fixture",
        evidence_tier=EvidenceTier.NATURAL,
        initial_observation=pre,
        steps=[step],
        terminal_outcome=TrajectoryOutcome(status="TIMEOUT", success=False),
    )


class TrajectorySchemaPipelineTests(unittest.TestCase):
    def test_projection_is_deterministic_and_preserves_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = JsonMemoryStore(Path(directory) / "memory.json")
            projector = TrajectoryMemoryProjector(store, window_size=8)
            first = projector.project([episode("one", True), episode("two", True)])
            ids = {item.id for item in store.list()}
            second = projector.project([episode("one", True), episode("two", True)])
            self.assertEqual(ids, {item.id for item in store.list()})
            self.assertEqual(first, second)
            transitions = [
                item for item in store.list() if item.metadata["memory_scope"] == "transition"
            ]
            windows = [
                item for item in store.list() if item.metadata["memory_scope"] == "window_summary"
            ]
            self.assertEqual(len(transitions), 2)
            self.assertEqual(windows[0].kind, MemoryKind.KNOWLEDGE)
            self.assertTrue(set(windows[0].metadata["source_memory_ids"]) <= ids)
            self.assertTrue(transitions[0].metadata["pre_artifacts"])

    def test_same_episode_id_in_two_games_does_not_collide(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = JsonMemoryStore(Path(directory) / "memory.json")
            first = episode("shared_id", True)
            second = replace(episode("shared_id", True), task_id="second-game")
            report = TrajectoryMemoryProjector(store).project([first, second])
            self.assertEqual(report.transition_memory_count, 2)
            self.assertEqual(len(store), 6)
            self.assertEqual(len({item.id for item in store.list()}), 6)

    def test_inducer_requires_repeated_memory_grounded_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = JsonMemoryStore(Path(directory) / "memory.json")
            TrajectoryMemoryProjector(store).project(
                [episode("effect_one", True), episode("effect_two", True), episode("noop", False)]
            )
            graph = PrototypeTrajectorySchemaInducer(min_support=2).induce(store.list())
            nodes = graph.list()
            self.assertEqual(len(nodes), 1)
            self.assertEqual(nodes[0].metadata["effect_class"], "effect")
            self.assertEqual(nodes[0].metadata["support_count"], 2)
            self.assertEqual(len(nodes[0].memory_index.source), 2)
            memory_ids = {item.id for item in store.list()}
            self.assertTrue(set(nodes[0].memory_index.source) <= memory_ids)


if __name__ == "__main__":
    unittest.main()
