from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from socialclaw.memory.assets import ContentAddressedArtifactStore
from socialclaw.memory.models import MemoryKind, MemoryRecord
from socialclaw.schema.layered_graph import LayeredSchemaGraph
from socialclaw.schema.window_induction import (
    ARCVisualTransitionProfiler,
    ProposalApplier,
    ProposalOperation,
    ProposalValidator,
    SchemaProposal,
    VisualTransitionProfile,
    WindowSchemaInductionScheduler,
)


def transition(memory_id: str, *, changed: bool, level: int = 1) -> MemoryRecord:
    return MemoryRecord(
        id=memory_id,
        task="fixture-game",
        kind=MemoryKind.EPISODE,
        metadata={
            "memory_scope": "transition",
            "game_id": "fixture-game",
            "action_name": "ACTION1",
            "target_role": "",
            "task_state_changed": changed,
            "task_changed_cells": 4 if changed else 0,
            "level_delta": 0,
            "environment_status": "NOT_FINISHED",
            "level": level,
        },
    )


def window(memory_id: str, start: int, source_ids: list[str]) -> MemoryRecord:
    return MemoryRecord(
        id=memory_id,
        task="fixture-game",
        kind=MemoryKind.KNOWLEDGE,
        metadata={
            "memory_scope": "window_summary",
            "episode_id": memory_id,
            "start_step": start,
            "source_memory_ids": source_ids,
        },
    )


class FixtureProfiler:
    def profile(self, record: MemoryRecord) -> VisualTransitionProfile:
        effect = "effect" if record.metadata["task_state_changed"] else "no_effect"
        region = "center" if effect == "effect" else "none"
        return VisualTransitionProfile(
            semantic_key=f"fixture-game|ACTION1|none|{effect}|{region}|local",
            base_key="fixture-game|ACTION1|none",
            game_id="fixture-game",
            action_name="ACTION1",
            target_role="",
            effect_class=effect,
            region=region,
            change_scale="local" if effect == "effect" else "none",
            changed_cells=4 if effect == "effect" else 0,
            level=int(record.metadata["level"]),
            trigger=f"fixture {effect} trigger",
            expectation=f"fixture {effect} expectation",
            action_text="ACTION1",
        )


class FakeProposalGenerator:
    def propose(self, records, graph, *, keyframes=()):
        return [
            SchemaProposal(
                operation=ProposalOperation.SKIP,
                semantic_key="fake-generator",
                evidence_memory_ids=[item.id for item in records],
                rationale="offline_fake_generator",
                metadata={"game_id": "fixture-game"},
            )
        ]


class WindowSchemaInductionTests(unittest.TestCase):
    def test_scheduler_accepts_an_offline_fake_generator(self) -> None:
        item = transition("one", changed=True)
        memories = [item, window("window_1", 0, [item.id])]
        graph, report, audit, _ = WindowSchemaInductionScheduler(
            profiler=FixtureProfiler(),
            generator_factory=lambda support, profiles, minimum: FakeProposalGenerator(),
        ).run(memories)
        self.assertEqual(len(graph.list()), 0)
        self.assertEqual(report["proposal_counts"], {"skip": 1})
        self.assertEqual(audit[0]["proposal"]["rationale"], "offline_fake_generator")

    def test_scheduler_creates_supports_revises_and_deduplicates_keyframes(self) -> None:
        memories = [
            transition("effect_1", changed=True),
            transition("effect_2", changed=True),
            transition("noop_1", changed=False),
            transition("noop_2", changed=False),
            transition("effect_3", changed=True, level=2),
            window("window_1", 0, ["effect_1", "effect_2"]),
            window("window_2", 1, ["noop_1", "noop_2"]),
            window("window_3", 2, ["effect_3"]),
        ]
        graph, report, audit, keyframes = WindowSchemaInductionScheduler(
            profiler=FixtureProfiler(), min_support=2
        ).run(memories)
        self.assertEqual(report["schema_count"], 2)
        self.assertEqual(report["proposal_counts"]["create"], 2)
        self.assertEqual(report["proposal_counts"]["revise"], 1)
        self.assertEqual(report["proposal_counts"]["support"], 1)
        self.assertEqual(len(keyframes), 2)
        self.assertTrue(all(item["accepted"] for item in audit))
        effect = next(node for node in graph.list() if node.metadata["effect_class"] == "effect")
        self.assertEqual(effect.metadata["support_count"], 3)
        self.assertEqual(effect.metadata["level_scope"], [1, 2])
        self.assertEqual(effect.metadata["paired_effect_classes"], ["no_effect"])
        self.assertEqual(effect.metadata["keyframe_memory_ids"], ["effect_1"])
        self.assertEqual(effect.metadata["negative_keyframe_memory_ids"], ["noop_1"])
        self.assertEqual(effect.memory_index.negative, ["noop_1", "noop_2"])
        graph.validate(memory_ids={item.id for item in memories})

    def test_skip_and_rejected_ungrounded_proposal_leave_graph_unchanged(self) -> None:
        item = transition("only_one", changed=True)
        memories = [item, window("window_1", 0, [item.id])]
        graph, report, audit, _ = WindowSchemaInductionScheduler(
            profiler=FixtureProfiler(), min_support=2
        ).run(memories)
        self.assertEqual(report["proposal_counts"], {"skip": 1})
        self.assertEqual(len(graph.list()), 0)
        bad = SchemaProposal(
            operation=ProposalOperation.CREATE,
            semantic_key="bad",
            evidence_memory_ids=["missing"],
            trigger="trigger",
            action_sequence=["ACTION1"],
            expectation="expectation",
            metadata={"game_id": "fixture-game"},
        )
        self.assertFalse(ProposalValidator({item.id: item}).validate(bad, graph).accepted)
        self.assertEqual(audit[0]["decision_reason"], "explicit_skip")

    def test_contradiction_operation_records_negative_evidence(self) -> None:
        positive = transition("positive", changed=True)
        negative = transition("negative", changed=False)
        graph = LayeredSchemaGraph()
        create = SchemaProposal(
            operation=ProposalOperation.CREATE,
            semantic_key="effect-key",
            evidence_memory_ids=[positive.id],
            trigger="trigger",
            action_sequence=["ACTION1"],
            expectation="effect",
            metadata={"game_id": "fixture-game", "level": 1},
        )
        target = ProposalApplier().apply(create, graph)
        contradict = SchemaProposal(
            operation=ProposalOperation.CONTRADICT,
            semantic_key="effect-key",
            evidence_memory_ids=[negative.id],
            target_schema_id=target,
            rationale="fixture conflict",
            metadata={"game_id": "fixture-game"},
        )
        validator = ProposalValidator({positive.id: positive, negative.id: negative})
        self.assertTrue(validator.validate(contradict, graph).accepted)
        ProposalApplier().apply(contradict, graph)
        self.assertEqual(graph.get(target).memory_index.negative, [negative.id])

    def test_arc_visual_profiler_reads_content_addressed_grids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ContentAddressedArtifactStore(root / "assets")
            pre = np.zeros((64, 64), dtype=np.int16)
            post = pre.copy()
            post[35:37, 28:30] = 2
            pre_ref = store.put_grid(pre, role="environment_state")
            post_ref = store.put_grid(post, role="environment_state")
            record = transition("visual", changed=True)
            record.task = "cd82-fixture"
            record.metadata.update(
                {
                    "game_id": "cd82-fixture",
                    "corpus_root": str(root),
                    "pre_artifacts": [pre_ref.to_dict()],
                    "post_artifacts": [post_ref.to_dict()],
                }
            )
            profile = ARCVisualTransitionProfiler().profile(record)
            self.assertEqual(profile.region, "canvas")
            self.assertEqual(profile.change_scale, "local")
            self.assertEqual(profile.changed_cells, 4)


if __name__ == "__main__":
    unittest.main()
