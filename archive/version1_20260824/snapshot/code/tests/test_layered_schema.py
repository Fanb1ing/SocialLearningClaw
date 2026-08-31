from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from socialclaw.memory import JsonMemoryStore, MemoryBank, MemoryRecord, MemoryStore
from socialclaw.schema import (
    LayeredSchemaGraph,
    LayeredSchemaStorage,
    LLMSchemaGenerator,
    MemoryIndex,
    SchemaManagementConfig,
    SchemaManager,
    SchemaNode,
    SchemaStatus,
    build_schema_system,
)


def episode(*, memory_id: str = "memory_1", success: bool = True) -> MemoryRecord:
    record = MemoryRecord(
        id=memory_id,
        task="Move the lever",
        context="The left lever is visible",
        outcome="The object moves upward",
        success=success,
        metadata={"schema_level": 2},
    )
    record.add_event(
        observation="Left lever at rest",
        action="ACTION1",
        result="Object moved upward",
    )
    return record


class MemoryLayerTests(unittest.TestCase):
    def test_json_memory_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.json"
            first = JsonMemoryStore(path)
            first.put(episode())
            second = JsonMemoryStore(path)
            loaded = second.get("memory_1")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.events[0].action, "ACTION1")

    def test_lexical_memory_retrieval(self) -> None:
        bank = MemoryBank(MemoryStore([episode()]))
        matches = bank.search("lever ACTION1", top_k=1)
        self.assertEqual(matches[0].record.id, "memory_1")
        self.assertGreater(matches[0].score, 0)


class LayeredGraphTests(unittest.TestCase):
    def test_layer_edges_are_symmetric_and_validated(self) -> None:
        parent = SchemaNode(index="general", level=0, description="General exploration")
        child = SchemaNode(index="specific", level=2, description="Specific lever rule")
        graph = LayeredSchemaGraph([parent, child])
        graph.connect_parent_child(
            "general", "specific", evidence_memory_id="memory_1"
        )
        graph.validate(memory_ids={"memory_1"})
        self.assertEqual(parent.related_schema_index.children, ["specific"])
        self.assertEqual(child.related_schema_index.parents, ["general"])
        self.assertEqual(
            child.related_schema_index.evidence["general"], ["memory_1"]
        )
        with self.assertRaises(ValueError):
            graph.connect_parent_child("specific", "general")

    def test_schema_storage_round_trip(self) -> None:
        node = SchemaNode.from_rule(
            level=2,
            trigger="lever visible",
            action_sequence=["ACTION1"],
            expectation="object moves upward",
            source_memory_id="memory_1",
        )
        graph = LayeredSchemaGraph([node])
        with tempfile.TemporaryDirectory() as directory:
            storage = LayeredSchemaStorage(Path(directory) / "schema.json")
            storage.save(graph)
            loaded = storage.load()
        self.assertEqual(loaded.get(node.index).memory_index.source, ["memory_1"])


class SchemaManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.record = episode()
        self.bank = MemoryBank(MemoryStore([self.record]))
        self.manager = SchemaManager(memory=self.bank)

    def test_memory_is_ground_truth_for_schema_generation(self) -> None:
        node = self.manager.learn(self.record.id)
        self.assertIsNotNone(node)
        self.assertEqual(node.level, 2)
        self.assertEqual(node.memory_index.source, [self.record.id])
        self.assertIn("Context/Perception", node.description)
        context = self.manager.context_block("lever ACTION1")
        self.assertIn(node.index, context)
        self.assertIn("reliability=", context)

    def test_feedback_updates_evidence_and_weight(self) -> None:
        node = self.manager.learn(self.record.id)
        original = node.reliability_weight
        self.manager.apply_feedback([node.index], memory_id=self.record.id, positive=True)
        self.assertGreater(node.reliability_weight, original)
        self.assertEqual(node.memory_index.positive, [self.record.id])
        self.manager.apply_feedback([node.index], memory_id=self.record.id, positive=False)
        self.assertEqual(node.memory_index.negative, [self.record.id])
        self.assertEqual(node.memory_index.positive, [])

    def test_specific_rules_are_more_feedback_sensitive(self) -> None:
        generic = SchemaNode(index="generic", level=0, description="same rule", reliability_weight=0.5)
        specific = SchemaNode(index="specific", level=3, description="same rule specific", reliability_weight=0.5)
        manager = SchemaManager(memory=self.bank, graph=LayeredSchemaGraph([generic, specific]))
        manager.apply_feedback([generic.index, specific.index], memory_id=self.record.id, positive=True)
        self.assertGreater(specific.reliability_weight, generic.reliability_weight)

    def test_forgetting_masks_isolated_stale_rule(self) -> None:
        node = self.manager.learn(self.record.id)
        node.last_accessed_at = (datetime.now(timezone.utc) - timedelta(days=365)).isoformat()
        manager = SchemaManager(
            memory=self.bank,
            graph=self.manager.graph,
            config=SchemaManagementConfig(daily_decay=0.02),
        )
        changed = manager.apply_forgetting(now=datetime.now(timezone.utc))
        self.assertIn(node.index, changed)
        self.assertIn(node.status, {SchemaStatus.MASKED, SchemaStatus.DEPRECATED})

    def test_task_mask_hides_unrelated_schema(self) -> None:
        unrelated = SchemaNode(index="math", level=1, description="Solve a quadratic equation")
        manager = SchemaManager(memory=self.bank, graph=LayeredSchemaGraph([unrelated]))
        masked = manager.update_task_mask("move the game lever", min_relevance=0.1)
        self.assertEqual(masked, ["math"])
        self.assertEqual(unrelated.status, SchemaStatus.MASKED)

    def test_consolidation_preserves_parent_and_evidence(self) -> None:
        parent = SchemaNode(index="parent", level=0, description="General action rule")
        left = SchemaNode(
            index="left",
            level=2,
            description="If lever is visible use ACTION1 and object moves up",
            memory_index=MemoryIndex(source=["memory_1"]),
        )
        right = SchemaNode(
            index="right",
            level=2,
            description="If lever is visible use ACTION1 and object moves up",
            memory_index=MemoryIndex(source=["memory_2"]),
        )
        graph = LayeredSchemaGraph([parent, left, right])
        graph.connect_parent_child("parent", "left")
        graph.connect_parent_child("parent", "right")
        bank = MemoryBank(MemoryStore([self.record, episode(memory_id="memory_2")]))
        manager = SchemaManager(memory=bank, graph=graph)
        merged = manager.consolidate()
        self.assertEqual(merged, [("left", "right")])
        self.assertIsNone(graph.get("right"))
        self.assertEqual(graph.get("left").memory_index.source, ["memory_1", "memory_2"])
        graph.validate()

    def test_complete_system_persists_both_layers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            system = build_schema_system(directory)
            node = system.remember_and_learn(episode())
            restored = build_schema_system(directory)
            self.assertIsNotNone(restored.memory.recall("memory_1"))
            self.assertIsNotNone(restored.graph.get(node.index))

    def test_llm_generator_drives_structured_creation(self) -> None:
        class FakeModel:
            def complete(self, messages, *, temperature, max_tokens):
                return SimpleNamespace(
                    text=(
                        '{"operation":"create","level":3,'
                        '"trigger":"left lever visible",'
                        '"action_sequence":["ACTION1"],'
                        '"expectation":"object moves upward",'
                        '"parent_ids":[],"similar_ids":[],"rationale":"observed transition"}'
                    )
                )

        manager = SchemaManager(
            memory=self.bank,
            generator=LLMSchemaGenerator(FakeModel()),
        )
        node = manager.learn(self.record.id)
        self.assertEqual(node.level, 3)
        self.assertEqual(node.action_sequence, ["ACTION1"])


if __name__ == "__main__":
    unittest.main()
