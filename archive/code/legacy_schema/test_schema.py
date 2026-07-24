"""Archived regression tests for the single-layer Concept/Relation schema."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from socialclaw.schema.arc_agi3_parser import compute_grid_diff
from socialclaw.schema.graph import Concept, Relation, SchemaGraph
from socialclaw.schema.storage import SchemaStorage


class SchemaGraphTests(unittest.TestCase):
    def test_stable_observation_id_preserves_learned_confidence(self) -> None:
        graph = SchemaGraph()
        graph.add_concept(Concept("object", "Object", "old", confidence=0.8))
        graph.add_concept(Concept("object", "Object", "new", confidence=0.6))
        concept = graph.get_concept("object")
        self.assertIsNotNone(concept)
        self.assertEqual(concept.description, "new")
        self.assertEqual(concept.confidence, 0.8)

    def test_exact_duplicate_relations_are_merged(self) -> None:
        graph = SchemaGraph()
        graph.add_relation(Relation("a", "b", "related", evidence=[{"step": 1}]))
        graph.add_relation(Relation("a", "b", "related", evidence=[{"step": 2}]))
        self.assertEqual(len(graph.list_relations()), 1)
        self.assertEqual(len(graph.list_relations()[0].evidence), 2)

    def test_shape_change_counts_as_grid_change(self) -> None:
        changed, regions = compute_grid_diff(np.zeros((2, 2)), np.zeros((3, 2)))
        self.assertTrue(changed)
        self.assertEqual(regions[0]["shape_before"], [2, 2])

    def test_storage_round_trip(self) -> None:
        graph = SchemaGraph()
        graph.add_concept(Concept("a", "A", "alpha"))
        embeddings = {"a": np.array([1.0, 0.0], dtype=np.float32)}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            storage = SchemaStorage(
                str(root / "concepts.jsonl"),
                str(root / "relations.jsonl"),
                str(root / "embeddings.npy"),
                str(root / "ids.json"),
            )
            storage.save(graph, embeddings)
            loaded_graph, loaded_embeddings = storage.load()
        self.assertEqual(loaded_graph.get_concept("a").name, "A")
        np.testing.assert_array_equal(loaded_embeddings["a"], embeddings["a"])


if __name__ == "__main__":
    unittest.main()
