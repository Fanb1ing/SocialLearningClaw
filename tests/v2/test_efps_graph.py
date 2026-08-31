from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from socialclaw.v2.efps import (
    EFPSGraph,
    EFPSGraphStorage,
    EvidenceRecord,
    GraphOperation,
    OperationKind,
)


class EFPSGraphTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = EFPSGraph()
        self.evidence = EvidenceRecord(
            evidence_id="evidence_1",
            kind="test_observation",
            episode_id="episode_1",
            step_index=None,
            observation_fingerprints=["fingerprint"],
        )
        self.graph.register_evidence(self.evidence)

    def test_transaction_is_evidence_grounded_and_rolls_back(self) -> None:
        self.graph.apply_transaction(
            [
                GraphOperation(
                    OperationKind.ADD_ENTITY,
                    {
                        "entity_id": "entity_1",
                        "label": "object",
                        "bbox": [0, 0, 1, 1],
                    },
                    [self.evidence.evidence_id],
                    "first observation",
                )
            ],
            actor="perception",
            step=0,
            mode="accommodation",
            summary="create one object",
        )
        before = self.graph.to_dict()
        with self.assertRaises(ValueError):
            self.graph.apply_transaction(
                [
                    GraphOperation(
                        OperationKind.UPDATE_ENTITY,
                        {"entity_id": "entity_1", "bbox": [1, 1, 2, 2]},
                        ["missing_evidence"],
                        "invalid update",
                    )
                ],
                actor="update_child_agent",
                step=1,
                mode="assimilation",
                summary="must fail",
            )
        self.assertEqual(self.graph.to_dict(), before)

    def test_atomic_storage_round_trip(self) -> None:
        self.graph.apply_transaction(
            [
                GraphOperation(
                    OperationKind.ADD_ENTITY,
                    {
                        "entity_id": "entity_1",
                        "label": "object",
                        "bbox": [0, 0, 1, 1],
                    },
                    [self.evidence.evidence_id],
                    "first observation",
                )
            ],
            actor="perception",
            step=0,
            mode="accommodation",
            summary="create one object",
        )
        with tempfile.TemporaryDirectory() as directory:
            storage = EFPSGraphStorage(directory)
            snapshot = storage.save(self.graph, label="test")
            restored = storage.load()
            self.assertTrue(snapshot.is_file())
            self.assertEqual(restored.to_dict(), self.graph.to_dict())
            self.assertEqual(len(list((Path(directory) / "snapshots").glob("*.json"))), 1)

    def test_evidence_id_resolves_to_semantic_entity_change(self) -> None:
        self.graph.apply_transaction(
            [
                GraphOperation(
                    OperationKind.ADD_ENTITY,
                    {
                        "entity_id": "entity_1",
                        "label": "visible object",
                        "bbox": [0, 0, 1, 1],
                    },
                    [self.evidence.evidence_id],
                    "first observation",
                )
            ],
            actor="update_child_agent",
            step=0,
            mode="accommodation",
            summary="create one object",
        )
        self.graph.annotate_evidence(
            self.evidence.evidence_id,
            semantic_summary="The visible object moved.",
            entity_changes=[
                {
                    "entity_id": "entity_1",
                    "label": "visible object",
                    "change_type": "moved",
                    "before": "left",
                    "after": "right",
                    "description": "The object moved right.",
                    "confidence": 0.9,
                }
            ],
            unassigned_visual_changes=[],
        )
        resolved = self.graph.resolve_evidence(self.evidence.evidence_id)
        self.assertEqual(resolved["semantic_summary"], "The visible object moved.")
        self.assertEqual(resolved["entity_changes"][0]["entity_id"], "entity_1")
        resolved["entity_changes"].clear()
        self.assertEqual(
            len(self.graph.resolve_evidence(self.evidence.evidence_id)["entity_changes"]),
            1,
        )

    def test_schema_requires_a_prototype_role_binding(self) -> None:
        before = self.graph.to_dict()
        with self.assertRaisesRegex(ValueError, "requires Prototype role bindings"):
            self.graph.apply_transaction(
                [
                    GraphOperation(
                        OperationKind.CREATE_SCHEMA,
                        {
                            "schema_id": "schema_invalid",
                            "name": "entity_bound_rule",
                            "role_bindings": {},
                            "action_pattern": {"action": "ACTION1", "arguments": {}},
                            "expected_changes": ["the observed object changes"],
                        },
                        [self.evidence.evidence_id],
                        "Schemas may not omit their Prototype type role",
                    )
                ],
                actor="update_child_agent",
                step=1,
                mode="accommodation",
                summary="reject a Schema without a Prototype role",
            )
        self.assertEqual(self.graph.to_dict(), before)


if __name__ == "__main__":
    unittest.main()
