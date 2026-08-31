from __future__ import annotations

import json
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
from socialclaw.v2.cognition import read_cognition, render_cognition_catalog


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

    def test_schema_is_a_complete_prototype_action_output_triple(self) -> None:
        self.graph.apply_transaction(
            [
                GraphOperation(
                    OperationKind.CREATE_PROTOTYPE,
                    {
                        "prototype_id": "prototype_1",
                        "name": "movable object",
                    },
                    [self.evidence.evidence_id],
                    "create input type",
                )
            ],
            actor="update_child_agent",
            step=0,
            mode="accommodation",
            summary="create prototype",
        )
        before = self.graph.to_dict()
        with self.assertRaisesRegex(ValueError, "has no output"):
            self.graph.apply_transaction(
                [
                    GraphOperation(
                        OperationKind.CREATE_SCHEMA,
                        {
                            "schema_id": "schema_invalid",
                            "prototype_id": "prototype_1",
                            "action": {"name": "ACTION1", "arguments": {}},
                            "output": "",
                        },
                        [self.evidence.evidence_id],
                        "Schemas may not omit the Output component",
                    )
                ],
                actor="update_child_agent",
                step=1,
                mode="accommodation",
                summary="reject an incomplete Schema triple",
            )
        self.assertEqual(self.graph.to_dict(), before)

        self.graph.apply_transaction(
            [
                GraphOperation(
                    OperationKind.CREATE_SCHEMA,
                    {
                        "schema_id": "schema_1",
                        "prototype_id": "prototype_1",
                        "action": {"name": "ACTION1", "arguments": {}},
                        "output": "the movable object shifts right",
                    },
                    [self.evidence.evidence_id],
                    "observed triple",
                )
            ],
            actor="update_child_agent",
            step=1,
            mode="accommodation",
            summary="create complete Schema triple",
        )
        schema = self.graph.schemas["schema_1"]
        self.assertEqual(schema.prototype_id, "prototype_1")
        self.assertEqual(schema.action["name"], "ACTION1")
        self.assertEqual(schema.output, "the movable object shifts right")

    def test_global_insight_is_evidence_grounded_and_revisable(self) -> None:
        self.graph.apply_transaction(
            [
                GraphOperation(
                    OperationKind.CREATE_INSIGHT,
                    {
                        "insight_id": "insight_1",
                        "kind": "constraint",
                        "statement": "An obstacle blocks movement into its cell.",
                        "scope": "global",
                        "confidence": 0.6,
                    },
                    [self.evidence.evidence_id],
                    "observed failed movement",
                )
            ],
            actor="update_child_agent",
            step=1,
            mode="accommodation",
            summary="learn global constraint",
        )
        insight = self.graph.insights["insight_1"]
        self.assertEqual(insight.kind.value, "constraint")
        self.assertEqual(insight.support_evidence_ids, ["evidence_1"])
        self.assertEqual(self.graph.counts()["insights"], 1)
        lookup = json.loads(
            read_cognition(
                self.graph, command="get_insight", item_id="insight_1"
            )
        )
        self.assertTrue(lookup["ok"])
        self.assertEqual(lookup["record"]["kind"], "constraint")
        catalog = render_cognition_catalog(self.graph)
        self.assertIn("Global Insights/Rules", catalog)
        self.assertIn("An obstacle blocks movement", catalog)

    def test_format_two_schema_migrates_to_triple(self) -> None:
        self.graph.apply_transaction(
            [
                GraphOperation(
                    OperationKind.CREATE_PROTOTYPE,
                    {"prototype_id": "prototype_1", "name": "button"},
                    [self.evidence.evidence_id],
                    "legacy prototype",
                )
            ],
            actor="update_child_agent",
            step=0,
            mode="accommodation",
            summary="create prototype",
        )
        payload = self.graph.to_dict()
        payload["format_version"] = 2
        payload.pop("insights")
        payload["schemas"] = {
            "schema_old": {
                "schema_id": "schema_old",
                "name": "press changes button",
                "role_bindings": {"target": ["prototype_1"]},
                "preconditions": [],
                "action_pattern": {"action": "ACTION1", "arguments": {}},
                "expected_changes": ["button changes color"],
                "invariants": [],
                "boundary_conditions": [],
                "support_evidence_ids": ["evidence_1"],
                "counter_evidence_ids": [],
                "confidence": 0.6,
                "status": "active",
                "revision_count": 0,
                "metadata": {},
            }
        }
        payload["relations"]["legacy_binding"] = {
            "relation_id": "legacy_binding",
            "relation_type": "binds_role_to",
            "source_id": "schema_old",
            "target_id": "prototype_1",
            "evidence_ids": ["evidence_1"],
            "metadata": {"role": "target"},
        }
        restored = EFPSGraph.from_dict(payload)
        restored.validate()
        schema = restored.schemas["schema_old"]
        self.assertEqual(schema.prototype_id, "prototype_1")
        self.assertEqual(schema.action["name"], "ACTION1")
        self.assertEqual(schema.output, "button changes color")
        self.assertEqual(restored.to_dict()["format_version"], 3)


if __name__ == "__main__":
    unittest.main()
