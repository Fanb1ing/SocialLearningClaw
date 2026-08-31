from __future__ import annotations

import unittest

from socialclaw.v2.agents.main_agent import MainAgent
from socialclaw.v2.agents.protocols import AgentCallAudit, ExplorationTurn
from socialclaw.v2.efps import (
    EFPSGraph,
    EvidenceRecord,
    GraphOperation,
    OperationKind,
)
from socialclaw.v2.model import ModelResult


class _InsightDecisionModel:
    @property
    def model_name(self) -> str:
        return "fixture-model"

    def generate(self, **kwargs):
        return ModelResult(
            data={
                "goal_hypotheses": [],
                "decision_mode": "insight",
                "selected_action": {"name": "ACTION1", "arguments": {}},
                "schemas_used": [],
                "schema_prediction": None,
                "insights_used": ["insight_wall"],
                "insight_application": "Avoid the blocked direction.",
                "exploration_hypothesis": None,
                "rationale": "Use the learned global constraint.",
            },
            model=self.model_name,
            usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        )


class InsightDecisionTests(unittest.TestCase):
    def test_main_can_act_from_an_insight_without_a_schema(self) -> None:
        graph = EFPSGraph()
        graph.register_evidence(
            EvidenceRecord(
                evidence_id="evidence_1",
                kind="public_transition",
                episode_id="episode_1",
                step_index=0,
                observation_fingerprints=["before", "after"],
            )
        )
        graph.apply_transaction(
            [
                GraphOperation(
                    OperationKind.CREATE_INSIGHT,
                    {
                        "insight_id": "insight_wall",
                        "kind": "constraint",
                        "statement": "A wall blocks movement into its cell.",
                        "scope": "global",
                    },
                    ["evidence_1"],
                    "observed blocked movement",
                )
            ],
            actor="update_child_agent",
            step=1,
            mode="accommodation",
            summary="learn constraint",
        )
        audit = AgentCallAudit(
            instruction_profile="fixture",
            received_refs=[],
            image_inputs=[],
            output="",
            model="fixture-model",
            usage={},
        )
        decision = MainAgent(_InsightDecisionModel()).decide(
            shared_input={
                "available_actions": [
                    {
                        "name": "ACTION1",
                        "arguments_schema": {
                            "type": "object",
                            "properties": {},
                            "required": [],
                        },
                    }
                ],
                "prompt_sections": {"Current learned cognition": "fixture"},
            },
            exploration=ExplorationTurn(advice="No probe needed.", audit=audit),
            images=[],
            graph=graph,
        )
        self.assertEqual(decision.decision_mode, "insight")
        self.assertEqual(decision.insight_ids, ["insight_wall"])
        self.assertEqual(decision.schema_ids, [])
        self.assertIsNone(decision.exploration_hypothesis)


if __name__ == "__main__":
    unittest.main()
