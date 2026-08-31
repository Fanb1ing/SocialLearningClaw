from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from ..cognition import cognition_tool, render_sections, section_metrics
from ..efps import EFPSGraph
from ..model import ModelImage, StructuredVisionModel
from .prompts import MAIN_INSTRUCTIONS
from .protocols import AgentCallAudit, ExplorationTurn, MainDecision
from .validation import bounded_probability, known_ids, legal_action


class MainAgent:
    """Game-agnostic orchestrator/planner and sole environment action selector."""

    instruction_profile = "main_agent_v2_generic"

    def __init__(
        self,
        model: StructuredVisionModel,
        *,
        artifact_root: str | Path | None = None,
    ) -> None:
        self.model = model
        self.artifact_root = artifact_root

    def decide(
        self,
        *,
        shared_input: Dict[str, Any],
        exploration: ExplorationTurn,
        images: List[ModelImage],
        graph: EFPSGraph,
    ) -> MainDecision:
        sections = dict(shared_input.get("prompt_sections") or {})
        sections["Exploration child advice"] = exploration.public_output()
        input_text = render_sections(sections)
        result = self.model.generate(
            instructions=MAIN_INSTRUCTIONS,
            payload=input_text,
            images=images,
            tools=[cognition_tool(graph, self.artifact_root)],
        )
        action = legal_action(
            result.data.get("selected_action"),
            list(shared_input.get("available_actions") or []),
        )
        schema_ids = known_ids(
            result.data.get("schemas_used") or [], graph.schemas
        )
        insight_ids = known_ids(
            result.data.get("insights_used") or [], graph.insights
        )
        schema_prediction = result.data.get("schema_prediction")
        insight_application = result.data.get("insight_application")
        if schema_ids:
            if not isinstance(schema_prediction, str) or not schema_prediction.strip():
                raise ValueError(
                    "Main Agent cited a Schema without a schema_prediction"
                )
            decision_mode = "schema"
            exploration_hypothesis = None
            if insight_ids and (
                not isinstance(insight_application, str)
                or not insight_application.strip()
            ):
                raise ValueError(
                    "Main Agent cited an Insight without insight_application"
                )
            if not insight_ids:
                insight_application = None
        elif insight_ids:
            schema_prediction = None
            exploration_hypothesis = None
            if (
                not isinstance(insight_application, str)
                or not insight_application.strip()
            ):
                raise ValueError(
                    "Main Agent cited an Insight without insight_application"
                )
            decision_mode = "insight"
        else:
            schema_prediction = None
            insight_application = None
            decision_mode = "explore"
            exploration_hypothesis = result.data.get("exploration_hypothesis")
            if not isinstance(exploration_hypothesis, str) or not exploration_hypothesis.strip():
                raise ValueError(
                    "An action without a Schema requires a Main-Agent-generated exploration hypothesis"
                )
        goals = []
        for raw in result.data.get("goal_hypotheses") or []:
            if not isinstance(raw, dict) or not str(raw.get("text") or "").strip():
                continue
            goals.append(
                {
                    "text": str(raw["text"]).strip(),
                    "confidence": bounded_probability(raw.get("confidence")),
                    "evidence_ids": known_ids(
                        raw.get("evidence_ids") or [], graph.evidence
                    ),
                }
            )
        audit = AgentCallAudit(
            instruction_profile=self.instruction_profile,
            received_refs=[
                "event.shared_decision_input",
                "event.agent_calls.exploration_agent.output",
            ],
            image_inputs=[item.audit_dict() for item in images],
            output=result.data,
            model=result.model,
            usage=result.usage,
            input_text=input_text,
            input_sections=section_metrics(sections),
            tool_trace=result.tool_trace,
            usage_rounds=result.usage_rounds,
        )
        return MainDecision(
            action=action,
            goal_hypotheses=goals,
            decision_mode=decision_mode,
            schema_ids=schema_ids,
            schema_prediction=str(schema_prediction).strip()
            if schema_prediction is not None
            else None,
            insight_ids=insight_ids,
            insight_application=str(insight_application).strip()
            if insight_application is not None
            else None,
            exploration_hypothesis=str(exploration_hypothesis).strip()
            if exploration_hypothesis is not None
            else None,
            rationale=str(result.data.get("rationale") or "").strip(),
            audit=audit,
        )


__all__ = ["MainAgent"]
