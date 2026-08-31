from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from ...trajectory import Action
from ..efps import GraphOperation


@dataclass(frozen=True)
class AgentCallAudit:
    instruction_profile: str
    received_refs: List[str]
    image_inputs: List[Dict[str, str]]
    output: Any
    model: str
    usage: Dict[str, int]
    input_text: str = ""
    input_sections: List[Dict[str, Any]] = field(default_factory=list)
    tool_trace: List[Dict[str, Any]] = field(default_factory=list)
    usage_rounds: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instruction_profile": self.instruction_profile,
            "received_refs": list(self.received_refs),
            "image_inputs": list(self.image_inputs),
            "output": self.output,
            "model": self.model,
            "usage": dict(self.usage),
            "input_text": self.input_text,
            "input_sections": list(self.input_sections),
            "tool_trace": list(self.tool_trace),
            "usage_rounds": list(self.usage_rounds),
        }


@dataclass(frozen=True)
class ExplorationTurn:
    advice: str
    audit: AgentCallAudit

    def public_output(self) -> str:
        return self.advice


@dataclass(frozen=True)
class MainDecision:
    action: Action
    goal_hypotheses: List[Dict[str, Any]]
    decision_mode: str
    schema_ids: List[str]
    schema_prediction: str | None
    insight_ids: List[str]
    insight_application: str | None
    exploration_hypothesis: str | None
    rationale: str
    audit: AgentCallAudit

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.to_dict(),
            "goal_hypotheses": list(self.goal_hypotheses),
            "decision_mode": self.decision_mode,
            "schema_ids": list(self.schema_ids),
            "schema_prediction": self.schema_prediction,
            "insight_ids": list(self.insight_ids),
            "insight_application": self.insight_application,
            "exploration_hypothesis": self.exploration_hypothesis,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class UpdateProposal:
    operations: List[GraphOperation]
    mode: str
    summary: str
    semantic_output: Dict[str, Any]
    transition_analysis: Dict[str, Any] | None
    audit: AgentCallAudit
    warnings: List[str] = field(default_factory=list)


__all__ = [
    "AgentCallAudit",
    "ExplorationTurn",
    "MainDecision",
    "UpdateProposal",
]
