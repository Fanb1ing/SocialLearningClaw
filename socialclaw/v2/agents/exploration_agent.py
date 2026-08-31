from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from ..cognition import cognition_tool, render_sections, section_metrics
from ..efps import EFPSGraph
from ..model import ModelImage, StructuredVisionModel
from .prompts import EXPLORATION_INSTRUCTIONS
from .protocols import AgentCallAudit, ExplorationTurn


class ExplorationAgent:
    """Game-agnostic child Agent that generates probes but cannot execute them."""

    instruction_profile = "exploration_agent_v2_generic"

    def __init__(
        self,
        model: StructuredVisionModel,
        *,
        artifact_root: str | Path | None = None,
    ) -> None:
        self.model = model
        self.artifact_root = artifact_root

    def propose(
        self,
        *,
        shared_input: Dict[str, Any],
        images: List[ModelImage],
        graph: EFPSGraph,
    ) -> ExplorationTurn:
        sections = dict(shared_input.get("prompt_sections") or {})
        input_text = render_sections(sections)
        result = self.model.generate_text(
            instructions=EXPLORATION_INSTRUCTIONS,
            payload=input_text,
            images=images,
            tools=[cognition_tool(graph, self.artifact_root)],
        )
        audit = AgentCallAudit(
            instruction_profile=self.instruction_profile,
            received_refs=["event.shared_decision_input"],
            image_inputs=[item.audit_dict() for item in images],
            output=result.text,
            model=result.model,
            usage=result.usage,
            input_text=input_text,
            input_sections=section_metrics(sections),
            tool_trace=result.tool_trace,
            usage_rounds=result.usage_rounds,
        )
        return ExplorationTurn(advice=result.text, audit=audit)


__all__ = ["ExplorationAgent"]
