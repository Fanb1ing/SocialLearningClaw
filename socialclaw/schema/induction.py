from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence

from ..memory import MemoryRecord
from .node import SchemaNode


@dataclass(frozen=True)
class SchemaProposal:
    """A validated operation proposed by the schema induction model."""

    operation: str
    level: int = 0
    trigger: str = ""
    action_sequence: List[str] = field(default_factory=list)
    expectation: str = ""
    target_schema_id: Optional[str] = None
    parent_ids: List[str] = field(default_factory=list)
    similar_ids: List[str] = field(default_factory=list)
    rationale: str = ""


class SchemaGenerator(Protocol):
    def propose(
        self, memory: MemoryRecord, candidates: Sequence[SchemaNode]
    ) -> SchemaProposal:
        ...

    def merge_description(self, left: SchemaNode, right: SchemaNode) -> str:
        ...


class ChatModel(Protocol):
    def complete(
        self,
        messages: List[Dict[str, Any]],
        *,
        temperature: float,
        max_tokens: int,
    ):
        ...


class LLMSchemaGenerator:
    """Structured LLM adapter for automatic schema induction and fusion."""

    def __init__(self, model: ChatModel, *, max_memory_chars: int = 8000) -> None:
        self.model = model
        self.max_memory_chars = max_memory_chars

    def propose(
        self, memory: MemoryRecord, candidates: Sequence[SchemaNode]
    ) -> SchemaProposal:
        candidate_payload = [
            {
                "index": node.index,
                "level": node.level,
                "description": node.description,
                "reliability_weight": node.reliability_weight,
            }
            for node in candidates
        ]
        prompt = (
            "You maintain a multi-level graph of learned world rules. Infer a reusable "
            "schema only from the supplied episode evidence. Smaller level means more "
            "general; larger level means more task-specific. Decide whether the evidence "
            "creates a node, merges into one candidate, or is too weak and should be skipped.\n"
            "Return ONLY one JSON object with keys: operation (create|merge|skip), level "
            "(non-negative integer), trigger, action_sequence (array of strings), expectation, "
            "target_schema_id (required only for merge), parent_ids, similar_ids, rationale. "
            "For merge, trigger/action_sequence/expectation must describe the consolidated "
            "rule supported by both the candidate and the new episode, not merely restate the episode. "
            "Only cite candidate IDs that are provided. Do not invent facts.\n\n"
            f"EPISODE MEMORY:\n{memory.text_for_retrieval()[:self.max_memory_chars]}\n\n"
            f"CANDIDATE SCHEMAS:\n{json.dumps(candidate_payload, ensure_ascii=False)}"
        )
        response = self.model.complete(
            [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=1200
        )
        return self._parse_proposal(response.text, candidates)

    def merge_description(self, left: SchemaNode, right: SchemaNode) -> str:
        prompt = (
            "Consolidate these two equivalent schema rules without adding facts. Preserve the "
            "format [Context/Perception: ...] [Action/Execution: ...] [Expectation: ...]. "
            "Return ONLY the consolidated description.\n\n"
            f"RULE A: {left.description}\nRULE B: {right.description}"
        )
        response = self.model.complete(
            [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=500
        )
        return (response.text or "").strip() or left.description

    @staticmethod
    def _parse_proposal(text: str, candidates: Sequence[SchemaNode]) -> SchemaProposal:
        data = _extract_json_object(text)
        operation = str(data.get("operation", "skip")).lower()
        if operation not in {"create", "merge", "skip"}:
            operation = "skip"
        candidate_ids = {node.index for node in candidates}
        raw_target = data.get("target_schema_id")
        target = raw_target if isinstance(raw_target, str) else None
        if operation == "merge" and target not in candidate_ids:
            operation = "skip"
            target = None
        actions = data.get("action_sequence", [])
        if isinstance(actions, str):
            actions = [actions]
        if not isinstance(actions, list):
            actions = []
        try:
            level = max(0, int(data.get("level", 0)))
        except (TypeError, ValueError):
            level = 0
        parent_ids = data.get("parent_ids", [])
        similar_ids = data.get("similar_ids", [])
        if not isinstance(parent_ids, list):
            parent_ids = []
        if not isinstance(similar_ids, list):
            similar_ids = []
        return SchemaProposal(
            operation=operation,
            level=level,
            trigger=str(data.get("trigger", "")).strip(),
            action_sequence=[str(item).strip() for item in actions if str(item).strip()],
            expectation=str(data.get("expectation", "")).strip(),
            target_schema_id=target,
            parent_ids=[
                item for item in parent_ids if isinstance(item, str) and item in candidate_ids
            ],
            similar_ids=[
                item for item in similar_ids if isinstance(item, str) and item in candidate_ids
            ],
            rationale=str(data.get("rationale", "")).strip(),
        )


def _extract_json_object(text: str) -> Dict[str, Any]:
    value = (text or "").strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value)
        value = re.sub(r"\s*```$", "", value)
    try:
        payload = json.loads(value)
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", value)
        if not match:
            return {}
        try:
            payload = json.loads(match.group(0))
            return payload if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            return {}
