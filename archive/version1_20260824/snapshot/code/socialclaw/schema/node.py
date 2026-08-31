from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List
import uuid

from ..memory.models import utc_now


class SchemaStatus(str, Enum):
    ACTIVE = "active"
    MASKED = "masked"
    DEPRECATED = "deprecated"


@dataclass
class MemoryIndex:
    """Evidence provenance for a schema rule."""

    source: List[str] = field(default_factory=list)
    positive: List[str] = field(default_factory=list)
    negative: List[str] = field(default_factory=list)

    def all(self) -> List[str]:
        return sorted({*self.source, *self.positive, *self.negative})


@dataclass
class RelatedSchemaIndex:
    """Typed neighbors in the layered graph.

    Parents are more general (smaller level), children more specific (larger
    level), and similar nodes normally live at the same level.
    """

    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    similar: List[str] = field(default_factory=list)
    evidence: Dict[str, List[str]] = field(default_factory=dict)

    def all(self) -> List[str]:
        return sorted({*self.parents, *self.children, *self.similar})


@dataclass
class SchemaNode:
    """A learned world rule grounded in concrete memory records."""

    level: int
    description: str
    index: str = field(default_factory=lambda: f"schema_{uuid.uuid4().hex[:12]}")
    trigger: str = ""
    action_sequence: List[str] = field(default_factory=list)
    expectation: str = ""
    memory_index: MemoryIndex = field(default_factory=MemoryIndex)
    related_schema_index: RelatedSchemaIndex = field(default_factory=RelatedSchemaIndex)
    reliability_weight: float = 0.5
    status: SchemaStatus = SchemaStatus.ACTIVE
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    last_accessed_at: str = field(default_factory=utc_now)
    access_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.level < 0:
            raise ValueError("Schema level must be non-negative")
        if not self.description.strip():
            raise ValueError("Schema description must not be empty")
        self.reliability_weight = min(1.0, max(0.0, float(self.reliability_weight)))

    @classmethod
    def from_rule(
        cls,
        *,
        level: int,
        trigger: str,
        action_sequence: List[str],
        expectation: str,
        source_memory_id: str,
        reliability_weight: float = 0.5,
    ) -> "SchemaNode":
        action_text = " -> ".join(action_sequence)
        description = (
            f"[Context/Perception: {trigger}] "
            f"[Action/Execution: {action_text}] "
            f"[Expectation: {expectation}]"
        )
        return cls(
            level=level,
            description=description,
            trigger=trigger,
            action_sequence=action_sequence,
            expectation=expectation,
            reliability_weight=reliability_weight,
            memory_index=MemoryIndex(source=[source_memory_id]),
        )

    def text_for_retrieval(self) -> str:
        return "\n".join(
            part
            for part in (
                self.description,
                f"Trigger: {self.trigger}" if self.trigger else "",
                f"Actions: {' -> '.join(self.action_sequence)}" if self.action_sequence else "",
                f"Expectation: {self.expectation}" if self.expectation else "",
            )
            if part
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SchemaNode":
        data = dict(payload)
        data["memory_index"] = MemoryIndex(**data.get("memory_index", {}))
        data["related_schema_index"] = RelatedSchemaIndex(
            **data.get("related_schema_index", {})
        )
        data["status"] = SchemaStatus(data.get("status", SchemaStatus.ACTIVE.value))
        return cls(**data)
