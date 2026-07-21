from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class MemoryKind(str, Enum):
    """Kinds of concrete information retained below the schema layer."""

    EPISODE = "episode"
    KNOWLEDGE = "knowledge"
    SKILL = "skill"


@dataclass
class MemoryEvent:
    """One observation-action-result transition inside an episode."""

    observation: str
    action: str
    result: str
    index: int = 0
    created_at: str = field(default_factory=utc_now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryRecord:
    """A durable, concrete experience from which schemas may be induced.

    Schema nodes reference ``id`` rather than copying this record. This keeps
    raw evidence available when a rule needs to be re-evaluated later.
    """

    task: str
    id: str = field(default_factory=lambda: f"memory_{uuid.uuid4().hex[:12]}")
    kind: MemoryKind = MemoryKind.EPISODE
    context: str = ""
    events: List[MemoryEvent] = field(default_factory=list)
    outcome: str = ""
    success: Optional[bool] = None
    feedback: str = ""
    tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_event(
        self,
        *,
        observation: str,
        action: str,
        result: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryEvent:
        event = MemoryEvent(
            index=len(self.events),
            observation=observation,
            action=action,
            result=result,
            metadata=dict(metadata or {}),
        )
        self.events.append(event)
        self.updated_at = utc_now()
        return event

    def text_for_retrieval(self) -> str:
        transitions = "\n".join(
            f"Observation: {event.observation}\nAction: {event.action}\nResult: {event.result}"
            for event in self.events
        )
        return "\n".join(
            part
            for part in (
                f"Task: {self.task}",
                f"Context: {self.context}" if self.context else "",
                transitions,
                f"Outcome: {self.outcome}" if self.outcome else "",
                f"Feedback: {self.feedback}" if self.feedback else "",
                f"Tags: {', '.join(self.tags)}" if self.tags else "",
            )
            if part
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["kind"] = self.kind.value
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "MemoryRecord":
        data = dict(payload)
        data["kind"] = MemoryKind(data.get("kind", MemoryKind.EPISODE.value))
        data["events"] = [MemoryEvent(**event) for event in data.get("events", [])]
        return cls(**data)
