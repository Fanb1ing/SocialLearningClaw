from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from ..memory.assets import MemoryArtifactRef


FORMAT_VERSION = 1
_SAFE_EPISODE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_copy(value):
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))
    except (TypeError, ValueError) as error:
        raise ValueError("Trajectory values must be JSON serializable") from error


class EvidenceTier(str, Enum):
    NATURAL = "natural"
    SOURCE_GUIDED_NATURAL = "source_guided_natural"
    STATE_INJECTED_PROBE = "state_injected_probe"
    SYNTHETIC = "synthetic"


@dataclass(frozen=True)
class Observation:
    text: str = ""
    structured: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[MemoryArtifactRef] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "structured", _json_copy(self.structured))
        object.__setattr__(self, "artifacts", list(self.artifacts))
        object.__setattr__(self, "metadata", _json_copy(self.metadata))
        if not self.text and not self.structured and not self.artifacts:
            raise ValueError("An observation must contain text, structured state, or artifacts")

    def content_fingerprint(self) -> str:
        payload = {
            "text": self.text,
            "structured": self.structured,
            "artifacts": [item.to_dict() for item in self.artifacts],
            "metadata": self.metadata,
        }
        encoded = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "structured": _json_copy(self.structured),
            "artifacts": [item.to_dict() for item in self.artifacts],
            "metadata": _json_copy(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Observation":
        return cls(
            text=str(payload.get("text", "")),
            structured=dict(payload.get("structured") or {}),
            artifacts=[
                MemoryArtifactRef.from_dict(item)
                for item in payload.get("artifacts", [])
            ],
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True)
class Action:
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Action name must not be empty")
        object.__setattr__(self, "arguments", _json_copy(self.arguments))
        object.__setattr__(self, "metadata", _json_copy(self.metadata))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "arguments": _json_copy(self.arguments),
            "metadata": _json_copy(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Action":
        return cls(
            name=str(payload["name"]),
            arguments=dict(payload.get("arguments") or {}),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True)
class Decision:
    response: str = ""
    rationale: str = ""
    retrieved_schema_ids: List[str] = field(default_factory=list)
    claimed_schema_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "retrieved_schema_ids", list(self.retrieved_schema_ids))
        object.__setattr__(self, "claimed_schema_ids", list(self.claimed_schema_ids))
        object.__setattr__(self, "metadata", _json_copy(self.metadata))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "response": self.response,
            "rationale": self.rationale,
            "retrieved_schema_ids": list(self.retrieved_schema_ids),
            "claimed_schema_ids": list(self.claimed_schema_ids),
            "metadata": _json_copy(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Decision":
        return cls(
            response=str(payload.get("response", "")),
            rationale=str(payload.get("rationale", "")),
            retrieved_schema_ids=[str(item) for item in payload.get("retrieved_schema_ids", [])],
            claimed_schema_ids=[str(item) for item in payload.get("claimed_schema_ids", [])],
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True)
class StepResult:
    observation: Observation
    environment_status: str
    state_delta: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.environment_status.strip():
            raise ValueError("environment_status must not be empty")
        object.__setattr__(self, "state_delta", _json_copy(self.state_delta))
        object.__setattr__(self, "metadata", _json_copy(self.metadata))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observation": self.observation.to_dict(),
            "environment_status": self.environment_status,
            "state_delta": _json_copy(self.state_delta),
            "metadata": _json_copy(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "StepResult":
        return cls(
            observation=Observation.from_dict(payload["observation"]),
            environment_status=str(payload["environment_status"]),
            state_delta=dict(payload.get("state_delta") or {}),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True)
class TrajectoryStep:
    step_index: int
    observation: Observation
    available_actions: List[Action]
    action: Action
    result: StepResult
    decision: Optional[Decision] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.step_index < 0:
            raise ValueError("step_index must be non-negative")
        object.__setattr__(self, "available_actions", list(self.available_actions))
        object.__setattr__(self, "metadata", _json_copy(self.metadata))
        available_names = {item.name for item in self.available_actions}
        if available_names and self.action.name not in available_names:
            raise ValueError(
                f"Action {self.action.name!r} is not in available actions {sorted(available_names)}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "observation": self.observation.to_dict(),
            "available_actions": [item.to_dict() for item in self.available_actions],
            "action": self.action.to_dict(),
            "result": self.result.to_dict(),
            "decision": self.decision.to_dict() if self.decision is not None else None,
            "metadata": _json_copy(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TrajectoryStep":
        decision = payload.get("decision")
        return cls(
            step_index=int(payload["step_index"]),
            observation=Observation.from_dict(payload["observation"]),
            available_actions=[Action.from_dict(item) for item in payload.get("available_actions", [])],
            action=Action.from_dict(payload["action"]),
            result=StepResult.from_dict(payload["result"]),
            decision=Decision.from_dict(decision) if isinstance(decision, dict) else None,
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True)
class TrajectoryOutcome:
    status: str
    success: Optional[bool] = None
    reward: Optional[float] = None
    details: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.status.strip():
            raise ValueError("Trajectory outcome status must not be empty")
        object.__setattr__(self, "metadata", _json_copy(self.metadata))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "success": self.success,
            "reward": self.reward,
            "details": self.details,
            "metadata": _json_copy(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TrajectoryOutcome":
        raw_reward = payload.get("reward")
        return cls(
            status=str(payload["status"]),
            success=payload.get("success"),
            reward=float(raw_reward) if raw_reward is not None else None,
            details=str(payload.get("details", "")),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True)
class TrajectoryEpisode:
    episode_id: str
    benchmark: str
    task_id: str
    split: str
    actor: str
    evidence_tier: EvidenceTier
    initial_observation: Observation
    provenance: Dict[str, Any] = field(default_factory=dict)
    steps: List[TrajectoryStep] = field(default_factory=list)
    terminal_outcome: Optional[TrajectoryOutcome] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not _SAFE_EPISODE_ID_RE.fullmatch(self.episode_id):
            raise ValueError(
                "episode_id must contain only letters, digits, dot, underscore, colon, or hyphen"
            )
        for field_name in ("benchmark", "task_id", "actor"):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} must not be empty")
        tier = self.evidence_tier
        if not isinstance(tier, EvidenceTier):
            tier = EvidenceTier(str(tier))
            object.__setattr__(self, "evidence_tier", tier)
        object.__setattr__(self, "provenance", _json_copy(self.provenance))
        object.__setattr__(self, "steps", list(self.steps))
        object.__setattr__(self, "metadata", _json_copy(self.metadata))
        self.validate()

    def validate(self) -> None:
        previous = self.initial_observation
        for expected_index, step in enumerate(self.steps):
            if step.step_index != expected_index:
                raise ValueError(
                    f"Non-contiguous step index: expected {expected_index}, got {step.step_index}"
                )
            if step.observation.content_fingerprint() != previous.content_fingerprint():
                raise ValueError(f"Observation continuity mismatch at step {step.step_index}")
            previous = step.result.observation

    def with_step(self, step: TrajectoryStep) -> "TrajectoryEpisode":
        if self.terminal_outcome is not None:
            raise ValueError("Cannot append a step to a finalized trajectory")
        return TrajectoryEpisode(
            episode_id=self.episode_id,
            benchmark=self.benchmark,
            task_id=self.task_id,
            split=self.split,
            actor=self.actor,
            evidence_tier=self.evidence_tier,
            initial_observation=self.initial_observation,
            provenance=self.provenance,
            steps=[*self.steps, step],
            terminal_outcome=None,
            metadata=self.metadata,
            created_at=self.created_at,
        )

    def finalized(self, outcome: TrajectoryOutcome) -> "TrajectoryEpisode":
        if self.terminal_outcome is not None:
            raise ValueError("Trajectory is already finalized")
        return TrajectoryEpisode(
            episode_id=self.episode_id,
            benchmark=self.benchmark,
            task_id=self.task_id,
            split=self.split,
            actor=self.actor,
            evidence_tier=self.evidence_tier,
            initial_observation=self.initial_observation,
            provenance=self.provenance,
            steps=self.steps,
            terminal_outcome=outcome,
            metadata=self.metadata,
            created_at=self.created_at,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "benchmark": self.benchmark,
            "task_id": self.task_id,
            "split": self.split,
            "actor": self.actor,
            "evidence_tier": self.evidence_tier.value,
            "initial_observation": self.initial_observation.to_dict(),
            "provenance": _json_copy(self.provenance),
            "steps": [step.to_dict() for step in self.steps],
            "terminal_outcome": (
                self.terminal_outcome.to_dict()
                if self.terminal_outcome is not None
                else None
            ),
            "metadata": _json_copy(self.metadata),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TrajectoryEpisode":
        outcome = payload.get("terminal_outcome")
        return cls(
            episode_id=str(payload["episode_id"]),
            benchmark=str(payload["benchmark"]),
            task_id=str(payload["task_id"]),
            split=str(payload.get("split", "")),
            actor=str(payload["actor"]),
            evidence_tier=EvidenceTier(str(payload["evidence_tier"])),
            initial_observation=Observation.from_dict(payload["initial_observation"]),
            provenance=dict(payload.get("provenance") or {}),
            steps=[TrajectoryStep.from_dict(item) for item in payload.get("steps", [])],
            terminal_outcome=(
                TrajectoryOutcome.from_dict(outcome)
                if isinstance(outcome, dict)
                else None
            ),
            metadata=dict(payload.get("metadata") or {}),
            created_at=str(payload.get("created_at") or utc_now()),
        )

    def envelope(self) -> Dict[str, Any]:
        return {"format_version": FORMAT_VERSION, "episode": self.to_dict()}
