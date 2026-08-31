from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List


def stable_id(prefix: str, payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(encoded).hexdigest()[:16]}"


class FeatureKind(str, Enum):
    INTRINSIC = "intrinsic"
    STATE = "state"
    AFFORDANCE = "affordance"
    RELATIONAL = "relational"


class EntityStatus(str, Enum):
    ACTIVE = "active"
    OCCLUDED = "occluded"
    DISAPPEARED = "disappeared"


class SchemaStatus(str, Enum):
    ACTIVE = "active"
    REVISED = "revised"
    RETIRED = "retired"


class RelationType(str, Enum):
    HAS_FEATURE = "has_feature"
    ASSERTS_FEATURE = "asserts_feature"
    INSTANCE_OF = "instance_of"
    DEFINED_BY = "defined_by"
    EXCLUDES = "excludes"
    BINDS_ROLE_TO = "binds_role_to"


@dataclass
class EvidenceRecord:
    evidence_id: str
    kind: str
    episode_id: str
    step_index: int | None
    observation_fingerprints: List[str]
    action: Dict[str, Any] = field(default_factory=dict)
    result: Dict[str, Any] = field(default_factory=dict)
    artifact_ids: List[str] = field(default_factory=list)
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    semantic_summary: str = ""
    entity_changes: List[Dict[str, Any]] = field(default_factory=list)
    unassigned_visual_changes: List[str] = field(default_factory=list)
    observation_refs: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Entity:
    entity_id: str
    label: str
    bbox: List[int]
    first_seen_step: int
    last_seen_step: int
    evidence_ids: List[str]
    status: EntityStatus = EntityStatus.ACTIVE
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureDefinition:
    feature_id: str
    name: str
    kind: FeatureKind
    description: str
    evidence_ids: List[str]


@dataclass
class FeatureAssertion:
    assertion_id: str
    subject_id: str
    feature_id: str
    value: Any
    confidence: float
    evidence_ids: List[str]
    first_observed_step: int
    last_observed_step: int
    history: List[Dict[str, Any]] = field(default_factory=list)
    description: str = ""


@dataclass
class Prototype:
    prototype_id: str
    name: str
    defining_feature_ids: List[str]
    optional_feature_ids: List[str]
    exclusion_feature_ids: List[str]
    member_confidences: Dict[str, float]
    evidence_ids: List[str]
    revision_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Schema:
    schema_id: str
    name: str
    role_bindings: Dict[str, List[str]]
    preconditions: List[str]
    action_pattern: Dict[str, Any]
    expected_changes: List[str]
    invariants: List[str]
    boundary_conditions: List[str]
    support_evidence_ids: List[str]
    counter_evidence_ids: List[str]
    confidence: float = 0.55
    status: SchemaStatus = SchemaStatus.ACTIVE
    revision_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Relation:
    relation_id: str
    relation_type: RelationType
    source_id: str
    target_id: str
    evidence_ids: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


def to_dict(value: Any) -> Dict[str, Any]:
    payload = asdict(value)
    for key, item in list(payload.items()):
        if isinstance(item, Enum):
            payload[key] = item.value
    return payload


__all__ = [
    "Entity",
    "EntityStatus",
    "EvidenceRecord",
    "FeatureAssertion",
    "FeatureDefinition",
    "FeatureKind",
    "Prototype",
    "Relation",
    "RelationType",
    "Schema",
    "SchemaStatus",
    "stable_id",
    "to_dict",
]
