from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List


class OperationKind(str, Enum):
    ADD_ENTITY = "add_entity"
    UPDATE_ENTITY = "update_entity"
    ADD_FEATURE_DEFINITION = "add_feature_definition"
    UPSERT_FEATURE_ASSERTION = "upsert_feature_assertion"
    CREATE_PROTOTYPE = "create_prototype"
    LINK_ENTITY_PROTOTYPE = "link_entity_prototype"
    CREATE_SCHEMA = "create_schema"
    ADD_SCHEMA_SUPPORT = "add_schema_support"
    ADD_SCHEMA_COUNTEREVIDENCE = "add_schema_counterevidence"
    REVISE_SCHEMA = "revise_schema"
    SKIP = "skip"


@dataclass(frozen=True)
class GraphOperation:
    kind: OperationKind
    payload: Dict[str, Any]
    evidence_ids: List[str]
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["kind"] = self.kind.value
        return payload


@dataclass(frozen=True)
class TransactionResult:
    transaction_id: str
    revision: int
    applied_operations: int
    skipped_operations: int
    mode: str
    summary: str
    operation_kinds: List[str] = field(default_factory=list)


__all__ = ["GraphOperation", "OperationKind", "TransactionResult"]
