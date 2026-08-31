from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

from .models import (
    Entity,
    EntityStatus,
    EvidenceRecord,
    FeatureAssertion,
    FeatureDefinition,
    FeatureKind,
    Insight,
    InsightKind,
    InsightStatus,
    Prototype,
    Relation,
    RelationType,
    Schema,
    SchemaStatus,
    stable_id,
    to_dict,
)
from .operations import GraphOperation, OperationKind, TransactionResult


COGNITION_CONTRACT_VERSION = 3


def _unique(values: Iterable[str]) -> List[str]:
    return sorted({str(item) for item in values if str(item)})


@dataclass
class EFPSGraph:
    """Typed Entity-Feature-Prototype-Schema graph with audited transactions."""

    evidence: Dict[str, EvidenceRecord] = field(default_factory=dict)
    entities: Dict[str, Entity] = field(default_factory=dict)
    feature_definitions: Dict[str, FeatureDefinition] = field(default_factory=dict)
    feature_assertions: Dict[str, FeatureAssertion] = field(default_factory=dict)
    prototypes: Dict[str, Prototype] = field(default_factory=dict)
    schemas: Dict[str, Schema] = field(default_factory=dict)
    insights: Dict[str, Insight] = field(default_factory=dict)
    relations: Dict[str, Relation] = field(default_factory=dict)
    revision: int = 0
    audit_log: List[Dict[str, Any]] = field(default_factory=list)

    def register_evidence(self, record: EvidenceRecord) -> None:
        existing = self.evidence.get(record.evidence_id)
        if existing is not None and to_dict(existing) != to_dict(record):
            raise ValueError(f"Evidence ID collision: {record.evidence_id}")
        self.evidence[record.evidence_id] = record

    def resolve_evidence(self, evidence_id: str) -> Dict[str, Any]:
        """Resolve a durable Evidence ID into its complete public audit record."""
        record = self._require(self.evidence, evidence_id, "evidence")
        return copy.deepcopy(to_dict(record))

    def annotate_evidence(
        self,
        evidence_id: str,
        *,
        semantic_summary: str,
        entity_changes: List[Dict[str, Any]],
        unassigned_visual_changes: List[str],
    ) -> None:
        """Attach Update-Agent semantics after their referenced Entities commit."""
        record = self._require(self.evidence, evidence_id, "evidence")
        normalized = []
        for change in entity_changes:
            value = copy.deepcopy(change)
            entity_id = str(value.get("entity_id") or "")
            if not entity_id:
                raise ValueError("A semantic entity change requires an Entity ID")
            self._require(self.entities, entity_id, "entity")
            normalized.append(value)
        record.semantic_summary = " ".join(str(semantic_summary).split())
        record.entity_changes = normalized
        record.unassigned_visual_changes = [
            " ".join(str(item).split())
            for item in unassigned_visual_changes
            if str(item).strip()
        ]

    def apply_transaction(
        self,
        operations: Iterable[GraphOperation],
        *,
        actor: str,
        step: int,
        mode: str,
        summary: str,
    ) -> TransactionResult:
        values = list(operations)
        if not values:
            raise ValueError("A graph transaction must contain at least one operation")
        candidate = copy.deepcopy(self)
        skipped = 0
        for operation in values:
            candidate._validate_operation_evidence(operation)
            if operation.kind == OperationKind.SKIP:
                skipped += 1
                continue
            candidate._apply(operation, step=step)
        candidate.revision += 1
        transaction_id = stable_id(
            "txn",
            {
                "revision": candidate.revision,
                "actor": actor,
                "step": step,
                "operations": [item.to_dict() for item in values],
            },
        )
        candidate.audit_log.append(
            {
                "transaction_id": transaction_id,
                "revision": candidate.revision,
                "step": step,
                "actor": actor,
                "mode": mode,
                "summary": summary,
                "operations": [item.to_dict() for item in values],
            }
        )
        candidate.validate()
        self.__dict__.update(candidate.__dict__)
        return TransactionResult(
            transaction_id=transaction_id,
            revision=self.revision,
            applied_operations=len(values) - skipped,
            skipped_operations=skipped,
            mode=mode,
            summary=summary,
            operation_kinds=[item.kind.value for item in values],
        )

    def _validate_operation_evidence(self, operation: GraphOperation) -> None:
        if operation.kind != OperationKind.SKIP and not operation.evidence_ids:
            raise ValueError(f"{operation.kind.value} requires durable evidence")
        missing = set(operation.evidence_ids) - set(self.evidence)
        if missing:
            raise ValueError(
                f"{operation.kind.value} references unknown evidence: {sorted(missing)}"
            )

    def _apply(self, operation: GraphOperation, *, step: int) -> None:
        payload = dict(operation.payload)
        evidence_ids = _unique(operation.evidence_ids)
        kind = operation.kind
        if kind == OperationKind.ADD_ENTITY:
            entity = Entity(
                entity_id=str(payload["entity_id"]),
                label=str(payload["label"]),
                bbox=[int(item) for item in payload["bbox"]],
                first_seen_step=int(payload.get("first_seen_step", step)),
                last_seen_step=int(payload.get("last_seen_step", step)),
                evidence_ids=evidence_ids,
                status=EntityStatus(payload.get("status", EntityStatus.ACTIVE.value)),
                metadata=dict(payload.get("metadata") or {}),
            )
            existing = self.entities.get(entity.entity_id)
            if existing is None:
                self.entities[entity.entity_id] = entity
            else:
                existing.last_seen_step = max(existing.last_seen_step, entity.last_seen_step)
                existing.evidence_ids = _unique([*existing.evidence_ids, *evidence_ids])
                existing.bbox = entity.bbox
                existing.status = entity.status
                existing.metadata.update(entity.metadata)
            return
        if kind == OperationKind.UPDATE_ENTITY:
            entity = self._require(self.entities, str(payload["entity_id"]), "entity")
            if "bbox" in payload:
                entity.bbox = [int(item) for item in payload["bbox"]]
            if "status" in payload:
                entity.status = EntityStatus(str(payload["status"]))
            entity.last_seen_step = int(payload.get("last_seen_step", step))
            entity.evidence_ids = _unique([*entity.evidence_ids, *evidence_ids])
            entity.metadata.update(dict(payload.get("metadata") or {}))
            return
        if kind == OperationKind.ADD_FEATURE_DEFINITION:
            definition = FeatureDefinition(
                feature_id=str(payload["feature_id"]),
                name=str(payload["name"]),
                kind=FeatureKind(str(payload["kind"])),
                description=str(payload["description"]),
                evidence_ids=evidence_ids,
            )
            existing = self.feature_definitions.get(definition.feature_id)
            if existing is None:
                self.feature_definitions[definition.feature_id] = definition
            else:
                if existing.name != definition.name or existing.kind != definition.kind:
                    raise ValueError(f"Conflicting feature definition: {definition.feature_id}")
                existing.evidence_ids = _unique([*existing.evidence_ids, *evidence_ids])
            return
        if kind == OperationKind.UPSERT_FEATURE_ASSERTION:
            subject_id = str(payload["subject_id"])
            feature_id = str(payload["feature_id"])
            self._require(self.entities, subject_id, "entity")
            self._require(self.feature_definitions, feature_id, "feature definition")
            assertion_id = str(
                payload.get("assertion_id")
                or stable_id("assertion", {"subject": subject_id, "feature": feature_id})
            )
            existing = self.feature_assertions.get(assertion_id)
            history_item = {
                "step": step,
                "value": payload.get("value"),
                "evidence_ids": evidence_ids,
                "description": str(payload.get("description") or ""),
            }
            if existing is None:
                assertion = FeatureAssertion(
                    assertion_id=assertion_id,
                    subject_id=subject_id,
                    feature_id=feature_id,
                    value=payload.get("value"),
                    confidence=float(payload.get("confidence", 1.0)),
                    evidence_ids=evidence_ids,
                    first_observed_step=step,
                    last_observed_step=step,
                    history=[history_item],
                    description=str(payload.get("description") or ""),
                )
                self.feature_assertions[assertion_id] = assertion
            else:
                existing.value = payload.get("value")
                existing.confidence = float(payload.get("confidence", existing.confidence))
                existing.last_observed_step = step
                if payload.get("description"):
                    existing.description = str(payload["description"])
                existing.evidence_ids = _unique([*existing.evidence_ids, *evidence_ids])
                existing.history.append(history_item)
            self._upsert_relation(
                RelationType.HAS_FEATURE,
                subject_id,
                assertion_id,
                evidence_ids,
            )
            self._upsert_relation(
                RelationType.ASSERTS_FEATURE,
                assertion_id,
                feature_id,
                evidence_ids,
            )
            return
        if kind == OperationKind.CREATE_PROTOTYPE:
            prototype = Prototype(
                prototype_id=str(payload["prototype_id"]),
                name=str(payload["name"]),
                defining_feature_ids=_unique(payload.get("defining_feature_ids", [])),
                optional_feature_ids=_unique(payload.get("optional_feature_ids", [])),
                exclusion_feature_ids=_unique(payload.get("exclusion_feature_ids", [])),
                member_confidences={
                    str(key): float(value)
                    for key, value in dict(payload.get("member_confidences") or {}).items()
                },
                evidence_ids=evidence_ids,
                metadata=dict(payload.get("metadata") or {}),
            )
            existing = self.prototypes.get(prototype.prototype_id)
            if existing is None:
                self.prototypes[prototype.prototype_id] = prototype
            else:
                if existing.name != prototype.name:
                    raise ValueError(f"Conflicting prototype: {prototype.prototype_id}")
                existing.evidence_ids = _unique([*existing.evidence_ids, *evidence_ids])
            for feature_id in prototype.defining_feature_ids:
                self._require(self.feature_definitions, feature_id, "feature definition")
                self._upsert_relation(
                    RelationType.DEFINED_BY,
                    prototype.prototype_id,
                    feature_id,
                    evidence_ids,
                )
            for feature_id in prototype.exclusion_feature_ids:
                self._require(self.feature_definitions, feature_id, "feature definition")
                self._upsert_relation(
                    RelationType.EXCLUDES,
                    prototype.prototype_id,
                    feature_id,
                    evidence_ids,
                )
            for entity_id, confidence in prototype.member_confidences.items():
                self._link_membership(entity_id, prototype.prototype_id, confidence, evidence_ids)
            return
        if kind == OperationKind.LINK_ENTITY_PROTOTYPE:
            self._link_membership(
                str(payload["entity_id"]),
                str(payload["prototype_id"]),
                float(payload.get("confidence", 1.0)),
                evidence_ids,
            )
            return
        if kind == OperationKind.CREATE_SCHEMA:
            schema = Schema(
                schema_id=str(payload["schema_id"]),
                prototype_id=str(payload["prototype_id"]),
                action=dict(payload.get("action") or {}),
                output=" ".join(str(payload.get("output") or "").split()),
                support_evidence_ids=evidence_ids,
                counter_evidence_ids=[],
                confidence=float(payload.get("confidence", 0.55)),
                metadata=dict(payload.get("metadata") or {}),
            )
            if schema.schema_id in self.schemas:
                raise ValueError(f"Schema already exists: {schema.schema_id}")
            self._require(self.prototypes, schema.prototype_id, "prototype")
            self.schemas[schema.schema_id] = schema
            self._upsert_relation(
                RelationType.TAKES_PROTOTYPE,
                schema.schema_id,
                schema.prototype_id,
                evidence_ids,
            )
            return
        if kind == OperationKind.ADD_SCHEMA_SUPPORT:
            schema = self._require(self.schemas, str(payload["schema_id"]), "schema")
            schema.support_evidence_ids = _unique(
                [*schema.support_evidence_ids, *evidence_ids]
            )
            schema.confidence = min(0.95, schema.confidence + 0.08 * len(evidence_ids))
            observed = dict(payload.get("metadata") or {})
            schema.metadata.update(observed)
            return
        if kind == OperationKind.ADD_SCHEMA_COUNTEREVIDENCE:
            schema = self._require(self.schemas, str(payload["schema_id"]), "schema")
            schema.counter_evidence_ids = _unique(
                [*schema.counter_evidence_ids, *evidence_ids]
            )
            schema.confidence = max(0.05, schema.confidence - 0.12 * len(evidence_ids))
            return
        if kind == OperationKind.REVISE_SCHEMA:
            schema = self._require(self.schemas, str(payload["schema_id"]), "schema")
            if "prototype_id" in payload:
                prototype_id = str(payload["prototype_id"])
                self._require(self.prototypes, prototype_id, "prototype")
                self.relations = {
                    key: relation
                    for key, relation in self.relations.items()
                    if not (
                        relation.relation_type
                        in {RelationType.TAKES_PROTOTYPE, RelationType.BINDS_ROLE_TO}
                        and relation.source_id == schema.schema_id
                    )
                }
                schema.prototype_id = prototype_id
                self._upsert_relation(
                    RelationType.TAKES_PROTOTYPE,
                    schema.schema_id,
                    prototype_id,
                    evidence_ids,
                )
            if "action" in payload:
                schema.action = dict(payload["action"])
            if "output" in payload:
                schema.output = " ".join(str(payload["output"]).split())
            schema.support_evidence_ids = _unique(
                [*schema.support_evidence_ids, *evidence_ids]
            )
            schema.metadata.update(dict(payload.get("metadata") or {}))
            schema.status = SchemaStatus.REVISED
            schema.revision_count += 1
            schema.confidence = min(0.95, schema.confidence + 0.05)
            return
        if kind == OperationKind.CREATE_INSIGHT:
            insight = Insight(
                insight_id=str(payload["insight_id"]),
                kind=InsightKind(str(payload.get("kind", InsightKind.OTHER.value))),
                statement=" ".join(str(payload.get("statement") or "").split()),
                scope=" ".join(str(payload.get("scope") or "global").split()),
                support_evidence_ids=evidence_ids,
                counter_evidence_ids=[],
                confidence=float(payload.get("confidence", 0.5)),
                metadata=dict(payload.get("metadata") or {}),
            )
            if insight.insight_id in self.insights:
                raise ValueError(f"Insight already exists: {insight.insight_id}")
            self.insights[insight.insight_id] = insight
            return
        if kind == OperationKind.ADD_INSIGHT_SUPPORT:
            insight = self._require(
                self.insights, str(payload["insight_id"]), "insight"
            )
            insight.support_evidence_ids = _unique(
                [*insight.support_evidence_ids, *evidence_ids]
            )
            insight.confidence = min(
                0.95, insight.confidence + 0.08 * len(evidence_ids)
            )
            insight.metadata.update(dict(payload.get("metadata") or {}))
            return
        if kind == OperationKind.ADD_INSIGHT_COUNTEREVIDENCE:
            insight = self._require(
                self.insights, str(payload["insight_id"]), "insight"
            )
            insight.counter_evidence_ids = _unique(
                [*insight.counter_evidence_ids, *evidence_ids]
            )
            insight.confidence = max(
                0.05, insight.confidence - 0.12 * len(evidence_ids)
            )
            return
        if kind == OperationKind.REVISE_INSIGHT:
            insight = self._require(
                self.insights, str(payload["insight_id"]), "insight"
            )
            if "kind" in payload:
                insight.kind = InsightKind(str(payload["kind"]))
            if "statement" in payload:
                insight.statement = " ".join(str(payload["statement"]).split())
            if "scope" in payload:
                insight.scope = " ".join(str(payload["scope"]).split())
            insight.support_evidence_ids = _unique(
                [*insight.support_evidence_ids, *evidence_ids]
            )
            insight.metadata.update(dict(payload.get("metadata") or {}))
            insight.status = InsightStatus.REVISED
            insight.revision_count += 1
            insight.confidence = min(0.95, insight.confidence + 0.05)
            return
        raise ValueError(f"Unsupported graph operation: {kind}")

    def _link_membership(
        self,
        entity_id: str,
        prototype_id: str,
        confidence: float,
        evidence_ids: List[str],
    ) -> None:
        self._require(self.entities, entity_id, "entity")
        prototype = self._require(self.prototypes, prototype_id, "prototype")
        prototype.member_confidences[entity_id] = max(0.0, min(1.0, confidence))
        prototype.evidence_ids = _unique([*prototype.evidence_ids, *evidence_ids])
        self._upsert_relation(
            RelationType.INSTANCE_OF,
            entity_id,
            prototype_id,
            evidence_ids,
            metadata={"confidence": prototype.member_confidences[entity_id]},
        )

    def _upsert_relation(
        self,
        relation_type: RelationType,
        source_id: str,
        target_id: str,
        evidence_ids: List[str],
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        value = dict(metadata or {})
        relation_id = stable_id(
            "relation",
            {
                "type": relation_type.value,
                "source": source_id,
                "target": target_id,
                "role": value.get("role", ""),
            },
        )
        existing = self.relations.get(relation_id)
        if existing is None:
            self.relations[relation_id] = Relation(
                relation_id=relation_id,
                relation_type=relation_type,
                source_id=source_id,
                target_id=target_id,
                evidence_ids=_unique(evidence_ids),
                metadata=value,
            )
        else:
            existing.evidence_ids = _unique([*existing.evidence_ids, *evidence_ids])
            existing.metadata.update(value)

    @staticmethod
    def _require(mapping: Dict[str, Any], item_id: str, kind: str):
        value = mapping.get(item_id)
        if value is None:
            raise KeyError(f"Unknown {kind}: {item_id}")
        return value

    def validate(self) -> None:
        evidence_ids = set(self.evidence)
        for kind, values in (
            ("entity", self.entities.values()),
            ("feature definition", self.feature_definitions.values()),
            ("feature assertion", self.feature_assertions.values()),
            ("prototype", self.prototypes.values()),
        ):
            for value in values:
                missing = set(value.evidence_ids) - evidence_ids
                if not value.evidence_ids or missing:
                    raise ValueError(
                        f"{kind} {next(iter(to_dict(value).values()))} has invalid evidence: {sorted(missing)}"
                    )
        for assertion in self.feature_assertions.values():
            self._require(self.entities, assertion.subject_id, "entity")
            self._require(self.feature_definitions, assertion.feature_id, "feature definition")
            if not 0.0 <= assertion.confidence <= 1.0:
                raise ValueError(f"Invalid assertion confidence: {assertion.assertion_id}")
        for prototype in self.prototypes.values():
            for feature_id in [
                *prototype.defining_feature_ids,
                *prototype.optional_feature_ids,
                *prototype.exclusion_feature_ids,
            ]:
                self._require(self.feature_definitions, feature_id, "feature definition")
            for entity_id, confidence in prototype.member_confidences.items():
                self._require(self.entities, entity_id, "entity")
                if not 0.0 <= confidence <= 1.0:
                    raise ValueError(f"Invalid membership confidence: {prototype.prototype_id}")
        for schema in self.schemas.values():
            support = set(schema.support_evidence_ids)
            counter = set(schema.counter_evidence_ids)
            if not support or (support | counter) - evidence_ids:
                raise ValueError(f"Schema {schema.schema_id} has invalid evidence")
            self._require(self.prototypes, schema.prototype_id, "prototype")
            if not schema.action or not str(schema.action.get("name") or ""):
                raise ValueError(f"Schema {schema.schema_id} has no action")
            if not schema.output:
                raise ValueError(f"Schema {schema.schema_id} has no output")
            if not 0.0 <= schema.confidence <= 1.0:
                raise ValueError(f"Schema {schema.schema_id} has invalid confidence")
            input_edges = [
                relation
                for relation in self.relations.values()
                if relation.relation_type == RelationType.TAKES_PROTOTYPE
                and relation.source_id == schema.schema_id
            ]
            if len(input_edges) != 1 or input_edges[0].target_id != schema.prototype_id:
                raise ValueError(
                    f"Schema {schema.schema_id} requires exactly one Prototype input edge"
                )
        for insight in self.insights.values():
            support = set(insight.support_evidence_ids)
            counter = set(insight.counter_evidence_ids)
            if not support or (support | counter) - evidence_ids:
                raise ValueError(f"Insight {insight.insight_id} has invalid evidence")
            if not insight.statement:
                raise ValueError(f"Insight {insight.insight_id} has no statement")
            if not 0.0 <= insight.confidence <= 1.0:
                raise ValueError(f"Insight {insight.insight_id} has invalid confidence")
        all_nodes = {
            *self.entities,
            *self.feature_definitions,
            *self.feature_assertions,
            *self.prototypes,
            *self.schemas,
            *self.insights,
        }
        endpoints = {
            RelationType.HAS_FEATURE: (set(self.entities), set(self.feature_assertions)),
            RelationType.ASSERTS_FEATURE: (
                set(self.feature_assertions),
                set(self.feature_definitions),
            ),
            RelationType.INSTANCE_OF: (set(self.entities), set(self.prototypes)),
            RelationType.DEFINED_BY: (set(self.prototypes), set(self.feature_definitions)),
            RelationType.EXCLUDES: (set(self.prototypes), set(self.feature_definitions)),
            RelationType.TAKES_PROTOTYPE: (set(self.schemas), set(self.prototypes)),
            RelationType.BINDS_ROLE_TO: (set(self.schemas), set(self.prototypes)),
        }
        for relation in self.relations.values():
            if relation.source_id not in all_nodes or relation.target_id not in all_nodes:
                raise ValueError(f"Dangling relation: {relation.relation_id}")
            allowed_source, allowed_target = endpoints[relation.relation_type]
            if relation.source_id not in allowed_source or relation.target_id not in allowed_target:
                raise ValueError(f"Invalid typed relation: {relation.relation_id}")
            if set(relation.evidence_ids) - evidence_ids:
                raise ValueError(f"Relation {relation.relation_id} has invalid evidence")

    def counts(self) -> Dict[str, int]:
        return {
            "evidence": len(self.evidence),
            "entities": len(self.entities),
            "feature_definitions": len(self.feature_definitions),
            "feature_assertions": len(self.feature_assertions),
            "prototypes": len(self.prototypes),
            "schemas": len(self.schemas),
            "insights": len(self.insights),
            "relations": len(self.relations),
            "revision": self.revision,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format_version": COGNITION_CONTRACT_VERSION,
            "architecture": "entity-feature-prototype-schema-plus-insight",
            "revision": self.revision,
            "evidence": {key: to_dict(value) for key, value in sorted(self.evidence.items())},
            "entities": {key: to_dict(value) for key, value in sorted(self.entities.items())},
            "feature_definitions": {
                key: to_dict(value) for key, value in sorted(self.feature_definitions.items())
            },
            "feature_assertions": {
                key: to_dict(value) for key, value in sorted(self.feature_assertions.items())
            },
            "prototypes": {key: to_dict(value) for key, value in sorted(self.prototypes.items())},
            "schemas": {key: to_dict(value) for key, value in sorted(self.schemas.items())},
            "insights": {key: to_dict(value) for key, value in sorted(self.insights.items())},
            "relations": {key: to_dict(value) for key, value in sorted(self.relations.items())},
            "audit_log": copy.deepcopy(self.audit_log),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "EFPSGraph":
        format_version = int(payload.get("format_version", 0))
        if format_version not in {2, 3}:
            raise ValueError("Unsupported EFPS graph format")
        schema_payloads = payload.get("schemas", {})
        schemas: Dict[str, Schema] = {}
        for key, value in schema_payloads.items():
            if format_version == 3:
                schemas[key] = Schema(
                    **{**value, "status": SchemaStatus(value["status"])}
                )
                continue
            role_bindings = dict(value.get("role_bindings") or {})
            prototype_id = next(
                (
                    str(prototype_id)
                    for prototype_ids in role_bindings.values()
                    for prototype_id in prototype_ids
                ),
                "",
            )
            legacy_action = dict(value.get("action_pattern") or {})
            action = {
                "name": str(
                    legacy_action.get("name") or legacy_action.get("action") or ""
                ),
                "arguments": dict(legacy_action.get("arguments") or {}),
            }
            output = "; ".join(
                str(item) for item in value.get("expected_changes") or []
            ) or "No explicit output was stored in graph format 2."
            schemas[key] = Schema(
                schema_id=str(value["schema_id"]),
                prototype_id=prototype_id,
                action=action,
                output=output,
                support_evidence_ids=list(value.get("support_evidence_ids") or []),
                counter_evidence_ids=list(value.get("counter_evidence_ids") or []),
                confidence=float(value.get("confidence", 0.55)),
                status=SchemaStatus(value.get("status", "active")),
                revision_count=int(value.get("revision_count", 0)),
                metadata={
                    **dict(value.get("metadata") or {}),
                    "migrated_from_format_2": {
                        "name": value.get("name"),
                        "role_bindings": role_bindings,
                        "preconditions": value.get("preconditions") or [],
                        "invariants": value.get("invariants") or [],
                        "boundary_conditions": value.get("boundary_conditions") or [],
                    },
                },
            )
        relations = {}
        for key, value in payload.get("relations", {}).items():
            relation_type = RelationType(value["relation_type"])
            if format_version == 2 and relation_type == RelationType.BINDS_ROLE_TO:
                continue
            relations[key] = Relation(
                **{**value, "relation_type": relation_type}
            )
        if format_version == 2:
            for schema in schemas.values():
                relation_id = stable_id(
                    "relation",
                    {
                        "type": RelationType.TAKES_PROTOTYPE.value,
                        "source": schema.schema_id,
                        "target": schema.prototype_id,
                        "role": "",
                    },
                )
                relations[relation_id] = Relation(
                    relation_id=relation_id,
                    relation_type=RelationType.TAKES_PROTOTYPE,
                    source_id=schema.schema_id,
                    target_id=schema.prototype_id,
                    evidence_ids=list(schema.support_evidence_ids),
                    metadata={"migrated_from_format_2": True},
                )
        return cls(
            evidence={key: EvidenceRecord(**value) for key, value in payload.get("evidence", {}).items()},
            entities={
                key: Entity(**{**value, "status": EntityStatus(value["status"])})
                for key, value in payload.get("entities", {}).items()
            },
            feature_definitions={
                key: FeatureDefinition(**{**value, "kind": FeatureKind(value["kind"])})
                for key, value in payload.get("feature_definitions", {}).items()
            },
            feature_assertions={
                key: FeatureAssertion(**value)
                for key, value in payload.get("feature_assertions", {}).items()
            },
            prototypes={key: Prototype(**value) for key, value in payload.get("prototypes", {}).items()},
            schemas=schemas,
            insights={
                key: Insight(
                    **{
                        **value,
                        "kind": InsightKind(value["kind"]),
                        "status": InsightStatus(value["status"]),
                    }
                )
                for key, value in payload.get("insights", {}).items()
            },
            relations=relations,
            revision=int(payload.get("revision", 0)),
            audit_log=list(payload.get("audit_log", [])),
        )


__all__ = ["COGNITION_CONTRACT_VERSION", "EFPSGraph"]
