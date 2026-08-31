from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Dict, Iterable, List

from ..trajectory import Observation
from .efps import EFPSGraph
from .model import ModelImage


def observation_receipt(observation: Observation) -> Dict[str, Any]:
    """Exact public observation metadata exposed to a cognitive Agent."""
    allowed = {
        key: observation.structured[key]
        for key in (
            "environment_status",
            "levels_completed",
            "available_action_ids",
            "grid_shape",
        )
        if key in observation.structured
    }
    return {
        "content_fingerprint": observation.content_fingerprint(),
        "public_text": observation.text,
        "public_state": allowed,
        "artifacts": [
            {
                "artifact_id": item.artifact_id,
                "role": item.role,
                "media_type": item.media_type,
                "relative_path": item.relative_path,
                "sha256": item.sha256,
                "metadata": item.metadata,
            }
            for item in observation.artifacts
        ],
    }


def observation_image(
    observation: Observation,
    *,
    asset_root: str | Path,
    label: str,
) -> ModelImage:
    reference = next(
        (
            item
            for item in observation.artifacts
            if item.media_type == "image/png" and item.role == "agent_view"
        ),
        None,
    )
    if reference is None:
        raise ValueError("Observation has no raw public PNG agent view")
    path = Path(asset_root) / reference.relative_path
    payload = path.read_bytes()
    return ModelImage(
        label=label,
        artifact_id=reference.artifact_id,
        sha256=reference.sha256,
        relative_path=reference.relative_path,
        data_url="data:image/png;base64," + base64.b64encode(payload).decode("ascii"),
    )


def cognition_view(graph: EFPSGraph) -> Dict[str, Any]:
    """Compact, read-only EFPS view sent to acting Agents."""
    feature_names = {
        item.feature_id: item.name for item in graph.feature_definitions.values()
    }
    entity_features: Dict[str, List[Dict[str, Any]]] = {
        entity_id: [] for entity_id in graph.entities
    }
    for assertion in graph.feature_assertions.values():
        entity_features.setdefault(assertion.subject_id, []).append(
            {
                "assertion_id": assertion.assertion_id,
                "feature_id": assertion.feature_id,
                "feature_name": feature_names.get(assertion.feature_id, "unknown"),
                "value": assertion.value,
                "confidence": assertion.confidence,
                "description": assertion.description,
                "last_observed_step": assertion.last_observed_step,
                "evidence_ids": list(assertion.evidence_ids),
            }
        )
    schemas = []
    cited_evidence = set()
    for item in graph.entities.values():
        cited_evidence.update(item.evidence_ids)
    for item in graph.feature_assertions.values():
        cited_evidence.update(item.evidence_ids)
    for item in graph.prototypes.values():
        cited_evidence.update(item.evidence_ids)
    for item in graph.schemas.values():
        cited_evidence.update(item.support_evidence_ids)
        cited_evidence.update(item.counter_evidence_ids)
        schemas.append(
            {
                "schema_id": item.schema_id,
                "name": item.name,
                "role_bindings": item.role_bindings,
                "preconditions": item.preconditions,
                "action_pattern": item.action_pattern,
                "expected_changes": item.expected_changes,
                "invariants": item.invariants,
                "boundary_conditions": item.boundary_conditions,
                "support_evidence_ids": item.support_evidence_ids,
                "counter_evidence_ids": item.counter_evidence_ids,
                "confidence": item.confidence,
                "status": item.status.value,
            }
        )
    return {
        "view_policy": (
            "read-only current Entity features, Prototype memberships, Schemas, "
            "and node-cited Evidence summaries; redundant relation edges, full "
            "assertion histories, and artifact bytes omitted"
        ),
        "revision": graph.revision,
        "counts": graph.counts(),
        "entities": [
            {
                "entity_id": item.entity_id,
                "label": item.label,
                "bbox": item.bbox,
                "status": item.status.value,
                "features": sorted(
                    entity_features.get(item.entity_id, []),
                    key=lambda value: value["feature_name"],
                ),
                "evidence_ids": item.evidence_ids,
            }
            for item in graph.entities.values()
        ],
        "prototypes": [
            {
                "prototype_id": item.prototype_id,
                "name": item.name,
                "defining_feature_ids": item.defining_feature_ids,
                "member_confidences": item.member_confidences,
                "evidence_ids": item.evidence_ids,
            }
            for item in graph.prototypes.values()
        ],
        "schemas": schemas,
        "evidence": [
            {
                "evidence_id": item.evidence_id,
                "kind": item.kind,
                "step_index": item.step_index,
                "action": item.action,
                "result": item.result,
                "artifact_ids": item.artifact_ids,
                "semantic_summary": item.semantic_summary,
                "entity_changes": item.entity_changes,
                "unassigned_visual_changes": item.unassigned_visual_changes,
            }
            for evidence_id, item in graph.evidence.items()
            if evidence_id in cited_evidence
        ],
    }


def unique_images(values: Iterable[ModelImage]) -> List[ModelImage]:
    result: List[ModelImage] = []
    seen = set()
    for item in values:
        if item.artifact_id in seen:
            continue
        seen.add(item.artifact_id)
        result.append(item)
    return result


__all__ = [
    "cognition_view",
    "observation_image",
    "observation_receipt",
    "unique_images",
]
