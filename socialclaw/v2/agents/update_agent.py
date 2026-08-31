from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..cognition import cognition_tool, render_sections, section_metrics
from ..efps import EFPSGraph, GraphOperation, OperationKind, stable_id
from ..model import ModelImage, StructuredVisionModel
from .prompts import UPDATE_INSTRUCTIONS
from .protocols import AgentCallAudit, UpdateProposal
from .validation import bounded_probability


_FEATURE_KINDS = {"intrinsic", "state", "affordance", "relational"}
_ENTITY_STATUSES = {"active", "occluded", "disappeared"}


def _clean_name(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


class UpdateAgent:
    """Game-agnostic child Agent that proposes evidence-grounded EFPS changes."""

    instruction_profile = "update_agent_v2_generic"

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
        update_input: Dict[str, Any],
        images: List[ModelImage],
        graph: EFPSGraph,
        evidence_id: str,
        episode_id: str,
        step: int,
        executed_action: Dict[str, Any] | None,
    ) -> UpdateProposal:
        total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        validation_warning = ""
        last_error: ValueError | None = None
        all_tool_trace: List[Dict[str, Any]] = []
        all_usage_rounds: List[Dict[str, Any]] = []
        sections = dict(update_input.get("prompt_sections") or {})
        input_text = render_sections(sections)
        for attempt in range(2):
            instructions = UPDATE_INSTRUCTIONS
            if attempt:
                instructions += (
                    "\n\nYour previous proposal was inconsistent with public_result and "
                    "the before/after images already present in the same input: "
                    f"{last_error}. No new observation or external fact is being provided. "
                    "Re-read the existing input and regenerate the complete JSON. Do not "
                    "assert an Entity change when its existing grid_changed field is false; "
                    "when that field is true, account for the visible difference with a "
                    "resolvable Entity change or unassigned_visual_changes."
                )
            result = self.model.generate(
                instructions=instructions,
                payload=input_text,
                images=images,
                tools=[cognition_tool(graph, self.artifact_root)],
            )
            all_tool_trace.extend(result.tool_trace)
            all_usage_rounds.extend(result.usage_rounds)
            for key in total_usage:
                total_usage[key] += int(result.usage.get(key) or 0)
            try:
                operations, warnings, local_entities = self._translate(
                    result.data,
                    graph=graph,
                    evidence_id=evidence_id,
                    episode_id=episode_id,
                    step=step,
                    grid_shape=self._grid_shape(update_input),
                    executed_action=executed_action,
                    scene_boundary=(
                        update_input.get("phase")
                        in {"public_level_boundary", "public_environment_reset"}
                        or int(
                            (update_input.get("public_result") or {}).get(
                                "level_delta"
                            )
                            or 0
                        )
                        > 0
                    ),
                )
                transition_analysis = self._transition_analysis(
                    result.data,
                    graph=graph,
                    local_entities=local_entities,
                    executed_action=executed_action,
                    public_result=dict(update_input.get("public_result") or {}),
                    warnings=warnings,
                )
            except ValueError as error:
                last_error = error
                if attempt == 0:
                    validation_warning = (
                        "Update output failed evidence validation once and was regenerated: "
                        f"{error}"
                    )
                    continue
                raise ValueError(
                    "Update Agent violated evidence constraints on both the initial "
                    "proposal and the correction retry"
                ) from error
            break
        else:  # pragma: no cover
            raise RuntimeError("Update validation retry loop did not produce a result")
        if validation_warning:
            warnings.append(validation_warning)
        if not operations:
            operations = [
                GraphOperation(
                    OperationKind.SKIP,
                    {},
                    [evidence_id],
                    "The update Agent proposed no valid evidence-grounded change.",
                )
            ]
        accommodating = {
            OperationKind.ADD_ENTITY,
            OperationKind.ADD_FEATURE_DEFINITION,
            OperationKind.CREATE_PROTOTYPE,
            OperationKind.CREATE_SCHEMA,
            OperationKind.REVISE_SCHEMA,
            OperationKind.ADD_SCHEMA_COUNTEREVIDENCE,
        }
        mode = (
            "accommodation"
            if any(item.kind in accommodating for item in operations)
            else "assimilation"
        )
        summary = _clean_name(result.data.get("scene_summary")) or (
            "No valid cognitive change was proposed."
        )
        audit = AgentCallAudit(
            instruction_profile=self.instruction_profile,
            received_refs=["event.update_input"],
            image_inputs=[item.audit_dict() for item in images],
            output=result.data,
            model=result.model,
            usage=total_usage,
            input_text=input_text,
            input_sections=section_metrics(sections),
            tool_trace=all_tool_trace,
            usage_rounds=all_usage_rounds,
        )
        return UpdateProposal(
            operations=operations,
            mode=mode,
            summary=summary,
            semantic_output=result.data,
            transition_analysis=transition_analysis,
            audit=audit,
            warnings=warnings,
        )

    @staticmethod
    def _grid_shape(payload: Dict[str, Any]) -> Tuple[int, int]:
        observation = payload.get("after_observation") or payload.get("observation") or {}
        state = observation.get("public_state") or {}
        shape = state.get("grid_shape") or [0, 0]
        if not isinstance(shape, list) or len(shape) != 2:
            return 0, 0
        return int(shape[0]), int(shape[1])

    def _translate(
        self,
        output: Dict[str, Any],
        *,
        graph: EFPSGraph,
        evidence_id: str,
        episode_id: str,
        step: int,
        grid_shape: Tuple[int, int],
        executed_action: Dict[str, Any] | None,
        scene_boundary: bool,
    ) -> Tuple[List[GraphOperation], List[str], Dict[str, str]]:
        evidence = [evidence_id]
        warnings: List[str] = []
        operations: List[GraphOperation] = []
        feature_by_name = {
            item.name.casefold(): item.feature_id
            for item in graph.feature_definitions.values()
        }
        entity_by_label = {
            item.label.casefold(): item.entity_id for item in graph.entities.values()
        }
        prototype_by_name = {
            item.name.casefold(): item.prototype_id for item in graph.prototypes.values()
        }
        schema_by_name = {
            item.name.casefold(): item.schema_id for item in graph.schemas.values()
        }
        local_entities: Dict[str, str] = {}

        valid_entities: List[Tuple[Dict[str, Any], str, List[int]]] = []
        for raw in output.get("entities") or []:
            if not isinstance(raw, dict):
                continue
            label = _clean_name(raw.get("label"))
            bbox = self._bbox(raw.get("bbox"), grid_shape)
            if not label or bbox is None:
                warnings.append("Discarded an entity without a valid label/bbox.")
                continue
            requested_id = str(raw.get("entity_id") or "")
            if requested_id and requested_id not in graph.entities:
                warnings.append(f"Ignored unknown entity_id {requested_id}.")
                requested_id = ""
            entity_id = requested_id or (
                None if scene_boundary else entity_by_label.get(label.casefold())
            )
            if not entity_id:
                entity_id = stable_id(
                    "entity",
                    {
                        "episode_id": episode_id,
                        "first_seen_step": step,
                        "label": label.casefold(),
                        "bbox": bbox,
                    },
                )
            ref = str(raw.get("ref") or entity_id)
            local_entities[ref] = entity_id
            entity_by_label[label.casefold()] = entity_id
            valid_entities.append((raw, entity_id, bbox))

            for feature in raw.get("features") or []:
                if not isinstance(feature, dict):
                    continue
                name = _clean_name(feature.get("name"))
                kind = str(feature.get("kind") or "state")
                if not name or kind not in _FEATURE_KINDS:
                    continue
                if name.casefold() not in feature_by_name:
                    feature_id = stable_id("feature", {"name": name.casefold()})
                    feature_by_name[name.casefold()] = feature_id
                    operations.append(
                        GraphOperation(
                            OperationKind.ADD_FEATURE_DEFINITION,
                            {
                                "feature_id": feature_id,
                                "name": name,
                                "kind": kind,
                                "description": _clean_name(feature.get("description"))
                                or "Agent-generated observable feature.",
                            },
                            evidence,
                            "Feature definition proposed by the update Agent from public evidence.",
                        )
                    )

        if scene_boundary:
            represented_entity_ids = {entity_id for _, entity_id, _ in valid_entities}
            operations[0:0] = [
                GraphOperation(
                    OperationKind.UPDATE_ENTITY,
                    {
                        "entity_id": entity.entity_id,
                        "bbox": list(entity.bbox),
                        "status": "disappeared",
                        "last_seen_step": step,
                    },
                    evidence,
                    "A public level/reset boundary ended the prior scene; this Entity was "
                    "not identified as visible in the recovered current scene.",
                )
                for entity in graph.entities.values()
                if entity.status.value == "active"
                and entity.entity_id not in represented_entity_ids
            ]

        for raw, entity_id, bbox in valid_entities:
            label = _clean_name(raw.get("label"))
            status = str(raw.get("status") or "active").lower()
            if status not in _ENTITY_STATUSES:
                status = "active"
            if entity_id in graph.entities:
                operations.append(
                    GraphOperation(
                        OperationKind.UPDATE_ENTITY,
                        {
                            "entity_id": entity_id,
                            "bbox": bbox,
                            "status": status,
                            "last_seen_step": step,
                        },
                        evidence,
                        "Update Agent matched this visual object to an existing Entity.",
                    )
                )
            else:
                operations.append(
                    GraphOperation(
                        OperationKind.ADD_ENTITY,
                        {
                            "entity_id": entity_id,
                            "label": label,
                            "bbox": bbox,
                            "status": status,
                            "first_seen_step": step,
                            "last_seen_step": step,
                        },
                        evidence,
                        "Update Agent proposed a visually grounded Entity.",
                    )
                )
            for feature in raw.get("features") or []:
                if not isinstance(feature, dict):
                    continue
                name = _clean_name(feature.get("name"))
                feature_id = feature_by_name.get(name.casefold())
                if not feature_id:
                    continue
                operations.append(
                    GraphOperation(
                        OperationKind.UPSERT_FEATURE_ASSERTION,
                        {
                            "subject_id": entity_id,
                            "feature_id": feature_id,
                            "value": feature.get("value"),
                            "confidence": bounded_probability(
                                feature.get("confidence"), default=0.5
                            ),
                            "description": _clean_name(feature.get("description")),
                        },
                        evidence,
                        "Update Agent asserted this feature from the current public evidence.",
                    )
                )

        for raw in output.get("prototypes") or []:
            if not isinstance(raw, dict):
                continue
            name = _clean_name(raw.get("name"))
            if not name:
                continue
            requested_id = str(raw.get("prototype_id") or "")
            if requested_id and requested_id not in graph.prototypes:
                warnings.append(f"Ignored unknown prototype_id {requested_id}.")
                requested_id = ""
            prototype_id = requested_id or prototype_by_name.get(name.casefold())
            member_ids = []
            for ref in raw.get("member_refs") or []:
                value = str(ref)
                entity_id = (
                    local_entities.get(value)
                    or (value if value in graph.entities else None)
                    or entity_by_label.get(value.casefold())
                )
                if entity_id and entity_id not in member_ids:
                    member_ids.append(entity_id)
            defining = [
                feature_by_name[item.casefold()]
                for item in map(_clean_name, raw.get("defining_feature_names") or [])
                if item.casefold() in feature_by_name
            ]
            if not prototype_id:
                prototype_id = stable_id("prototype", {"name": name.casefold()})
                prototype_by_name[name.casefold()] = prototype_id
                operations.append(
                    GraphOperation(
                        OperationKind.CREATE_PROTOTYPE,
                        {
                            "prototype_id": prototype_id,
                            "name": name,
                            "defining_feature_ids": defining,
                            "member_confidences": {
                                entity_id: 1.0 for entity_id in member_ids
                            },
                        },
                        evidence,
                        "Update Agent compressed evidenced commonality into a Prototype.",
                    )
                )
            else:
                prototype_by_name[name.casefold()] = prototype_id
                existing_members = graph.prototypes[prototype_id].member_confidences
                for entity_id in member_ids:
                    if entity_id not in existing_members:
                        operations.append(
                            GraphOperation(
                                OperationKind.LINK_ENTITY_PROTOTYPE,
                                {
                                    "entity_id": entity_id,
                                    "prototype_id": prototype_id,
                                    "confidence": 1.0,
                                },
                                evidence,
                                "Update Agent assigned a newly observed member to an existing Prototype.",
                            )
                        )

        if executed_action is None:
            if output.get("schema_updates"):
                warnings.append(
                    "Discarded Schema proposals from an observation without an action transition."
                )
            return operations, warnings, local_entities

        executed_name = str(executed_action.get("name") or "")
        for raw in output.get("schema_updates") or []:
            if not isinstance(raw, dict):
                continue
            operation = str(raw.get("operation") or "").lower()
            name = _clean_name(raw.get("name"))
            requested_id = str(raw.get("schema_id") or "")
            schema_id = (
                requested_id if requested_id in graph.schemas else schema_by_name.get(name.casefold())
            )
            action_pattern = raw.get("action") or {}
            if not isinstance(action_pattern, dict):
                action_pattern = {}
            if str(action_pattern.get("name") or "") != executed_name:
                warnings.append(
                    "Discarded a Schema update whose action did not match the evidenced action."
                )
                continue
            normalized_action = {
                "action": executed_name,
                "arguments": dict(action_pattern.get("arguments") or {}),
            }
            role_bindings: Dict[str, List[str]] = {}
            for binding in raw.get("role_bindings") or []:
                if not isinstance(binding, dict):
                    continue
                role = _clean_name(binding.get("role"))
                target = str(binding.get("prototype") or "")
                prototype_id = (
                    target
                    if target in graph.prototypes or target in prototype_by_name.values()
                    else prototype_by_name.get(target.casefold())
                )
                if role and prototype_id:
                    role_bindings.setdefault(role, []).append(prototype_id)
            if operation == "revise" and schema_id and not role_bindings:
                role_bindings = {
                    role: list(prototype_ids)
                    for role, prototype_ids in graph.schemas[
                        schema_id
                    ].role_bindings.items()
                }
            common = {
                "role_bindings": role_bindings,
                "preconditions": [str(item) for item in raw.get("preconditions") or []],
                "action_pattern": normalized_action,
                "expected_changes": [str(item) for item in raw.get("expected_changes") or []],
                "invariants": [str(item) for item in raw.get("invariants") or []],
                "boundary_conditions": [
                    str(item) for item in raw.get("boundary_conditions") or []
                ],
            }
            if operation in {"create", "revise"} and not role_bindings:
                warnings.append(
                    "Discarded a Schema update without a resolved Prototype role binding."
                )
                continue
            if operation == "create" and not schema_id and name:
                schema_id = stable_id(
                    "schema",
                    {"name": name.casefold(), "action": normalized_action},
                )
                schema_by_name[name.casefold()] = schema_id
                operations.append(
                    GraphOperation(
                        OperationKind.CREATE_SCHEMA,
                        {
                            "schema_id": schema_id,
                            "name": name,
                            "role_bindings": role_bindings,
                            **common,
                            "confidence": bounded_probability(
                                raw.get("confidence"), default=0.55
                            ),
                            "metadata": {"agent_generated": True},
                        },
                        evidence,
                        _clean_name(raw.get("reason"))
                        or "Update Agent proposed a new grounded Schema.",
                    )
                )
            elif schema_id and operation == "support":
                operations.append(
                    GraphOperation(
                        OperationKind.ADD_SCHEMA_SUPPORT,
                        {"schema_id": schema_id},
                        evidence,
                        _clean_name(raw.get("reason"))
                        or "The transition supports an existing Schema.",
                    )
                )
            elif schema_id and operation == "counterevidence":
                operations.append(
                    GraphOperation(
                        OperationKind.ADD_SCHEMA_COUNTEREVIDENCE,
                        {"schema_id": schema_id},
                        evidence,
                        _clean_name(raw.get("reason"))
                        or "The transition contradicts an existing Schema.",
                    )
                )
            elif schema_id and operation == "revise":
                operations.append(
                    GraphOperation(
                        OperationKind.REVISE_SCHEMA,
                        {"schema_id": schema_id, **common},
                        evidence,
                        _clean_name(raw.get("reason"))
                        or "The transition requires revising an existing Schema.",
                    )
                )
            else:
                warnings.append("Discarded an unresolved Schema update.")
        return operations, warnings, local_entities

    @staticmethod
    def _transition_analysis(
        output: Dict[str, Any],
        *,
        graph: EFPSGraph,
        local_entities: Dict[str, str],
        executed_action: Dict[str, Any] | None,
        public_result: Dict[str, Any],
        warnings: List[str],
    ) -> Dict[str, Any] | None:
        if executed_action is None:
            if output.get("transition_analysis") is not None:
                warnings.append(
                    "Ignored transition_analysis for an initial observation."
                )
            return None

        raw = output.get("transition_analysis")
        if not isinstance(raw, dict):
            raise ValueError(
                "Update Agent must analyze Entity-level effects after every action"
            )
        summary = _clean_name(raw.get("summary"))
        if not summary:
            raise ValueError("transition_analysis requires a semantic summary")

        label_to_id = {
            item.label.casefold(): item.entity_id for item in graph.entities.values()
        }
        proposed_labels: Dict[str, str] = {}
        for entity in output.get("entities") or []:
            if not isinstance(entity, dict):
                continue
            ref = str(entity.get("ref") or "")
            label = _clean_name(entity.get("label"))
            entity_id = local_entities.get(ref)
            if entity_id and label:
                proposed_labels[entity_id] = label
                label_to_id[label.casefold()] = entity_id

        allowed_types = {
            "appeared",
            "disappeared",
            "moved",
            "state_changed",
            "feature_changed",
        }
        entity_changes: List[Dict[str, Any]] = []
        unresolved: List[str] = []
        for change in raw.get("entity_changes") or []:
            if not isinstance(change, dict):
                continue
            entity_ref = str(
                change.get("entity_id") or change.get("entity_ref") or ""
            )
            label = _clean_name(change.get("label"))
            entity_id = ""
            if entity_ref in graph.entities:
                entity_id = entity_ref
            elif entity_ref in local_entities:
                entity_id = local_entities[entity_ref]
            elif entity_ref in local_entities.values():
                entity_id = entity_ref
            elif label:
                entity_id = label_to_id.get(label.casefold(), "")
            if not entity_id:
                unresolved.append(
                    _clean_name(change.get("description"))
                    or f"Unresolved changed visual object: {label or entity_ref or 'unknown'}"
                )
                warnings.append(
                    f"Could not resolve transition Entity: {label or entity_ref or 'unknown'}."
                )
                continue
            change_type = str(change.get("change_type") or "")
            if change_type not in allowed_types:
                warnings.append(
                    f"Discarded unsupported Entity change type: {change_type or 'missing'}."
                )
                continue
            resolved_label = (
                graph.entities[entity_id].label
                if entity_id in graph.entities
                else proposed_labels.get(entity_id, label)
            )
            description = _clean_name(change.get("description"))
            before = _clean_name(change.get("before"))
            after = _clean_name(change.get("after"))
            if not description and not (before or after):
                warnings.append(
                    f"Discarded an unexplained change for Entity {entity_id}."
                )
                continue
            entity_changes.append(
                {
                    "entity_id": entity_id,
                    "label": resolved_label,
                    "change_type": change_type,
                    "before": before,
                    "after": after,
                    "description": description,
                    "confidence": bounded_probability(
                        change.get("confidence"), default=0.5
                    ),
                }
            )

        unassigned = [
            _clean_name(item)
            for item in raw.get("unassigned_visual_changes") or []
            if _clean_name(item)
        ]
        unassigned.extend(unresolved)
        grid_changed = bool(public_result.get("grid_changed"))
        if grid_changed and not entity_changes and not unassigned:
            raise ValueError(
                "A changed grid requires Entity changes or an explicit unassigned visual change"
            )
        if not grid_changed and entity_changes:
            raise ValueError(
                "Entity changes cannot be asserted when the public grid did not change"
            )
        return {
            "summary": summary,
            "entity_changes": entity_changes,
            "unassigned_visual_changes": unassigned,
        }

    @staticmethod
    def _bbox(value: Any, shape: Tuple[int, int]) -> List[int] | None:
        if not isinstance(value, list) or len(value) != 4:
            return None
        try:
            left, top, right, bottom = [int(item) for item in value]
        except (TypeError, ValueError):
            return None
        height, width = shape
        if (
            height <= 0
            or width <= 0
            or left < 0
            or top < 0
            or right < left
            or bottom < top
            or right >= width
            or bottom >= height
        ):
            return None
        return [left, top, right, bottom]


__all__ = ["UpdateAgent"]
