from __future__ import annotations

import base64
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .efps import EFPSGraph
from .efps.models import to_dict
from .model import ModelImage, ModelTool, ModelToolResult


def _clean(value: Any, limit: int = 180) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _value(value: Any, limit: int = 100) -> str:
    if isinstance(value, str):
        return _clean(value, limit)
    return _clean(json.dumps(value, ensure_ascii=False, sort_keys=True), limit)


def _refs(values: Iterable[str], *, limit: int = 2) -> str:
    items = list(dict.fromkeys(str(item) for item in values if str(item)))
    if not items:
        return "none"
    visible = items[-limit:]
    suffix = f" (+{len(items) - limit} earlier)" if len(items) > limit else ""
    return ", ".join(visible) + suffix


def render_cognition_catalog(graph: EFPSGraph) -> str:
    """Render every EFPS node class needed for decisions without raw graph JSON."""
    visible_entities = {
        key: item
        for key, item in graph.entities.items()
        if item.status.value == "active"
    }
    feature_names = {
        item.feature_id: item.name for item in graph.feature_definitions.values()
    }
    assertions: Dict[str, List[str]] = {key: [] for key in visible_entities}
    for item in graph.feature_assertions.values():
        if item.subject_id not in visible_entities:
            continue
        name = feature_names.get(item.feature_id, item.feature_id)
        description = _clean(item.description, 80)
        rendered = f"{name}={_value(item.value)} (confidence={item.confidence:.2f}"
        if description and description.casefold() != name.casefold():
            rendered += f", {description}"
        rendered += ")"
        assertions.setdefault(item.subject_id, []).append(rendered)

    prototype_memberships: Dict[str, List[str]] = {key: [] for key in visible_entities}
    for prototype in graph.prototypes.values():
        for entity_id, confidence in prototype.member_confidences.items():
            if entity_id in visible_entities:
                prototype_memberships.setdefault(entity_id, []).append(
                    f"{prototype.prototype_id}:{prototype.name} ({confidence:.2f})"
                )

    lines = [
        f"EFPS revision {graph.revision}; {len(visible_entities)} currently visible Entities "
        f"({len(graph.entities)} stored), "
        f"{len(graph.prototypes)} Prototypes, {len(graph.schemas)} Schemas.",
        "Entities (currently visible):",
    ]
    if not visible_entities:
        lines.append("- none")
    for entity in visible_entities.values():
        features = "; ".join(assertions.get(entity.entity_id) or []) or "none yet"
        memberships = "; ".join(prototype_memberships.get(entity.entity_id) or [])
        relation_text = f"; instance_of={memberships}" if memberships else ""
        lines.append(
            f"- {entity.entity_id} — {entity.label}; bbox={entity.bbox}; "
            f"important features: {features}"
            f"{relation_text}; evidence={_refs(entity.evidence_ids)}"
        )

    lines.append("Prototypes (all current prototypes):")
    if not graph.prototypes:
        lines.append("- none")
    for prototype in graph.prototypes.values():
        defining = [
            feature_names.get(item, item) for item in prototype.defining_feature_ids
        ]
        optional = [
            feature_names.get(item, item) for item in prototype.optional_feature_ids
        ]
        excluded = [
            feature_names.get(item, item) for item in prototype.exclusion_feature_ids
        ]
        members = [
            f"{entity_id}:{visible_entities[entity_id].label} ({confidence:.2f})"
            for entity_id, confidence in prototype.member_confidences.items()
            if entity_id in visible_entities
        ]
        optional_text = f"; optional={', '.join(optional)}" if optional else ""
        excluded_text = f"; excludes={', '.join(excluded)}" if excluded else ""
        lines.append(
            f"- {prototype.prototype_id} — {prototype.name}; "
            f"members={'; '.join(members) or 'none currently visible'}; "
            f"defined_by={', '.join(defining) or 'none'}"
            f"{optional_text}{excluded_text}; "
            f"evidence={_refs(prototype.evidence_ids)}"
        )

    lines.append("Schemas (all current schemas):")
    if not graph.schemas:
        lines.append("- none")
    for schema in graph.schemas.values():
        bindings = []
        for role, prototype_ids in schema.role_bindings.items():
            names = [
                f"{item}:{graph.prototypes[item].name}"
                if item in graph.prototypes
                else item
                for item in prototype_ids
            ]
            bindings.append(f"{role} -> {', '.join(names)}")
        lines.append(
            f"- {schema.schema_id} — {schema.name}; status={schema.status.value}; "
            f"confidence={schema.confidence:.2f}; roles={'; '.join(bindings) or 'none'}; "
            f"when={_clean('; '.join(schema.preconditions), 220) or 'unspecified'}; "
            f"action={_value(schema.action_pattern, 140)}; "
            f"expect={_clean('; '.join(schema.expected_changes), 220) or 'unspecified'}; "
            f"boundaries={_clean('; '.join(schema.boundary_conditions), 180) or 'none'}; "
            f"support={_refs(schema.support_evidence_ids)}; "
            f"counter={_refs(schema.counter_evidence_ids)}"
        )
    return "\n".join(lines)


def render_public_observation(observation: Mapping[str, Any]) -> str:
    state = observation.get("public_state") or {}
    return (
        f"status={state.get('environment_status')}; "
        f"levels_completed={state.get('levels_completed')}; "
        f"grid_shape={state.get('grid_shape')}; "
        f"available_action_ids={state.get('available_action_ids')}. "
        "The attached current_public_observation image is the visual content."
    )


def render_action_contracts(contracts: Sequence[Mapping[str, Any]]) -> str:
    lines = []
    for item in contracts:
        schema = item.get("arguments_schema") or {}
        properties = schema.get("properties") or {}
        if not properties:
            lines.append(f"- {item.get('name')}: no arguments")
            continue
        values = []
        for name, definition in properties.items():
            bounds = ""
            if "minimum" in definition or "maximum" in definition:
                bounds = f" [{definition.get('minimum')}..{definition.get('maximum')}]"
            description = _clean(definition.get("description"), 120)
            values.append(
                f"{name}: {definition.get('type', 'unknown')}{bounds}"
                + (f"; {description}" if description else "")
            )
        lines.append(f"- {item.get('name')}: " + " | ".join(values))
    return "\n".join(lines) or "- none"


def render_recent_transitions(values: Sequence[Mapping[str, Any]], limit: int = 3) -> str:
    if not values:
        return "- none yet"
    lines = []
    for item in values[-limit:]:
        result = item.get("public_result") or {}
        semantic = item.get("semantic_result") or {}
        action = item.get("action") or {}
        arguments = action.get("arguments") or {}
        action_text = str(action.get("name") or "unknown")
        if arguments:
            action_text += f"({_value(arguments, 100)})"
        lines.append(
            f"- Step {item.get('step')}: {action_text}; changed={result.get('grid_changed')}, "
            f"cells={result.get('changed_cells')}, bounds={result.get('changed_bounds')}, "
            f"level_delta={result.get('level_delta')}, status={result.get('environment_status')}; "
            f"Entity-level observation: {_clean(semantic.get('summary'), 300) or 'none'}; "
            f"Evidence={item.get('evidence_id')}"
        )
    return "\n".join(lines)


def render_sections(sections: Mapping[str, str]) -> str:
    return "\n\n".join(
        f"## {name}\n{value.strip()}" for name, value in sections.items()
    ).strip()


def section_metrics(sections: Mapping[str, str]) -> List[Dict[str, Any]]:
    total = sum(len(value) for value in sections.values()) or 1
    return [
        {
            "section": name,
            "characters": len(value),
            "utf8_bytes": len(value.encode("utf-8")),
            "character_share": round(len(value) / total, 6),
        }
        for name, value in sections.items()
    ]


def decision_sections(
    *,
    observation: Mapping[str, Any],
    contracts: Sequence[Mapping[str, Any]],
    recent_transitions: Sequence[Mapping[str, Any]],
    graph: EFPSGraph,
    level_budget: Mapping[str, Any] | None = None,
) -> "OrderedDict[str, str]":
    sections: "OrderedDict[str, str]" = OrderedDict(
        [
            ("Current public observation", render_public_observation(observation)),
            ("Legal public actions", render_action_contracts(contracts)),
            ("Recent public transitions", render_recent_transitions(recent_transitions)),
            ("Current learned cognition", render_cognition_catalog(graph)),
            (
                "On-demand memory",
                "Use read_cognition(command, id, optional feature_id) only with an exact "
                "ID copied from this catalog. Commands: get_entity, get_prototype, "
                "get_schema, get_evidence, get_feature_history, get_relations, "
                "get_artifact. It returns stored JSON records; get_artifact attaches the "
                "saved public PNG. It does no natural-language search or summarization.",
            ),
        ]
    )
    if level_budget:
        sections["Current level action budget"] = (
            f"level={level_budget.get('level')}; "
            f"used={level_budget.get('actions_used')}; "
            f"remaining={level_budget.get('actions_remaining')}; "
            f"maximum={level_budget.get('max_actions')}. "
            "Completing the level resets this budget; an environment recovery reset does not."
        )
    return sections


def update_sections(
    *,
    phase: str,
    evidence_id: str,
    graph: EFPSGraph,
    observation: Mapping[str, Any] | None = None,
    before_observation: Mapping[str, Any] | None = None,
    after_observation: Mapping[str, Any] | None = None,
    executed_action: Mapping[str, Any] | None = None,
    public_result: Mapping[str, Any] | None = None,
    main_decision: Mapping[str, Any] | None = None,
) -> "OrderedDict[str, str]":
    sections: "OrderedDict[str, str]" = OrderedDict()
    sections["Update phase"] = f"phase={phase}; durable Evidence ID for this update={evidence_id}"
    if observation is not None:
        sections["Current public observation"] = render_public_observation(observation)
        if phase == "public_environment_reset":
            sections["Public reset recovery"] = (
                "The public environment reported GAME_OVER and the runtime reset the same "
                "level. This is recovery bookkeeping, not an Agent-selected action. Re-read "
                "the complete current image; do not create an ENV_RESET action Schema."
            )
    else:
        sections["Before public observation"] = render_public_observation(
            before_observation or {}
        )
        sections["Executed public action"] = _value(executed_action or {}, 300)
        sections["Public transition result"] = _value(public_result or {}, 400)
        sections["After public observation"] = render_public_observation(
            after_observation or {}
        )
        if phase == "public_level_boundary":
            sections["Public level boundary"] = (
                "public_result.level_delta is positive: the action completed the previous "
                "level, and the attached after image may already be the next level. Treat "
                "completion as the terminal action effect; treat the after image as a new "
                "current scene rather than as an ordinary scene-wide action effect."
            )
        decision = main_decision or {}
        sections["Decision context relevant to learning"] = (
            f"mode={decision.get('decision_mode')}; schemas_used={decision.get('schema_ids')}; "
            f"schema_prediction={_clean(decision.get('schema_prediction'), 10000) or 'none'}; "
            f"exploration_hypothesis={_clean(decision.get('exploration_hypothesis'), 10000) or 'none'}"
        )
    sections["Current learned cognition"] = render_cognition_catalog(graph)
    sections["On-demand memory"] = (
        "Use read_cognition(command, id, optional feature_id) only with an exact stored "
        "ID. Commands: get_entity, get_prototype, get_schema, get_evidence, "
        "get_feature_history, get_relations, get_artifact. Results are stored JSON "
        "records; get_artifact attaches an agent-visible saved public PNG. There is no "
        "natural-language search, ranking, or summarization."
    )
    return sections


def _json_result(command: str, item_id: str, **values: Any) -> str:
    return json.dumps(
        {"ok": True, "command": command, "id": item_id, **values},
        ensure_ascii=False,
        sort_keys=True,
    )


def _not_found(command: str, item_id: str) -> str:
    return json.dumps(
        {"ok": False, "command": command, "id": item_id, "error": "not_found"},
        ensure_ascii=False,
        sort_keys=True,
    )


def _relations_for(graph: EFPSGraph, node_id: str) -> List[Dict[str, Any]]:
    return [
        to_dict(item)
        for item in graph.relations.values()
        if item.source_id == node_id or item.target_id == node_id
    ]


def _agent_evidence_record(record: Any) -> Dict[str, Any]:
    value = to_dict(record)
    allowed_artifacts = {
        item["artifact_id"]
        for item in value.get("artifacts", [])
        if item.get("role") == "agent_view"
    }
    value["artifacts"] = [
        item
        for item in value.get("artifacts", [])
        if item.get("artifact_id") in allowed_artifacts
    ]
    value["artifact_ids"] = sorted(allowed_artifacts)
    value["observation_refs"] = [
        {
            **item,
            "artifact_ids": [
                artifact_id
                for artifact_id in item.get("artifact_ids", [])
                if artifact_id in allowed_artifacts
            ],
        }
        for item in value.get("observation_refs", [])
    ]
    return value


def read_cognition(
    graph: EFPSGraph,
    *,
    command: str,
    item_id: str,
    feature_id: str | None = None,
) -> str:
    """Execute one exact dictionary lookup; never score, infer, or summarize."""
    if command == "get_entity":
        item = graph.entities.get(item_id)
        if item is None:
            return _not_found(command, item_id)
        assertions = [
            to_dict(value)
            for value in graph.feature_assertions.values()
            if value.subject_id == item_id
        ]
        return _json_result(
            command,
            item_id,
            record=to_dict(item),
            feature_assertions=assertions,
            relations=_relations_for(graph, item_id),
        )
    if command == "get_prototype":
        item = graph.prototypes.get(item_id)
        if item is None:
            return _not_found(command, item_id)
        feature_ids = {
            *item.defining_feature_ids,
            *item.optional_feature_ids,
            *item.exclusion_feature_ids,
        }
        return _json_result(
            command,
            item_id,
            record=to_dict(item),
            feature_definitions=[
                to_dict(graph.feature_definitions[value])
                for value in sorted(feature_ids)
                if value in graph.feature_definitions
            ],
            relations=_relations_for(graph, item_id),
        )
    if command == "get_schema":
        item = graph.schemas.get(item_id)
        return (
            _json_result(command, item_id, record=to_dict(item))
            if item is not None
            else _not_found(command, item_id)
        )
    if command == "get_evidence":
        item = graph.evidence.get(item_id)
        return (
            _json_result(command, item_id, record=_agent_evidence_record(item))
            if item is not None
            else _not_found(command, item_id)
        )
    if command == "get_feature_history":
        if item_id not in graph.entities:
            return _not_found(command, item_id)
        records = [
            to_dict(item)
            for item in graph.feature_assertions.values()
            if item.subject_id == item_id
            and (feature_id is None or item.feature_id == feature_id)
        ]
        return _json_result(command, item_id, records=records)
    if command == "get_relations":
        all_ids = {
            *graph.entities,
            *graph.feature_definitions,
            *graph.feature_assertions,
            *graph.prototypes,
            *graph.schemas,
        }
        return (
            _json_result(command, item_id, records=_relations_for(graph, item_id))
            if item_id in all_ids
            else _not_found(command, item_id)
        )
    return json.dumps(
        {"ok": False, "command": command, "id": item_id, "error": "invalid_command"},
        ensure_ascii=False,
        sort_keys=True,
    )


def _artifact_result(
    graph: EFPSGraph, artifact_root: Path | None, item_id: str
) -> ModelToolResult:
    descriptor = None
    evidence_ids = []
    observation_phases = []
    for evidence in graph.evidence.values():
        match = next(
            (
                item
                for item in evidence.artifacts
                if item.get("artifact_id") == item_id
            ),
            None,
        )
        if match is None:
            continue
        descriptor = dict(match)
        evidence_ids.append(evidence.evidence_id)
        observation_phases.extend(
            str(item.get("phase"))
            for item in evidence.observation_refs
            if item_id in item.get("artifact_ids", [])
        )
    if descriptor is None:
        return ModelToolResult(_not_found("get_artifact", item_id))
    if descriptor.get("role") != "agent_view" or descriptor.get("media_type") != "image/png":
        return ModelToolResult(
            json.dumps(
                {
                    "ok": False,
                    "command": "get_artifact",
                    "id": item_id,
                    "error": "not_agent_visible",
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    record = {
        **descriptor,
        "evidence_ids": sorted(set(evidence_ids)),
        "observation_phases": sorted(set(observation_phases)),
    }
    text = _json_result("get_artifact", item_id, record=record)
    if artifact_root is None:
        return ModelToolResult(text)
    root = artifact_root.resolve()
    path = (root / str(descriptor["relative_path"])).resolve()
    if root not in path.parents or not path.is_file():
        return ModelToolResult(
            json.dumps(
                {
                    "ok": False,
                    "command": "get_artifact",
                    "id": item_id,
                    "error": "stored_file_unavailable",
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    image = ModelImage(
        label=f"stored_public_artifact:{item_id}",
        artifact_id=item_id,
        sha256=str(descriptor["sha256"]),
        relative_path=str(descriptor["relative_path"]),
        data_url="data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii"),
    )
    return ModelToolResult(text=text, images=[image])


def cognition_tool(
    graph: EFPSGraph, artifact_root: str | Path | None = None
) -> ModelTool:
    root = Path(artifact_root) if artifact_root is not None else None

    def execute(arguments: Dict[str, Any]) -> str | ModelToolResult:
        command = str(arguments.get("command") or "")
        item_id = str(arguments.get("id") or "")
        if command == "get_artifact":
            return _artifact_result(graph, root, item_id)
        return read_cognition(
            graph,
            command=command,
            item_id=item_id,
            feature_id=(
                str(arguments["feature_id"])
                if arguments.get("feature_id")
                else None
            ),
        )

    return ModelTool(
        name="read_cognition",
        description=(
            "Perform one exact read from the learned EFPS/Evidence store. Supply a fixed "
            "command and an exact ID copied from the catalog. There is no natural-language "
            "search, similarity ranking, summarization, or hidden inference. get_artifact "
            "returns the exact stored agent-visible public PNG when available."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": [
                        "get_entity",
                        "get_prototype",
                        "get_schema",
                        "get_evidence",
                        "get_feature_history",
                        "get_relations",
                        "get_artifact",
                    ],
                },
                "id": {
                    "type": "string",
                    "description": "Exact Entity/Prototype/Schema/Evidence/artifact ID.",
                },
                "feature_id": {
                    "type": "string",
                    "description": "Optional exact filter for get_feature_history only.",
                },
            },
            "required": ["command", "id"],
            "additionalProperties": False,
        },
        execute=execute,
    )


__all__ = [
    "cognition_tool",
    "decision_sections",
    "read_cognition",
    "render_cognition_catalog",
    "render_sections",
    "section_metrics",
    "update_sections",
]
