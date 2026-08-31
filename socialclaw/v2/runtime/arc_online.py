from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import copy
from typing import Any, Callable, Dict, List, Tuple

from ...trajectory import EvidenceTier, Observation, replay_arc_episode
from ...trajectory.corpus import write_json_atomic
from ..agents import ExplorationAgent, MainAgent, UpdateAgent
from ..agents.prompts import INSTRUCTION_PROFILES
from ..cognition import decision_sections, update_sections
from ..context import (
    cognition_view,
    observation_image,
    observation_receipt,
    unique_images,
)
from ..efps import EFPSGraph, EvidenceRecord, stable_id
from ..model import ModelImage, StructuredVisionModel
from ..reporting import build_process_markdown
from ..usage import build_usage_report, usage_markdown
from .public_arc import PublicARCSession


def _artifact_ids(observation: Observation) -> List[str]:
    return sorted({item.artifact_id for item in observation.artifacts})


def _artifact_records(*observations: Observation) -> List[Dict[str, Any]]:
    values: Dict[str, Dict[str, Any]] = {}
    for observation in observations:
        for item in observation.artifacts:
            values.setdefault(
                item.artifact_id,
                {
                    "artifact_id": item.artifact_id,
                    "role": item.role,
                    "media_type": item.media_type,
                    "relative_path": item.relative_path,
                    "sha256": item.sha256,
                    "metadata": dict(item.metadata),
                },
            )
    return [values[key] for key in sorted(values)]


def _observation_ref(phase: str, observation: Observation) -> Dict[str, Any]:
    return {
        "phase": phase,
        "fingerprint": observation.content_fingerprint(),
        "artifact_ids": _artifact_ids(observation),
    }


def _initial_evidence(episode_id: str, observation: Observation) -> EvidenceRecord:
    evidence_id = stable_id(
        "evidence",
        {
            "episode_id": episode_id,
            "kind": "initial_public_observation",
            "fingerprint": observation.content_fingerprint(),
        },
    )
    return EvidenceRecord(
        evidence_id=evidence_id,
        kind="initial_public_observation",
        episode_id=episode_id,
        step_index=None,
        observation_fingerprints=[observation.content_fingerprint()],
        artifact_ids=_artifact_ids(observation),
        artifacts=_artifact_records(observation),
        observation_refs=[_observation_ref("current", observation)],
    )


def _transition_evidence(episode_id: str, step) -> EvidenceRecord:
    evidence_id = stable_id(
        "evidence",
        {
            "episode_id": episode_id,
            "step_index": step.step_index,
            "before": step.observation.content_fingerprint(),
            "action": step.action.to_dict(),
            "after": step.result.observation.content_fingerprint(),
        },
    )
    return EvidenceRecord(
        evidence_id=evidence_id,
        kind="public_transition",
        episode_id=episode_id,
        step_index=step.step_index,
        observation_fingerprints=[
            step.observation.content_fingerprint(),
            step.result.observation.content_fingerprint(),
        ],
        action=step.action.to_dict(),
        result={
            "environment_status": step.result.environment_status,
            "state_delta": step.result.state_delta,
        },
        artifact_ids=sorted(
            {
                *_artifact_ids(step.observation),
                *_artifact_ids(step.result.observation),
            }
        ),
        artifacts=_artifact_records(step.observation, step.result.observation),
        observation_refs=[
            _observation_ref("before", step.observation),
            _observation_ref("after", step.result.observation),
        ],
    )


def _reset_evidence(episode_id: str, step, reset_index: int) -> EvidenceRecord:
    evidence_id = stable_id(
        "evidence",
        {
            "episode_id": episode_id,
            "kind": "public_environment_reset",
            "reset_index": reset_index,
            "before": step.observation.content_fingerprint(),
            "after": step.result.observation.content_fingerprint(),
        },
    )
    return EvidenceRecord(
        evidence_id=evidence_id,
        kind="public_environment_reset",
        episode_id=episode_id,
        step_index=step.step_index,
        observation_fingerprints=[
            step.observation.content_fingerprint(),
            step.result.observation.content_fingerprint(),
        ],
        action=step.action.to_dict(),
        result={
            "environment_status": step.result.environment_status,
            "state_delta": step.result.state_delta,
        },
        artifact_ids=sorted(
            {
                *_artifact_ids(step.observation),
                *_artifact_ids(step.result.observation),
            }
        ),
        artifacts=_artifact_records(step.observation, step.result.observation),
        observation_refs=[
            _observation_ref("before_reset", step.observation),
            _observation_ref("after_reset", step.result.observation),
        ],
    )


def _relabel(image: ModelImage, label: str) -> ModelImage:
    return ModelImage(
        label=label,
        artifact_id=image.artifact_id,
        sha256=image.sha256,
        relative_path=image.relative_path,
        data_url=image.data_url,
    )


def _supporting_images(
    graph: EFPSGraph,
    catalog: Dict[str, List[ModelImage]],
    *,
    limit: int = 1,
) -> List[ModelImage]:
    evidence_ids: List[str] = []
    for schema in graph.schemas.values():
        for evidence_id in schema.support_evidence_ids:
            if evidence_id not in evidence_ids:
                evidence_ids.append(evidence_id)
    result = []
    for evidence_id in evidence_ids[-limit:]:
        images = catalog.get(evidence_id) or []
        for index, item in enumerate(images):
            phase = "before" if index == 0 else "after"
            result.append(
                _relabel(item, f"schema_evidence:{evidence_id}:{phase}")
            )
    return result


def _catalog_item(
    catalog: Dict[str, Dict[str, Any]],
    section: str,
    value: Any,
) -> str:
    item_id = stable_id(section.rstrip("s"), value)
    stored = cognition_input_receipt(value, view_id=item_id) if section == "cognition_views" else value
    catalog[section].setdefault(item_id, stored)
    return f"input_catalog.{section}.{item_id}"


def cognition_input_receipt(
    view: Dict[str, Any],
    *,
    view_id: str | None = None,
) -> Dict[str, Any]:
    """Compact audit source used to build the prose cognition catalog.

    This is not the model payload. Each AgentCallAudit.input_text is the exact
    text sent to the model; the hash here only identifies its structured source.
    """
    identity = view_id or stable_id("cognition_view", view)
    source_fields = [
        "entities: id,label,bbox,status,current feature assertions,evidence_ids",
        "prototypes: id,name,defining features,members,evidence_ids",
        "schemas: Prototype-Action-Output triple,confidence,evidence_ids",
        "insights: kind,statement,scope,confidence,evidence_ids",
        "evidence: kind,step,action,result,Entity semantic changes,artifact_ids",
    ]
    if "relations" in view:
        source_fields.append(
            "relations: type,source,target,metadata,evidence_ids (legacy run view)"
        )
    return {
        "view_id": identity,
        "role": "audit_source_not_model_payload",
        "expanded_view_hash": stable_id("expanded_cognition", view),
        "revision": view.get("revision"),
        "counts": view.get("counts"),
        "view_policy": view.get("view_policy"),
        "source_fields": source_fields,
        "entities_sent": copy.deepcopy(view.get("entities") or []),
        "prototypes_sent": copy.deepcopy(view.get("prototypes") or []),
        "schemas_sent": [
            {
                "schema_id": item.get("schema_id"),
                "prototype_id": item.get("prototype_id"),
                "action": item.get("action"),
                "output": item.get("output"),
                "confidence": item.get("confidence"),
                "support_evidence_ids": item.get("support_evidence_ids"),
                "counter_evidence_ids": item.get("counter_evidence_ids"),
            }
            for item in view.get("schemas") or []
        ],
        "insights_sent": copy.deepcopy(view.get("insights") or []),
        "evidence_sent": copy.deepcopy(view.get("evidence") or []),
        "full_view_saved_elsewhere": False,
        "full_state_source": "cognition/graph.json plus audit_log and assertion histories",
    }


def compact_timeline_inputs(
    events: List[Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    """Deduplicate exact Agent inputs without losing audit reconstructability."""
    catalog: Dict[str, Dict[str, Any]] = {
        "observations": {},
        "cognition_views": {},
        "action_contracts": {},
        "evidence": {},
    }
    values = copy.deepcopy(events)
    for event in values:
        initial = event.get("update_input")
        if event.get("step") == 0 and isinstance(initial, dict):
            initial["observation_ref"] = _catalog_item(
                catalog, "observations", initial.pop("observation")
            )
            initial["cognition_before_ref"] = _catalog_item(
                catalog, "cognition_views", initial.pop("cognition_before")
            )
            initial["evidence_ref"] = _catalog_item(
                catalog, "evidence", initial.pop("evidence")
            )
            continue
        shared = event.get("shared_decision_input") or {}
        shared["current_observation_ref"] = _catalog_item(
            catalog, "observations", shared.pop("current_observation")
        )
        shared["available_actions_ref"] = _catalog_item(
            catalog, "action_contracts", shared.pop("available_actions")
        )
        shared["cognition_ref"] = _catalog_item(
            catalog, "cognition_views", shared.pop("cognition")
        )
        for recent in shared.get("recent_public_transitions") or []:
            if "evidence" in recent:
                recent["evidence_ref"] = _catalog_item(
                    catalog, "evidence", recent.pop("evidence")
                )
        update = event.get("update_input") or {}
        update["before_observation_ref"] = _catalog_item(
            catalog, "observations", update.pop("before_observation")
        )
        update["after_observation_ref"] = _catalog_item(
            catalog, "observations", update.pop("after_observation")
        )
        update["cognition_before_update_ref"] = _catalog_item(
            catalog, "cognition_views", update.pop("cognition_before_update")
        )
        if update.get("main_decision") == event.get("decision"):
            update.pop("main_decision")
            update["main_decision_ref"] = "event.decision"
        environment = event.get("environment_transition") or {}
        if "evidence" in environment:
            environment["evidence_ref"] = _catalog_item(
                catalog, "evidence", environment.pop("evidence")
            )
        reset = event.get("environment_reset") or {}
        reset_input = reset.get("update_input") or {}
        if "observation" in reset_input:
            reset_input["observation_ref"] = _catalog_item(
                catalog, "observations", reset_input.pop("observation")
            )
        if "cognition_before" in reset_input:
            reset_input["cognition_before_ref"] = _catalog_item(
                catalog, "cognition_views", reset_input.pop("cognition_before")
            )
        if "evidence" in reset_input:
            reset_input["evidence_ref"] = _catalog_item(
                catalog, "evidence", reset_input.pop("evidence")
            )
        if "evidence" in reset:
            reset["evidence_ref"] = _catalog_item(
                catalog, "evidence", reset.pop("evidence")
            )
    return catalog, values


def _usage_totals(events: List[Dict[str, Any]]) -> Dict[str, int]:
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for event in events:
        for call in (event.get("agent_calls") or {}).values():
            for key in totals:
                totals[key] += int((call.get("usage") or {}).get(key, 0))
    return totals


def _write_partial_checkpoint(
    output: Path,
    *,
    game_id: str,
    model_name: str,
    graph: EFPSGraph,
    events: List[Dict[str, Any]],
    compact_process: bool = False,
) -> None:
    """Preserve the last complete cognitive step during costly live runs."""
    input_catalog, compact_events = compact_timeline_inputs(events)
    actions = max(0, len(events) - 1)
    payload = {
        "format_version": 2,
        "description": "Incomplete checkpoint; not a finalized experiment result.",
        "process_prompt_detail": "compact" if compact_process else "full",
        "instruction_profiles": INSTRUCTION_PROFILES,
        "input_catalog": input_catalog,
        "summary": {
            "game_id": game_id,
            "model": model_name,
            "success": False,
            "actions": actions,
            "model_calls": sum(
                len(event.get("agent_calls") or {}) for event in events
            ),
            "final_cognition": graph.counts(),
            "checkpoint_status": "INCOMPLETE",
        },
        "events": compact_events,
    }
    write_json_atomic(output / "timeline.partial.json", payload)
    write_json_atomic(output / "cognition" / "graph.partial.json", graph.to_dict())
    (output / "process.partial.md").write_text(
        build_process_markdown(payload), encoding="utf-8"
    )


def run_arc_online(
    output_dir: str | Path,
    *,
    game_id: str,
    model: StructuredVisionModel,
    max_steps: int = 30,
    stop_after_levels: int | None = 1,
    reset_on_game_over: bool = True,
    compact_process: bool = False,
    env: Any | None = None,
    replay_fn: Callable[[str | Path, Any], Dict[str, Any]] = replay_arc_episode,
) -> Dict[str, Any]:
    """Run the same zero-prior cognitive loop on any public ARC game ID."""
    if max_steps < 1 or (stop_after_levels is not None and stop_after_levels < 1):
        raise ValueError(
            "max_steps must be positive and stop_after_levels must be positive or None"
        )
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    output.mkdir(parents=True)
    episode_id = stable_id(
        "v2_episode",
        {"game_id": game_id, "model": model.model_name, "output": output.name},
    )
    session = PublicARCSession(
        game_id=game_id,
        root=output / "trajectory",
        episode_id=episode_id,
        actor="v2_generic_visual_main_agent",
        env=env,
        provenance={
            "agent_input": "public_visual_observation_and_public_action_contracts",
            "runtime_game_rules": 0,
            "goal_supplied_to_agent": False,
            "gold_reads": 0,
            "game_source_reads_by_agents": 0,
            "special_coordinate_reads": 0,
            "precomputed_route_reads": 0,
            "model": model.model_name,
        },
        metadata={
            "stop_after_public_level_delta": stop_after_levels,
            "max_agent_actions_per_level": max_steps,
            "reset_on_game_over": reset_on_game_over,
        },
        episode_created_at=getattr(model, "episode_created_at", None),
    )
    graph = EFPSGraph()
    exploration_agent = ExplorationAgent(model, artifact_root=session.asset_root)
    main_agent = MainAgent(model, artifact_root=session.asset_root)
    update_agent = UpdateAgent(model, artifact_root=session.asset_root)
    events: List[Dict[str, Any]] = []
    recent_transitions: List[Dict[str, Any]] = []
    evidence_images: Dict[str, List[ModelImage]] = {}
    initial_levels = int(session.observation.structured["levels_completed"])

    initial_evidence = _initial_evidence(episode_id, session.observation)
    graph.register_evidence(initial_evidence)
    initial_image = observation_image(
        session.observation,
        asset_root=session.asset_root,
        label="current_public_observation",
    )
    evidence_images[initial_evidence.evidence_id] = [initial_image]
    initial_input = {
        "phase": "initial_observation",
        "observation": observation_receipt(session.observation),
        "evidence_id": initial_evidence.evidence_id,
        "evidence": graph.resolve_evidence(initial_evidence.evidence_id),
        "cognition_before": cognition_view(graph),
    }
    initial_input["prompt_sections"] = update_sections(
        phase="initial_observation",
        evidence_id=initial_evidence.evidence_id,
        graph=graph,
        observation=observation_receipt(session.observation),
    )
    before_counts = graph.counts()
    initial_update = update_agent.propose(
        update_input=initial_input,
        images=[initial_image],
        graph=graph,
        evidence_id=initial_evidence.evidence_id,
        episode_id=episode_id,
        step=0,
        executed_action=None,
    )
    initial_transaction = graph.apply_transaction(
        initial_update.operations,
        actor="update_child_agent",
        step=0,
        mode=initial_update.mode,
        summary=initial_update.summary,
    )
    events.append(
        {
            "step": 0,
            "phase": "initial_cognition",
            "update_input": initial_input,
            "agent_calls": {"update_agent": initial_update.audit.to_dict()},
            "cognitive_update": {
                "transaction": asdict(initial_transaction),
                "counts_before": before_counts,
                "counts_after": graph.counts(),
                "warnings": initial_update.warnings,
            },
        }
    )
    _write_partial_checkpoint(
        output,
        game_id=game_id,
        model_name=model.model_name,
        graph=graph,
        events=events,
        compact_process=compact_process,
    )

    completed = False
    game_won = False
    termination_reason = ""
    failure_reason = ""
    agent_actions = 0
    actions_in_level = 0
    runtime_resets = 0
    resets_in_level = 0
    level_results: List[Dict[str, Any]] = []
    while True:
        if actions_in_level >= max_steps:
            level_results.append(
                {
                    "level": initial_levels + len(level_results) + 1,
                    "success": False,
                    "actions": actions_in_level,
                    "resets": resets_in_level,
                    "reason": "step_limit",
                }
            )
            termination_reason = "level_step_limit"
            failure_reason = (
                f"Level {level_results[-1]['level']} did not complete within "
                f"{max_steps} Agent actions."
            )
            break
        agent_actions += 1
        step_number = agent_actions
        current = session.observation
        current_image = observation_image(
            current,
            asset_root=session.asset_root,
            label="current_public_observation",
        )
        supporting: List[ModelImage] = []
        exploration_images = [current_image]
        main_images = [current_image]
        observation_context = observation_receipt(current)
        action_contracts = session.available_action_contracts()
        level_budget = {
            "level": int(current.structured.get("levels_completed") or 0) + 1,
            "actions_used": actions_in_level,
            "actions_remaining": max_steps - actions_in_level,
            "max_actions": max_steps,
        }
        shared_decision_input = {
            "step": step_number,
            "current_observation": observation_context,
            "available_actions": action_contracts,
            "cognition": cognition_view(graph),
            "recent_public_transitions": recent_transitions[-8:],
            "attached_schema_evidence_images": [
                item.audit_dict() for item in supporting
            ],
            "level_budget": level_budget,
            "prompt_sections": decision_sections(
                observation=observation_context,
                contracts=action_contracts,
                recent_transitions=recent_transitions,
                graph=graph,
                level_budget=level_budget,
            ),
        }
        exploration = exploration_agent.propose(
            shared_input=shared_decision_input,
            images=exploration_images,
            graph=graph,
        )
        decision = main_agent.decide(
            shared_input=shared_decision_input,
            exploration=exploration,
            images=main_images,
            graph=graph,
        )
        transition = session.execute(
            decision.action,
            rationale=decision.rationale,
            schemas_used=decision.schema_ids,
            decision_metadata={
                "decision_mode": decision.decision_mode,
                "schema_prediction": decision.schema_prediction,
                "insight_ids": decision.insight_ids,
                "insight_application": decision.insight_application,
                "exploration_hypothesis": decision.exploration_hypothesis,
                "model_calls": 2,
            },
        )
        actions_in_level += 1
        evidence = _transition_evidence(episode_id, transition)
        graph.register_evidence(evidence)
        before_image = observation_image(
            transition.observation,
            asset_root=session.asset_root,
            label="transition_before",
        )
        after_image = observation_image(
            transition.result.observation,
            asset_root=session.asset_root,
            label="transition_after",
        )
        evidence_images[evidence.evidence_id] = [before_image, after_image]
        level_delta = int(transition.result.state_delta.get("level_delta") or 0)
        update_phase = (
            "public_level_boundary" if level_delta > 0 else "public_transition"
        )
        update_input = {
            "phase": update_phase,
            "step": step_number,
            "before_observation": observation_receipt(transition.observation),
            "executed_action": transition.action.to_dict(),
            "after_observation": observation_receipt(
                transition.result.observation
            ),
            "public_result": transition.result.state_delta,
            "main_decision": decision.to_dict(),
            "evidence_id": evidence.evidence_id,
            "cognition_before_update": cognition_view(graph),
        }
        update_input["prompt_sections"] = update_sections(
            phase=update_phase,
            evidence_id=evidence.evidence_id,
            graph=graph,
            before_observation=update_input["before_observation"],
            after_observation=update_input["after_observation"],
            executed_action=update_input["executed_action"],
            public_result=update_input["public_result"],
            main_decision=update_input["main_decision"],
        )
        counts_before_update = graph.counts()
        update = update_agent.propose(
            update_input=update_input,
            images=unique_images([before_image, after_image]),
            graph=graph,
            evidence_id=evidence.evidence_id,
            episode_id=episode_id,
            step=step_number,
            executed_action=transition.action.to_dict(),
        )
        update_transaction = graph.apply_transaction(
            update.operations,
            actor="update_child_agent",
            step=step_number,
            mode=update.mode,
            summary=update.summary,
        )
        if update.transition_analysis is None:  # pragma: no cover - guarded by UpdateAgent
            raise ValueError("A public action transition requires semantic analysis")
        graph.annotate_evidence(
            evidence.evidence_id,
            semantic_summary=update.transition_analysis["summary"],
            entity_changes=update.transition_analysis["entity_changes"],
            unassigned_visual_changes=update.transition_analysis[
                "unassigned_visual_changes"
            ],
        )
        resolved_evidence = graph.resolve_evidence(evidence.evidence_id)
        event = {
            "step": step_number,
            "phase": "act_observe_update",
            "shared_decision_input": shared_decision_input,
            "agent_calls": {
                "exploration_agent": exploration.audit.to_dict(),
                "main_agent": decision.audit.to_dict(),
                "update_agent": update.audit.to_dict(),
            },
            "decision": decision.to_dict(),
            "environment_transition": {
                "action": transition.action.to_dict(),
                "result": transition.result.state_delta,
                "semantic_result": update.transition_analysis,
                "evidence_id": evidence.evidence_id,
                "evidence": resolved_evidence,
            },
            "update_input": update_input,
            "cognitive_update": {
                "transaction": asdict(update_transaction),
                "counts_before": counts_before_update,
                "counts_after": graph.counts(),
                "warnings": update.warnings,
            },
        }
        events.append(event)
        recent_transitions.append(
            {
                "step": step_number,
                "action": transition.action.to_dict(),
                "public_result": transition.result.state_delta,
                "semantic_result": update.transition_analysis,
                "evidence_id": evidence.evidence_id,
                "evidence": resolved_evidence,
            }
        )
        _write_partial_checkpoint(
            output,
            game_id=game_id,
            model_name=model.model_name,
            graph=graph,
            events=events,
            compact_process=compact_process,
        )
        levels_completed = int(
            transition.result.observation.structured["levels_completed"]
        )
        environment_status = str(
            transition.result.state_delta.get("environment_status") or ""
        )
        if level_delta > 0:
            level_results.append(
                {
                    "level": levels_completed,
                    "success": True,
                    "actions": actions_in_level,
                    "resets": resets_in_level,
                    "reason": "level_completed",
                }
            )
            actions_in_level = 0
            resets_in_level = 0
            completed_count = levels_completed - initial_levels
            if environment_status == "WIN":
                completed = True
                game_won = True
                termination_reason = "game_win"
                break
            if (
                stop_after_levels is not None
                and completed_count >= stop_after_levels
            ):
                completed = True
                termination_reason = "requested_level_boundary"
                break
            continue
        if environment_status == "WIN":
            level_results.append(
                {
                    "level": levels_completed or (initial_levels + len(level_results) + 1),
                    "success": True,
                    "actions": actions_in_level,
                    "resets": resets_in_level,
                    "reason": "game_win",
                }
            )
            completed = True
            game_won = True
            termination_reason = "game_win"
            break
        if environment_status == "GAME_OVER":
            if reset_on_game_over and actions_in_level < max_steps:
                runtime_resets += 1
                resets_in_level += 1
                reset_step = session.reset_after_game_over()
                reset_evidence = _reset_evidence(
                    episode_id, reset_step, runtime_resets
                )
                graph.register_evidence(reset_evidence)
                reset_before_image = observation_image(
                    reset_step.observation,
                    asset_root=session.asset_root,
                    label="reset_before_game_over",
                )
                reset_after_image = observation_image(
                    reset_step.result.observation,
                    asset_root=session.asset_root,
                    label="reset_after_current_level",
                )
                evidence_images[reset_evidence.evidence_id] = [
                    reset_before_image,
                    reset_after_image,
                ]
                reset_input = {
                    "phase": "public_environment_reset",
                    "step": step_number,
                    "observation": observation_receipt(
                        reset_step.result.observation
                    ),
                    "public_result": reset_step.result.state_delta,
                    "evidence_id": reset_evidence.evidence_id,
                    "evidence": graph.resolve_evidence(reset_evidence.evidence_id),
                    "cognition_before": cognition_view(graph),
                    "level_budget": {
                        "level": levels_completed + 1,
                        "actions_used": actions_in_level,
                        "actions_remaining": max_steps - actions_in_level,
                        "max_actions": max_steps,
                    },
                }
                reset_input["prompt_sections"] = update_sections(
                    phase="public_environment_reset",
                    evidence_id=reset_evidence.evidence_id,
                    graph=graph,
                    observation=reset_input["observation"],
                )
                reset_counts_before = graph.counts()
                reset_update = update_agent.propose(
                    update_input=reset_input,
                    images=[reset_after_image],
                    graph=graph,
                    evidence_id=reset_evidence.evidence_id,
                    episode_id=episode_id,
                    step=step_number,
                    executed_action=None,
                )
                reset_transaction = graph.apply_transaction(
                    reset_update.operations,
                    actor="update_child_agent_after_runtime_reset",
                    step=step_number,
                    mode=reset_update.mode,
                    summary=reset_update.summary,
                )
                reset_semantic = {
                    "summary": (
                        "The public environment was reset after GAME_OVER on the same "
                        f"level; {actions_in_level}/{max_steps} Agent actions remain "
                        "consumed for this level."
                    ),
                    "entity_changes": [],
                    "unassigned_visual_changes": [],
                }
                graph.annotate_evidence(
                    reset_evidence.evidence_id,
                    semantic_summary=reset_semantic["summary"],
                    entity_changes=[],
                    unassigned_visual_changes=[],
                )
                resolved_reset_evidence = graph.resolve_evidence(
                    reset_evidence.evidence_id
                )
                event["agent_calls"]["recovery_update_agent"] = (
                    reset_update.audit.to_dict()
                )
                event["environment_reset"] = {
                    "trajectory_step_index": reset_step.step_index,
                    "result": reset_step.result.state_delta,
                    "semantic_result": reset_semantic,
                    "evidence_id": reset_evidence.evidence_id,
                    "evidence": resolved_reset_evidence,
                    "update_input": reset_input,
                    "cognitive_update": {
                        "transaction": asdict(reset_transaction),
                        "counts_before": reset_counts_before,
                        "counts_after": graph.counts(),
                        "warnings": reset_update.warnings,
                    },
                }
                recent_transitions.append(
                    {
                        "step": f"{step_number}R{runtime_resets}",
                        "action": reset_step.action.to_dict(),
                        "public_result": reset_step.result.state_delta,
                        "semantic_result": reset_semantic,
                        "evidence_id": reset_evidence.evidence_id,
                        "evidence": resolved_reset_evidence,
                    }
                )
                _write_partial_checkpoint(
                    output,
                    game_id=game_id,
                    model_name=model.model_name,
                    graph=graph,
                    events=events,
                    compact_process=compact_process,
                )
                continue
            level_results.append(
                {
                    "level": levels_completed + 1,
                    "success": False,
                    "actions": actions_in_level,
                    "resets": resets_in_level,
                    "reason": "game_over",
                }
            )
            termination_reason = "game_over"
            failure_reason = (
                f"Level {levels_completed + 1} reached GAME_OVER after "
                f"{actions_in_level}/{max_steps} Agent actions"
                + (
                    "; no budget remained for another recovery reset."
                    if reset_on_game_over
                    else "."
                )
            )
            break

    episode_status = {
        "game_win": "GAME_WIN",
        "requested_level_boundary": "REQUESTED_LEVEL_BOUNDARY",
        "game_over": "GAME_OVER",
        "level_step_limit": "LEVEL_STEP_LIMIT",
    }.get(termination_reason, "STOPPED")
    episode = session.finish(
        status=episode_status,
        success=completed,
        metadata={
            "graph_revision": graph.revision,
            "termination_reason": termination_reason,
            "max_agent_actions_per_level": max_steps,
            "runtime_resets": runtime_resets,
        },
        details=failure_reason or termination_reason,
    )
    replay = replay_fn(output / "trajectory", episode)
    graph.validate()
    usage = _usage_totals(events)
    detailed_usage = build_usage_report(events)
    levels_attempted = len(level_results)
    levels_passed = sum(1 for item in level_results if item.get("success"))
    summary = {
        "game_id": game_id,
        "model": model.model_name,
        "success": completed,
        "actions": agent_actions,
        "trajectory_steps": len(episode.steps),
        "runtime_resets": runtime_resets,
        "max_steps_per_level": max_steps,
        "stop_after_levels": stop_after_levels,
        "game_won": game_won,
        "termination_reason": termination_reason,
        "failure_reason": failure_reason,
        "levels_attempted": levels_attempted,
        "levels_passed": levels_passed,
        "level_pass_rate": (
            levels_passed / levels_attempted if levels_attempted else 0.0
        ),
        "level_results": level_results,
        "public_levels_completed": int(
            session.observation.structured["levels_completed"]
        )
        - initial_levels,
        "final_cognition": graph.counts(),
        "model_calls": sum(
            len(event.get("agent_calls") or {}) for event in events
        ),
        "provider_requests": detailed_usage["totals"]["provider_requests"],
        "cognition_tool_calls": detailed_usage["totals"]["tool_calls"],
        "usage": usage,
        "trajectory_replay": replay,
        "forbidden_reads": {
            "gold": 0,
            "game_source_by_agents": 0,
            "special_coordinates": 0,
            "goal_mask": 0,
            "precomputed_route": 0,
        },
    }
    input_catalog, compact_events = compact_timeline_inputs(events)
    write_json_atomic(output / "cognition" / "graph.json", graph.to_dict())
    timeline_payload = {
        "format_version": 2,
        "description": (
            "Each event records shared inputs once, then identifies exactly "
            "which Agent received them and which durable images were attached."
        ),
        "process_prompt_detail": "compact" if compact_process else "full",
        "instruction_profiles": INSTRUCTION_PROFILES,
        "input_catalog": input_catalog,
        "summary": summary,
        "events": compact_events,
    }
    write_json_atomic(output / "timeline.json", timeline_payload)
    write_json_atomic(output / "token_usage.json", detailed_usage)
    (output / "token_usage.md").write_text(
        usage_markdown(detailed_usage), encoding="utf-8"
    )
    (output / "process.md").write_text(
        build_process_markdown(timeline_payload), encoding="utf-8"
    )
    (output / "report.md").write_text(
        _report(summary, events), encoding="utf-8"
    )
    for checkpoint in (
        output / "timeline.partial.json",
        output / "process.partial.md",
        output / "cognition" / "graph.partial.json",
    ):
        checkpoint.unlink(missing_ok=True)
    return summary


def _report(summary: Dict[str, Any], events: List[Dict[str, Any]]) -> str:
    rows = []
    for event in events:
        if event["step"] == 0:
            after = event["cognitive_update"]["counts_after"]
            rows.append(
                f"| 0 | 初始画面 | Update 子 Agent 形成首批观察概念 | - | {after['entities']} | {after['prototypes']} | {after['schemas']} | {after['insights']} |"
            )
            continue
        decision = event["decision"]
        result = event["environment_transition"]["result"]
        after = event["cognitive_update"]["counts_after"]
        hypothesis = (
            decision["schema_prediction"]
            if decision["schema_prediction"] is not None
            else decision.get("insight_application")
            or decision["exploration_hypothesis"]
        )
        rows.append(
            "| {step} | {action} | {mode}: {hypothesis} | changed={changed}, level_delta={delta} | {entities} | {prototypes} | {schemas} | {insights} |".format(
                step=event["step"],
                action=decision["action"]["name"],
                mode=decision["decision_mode"],
                hypothesis=str(hypothesis or "")[:100].replace("|", "/"),
                changed=result.get("changed_cells"),
                delta=result.get("level_delta"),
                entities=after["entities"],
                prototypes=after["prototypes"],
                schemas=after["schemas"],
                insights=after["insights"],
            )
        )
    return "\n".join(
        [
            "# V2 通用视觉 Agent：在线 Level 测试",
            "",
            f"- 游戏：`{summary['game_id']}`（仅由评测 harness 选择，未传给认知 Agent 作为规则）",
            f"- 模型：`{summary['model']}`",
            f"- 结果：`{summary.get('termination_reason')}`；game won={summary.get('game_won')}；"
            f"failure=`{summary.get('failure_reason') or '无'}`",
            f"- 关卡：通过 {summary.get('levels_passed', 0)}/{summary.get('levels_attempted', 0)}；"
            f"通关率={summary.get('level_pass_rate', 0.0):.2%}；每关最多 {summary.get('max_steps_per_level')} actions",
            f"- 动作数：{summary['actions']}；环境恢复：{summary.get('runtime_resets', 0)}；"
            f"trajectory steps：{summary.get('trajectory_steps', summary['actions'])}；模型调用：{summary['model_calls']}",
            f"- provider requests：{summary.get('provider_requests', summary['model_calls'])}；"
            f"read_cognition calls：{summary.get('cognition_tool_calls', 0)}",
            f"- token：input={summary['usage']['input_tokens']:,}；"
            f"output={summary['usage']['output_tokens']:,}；total={summary['usage']['total_tokens']:,}",
            f"- 最终认知：{summary['final_cognition']}",
            f"- 分关结果：`{summary.get('level_results') or []}`",
            "- 输入边界：原始公开画面、公开环境状态、SDK 公开动作参数合同、公开转移差异、只读 EFPS。",
            "- 未提供：目标、对象标签、动作语义、Gold、游戏源码、专用坐标、goal mask 或路线。",
            "",
            "| Step | 动作 | 当时的 Agent 预测/假设 | 公开结果 | Entity | Prototype | Schema | Insight |",
            "|---:|---|---|---|---:|---:|---:|---:|",
            *rows,
            "",
            "逐时刻的人类可读输入、输出、触发原因和图片索引在 `process.md`；"
            "`token_usage.md`/`token_usage.json` 给出逐 Agent、逐步、逐 provider request、"
            "逐输入区段和工具调用统计；`timeline.json` 保留机器审计细节。另存最终 "
            "`cognition/graph.json` 和可 replay 的 `trajectory/`。",
            "",
        ]
    )


__all__ = ["run_arc_online"]
