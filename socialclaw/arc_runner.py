"""ARC-AGI-3 runner backed by the memory-grounded layered schema system."""

from __future__ import annotations

import base64
import io
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .agent.base import AgentAttempt, ReasoningTrace, Usage
from .dataset.arc_agi3 import ARCAGI3EnvWrapper
from .dataset.base import EvalResult, Problem
from .human_io import HumanIO
from .llm import OpenAIChatClient
from .logging import write_episode, write_step, write_trajectory
from .memory import MemoryRecord
from .schema import SchemaManager, build_schema_system
from .schema.arc_agi3_parser import color_name, compute_grid_diff, extract_grid_objects
from .types import AttemptRecord, Episode
from .utils import make_run_dir, save_cmd


_ARC_SYSTEM_PROMPT = """You are an ARC-AGI-3 game-playing agent. Observe the full grid image and choose one available action.

Coordinates use (x, y): x increases left-to-right and y increases top-to-bottom.
ACTION1/2/3/4 usually move up/down/left/right. ACTION6 is a click and requires {"x": col, "y": row}. Other actions require no coordinates unless the environment evidence proves otherwise.

Use retrieved schemas only when their trigger matches the current grid. Current observations override stale schemas. Avoid repeating actions that repeatedly have no effect.

Return ONLY raw JSON:
{"action":"ACTION_NAME","data":{},"reasoning":{"schemas_used":["schema_id"],"explanation":"brief reason"}}
Omit data for actions that do not need it."""


def _grid_summary(grid: np.ndarray, *, max_objects: int = 30) -> str:
    objects = extract_grid_objects(grid)
    lines = [f"Grid shape: {grid.shape[0]} rows x {grid.shape[1]} columns."]
    if not objects:
        lines.append("No non-background connected objects detected.")
        return "\n".join(lines)
    lines.append(f"Detected {len(objects)} non-background connected objects:")
    for index, obj in enumerate(objects[:max_objects]):
        left, top = obj["top_left"]
        right, bottom = obj["bottom_right"]
        lines.append(
            f"- object_{index}: color={color_name(obj['color'])}; "
            f"bounds=({left},{top})-({right},{bottom}); area={obj['area']}"
        )
    if len(objects) > max_objects:
        lines.append(f"- {len(objects) - max_objects} additional objects omitted from text; see image.")
    return "\n".join(lines)


def _history_summary(history: Sequence[Dict[str, Any]]) -> str:
    if not history:
        return "No actions have been taken in this level attempt."
    totals: Dict[str, int] = {}
    no_effect: Dict[str, int] = {}
    for item in history:
        action = str(item["action"])
        totals[action] = totals.get(action, 0) + 1
        if not item.get("grid_changed", False):
            no_effect[action] = no_effect.get(action, 0) + 1
    lines = [
        "Action counts: "
        + ", ".join(
            f"{action}={count} (no-effect={no_effect.get(action, 0)})"
            for action, count in sorted(totals.items())
        ),
        "Recent transitions:",
    ]
    for item in history[-5:]:
        lines.append(
            f"- step {item['step']}: {item['action']} data={item.get('action_data', {})} "
            f"changed={item.get('grid_changed')} state={item.get('state')}"
        )
    return "\n".join(lines)


def _build_arc_prompt(
    *,
    game_id: str,
    level: int,
    step: int,
    grid_summary: str,
    available_actions: Sequence[str],
    history: Sequence[Dict[str, Any]],
    schema_context: str,
) -> str:
    context = schema_context or "(No learned schema is currently available.)"
    return (
        f"{_ARC_SYSTEM_PROMPT}\n\n"
        f"{context}\n\n"
        f"Game: {game_id}; level: {level}; step: {step}\n"
        f"Available actions: {', '.join(available_actions)}\n\n"
        f"Current observation:\n{grid_summary}\n\n"
        f"History:\n{_history_summary(history)}\n\n"
        "Choose the next action."
    )


def _parse_agent_action(answer_text: str, available_actions: Sequence[Any]):
    action_name = ""
    data: Dict[str, Any] = {}
    trace = ReasoningTrace(concepts=[], relations=[], explanation="")
    text = (answer_text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            payload = json.loads(text[start : end + 1])
            action_name = str(payload.get("action", "")).upper()
            raw_data = payload.get("data") or {}
            data = raw_data if isinstance(raw_data, dict) else {}
            reasoning = payload.get("reasoning") or {}
            if isinstance(reasoning, dict):
                schemas = reasoning.get("schemas_used", reasoning.get("concepts_used", []))
                trace = ReasoningTrace(
                    concepts=[str(item) for item in schemas if str(item).strip()]
                    if isinstance(schemas, list)
                    else [],
                    relations=[],
                    explanation=str(reasoning.get("explanation", "")),
                )
            elif isinstance(reasoning, str):
                trace = ReasoningTrace(concepts=[], relations=[], explanation=reasoning)
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    if not action_name:
        upper = (answer_text or "").upper()
        for action in available_actions:
            if action.name.upper() in upper:
                action_name = action.name.upper()
                break
    chosen = next(
        (action for action in available_actions if action.name.upper() == action_name),
        available_actions[0] if available_actions else None,
    )
    if chosen is not None and chosen.name != "ACTION6":
        data = {}
    return chosen, data, trace


def _schema_query(
    *,
    game_id: str,
    level: int,
    grid_summary: str,
    available_actions: Sequence[str],
    history: Sequence[Dict[str, Any]],
) -> str:
    return (
        f"ARC-AGI-3 game={game_id}; level={level}.\n"
        f"Available actions: {', '.join(available_actions)}.\n"
        f"{grid_summary}\n{_history_summary(history)}"
    )


def _transition_result(
    *,
    changed: bool,
    changed_regions: Sequence[Dict[str, Any]],
    state: str,
    post_summary: str,
) -> str:
    regions = json.dumps(list(changed_regions), ensure_ascii=False, default=str)
    return (
        f"Environment state={state}; grid_changed={changed}; "
        f"changed_regions={regions}. Post-action observation: {post_summary}"
    )


def _learn_transition(
    *,
    manager: SchemaManager,
    game_id: str,
    level: int,
    step: int,
    query: str,
    observation: str,
    action: str,
    result: str,
    state: str,
) -> tuple[str, Optional[str]]:
    record = MemoryRecord(
        task=f"Learn the transition rules of ARC-AGI-3 game {game_id}",
        context=query,
        outcome=result,
        success=True if state.endswith("WIN") else False if state.endswith("GAME_OVER") else None,
        feedback="direct environment transition; no hidden rule or gold schema supplied",
        tags=["arc_agi3", game_id, f"level_{level}"],
        metadata={"schema_level": 3, "level": level, "step": step},
    )
    record.add_event(observation=observation, action=action, result=result)
    node = manager.remember_and_learn(record)
    if node is not None:
        manager.apply_feedback([node.index], memory_id=record.id, positive=True)
    return record.id, node.index if node is not None else None


def _apply_level_feedback(
    manager: SchemaManager,
    *,
    schema_ids: Sequence[str],
    game_id: str,
    level: int,
    outcome: str,
) -> Optional[str]:
    valid_ids = [schema_id for schema_id in sorted(set(schema_ids)) if manager.graph.get(schema_id)]
    if not valid_ids:
        return None
    positive = outcome == "WIN"
    record = MemoryRecord(
        task=f"Complete ARC-AGI-3 game {game_id} level {level}",
        context=f"Schemas used during the level: {', '.join(valid_ids)}",
        outcome=f"Level outcome={outcome}",
        success=positive,
        feedback=f"environment outcome={outcome}",
        tags=["arc_agi3", game_id, f"level_{level}", outcome.lower()],
        metadata={"schema_level": 3, "feedback_kind": "environment"},
    )
    manager.memory.remember(record)
    manager.apply_feedback(
        valid_ids, memory_id=record.id, positive=positive
    )
    return record.id


def _schema_confidence(manager: SchemaManager, schema_ids: Sequence[str]) -> float:
    weights = [
        node.reliability_weight
        for schema_id in set(schema_ids)
        if (node := manager.graph.get(schema_id)) is not None
    ]
    return sum(weights) / len(weights) if weights else 0.0


def _all_schema_context(manager: SchemaManager) -> tuple[str, List[str]]:
    nodes = manager.graph.list()
    if not nodes:
        return "", []
    lines = ["=== All active world schemas ==="]
    for node in nodes:
        lines.append(
            f"- [{node.index}] level={node.level} reliability={node.reliability_weight:.3f}: "
            f"{node.description}"
        )
    return "\n".join(lines), [node.index for node in nodes]


def run_arc_agi3(
    *,
    game_id: str,
    agent,
    embedder,
    schema_llm: Optional[OpenAIChatClient] = None,
    max_steps_per_level: int = 200,
    max_retries_per_level: int = 1,
    schema_dir: str = "schema_arc_agi3",
    runs_dir: str = "outputs",
    reset_schema: bool = False,
    render_mode: Optional[str] = None,
    auto_yes: bool = True,
    always_ask_correction: bool = False,
    correction_conf_threshold: float = 1.1,
    use_llm_concepts: bool = False,
    no_retrieval: bool = False,
    model: Optional[str] = None,
) -> str:
    """Run ARC with online MemoryRecord -> LayeredSchemaGraph learning."""
    if max_retries_per_level < 1:
        raise ValueError("max_retries_per_level must be >= 1")
    model_name = model or "unknown_model"
    run_dir = make_run_dir(runs_dir, "arc_agi3/schema", model_name)
    save_cmd(run_dir)
    state_dir = Path(run_dir) / "schema" if schema_dir == "schema_arc_agi3" else Path(schema_dir)
    if reset_schema and state_dir.exists():
        shutil.rmtree(state_dir)
    manager = build_schema_system(state_dir, llm=schema_llm, embedder=embedder)
    if use_llm_concepts:
        print("[ARC schema] --use-llm-concepts is obsolete after layered-schema migration; induction now reads transition memory.")

    env = ARCAGI3EnvWrapper(game_id, render_mode=render_mode)
    initial = env.reset()
    win_levels = initial.win_levels if initial else 1
    human_io = HumanIO(auto_yes=auto_yes)
    current_level = 1
    terminate = False

    while current_level <= win_levels and not terminate:
        level_won = False
        for level_attempt in range(1, max_retries_per_level + 1):
            level_problem_id = f"{game_id}_L{current_level}"
            artifact_id = (
                level_problem_id
                if max_retries_per_level == 1
                else f"{level_problem_id}_A{level_attempt}"
            )
            problem = Problem(
                id=level_problem_id,
                prompt="",
                problem_type="arc_grid",
                meta={
                    "game_id": game_id,
                    "level": current_level,
                    "attempt": level_attempt,
                    "win_levels": win_levels,
                },
            )
            obs = env.reset()
            history: List[Dict[str, Any]] = []
            trajectory: List[Dict[str, Any]] = []
            attempts: List[AttemptRecord] = []
            evals: List[EvalResult] = []
            flags = ["layered_schema"]
            used_schema_ids: List[str] = []
            stop_reason: Optional[str] = None
            last_trace: Optional[ReasoningTrace] = None

            for step in range(max_steps_per_level):
                if obs is None or obs.is_empty() or not obs.frame:
                    stop_reason = "empty_observation"
                    evals.append(EvalResult(False, "", "WIN", "EMPTY_OBSERVATION"))
                    break
                grid = obs.frame[-1]
                available = env.get_available_actions(obs)
                if not available:
                    stop_reason = "no_actions"
                    evals.append(EvalResult(False, "", "WIN", "NO_ACTIONS"))
                    break
                available_names = [action.name for action in available]
                observation_summary = _grid_summary(grid)
                query = _schema_query(
                    game_id=game_id,
                    level=current_level,
                    grid_summary=observation_summary,
                    available_actions=available_names,
                    history=history,
                )
                if no_retrieval:
                    schema_context, injected_ids = _all_schema_context(manager)
                else:
                    matches = manager.retrieve(query, top_k=10)
                    injected_ids = [match.node.index for match in matches]
                    schema_context = manager.format_context(matches)
                prompt = _build_arc_prompt(
                    game_id=game_id,
                    level=current_level,
                    step=step,
                    grid_summary=observation_summary,
                    available_actions=available_names,
                    history=history,
                    schema_context=schema_context,
                )
                if step == 0:
                    problem.prompt = prompt

                image = ARCAGI3EnvWrapper.grid_to_image(grid, cell_size=8)
                image_dir = Path(run_dir) / artifact_id
                image_dir.mkdir(parents=True, exist_ok=True)
                image_name = f"step_{step:03d}.png"
                image.save(image_dir / image_name)
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                image_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

                try:
                    attempt = agent.answer(
                        prompt=prompt,
                        meta={
                            "game_id": game_id,
                            "level": current_level,
                            "step": step,
                            "image_data_url": image_url,
                        },
                        response_format="json_object",
                    )
                except Exception as error:
                    attempt = AgentAttempt(
                        answer_text='{"action":"ACTION1","reasoning":{"schemas_used":[],"explanation":"API fallback"}}',
                        reasoning_trace=ReasoningTrace([], [], "API fallback"),
                        usage=Usage(0, 0, 0),
                        raw={"error": str(error)},
                    )
                    flags.append("action_api_fallback")
                chosen, action_data, trace = _parse_agent_action(
                    attempt.answer_text, available
                )
                attempt.reasoning_trace = trace
                last_trace = trace
                claimed_schema_ids = [
                    schema_id
                    for schema_id in trace.concepts
                    if schema_id in injected_ids
                ]
                used_schema_ids.extend(claimed_schema_ids or injected_ids)
                pre_grid = grid.copy()
                obs = env.step(chosen, data=action_data) if chosen else None
                post_grid = obs.frame[-1] if obs and obs.frame else None
                changed, regions = compute_grid_diff(pre_grid, post_grid)
                state = str(obs.state) if obs else "UNKNOWN"
                post_summary = _grid_summary(post_grid) if post_grid is not None else "No post-action grid."
                result_text = _transition_result(
                    changed=changed,
                    changed_regions=regions,
                    state=state,
                    post_summary=post_summary,
                )
                action_name = chosen.name if chosen else "NONE"
                action_text = action_name + (
                    f" data={json.dumps(action_data, sort_keys=True)}" if action_data else ""
                )
                try:
                    memory_id, learned_schema_id = _learn_transition(
                        manager=manager,
                        game_id=game_id,
                        level=current_level,
                        step=step + 1,
                        query=query,
                        observation=observation_summary,
                        action=action_text,
                        result=result_text,
                        state=state,
                    )
                except Exception as error:
                    memory_id, learned_schema_id = "", None
                    flags.append("schema_induction_error")
                    print(f"[ARC schema] induction failed at step {step + 1}: {error}")

                history_item = {
                    "step": step + 1,
                    "action": action_name,
                    "action_data": action_data,
                    "state": state,
                    "grid_changed": changed,
                }
                history.append(history_item)
                trajectory.append(
                    {
                        **history_item,
                        "memory_id": memory_id,
                        "schema_node_learned": learned_schema_id,
                        "schema_nodes_injected": injected_ids,
                    }
                )
                attempts.append(
                    AttemptRecord(
                        input_prompt=prompt,
                        answer_text=attempt.answer_text,
                        reasoning_trace=trace,
                        usage=asdict(attempt.usage),
                    )
                )
                write_step(
                    run_dir,
                    artifact_id,
                    step + 1,
                    {
                        "prompt": prompt,
                        "answer_text": attempt.answer_text,
                        "reasoning_trace": asdict(trace),
                        "usage": asdict(attempt.usage),
                        "action": action_name,
                        "action_data": action_data,
                        "state": state,
                        "grid_image": image_name,
                        "grid_changed": changed,
                        "changed_regions": regions,
                        "memory_id": memory_id,
                        "schema_nodes_injected": injected_ids,
                        "schema_node_learned": learned_schema_id,
                    },
                )
                print(
                    f"[ARC schema] L{current_level} step={step + 1} action={action_name} "
                    f"state={state} changed={changed} learned={learned_schema_id or '-'}"
                )

                if obs and obs.state == env.GameState.WIN:
                    evals.append(EvalResult(True, action_name, "WIN", ""))
                    level_won = True
                    break
                if obs and obs.state == env.GameState.GAME_OVER:
                    evals.append(EvalResult(False, action_name, "WIN", "GAME_OVER"))
                    break
            else:
                stop_reason = "max_steps"
                evals.append(EvalResult(False, "", "WIN", "TIMEOUT"))

            outcome = (
                "WIN"
                if evals and evals[-1].correct
                else str(evals[-1].details or "FAILED").upper()
                if evals
                else "FAILED"
            )
            feedback_memory_id = _apply_level_feedback(
                manager,
                schema_ids=used_schema_ids,
                game_id=game_id,
                level=current_level,
                outcome=outcome,
            )
            if feedback_memory_id:
                flags.append("schema_reinforce" if level_won else "schema_weaken")
            confidence = _schema_confidence(manager, used_schema_ids)

            if (
                not level_won
                and last_trace is not None
                and (always_ask_correction or confidence > correction_conf_threshold)
            ):
                last_attempt = attempts[-1] if attempts else None
                if last_attempt is not None:
                    correction = human_io.ask_correction(
                        problem=problem,
                        attempt=AgentAttempt(
                            answer_text=last_attempt.answer_text,
                            reasoning_trace=last_attempt.reasoning_trace,
                            usage=Usage(**last_attempt.usage),
                            raw={},
                        ),
                        reasoning_confidence=confidence,
                        eval=evals[-1],
                    )
                    if correction.strip():
                        correction_record = MemoryRecord(
                            task=f"Correct ARC-AGI-3 game {game_id} level {current_level} schema",
                            context=problem.prompt,
                            outcome="Human supplied a correction after failure.",
                            success=True,
                            feedback="human correction",
                            tags=["arc_agi3", "human_feedback"],
                            metadata={"schema_level": 3},
                        )
                        correction_record.add_event(
                            observation=problem.prompt,
                            action=correction,
                            result="Correction should guide the next attempt.",
                        )
                        corrected_node = manager.remember_and_learn(correction_record)
                        if corrected_node:
                            manager.apply_feedback(
                                [corrected_node.index],
                                memory_id=correction_record.id,
                                positive=True,
                            )
                        flags.append("human_correction")

            episode = Episode(
                problem=problem,
                attempts=attempts,
                evals=evals,
                reasoning_trace=last_trace,
                reasoning_confidence=round(confidence, 4),
                flags=flags,
                stop_reason=stop_reason,
                model=model,
            )
            write_episode(run_dir, episode, subdir=artifact_id)
            write_trajectory(run_dir, artifact_id, trajectory)
            try:
                manager.run_maintenance()
            except Exception as error:
                print(f"[ARC schema] maintenance failed: {error}")

            if level_won:
                current_level += 1
                break
            if level_attempt == max_retries_per_level:
                terminate = True
        if not level_won and terminate:
            break

    scorecard = env.get_scorecard()
    if scorecard:
        print(f"Final Score: {scorecard.score}, Actions: {scorecard.total_actions}")
    return run_dir


__all__ = ["run_arc_agi3"]
