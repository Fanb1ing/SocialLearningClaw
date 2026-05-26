from __future__ import annotations

import base64
import io
import json
import os
import shutil
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import numpy as np

from .agent.base import AgentAttempt, ReasoningTrace, Usage
from .dataset.arc_agi3 import ARCAGI3EnvWrapper
from .dataset.base import EvalResult, Problem
from .human_io import HumanIO
from .logging import write_episode, write_step, write_trajectory
from .schema.arc_agi3_parser import (
    build_action_effect_concepts_and_relations,
    build_spatial_relations,
    color_name,
    compute_grid_diff,
    diff_objects_to_rules,
    extract_grid_objects,
    llm_extract_grid_concepts,
    objects_to_concepts,
)
from .schema.graph import Concept, Relation, SchemaGraph
from .schema.initializer import SchemaInitializer
from .schema.retriever import SchemaRetriever
from .schema.storage import SchemaStorage
from .types import AttemptRecord, Episode
from .utils import add_concepts_with_embeddings, add_relations_resolved, make_run_dir, resolve_relation_names, save_cmd


_DEFAULT_COLORS = {
    0: "Black", 1: "Blue", 2: "Red", 3: "Green", 4: "Yellow",
    5: "Gray", 6: "Pink", 7: "Orange", 8: "Cyan", 9: "Maroon",
    10: "Beige", 11: "Lime", 12: "Indigo", 13: "Brown", 14: "Magenta", 15: "White",
}


def _reset_schema_dir(schema_dir: str) -> None:
    if os.path.exists(schema_dir):
        shutil.rmtree(schema_dir)
    os.makedirs(schema_dir, exist_ok=True)


def _encode_text(embedder, text: str):
    emb = embedder.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    if emb.ndim == 2:
        emb = emb[0]
    return emb


def _retrieve_relevant_concepts(
    grid: np.ndarray,
    graph: SchemaGraph,
    embeddings: dict,
    embedder,
    top_k: int = 10,
):
    """Retrieve schema concepts most relevant to the current grid state."""
    objects = extract_grid_objects(grid)
    if not objects:
        query_text = "ARC grid: empty or background only"
    else:
        colour_counts: dict[str, int] = {}
        total_area = 0
        for o in objects:
            cname = color_name(o["color"])
            colour_counts[cname] = colour_counts.get(cname, 0) + 1
            total_area += o["area"]
        parts = [f"{c}({n})" for c, n in colour_counts.items()]
        query_text = f"ARC grid with {len(objects)} objects, colours {', '.join(parts)}, total_area={total_area}"

    def _fallback_top_k():
        all_concepts = graph.list_concepts()
        top = sorted(all_concepts, key=lambda c: c.confidence, reverse=True)[:top_k]
        kept = {c.id for c in top}
        rels = [r for r in graph.list_relations() if r.source in kept or r.target in kept]
        return top, rels

    if not embeddings:
        return _fallback_top_k()

    try:
        query_emb = _encode_text(embedder, query_text)
    except Exception:
        return _fallback_top_k()

    concept_ids = list(embeddings.keys())
    if not concept_ids:
        return _fallback_top_k()

    concept_embs = np.stack([embeddings[cid] for cid in concept_ids])
    sims = concept_embs @ query_emb
    top_indices = np.argsort(sims)[::-1][:top_k]
    top_concept_ids = {concept_ids[i] for i in top_indices if sims[i] > 0.1}

    if not top_concept_ids:
        return _fallback_top_k()

    top_concepts = [graph.get_concept(cid) for cid in top_concept_ids]
    top_concepts = [c for c in top_concepts if c is not None]
    top_relations = [r for r in graph.list_relations() if r.source in top_concept_ids or r.target in top_concept_ids]
    return top_concepts, top_relations


def _build_arc_prompt(
    *,
    game_id: str,
    level: int,
    step: int,
    grid_text: str,
    available_actions: List[str],
    schema_concepts: List[Concept],
    schema_relations: List[Relation],
    history: List[Dict],
    max_concepts: int = 10,
) -> str:
    """Build a prompt for ARC-AGI-3 agent decision."""
    sorted_concepts = schema_concepts
    kept_ids = {c.id for c in sorted_concepts}

    rule_rels: List[Relation] = []
    spatial_rels: List[Relation] = []
    for r in schema_relations:
        if r.source in kept_ids or r.target in kept_ids:
            if r.relation_type.startswith("transformed_by_"):
                rule_rels.append(r)
            else:
                spatial_rels.append(r)

    concept_id_to_name = {c.id: c.name for c in sorted_concepts}
    concept_blocks = []
    for c in sorted_concepts:
        blk = f"- {c.name} (confidence={c.confidence:.2f}): {c.description}"
        nbrs = [r for r in spatial_rels if r.source == c.id or r.target == c.id]
        if nbrs:
            blk += "\n  relations: " + ", ".join(
                f"{r.relation_type}({concept_id_to_name.get(r.target if r.source == c.id else r.source, '?')})"
                for r in nbrs
            )
        concept_blocks.append(blk)

    rule_blocks: List[str] = []
    for r in rule_rels:
        src_c = next((c for c in sorted_concepts if c.id == r.source), None)
        tgt_c = next((c for c in sorted_concepts if c.id == r.target), None)
        if not src_c or not tgt_c:
            continue
        action = r.relation_type.replace("transformed_by_", "")
        if src_c.name == tgt_c.name and src_c.description == tgt_c.description:
            continue
        if src_c.name == tgt_c.name:
            rule_blocks.append(f"- {src_c.name} was affected by {action}")
        else:
            rule_blocks.append(f"- {src_c.name} became {tgt_c.name} after {action}")

    history_text = ""
    if history:
        # Per-action-type no-effect count
        action_no_effect: dict = {}
        action_total: dict = {}
        for h in history:
            act = h["action"]
            action_total[act] = action_total.get(act, 0) + 1
            if not h.get("grid_changed", True):
                action_no_effect[act] = action_no_effect.get(act, 0) + 1

        history_text = "\nAction summary so far:\n"
        for act in sorted(action_total):
            total = action_total[act]
            no_eff = action_no_effect.get(act, 0)
            history_text += f"  {act}: tried {total}x, no-effect {no_eff}x\n"

        # Warn when an action has only failed
        only_failing = [a for a in action_total if action_no_effect.get(a, 0) == action_total[a]]
        never_tried = [a for a in available_actions if a not in action_total]
        if only_failing:
            history_text += f"  NOTE: {', '.join(only_failing)} have had NO effect in any attempt so far.\n"
        if never_tried:
            history_text += f"  NOTE: You have NOT yet tried: {', '.join(never_tried)}. Consider exploring these.\n"

        history_text += "\nRecent history (last 5 steps):\n"
        for h in history[-5:]:
            act = h["action"]
            data = h.get("action_data", {})
            changed = h.get("grid_changed", "unknown")
            if act == "ACTION6" and data:
                history_text += f"  Step {h['step']}: action={act} at ({data.get('x', '?')},{data.get('y', '?')}) -> state={h['state']}, grid_changed={changed}\n"
            else:
                history_text += f"  Step {h['step']}: action={act} -> state={h['state']}, grid_changed={changed}\n"

    action_effect_blocks: List[str] = []
    for r in schema_relations:
        if r.weight < 0.3:
            continue
        if r.relation_type == "no_effect":
            src_c = next((c for c in schema_concepts if c.id == r.source), None)
            if src_c:
                action_effect_blocks.append(f"- {src_c.name} had no effect on grid")
        elif r.relation_type.startswith("changed_") or r.relation_type.startswith("affected"):
            src_c = next((c for c in schema_concepts if c.id == r.source), None)
            tgt_c = next((c for c in schema_concepts if c.id == r.target), None)
            if src_c and tgt_c:
                action_effect_blocks.append(f"- {src_c.name} {r.relation_type.replace('_', ' ')} {tgt_c.name}")
    seen_ae: set = set()
    deduped_ae: List[str] = []
    for line in action_effect_blocks:
        if line not in seen_ae:
            seen_ae.add(line)
            deduped_ae.append(line)
    action_effect_blocks = deduped_ae

    system = (
        "You are an ARC-AGI-3 game-playing agent. You observe a grid world and choose actions to solve puzzles.\n\n"
        "Grid coordinate system:\n"
        "  - col (x): 0 = leftmost column, increases to the RIGHT\n"
        "  - row (y): 0 = topmost row, increases DOWNWARD\n"
        "  - All positions are written as (col, row) = (x, y)\n"
        "  - The image has sparse gridlines every 8 cells to help locate positions\n\n"
        "Action definitions:\n"
        "- ACTION1 = move up: row decreases by 1. No data needed.\n"
        "- ACTION2 = move down: row increases by 1. No data needed.\n"
        "- ACTION3 = move left: col decreases by 1. No data needed.\n"
        "- ACTION4 = move right: col increases by 1. No data needed.\n"
        "- ACTION6 = click at specific (col, row). MUST provide data: {\"x\": col_int, \"y\": row_int}.\n"
        "- ACTION7 = special action. No data needed.\n\n"
        "Rules:\n"
        "1. Analyze the current grid image and the known objects/rules below.\n"
        "2. Choose the best action from the available list.\n"
        "3. ONLY provide x/y coordinates for ACTION6 (click). Never attach data to move actions.\n"
        "4. Each level has a limited step budget. Avoid unnecessary moves; be direct and efficient.\n"
        "5. IMPORTANT: If one action type (e.g. ACTION6) has been tried multiple times with NO grid "
        "changes, STOP using that action type and try a DIFFERENT action type "
        "(ACTION1/ACTION2/ACTION3/ACTION4/ACTION7). Repeating a failing action is wasteful.\n\n"
        "Output format (strict JSON):\n"
        '{\n'
        '  "action": "ACTION_NAME",\n'
        '  "data": {"x": 10, "y": 20},\n'
        '  "reasoning": {\n'
        '    "concepts_used": ["Concept1", "Concept2"],\n'
        '    "reasoning_path": ["Concept1 -> relation_type -> Concept2"],\n'
        '    "explanation": "Why I chose this action based on the grid and schema."\n'
        '  }\n'
        '}\n'
        "If action does not need coordinates, omit the 'data' field or set it to {}.\n"
        "CRITICAL: Output ONLY raw JSON. Do not write any text before or after the JSON object. "
        "Do not use markdown code blocks. Do not explain your reasoning outside the JSON.\n"
    )

    user = (
        f"Game: {game_id} | Level: {level} | Step: {step}\n"
        f"Grid (see attached image): {grid_text}\n\n"
        f"Available actions: {', '.join(available_actions)}\n"
        f"{history_text}\n"
    )
    if concept_blocks:
        user += "Top relevant objects from schema:\n" + "\n".join(concept_blocks) + "\n\n"
    if rule_blocks:
        user += "Learned transformation rules:\n" + "\n".join(rule_blocks) + "\n\n"
    if action_effect_blocks:
        user += "Learned action effects:\n" + "\n".join(action_effect_blocks) + "\n\n"
    user += "Choose your next action:"

    return system + "\n---\n" + user


def _parse_agent_action(answer_text: str, available_actions: List):
    """Parse LLM JSON output into (action, data, reasoning_trace)."""
    action_name = None
    data = {}
    reasoning_trace = ReasoningTrace(concepts=[], relations=[], explanation="")

    text = answer_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(text[start:end + 1])
            action_name = parsed.get("action", "")
            data = parsed.get("data") or {}
            raw_reasoning = parsed.get("reasoning", "")

            if isinstance(raw_reasoning, dict):
                concepts = raw_reasoning.get("concepts_used", [])
                paths = raw_reasoning.get("reasoning_path", [])
                explanation = raw_reasoning.get("explanation", "")
            elif isinstance(raw_reasoning, str):
                concepts, paths, explanation = [], [], raw_reasoning
            else:
                concepts, paths, explanation = [], [], ""

            concepts = [c.strip() for c in concepts if isinstance(c, str) and c.strip()]

            relations = []
            for p in paths:
                if not isinstance(p, str):
                    continue
                p = p.strip()
                if not p:
                    continue
                normalized = p.replace("→", "->")
                parts = [part.strip() for part in normalized.split("->") if part.strip()]
                if len(parts) >= 3:
                    i = 0
                    while i + 2 < len(parts):
                        relations.append((parts[i], parts[i + 2], parts[i + 1]))
                        i += 2
                elif len(parts) == 2:
                    relations.append((parts[0], parts[1], "related"))

            reasoning_trace = ReasoningTrace(concepts=concepts, relations=relations, explanation=explanation)
        except Exception:
            pass

    if not action_name:
        search_text = answer_text.upper()
        for act in available_actions:
            if act.name in search_text:
                action_name = act.name
                break

    chosen_action = None
    if action_name:
        upper_name = action_name.upper()
        for act in available_actions:
            if act.name == upper_name:
                chosen_action = act
                break

    if chosen_action is None and available_actions:
        chosen_action = available_actions[0]

    if chosen_action and chosen_action.name != "ACTION6":
        data = {}

    return chosen_action, data or {}, reasoning_trace


def _reinforce_level_rules(graph: SchemaGraph, concepts: List[Concept], delta: float = 0.05):
    for c in concepts:
        existing = graph.get_concept(c.id)
        if existing:
            graph.update_concept(c.id, confidence=min(0.95, existing.confidence + delta))


def _correct_level_rules(graph: SchemaGraph, concepts: List[Concept], delta: float = -0.05):
    for c in concepts:
        existing = graph.get_concept(c.id)
        if existing:
            graph.update_concept(c.id, confidence=max(0.1, existing.confidence + delta))


def run_arc_agi3(
    *,
    game_id: str,
    agent,
    embedder,
    max_steps_per_level: int = 200,
    max_retries_per_level: int = 3,
    schema_dir: str = "schema_arc_agi3",
    runs_dir: str = "runs",
    reset_schema: bool = False,
    render_mode: Optional[str] = None,
    auto_yes: bool = False,
    always_ask_correction: bool = False,
    correction_conf_threshold: float = -1.0,
    use_llm_concepts: bool = False,
    no_retrieval: bool = False,
    model: Optional[str] = None,
) -> str:
    """Run ARC-AGI-3 interactive loop with schema-based reasoning."""
    print(f"[ARC] Creating environment for game={game_id} ...")
    env_wrapper = ARCAGI3EnvWrapper(game_id, render_mode=render_mode)
    print("[ARC] Environment created. Getting initial observation via reset() ...")
    obs = env_wrapper.reset()
    win_levels = obs.win_levels if obs else 1
    print(f"[ARC] win_levels={win_levels}, initial state={obs.state if obs else 'N/A'}")

    model_str = model or "unknown_model"
    run_dir = make_run_dir(runs_dir, "arc_agi3", model_str)
    save_cmd(run_dir)

    # Schema is scoped to the run directory.
    # If user explicitly set schema_dir, respect it; otherwise use run_dir/schema.
    if schema_dir == "schema_arc_agi3":
        schema_dir = os.path.join(run_dir, "schema")

    if reset_schema:
        _reset_schema_dir(schema_dir)
    os.makedirs(schema_dir, exist_ok=True)
    storage = SchemaStorage(
        concepts_path=os.path.join(schema_dir, "concepts.jsonl"),
        relations_path=os.path.join(schema_dir, "relations.jsonl"),
        embeddings_path=os.path.join(schema_dir, "concept_embeddings.npy"),
        concept_ids_path=os.path.join(schema_dir, "concept_ids.json"),
    )
    graph, embeddings = storage.load()

    retriever = SchemaRetriever(graph, embeddings, embedder, agent=agent)
    initializer = SchemaInitializer(agent)
    human_io = HumanIO(auto_yes=auto_yes)

    current_level = 0
    total_steps = 0
    level_retries = 0

    while current_level < win_levels:
        print(f"\n=== Level {current_level + 1}/{win_levels} ===")
        print(f"[ARC] Resetting level {current_level + 1} ...")
        obs = env_wrapper.reset()
        print(f"[ARC] Level {current_level + 1} initial state={obs.state if obs else 'N/A'}")

        level_problem = Problem(
            id=f"{game_id}_L{current_level + 1}",
            prompt="",
            problem_type="arc_grid",
            meta={"game_id": game_id, "level": current_level + 1, "win_levels": win_levels},
        )

        step = 0
        prev_objects = []
        prev_concepts = []
        history = []
        trajectory: List[Dict[str, Any]] = []
        attempts: List[AttemptRecord] = []
        evals: List[EvalResult] = []
        flags: List[str] = []
        stop_reason: Optional[str] = None
        last_trace: Optional[ReasoningTrace] = None

        while step < max_steps_per_level:
            if obs is None or obs.is_empty():
                print(f"[ARC] Observation is None/empty at step {step}, breaking.")
                break

            grid = obs.frame[0] if obs.frame else None
            if grid is None:
                print(f"[ARC] Grid is None at step {step}, breaking.")
                break

            available_actions = env_wrapper.get_available_actions(obs)
            h, w = grid.shape
            grid_summary = f"{h}x{w} full grid — see attached image"
            print(f"[ARC] Step {step}: available_actions={[a.name for a in available_actions]}, grid={h}x{w}")

            # Render grid as image and encode to base64
            grid_img = ARCAGI3EnvWrapper.grid_to_image(grid, cell_size=8)
            img_dir = os.path.join(run_dir, level_problem.id)
            os.makedirs(img_dir, exist_ok=True)
            img_path = os.path.join(img_dir, f"step_{step:03d}.png")
            grid_img.save(img_path)

            img_buffer = io.BytesIO()
            grid_img.save(img_buffer, format="PNG")
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode("ascii")
            image_data_url = f"data:image/png;base64,{img_base64}"

            # ---- Step 0: populate schema from initial observation BEFORE building prompt ----
            # This ensures step 0's prompt already contains the initial scene concepts.
            if step == 0:
                print(f"[ARC] Level {current_level + 1} step 0: initializing schema from initial observation ...")
                if use_llm_concepts:
                    try:
                        llm_concepts, llm_relations = llm_extract_grid_concepts(
                            grid_img, agent, level=current_level, step=0, grid_shape=grid.shape
                        )
                        add_concepts_with_embeddings(graph, embeddings, embedder, llm_concepts)
                        for r in llm_relations:
                            graph.add_relation(r)
                        objects_summary = "\n".join(
                            f"- {c.name}: {c.description}" for c in llm_concepts
                        ) if llm_concepts else "(no objects detected by LLM)"
                        prev_objects = []
                        prev_concepts = llm_concepts
                    except Exception as e:
                        print(f"[ARC] LLM concept extraction failed: {e}. Falling back to BFS.")
                        init_objects = extract_grid_objects(grid)
                        init_concepts = objects_to_concepts(init_objects, level=current_level, step=0)
                        init_relations = build_spatial_relations(init_objects, init_concepts)
                        add_concepts_with_embeddings(graph, embeddings, embedder, init_concepts)
                        for r in init_relations:
                            if not graph.get_relation(r.source, r.target, r.relation_type):
                                graph.add_relation(r)
                        prev_objects = init_objects
                        prev_concepts = init_concepts
                        objects_summary = "\n".join(
                            f"- {color_name(o['color'])} blob at col={o['top_left'][0]}-{o['bottom_right'][0]},"
                            f" row={o['top_left'][1]}-{o['bottom_right'][1]}, area={o['area']}"
                            for o in init_objects
                        ) if init_objects else "(no objects detected)"
                else:
                    init_objects = extract_grid_objects(grid)
                    init_concepts = objects_to_concepts(init_objects, level=current_level, step=0)
                    init_relations = build_spatial_relations(init_objects, init_concepts)
                    add_concepts_with_embeddings(graph, embeddings, embedder, init_concepts)
                    for r in init_relations:
                        if not graph.get_relation(r.source, r.target, r.relation_type):
                            graph.add_relation(r)
                    prev_objects = init_objects
                    prev_concepts = init_concepts
                    objects_summary = "\n".join(
                        f"- {color_name(o['color'])} blob at col={o['top_left'][0]}-{o['bottom_right'][0]},"
                        f" row={o['top_left'][1]}-{o['bottom_right'][1]}, area={o['area']}"
                        for o in init_objects
                    ) if init_objects else "(no objects detected)"

                storage.save(graph, embeddings)

                # Check schema sufficiency (now AFTER initial concepts are in schema)
                init_problem = Problem(
                    id=level_problem.id,
                    prompt=f"ARC-AGI-3 {game_id} Level {current_level + 1} initial objects:\n{objects_summary}",
                    problem_type="arc_grid",
                    meta=level_problem.meta,
                )
                print(f"[ARC] Checking schema sufficiency ...")
                try:
                    result = retriever.retrieve(init_problem, top_k=5, threshold=0.75)
                    sufficient = retriever.is_sufficient(result)
                except Exception as e:
                    print(f"[ARC] Schema retrieval failed: {e}. Treating as sufficient.")
                    result = None
                    sufficient = True

                print(f"[ARC] Schema sufficient={sufficient}, missing={result.missing if result else []}")

                if not sufficient and result:
                    if auto_yes:
                        print(f"[auto-yes] Schema insufficient for level {current_level + 1}, auto-generating ...")
                        try:
                            auto_concepts, auto_relations = initializer.generate_schema(init_problem)
                            add_concepts_with_embeddings(graph, embeddings, embedder, auto_concepts)
                            add_relations_resolved(graph, auto_relations)
                            flags.append("agent_auto_init")
                        except Exception as e:
                            print(f"[ARC] Auto-generate schema failed: {e}. Proceeding with current schema.")
                    else:
                        missing_desc = initializer.describe_missing(init_problem, result.matched, missing=result.missing)
                        human_answer = human_io.ask(
                            question=missing_desc["question"],
                            context=missing_desc["context"],
                            hint=missing_desc["hint"],
                        )
                        if human_answer.strip():
                            try:
                                new_concepts, new_relations = initializer.parse_human_answer(human_answer, init_problem)
                                add_concepts_with_embeddings(graph, embeddings, embedder, new_concepts)
                                add_relations_resolved(graph, new_relations)
                                flags.append("human_init_concepts")
                            except Exception as e:
                                print(f"[ARC] Parse human answer failed: {e}. Proceeding with current schema.")

            # ---- Retrieve relevant schema concepts (step 0 now has initial concepts) ----
            if no_retrieval:
                top_concepts = graph.list_concepts()
                top_relations = graph.list_relations()
            else:
                top_concepts, top_relations = _retrieve_relevant_concepts(
                    grid, graph, embeddings, embedder, top_k=10
                )

            # Always inject recent action-effect concepts
            ae_concepts = [
                c for c in graph.list_concepts()
                if c.source == "action_effect" and c.category == f"level_{current_level}"
            ]
            ae_concepts = ae_concepts[-5:] if len(ae_concepts) > 5 else ae_concepts
            ae_ids = {c.id for c in ae_concepts}
            ae_relations = [
                r for r in graph.list_relations()
                if (r.source in ae_ids or r.target in ae_ids)
                and (r.relation_type == "no_effect" or r.relation_type.startswith("changed_") or r.relation_type.startswith("affected"))
            ]

            # Always inject human feedback concepts
            human_concepts = [
                c for c in graph.list_concepts()
                if c.source in ("human_feedback", "human_init_concepts")
            ]
            human_concepts = human_concepts[-5:] if len(human_concepts) > 5 else human_concepts
            human_ids = {c.id for c in human_concepts}
            human_relations = [
                r for r in graph.list_relations()
                if r.source in human_ids or r.target in human_ids
            ]

            # Merge (dedup)
            seen_cids: set = set()
            merged_concepts: List[Concept] = []
            for c in top_concepts + ae_concepts + human_concepts:
                if c.id not in seen_cids:
                    seen_cids.add(c.id)
                    merged_concepts.append(c)
            seen_rkeys: set = set()
            merged_relations: List[Relation] = []
            for r in top_relations + ae_relations + human_relations:
                rkey = (r.source, r.target, r.relation_type)
                if rkey not in seen_rkeys:
                    seen_rkeys.add(rkey)
                    merged_relations.append(r)

            prompt = _build_arc_prompt(
                game_id=game_id,
                level=current_level + 1,
                step=step,
                grid_text=grid_summary,
                available_actions=[a.name for a in available_actions],
                schema_concepts=merged_concepts,
                schema_relations=merged_relations,
                history=history,
            )

            if step == 0:
                level_problem.prompt = prompt

            # Agent decides action
            print(f"[ARC] Step {step}: calling LLM ...")
            try:
                attempt = agent.answer(
                    prompt=prompt,
                    meta={
                        "game_id": game_id,
                        "level": current_level,
                        "step": step,
                        "image_data_url": image_data_url,
                    },
                    response_format="json_object",
                )
            except Exception as e:
                print(f"[ARC] Step {step}: LLM call FAILED: {e}")
                attempt = AgentAttempt(
                    answer_text='{"action": "ACTION1", "reasoning": {"concepts_used": [], "reasoning_path": [], "explanation": "API error fallback"}}',
                    reasoning_trace=ReasoningTrace(concepts=[], relations=[], explanation="API error fallback"),
                    usage=Usage(input_tokens=0, output_tokens=0, total_tokens=0),
                    raw={"error": str(e)},
                )

            chosen_action, action_data, reasoning_trace = _parse_agent_action(
                attempt.answer_text, available_actions
            )
            attempt.reasoning_trace = reasoning_trace

            # Execute action
            pre_grid = grid.copy() if grid is not None else None
            if chosen_action:
                obs = env_wrapper.step(chosen_action, data=action_data)
            else:
                obs = env_wrapper.step(available_actions[0])

            step += 1
            total_steps += 1

            state_str = str(obs.state) if obs else "UNKNOWN"
            post_grid = obs.frame[0] if obs and obs.frame else None
            grid_changed, changed_regions = compute_grid_diff(pre_grid, post_grid)

            # Extract objects from post-action grid and update schema
            if not grid_changed and prev_concepts:
                post_objects = prev_objects
                post_concepts = prev_concepts
                post_relations = []
                print(f"[ARC] Step {step}: grid unchanged, reusing {len(post_concepts)} previous concepts")
            elif use_llm_concepts:
                try:
                    if post_grid is not None:
                        post_grid_img = ARCAGI3EnvWrapper.grid_to_image(post_grid, cell_size=8)
                        post_concepts, post_relations = llm_extract_grid_concepts(
                            post_grid_img, agent, level=current_level, step=step - 1,
                            grid_shape=post_grid.shape,
                        )
                        post_objects = []
                    else:
                        post_concepts, post_relations, post_objects = [], [], []
                except Exception as e:
                    print(f"[ARC] Step {step}: LLM concept extraction failed: {e}. Falling back to BFS.")
                    post_objects = extract_grid_objects(post_grid) if post_grid is not None else []
                    post_concepts = objects_to_concepts(post_objects, level=current_level, step=step - 1)
                    post_relations = build_spatial_relations(post_objects, post_concepts)
            else:
                post_objects = extract_grid_objects(post_grid) if post_grid is not None else []
                post_concepts = objects_to_concepts(post_objects, level=current_level, step=step - 1)
                post_relations = build_spatial_relations(post_objects, post_concepts)

            ae_concepts_new, ae_relations_new = build_action_effect_concepts_and_relations(
                action_name=chosen_action.name if chosen_action else "NONE",
                action_data=action_data,
                step=step,
                level=current_level,
                grid_changed=grid_changed,
                changed_regions=changed_regions,
                pre_objects=prev_objects,
                pre_concepts=prev_concepts,
                post_objects=post_objects,
                post_concepts=post_concepts,
            )

            if grid_changed or not prev_concepts:
                # Only add/recompute when grid actually changed or on first observation
                add_concepts_with_embeddings(graph, embeddings, embedder, post_concepts)
                for r in post_relations:
                    if not graph.get_relation(r.source, r.target, r.relation_type):
                        graph.add_relation(r)
            add_concepts_with_embeddings(graph, embeddings, embedder, ae_concepts_new)
            for r in ae_relations_new:
                graph.add_relation(r)

            storage.save(graph, embeddings)

            history.append({
                "step": step,
                "action": chosen_action.name if chosen_action else "NONE",
                "state": state_str,
                "action_data": action_data,
                "grid_changed": grid_changed,
            })

            trajectory.append({
                "step": step,
                "action": chosen_action.name if chosen_action else "NONE",
                "x": action_data.get("x"),
                "y": action_data.get("y"),
                "state": state_str,
                "grid_changed": grid_changed,
                "schema_concepts_added": [c.name for c in (post_concepts + ae_concepts_new)],
            })

            attempts.append(AttemptRecord(
                input_prompt=prompt,
                answer_text=attempt.answer_text,
                reasoning_trace=attempt.reasoning_trace,
                usage=asdict(attempt.usage) if attempt.usage else {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            ))
            last_trace = attempt.reasoning_trace

            print(f"  Step {step}: action={chosen_action.name if chosen_action else 'NONE'}, state={state_str}, grid_changed={grid_changed}")

            step_data = {
                "prompt": prompt,
                "answer_text": attempt.answer_text,
                "reasoning_trace": asdict(attempt.reasoning_trace) if attempt.reasoning_trace else {},
                "usage": asdict(attempt.usage) if attempt.usage else {},
                "action": chosen_action.name if chosen_action else "NONE",
                "action_data": action_data,
                "state": state_str,
                "grid_image": f"step_{step:03d}.png",
                "schema_concepts_injected": [c.name for c in merged_concepts],
                "schema_relations_injected": [r.relation_type for r in merged_relations],
                "grid_changed": grid_changed,
            }
            write_step(run_dir, level_problem.id, step, step_data)

            prev_objects = post_objects
            prev_concepts = post_concepts

            if obs and obs.state == env_wrapper.GameState.WIN:
                print(f"  Level {current_level + 1} WON in {step} steps!")
                evals.append(EvalResult(correct=True, pred=chosen_action.name if chosen_action else "NONE", gold="WIN", details=""))
                flags.append("schema_reinforce")
                _reinforce_level_rules(graph, prev_concepts, delta=0.05)
                break

            if obs and obs.state == env_wrapper.GameState.GAME_OVER:
                print(f"  Level {current_level + 1} GAME_OVER at step {step}")
                evals.append(EvalResult(correct=False, pred=chosen_action.name if chosen_action else "NONE", gold="WIN", details="GAME_OVER"))
                flags.append("schema_correct")
                _correct_level_rules(graph, prev_concepts, delta=-0.05)
                break

        else:
            print(f"  Level {current_level + 1} TIMEOUT after {step} steps")
            evals.append(EvalResult(correct=False, pred="", gold="WIN", details="TIMEOUT"))
            stop_reason = "max_iters"

        reasoning_confidence = 0.0
        if last_trace:
            reasoning_confidence = graph.compute_confidence(last_trace)

        # High-confidence error -> ask human for correction
        last_eval = evals[-1] if evals else None
        if (
            last_eval
            and not last_eval.correct
            and last_trace
            and (reasoning_confidence > correction_conf_threshold or always_ask_correction)
        ):
            print(f"[ARC] High-confidence error (confidence={reasoning_confidence:.3f}). Asking human for correction ...")
            try:
                last_record = attempts[-1]
                synthetic_attempt = AgentAttempt(
                    answer_text=last_record.answer_text,
                    reasoning_trace=last_record.reasoning_trace,
                    usage=Usage(
                        input_tokens=last_record.usage.get("input_tokens", 0),
                        output_tokens=last_record.usage.get("output_tokens", 0),
                        total_tokens=last_record.usage.get("total_tokens", 0),
                    ),
                    raw={},
                )
                correction = human_io.ask_correction(
                    problem=level_problem,
                    attempt=synthetic_attempt,
                    reasoning_confidence=reasoning_confidence,
                    eval=last_eval,
                )
                if correction.strip():
                    try:
                        corrected = initializer.parse_correction(correction, level_problem)
                        for c in corrected.get("add_concepts", []):
                            add_concepts_with_embeddings(graph, embeddings, embedder, [c])
                        add_relations_resolved(graph, corrected.get("add_relations", []))
                        for upd in corrected.get("update_concepts", []):
                            cid = upd.get("id")
                            if cid:
                                graph.update_concept(cid, **{k: v for k, v in upd.items() if k != "id"})
                        flags.append("human_correction")
                        storage.save(graph, embeddings)
                        print(f"[ARC] Applied human correction, schema saved.")
                    except Exception as e:
                        print(f"[ARC] Parse correction failed: {e}. Skipping.")
            except Exception as e:
                print(f"[ARC] Ask correction failed: {e}. Skipping.")

        episode = Episode(
            problem=level_problem,
            attempts=attempts,
            evals=evals,
            reasoning_trace=last_trace,
            reasoning_confidence=round(reasoning_confidence, 4),
            flags=flags,
            stop_reason=stop_reason,
            model=model,
        )
        ep_path = write_episode(run_dir, episode)
        print(f"  Episode saved: {ep_path}")

        traj_path = write_trajectory(run_dir, level_problem.id, trajectory)
        print(f"  Trajectory saved: {traj_path}")

        storage.save(graph, embeddings)

        last_eval = evals[-1] if evals else None
        if last_eval and last_eval.correct:
            current_level += 1
            level_retries = 0
        elif last_eval and last_eval.details == "GAME_OVER":
            level_retries += 1
            if level_retries >= max_retries_per_level:
                print(f"  Max retries ({max_retries_per_level}) reached for level {current_level + 1}, stopping run.")
                break
            print(f"  Retrying level {current_level + 1} (retry {level_retries}/{max_retries_per_level})...")
        elif stop_reason == "max_iters":
            if "human_correction" in flags:
                level_retries += 1
                if level_retries >= max_retries_per_level:
                    print(f"  Max retries reached after human correction for level {current_level + 1}, stopping.")
                    break
                print(f"  TIMEOUT but human correction received — retrying (retry {level_retries}/{max_retries_per_level})...")
            else:
                print(f"  Level {current_level + 1} TIMEOUT — terminating run.")
                break
        else:
            current_level += 1
            level_retries = 0

    scorecard = env_wrapper.get_scorecard()
    if scorecard:
        print(f"\nFinal Score: {scorecard.score}, Actions: {scorecard.total_actions}")

    return run_dir
