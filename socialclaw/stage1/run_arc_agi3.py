from __future__ import annotations

import argparse
import json
import os
import shutil
import warnings
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from .agent.openai_compatible import OpenAICompatibleAgent
from .dataset.arc_agi3 import ARCAGI3EnvWrapper
from .prompt_builder import build_prompt
from .schema.arc_agi3_parser import (
    build_spatial_relations,
    diff_objects_to_rules,
    extract_grid_objects,
    objects_to_concepts,
)
from .schema.graph import Concept, Relation, SchemaGraph
from .schema.storage import SchemaStorage


_DEFAULT_COLORS = {
    0: "Black", 1: "Blue", 2: "Red", 3: "Green", 4: "Yellow",
    5: "Gray", 6: "Pink", 7: "Orange", 8: "Cyan", 9: "Maroon",
    10: "Beige", 11: "Lime", 12: "Indigo", 13: "Brown", 14: "Magenta", 15: "White",
}


def _load_dotenv(dotenv_path: str) -> None:
    if not os.path.exists(dotenv_path):
        return
    with open(dotenv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


def _reset_schema_dir(schema_dir: str) -> None:
    if os.path.exists(schema_dir):
        shutil.rmtree(schema_dir)
    os.makedirs(schema_dir, exist_ok=True)


def _encode_text(embedder, text: str):
    emb = embedder.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    if emb.ndim == 2:
        emb = emb[0]
    return emb


def _add_concepts_with_embeddings(graph, embeddings, embedder, concepts):
    for c in concepts:
        graph.add_concept(c)
        try:
            emb = _encode_text(embedder, f"{c.name}: {c.description}")
            embeddings[c.id] = emb
        except Exception:
            pass


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
) -> str:
    """Build a prompt for ARC-AGI-3 agent decision."""
    concept_blocks = []
    for c in schema_concepts:
        blk = f"- {c.name} ({c.category}): {c.description}"
        nbrs = [r for r in schema_relations if r.source == c.id or r.target == c.id]
        if nbrs:
            blk += "\n  relations: " + ", ".join(
                f"{r.relation_type}({r.target if r.source == c.id else r.source})"
                for r in nbrs[:3]
            )
        concept_blocks.append(blk)

    history_text = ""
    if history:
        history_text = "\nRecent history (last 5 steps):\n"
        for h in history[-5:]:
            history_text += f"  Step {h['step']}: action={h['action']} -> state={h['state']}\n"

    system = (
        "You are an ARC-AGI-3 game-playing agent. You observe a grid world and choose actions to solve puzzles.\n\n"
        "Rules:\n"
        "1. Analyze the current grid and the known object/rules below.\n"
        "2. Choose the best action from the available list.\n"
        "3. For complex actions (like click), provide x and y coordinates (0-63).\n\n"
        "Output format (strict JSON):\n"
        '{"action": "ACTION_NAME", "data": {"x": 10, "y": 20}, "reasoning": "brief explanation"}\n'
        "If action is not complex, omit the 'data' field or leave it empty.\n"
    )

    user = (
        f"Game: {game_id} | Level: {level} | Step: {step}\n"
        f"Grid (center 16x16, values are color indices 0-15):\n{grid_text}\n\n"
        f"Available actions: {', '.join(available_actions)}\n"
        f"Color palette: {json.dumps(_DEFAULT_COLORS)}\n"
        f"{history_text}\n"
    )
    if concept_blocks:
        user += (
            "Known objects/rules from schema:\n"
            + "\n".join(concept_blocks)
            + "\n\n"
        )
    user += "Choose your next action:"

    return system + "\n---\n" + user


def _parse_agent_action(answer_text: str, available_actions: List):
    """Parse LLM JSON output into (action, data)."""
    action_name = None
    data = {}
    reasoning = ""

    # Try JSON extraction
    text = answer_text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(text[start:end + 1])
            action_name = parsed.get("action", "")
            data = parsed.get("data") or {}
            reasoning = parsed.get("reasoning", "")
        except Exception:
            pass

    # Fallback: search for action name in text
    if not action_name:
        for act in available_actions:
            if act.name in text.upper():
                action_name = act.name
                break

    # Match to actual action object
    chosen_action = None
    if action_name:
        for act in available_actions:
            if act.name == action_name.upper():
                chosen_action = act
                break

    if chosen_action is None and available_actions:
        chosen_action = available_actions[0]

    return chosen_action, data or {}, reasoning


def run_arc_agi3(
    *,
    game_id: str,
    agent,
    embedder,
    max_steps_per_level: int = 200,
    schema_dir: str = "schema_arc_agi3",
    runs_dir: str = "runs_arc_agi3",
    reset_schema: bool = False,
    render_mode: Optional[str] = None,
) -> str:
    """Run ARC-AGI-3 interactive loop with schema-based reasoning."""

    # Init environment
    print(f"[ARC] Creating environment for game={game_id} ...")
    env_wrapper = ARCAGI3EnvWrapper(game_id, render_mode=render_mode)
    print("[ARC] Environment created. Getting initial observation via reset() ...")
    obs = env_wrapper.reset()
    win_levels = obs.win_levels if obs else 1
    print(f"[ARC] win_levels={win_levels}, initial state={obs.state if obs else 'N/A'}")

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(runs_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Schema is scoped to the run directory (like episode logs).
    # If user explicitly set schema_dir, respect it; otherwise use run_dir/schema.
    if schema_dir == "schema_arc_agi3":
        schema_dir = os.path.join(run_dir, "schema")

    # Init schema
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

    # Episode log
    episode = {
        "game_id": game_id,
        "levels": [],
        "created_at": datetime.utcnow().isoformat() + "Z",
    }

    current_level = 0
    total_steps = 0

    while current_level < win_levels:
        print(f"\n=== Level {current_level + 1}/{win_levels} ===")
        level_log = {
            "level": current_level + 1,
            "steps": [],
            "result": None,
        }

        # Reset at start of each level and get initial observation
        print(f"[ARC] Resetting level {current_level + 1} ...")
        obs = env_wrapper.reset()
        print(f"[ARC] Level {current_level + 1} initial state={obs.state if obs else 'N/A'}")

        step = 0
        prev_objects = []
        prev_concepts = []
        history = []

        while step < max_steps_per_level:
            if obs is None or obs.is_empty():
                print(f"[ARC] Observation is None/empty at step {step}, breaking.")
                break

            grid = obs.frame[0] if obs.frame else None
            if grid is None:
                print(f"[ARC] Grid is None at step {step}, breaking.")
                break

            # 1. Extract objects -> schema concepts
            print(f"[ARC] Step {step}: extracting objects from grid ...")
            objects = extract_grid_objects(grid)
            concepts = objects_to_concepts(objects, level=current_level, step=step)
            relations = build_spatial_relations(objects, concepts)
            print(f"[ARC] Step {step}: extracted {len(objects)} objects, {len(relations)} relations")

            # Add new concepts/relations to schema
            _add_concepts_with_embeddings(graph, embeddings, embedder, concepts)
            for r in relations:
                graph.add_relation(r)

            # Diff rules from previous step
            if prev_objects and prev_concepts:
                rules = diff_objects_to_rules(
                    prev_objects, prev_concepts, objects, concepts,
                    action_name=history[-1]["action"] if history else "INIT",
                )
                for r in rules:
                    graph.add_relation(r)

            prev_objects = objects
            prev_concepts = concepts

            # 2. Build prompt with schema subgraph
            available_actions = env_wrapper.get_available_actions(obs)
            grid_text = ARCAGI3EnvWrapper.grid_to_text(grid, max_size=16)
            print(f"[ARC] Step {step}: available_actions={[a.name for a in available_actions]}")

            subgraph = graph.subgraph([c.id for c in concepts], depth=1)
            prompt = _build_arc_prompt(
                game_id=game_id,
                level=current_level + 1,
                step=step,
                grid_text=grid_text,
                available_actions=[a.name for a in available_actions],
                schema_concepts=subgraph.list_concepts(),
                schema_relations=subgraph.list_relations(),
                history=history,
            )

            # 3. Agent decides action
            print(f"[ARC] Step {step}: calling LLM (timeout={agent.timeout_s}s) ...")
            attempt = agent.answer(
                prompt=prompt,
                meta={"game_id": game_id, "level": current_level, "step": step},
            )
            print(f"[ARC] Step {step}: LLM returned, answer_text length={len(attempt.answer_text)}")

            chosen_action, action_data, reasoning = _parse_agent_action(
                attempt.answer_text, available_actions
            )
            print(f"[ARC] Step {step}: parsed action={chosen_action.name if chosen_action else 'NONE'}")

            # 4. Execute action and get next observation
            print(f"[ARC] Step {step}: executing action ...")
            if chosen_action:
                obs = env_wrapper.step(chosen_action, data=action_data)
            else:
                obs = env_wrapper.step(available_actions[0])
            print(f"[ARC] Step {step}: action executed, new state={obs.state if obs else 'N/A'}")

            step += 1
            total_steps += 1

            state_str = str(obs.state) if obs else "UNKNOWN"
            history.append({"step": step, "action": chosen_action.name if chosen_action else "NONE", "state": state_str})

            level_log["steps"].append({
                "step": step,
                "action": chosen_action.name if chosen_action else "NONE",
                "action_data": action_data,
                "reasoning": reasoning,
                "answer_text": attempt.answer_text,
                "state": state_str,
            })

            print(f"  Step {step}: action={chosen_action.name if chosen_action else 'NONE'}, state={state_str}")

            # 5. Check level outcome
            if obs and obs.state == env_wrapper.GameState.WIN:
                print(f"  Level {current_level + 1} WON in {step} steps!")
                level_log["result"] = "WIN"
                # Reinforce schema rules used in this level
                _reinforce_level_rules(graph, prev_concepts, delta=0.05)
                break

            if obs and obs.state == env_wrapper.GameState.GAME_OVER:
                print(f"  Level {current_level + 1} GAME_OVER at step {step}")
                level_log["result"] = "GAME_OVER"
                # Correct schema rules (reduce confidence)
                _correct_level_rules(graph, prev_concepts, delta=-0.05)
                break

        if level_log["result"] is None:
            level_log["result"] = "TIMEOUT"
            print(f"  Level {current_level + 1} TIMEOUT after {step} steps")

        episode["levels"].append(level_log)

        # Save schema after each level
        storage.save(graph, embeddings)

        # Log episode
        with open(os.path.join(run_dir, "episode.json"), "w", encoding="utf-8") as f:
            json.dump(episode, f, ensure_ascii=False, indent=2)

        if level_log["result"] == "WIN":
            current_level += 1
        elif level_log["result"] == "GAME_OVER":
            # Retry same level (schema may have been corrected)
            print(f"  Retrying level {current_level + 1}...")
        else:
            # Timeout: move to next level or stop
            current_level += 1

    scorecard = env_wrapper.get_scorecard()
    if scorecard:
        print(f"\nFinal Score: {scorecard.score}, Actions: {scorecard.total_actions}")

    return run_dir


def _reinforce_level_rules(graph: SchemaGraph, concepts: List[Concept], delta: float = 0.05):
    """Boost confidence of concepts used in a winning level."""
    for c in concepts:
        existing = graph.get_concept(c.id)
        if existing:
            new_conf = min(0.95, existing.confidence + delta)
            graph.update_concept(c.id, confidence=new_conf)


def _correct_level_rules(graph: SchemaGraph, concepts: List[Concept], delta: float = -0.05):
    """Reduce confidence of concepts used in a failed level."""
    for c in concepts:
        existing = graph.get_concept(c.id)
        if existing:
            new_conf = max(0.1, existing.confidence + delta)
            graph.update_concept(c.id, confidence=new_conf)


def main() -> None:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    dotenv_path = os.path.join(project_root, ".env")
    _load_dotenv(dotenv_path)

    p = argparse.ArgumentParser(description="Run ARC-AGI-3 with schema-based reasoning")
    p.add_argument("--game-id", required=True, help="ARC-AGI-3 game ID (e.g. sk48-d8078629)")
    p.add_argument("--base-url", default="https://openrouter.ai/api/v1", help="OpenAI-compatible base URL")
    p.add_argument("--api-key", default="", help="API key")
    p.add_argument("--model", default="moonshotai/kimi-k2.6", help="LLM model name")
    p.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5", help="Embedding model")
    p.add_argument("--max-steps", type=int, default=200, help="Max steps per level")
    p.add_argument("--schema-dir", default="schema_arc_agi3", help="Schema directory. By default schema is saved inside run_dir/schema for per-run isolation. Set explicitly to reuse a previous schema.")
    p.add_argument("--runs-dir", default="runs_arc_agi3")
    p.add_argument("--reset-schema", action="store_true", help="Clear schema before run (useful when reusing a schema-dir)")
    p.add_argument("--render", action="store_true", help="Enable terminal rendering")
    args = p.parse_args()

    api_key = (args.api_key or "").strip()
    if not api_key:
        for k in ["OPENROUTER_API_KEY", "OPENAI_API_KEY", "API_KEY"]:
            v = (os.environ.get(k) or "").strip()
            if v:
                api_key = v
                break

    if not api_key:
        raise SystemExit("Missing API key.")

    print(f"Loading embedding model: {args.embed_model} ...")
    embedder = SentenceTransformer(args.embed_model)

    agent = OpenAICompatibleAgent(
        base_url=args.base_url,
        api_key=api_key,
        model=args.model,
    )

    run_dir = run_arc_agi3(
        game_id=args.game_id,
        agent=agent,
        embedder=embedder,
        max_steps_per_level=args.max_steps,
        schema_dir=args.schema_dir,
        runs_dir=args.runs_dir,
        reset_schema=args.reset_schema,
        render_mode="terminal" if args.render else None,
    )
    print(f"Run completed: {run_dir}")


if __name__ == "__main__":
    main()
