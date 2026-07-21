#!/usr/bin/env python3
"""ARC-AGI-3 baseline runner — naive / icl / rag / withrule.

Modes
-----
naive      : grid image + actions + history only; no extra context.
icl        : loads configs/arc_agi3/icl/{game}.md and injects as examples.
rag        : online episode buffer; at each step retrieves top-k similar past
             (grid_description, action, outcome) records via embedding cosine sim.
withrule   : loads configs/arc_agi3/rules/{game}.md (human-written rules).
"""

from __future__ import annotations

import base64
import io
import json
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

from socialclaw.agent.base import AgentAttempt, ReasoningTrace, Usage
from socialclaw.agent.openai_compatible import OpenAICompatibleAgent
from socialclaw.dataset.arc_agi3 import ARCAGI3EnvWrapper
from socialclaw.dataset.base import EvalResult, Problem
from socialclaw.logging import write_episode, write_step, write_trajectory
from socialclaw.schema.arc_agi3_parser import (
    color_name,
    compute_grid_diff,
    extract_grid_objects,
)
from socialclaw.types import AttemptRecord, Episode
from socialclaw.utils import make_run_dir, save_cmd


# ---------------------------------------------------------------------------
# RAG buffer
# ---------------------------------------------------------------------------

class RAGBuffer:
    """Flat experience buffer with embedding-based retrieval.

    Stores (grid_description, action, outcome, grid_changed) tuples accumulated
    during the current game run.  At each step we embed the current grid
    description and return the top-k most similar past records.
    """

    def __init__(self, embedder):
        self._embedder = embedder
        self._records: List[Dict] = []
        self._embs: List[np.ndarray] = []

    def _encode(self, text: str) -> np.ndarray:
        emb = self._embedder.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return emb[0] if emb.ndim == 2 else emb

    def add(self, grid_desc: str, action: str, outcome: str, grid_changed: bool, step: int, level: int):
        self._embs.append(self._encode(grid_desc))
        self._records.append({
            "grid_desc": grid_desc,
            "action": action,
            "outcome": outcome,
            "grid_changed": grid_changed,
            "step": step,
            "level": level,
        })

    def retrieve(self, grid_desc: str, top_k: int = 5) -> List[Dict]:
        if not self._records:
            return []
        query = self._encode(grid_desc)
        matrix = np.stack(self._embs)
        sims = matrix @ query
        indices = np.argsort(sims)[::-1][:top_k]
        return [self._records[i] for i in indices if sims[i] > 0.1]


# ---------------------------------------------------------------------------
# Grid description helper (for RAG embedding / ICL text)
# ---------------------------------------------------------------------------

def _grid_desc(grid: np.ndarray) -> str:
    objects = extract_grid_objects(grid)
    if not objects:
        return f"empty {grid.shape[0]}x{grid.shape[1]} grid"
    colour_counts: Dict[str, int] = {}
    for o in objects:
        cname = color_name(o["color"])
        colour_counts[cname] = colour_counts.get(cname, 0) + 1
    parts = ", ".join(f"{c}({n})" for c, n in sorted(colour_counts.items()))
    return f"{grid.shape[0]}x{grid.shape[1]} grid: {len(objects)} objects [{parts}]"


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

_SYSTEM_BASE = (
    "You are an ARC-AGI-3 game-playing agent. You observe a grid world and choose actions to solve puzzles.\n\n"
    "Grid coordinate system:\n"
    "  - col (x): 0 = leftmost column, increases to the RIGHT\n"
    "  - row (y): 0 = topmost row, increases DOWNWARD\n"
    "  - All positions are written as (col, row) = (x, y)\n"
    "  - The image has sparse gridlines every 8 cells to help locate positions\n\n"
    "Action definitions:\n"
    "  - ACTION1 = move up (row - 1). No data needed.\n"
    "  - ACTION2 = move down (row + 1). No data needed.\n"
    "  - ACTION3 = move left (col - 1). No data needed.\n"
    "  - ACTION4 = move right (col + 1). No data needed.\n"
    "  - ACTION6 = click at (col, row). MUST provide data: {\"x\": col_int, \"y\": row_int}.\n"
    "  - ACTION7 = special action. No data needed.\n\n"
    "Rules:\n"
    "  1. Analyze the current grid image and choose the best action.\n"
    "  2. ONLY provide x/y for ACTION6. Never attach data to movement actions.\n"
    "  3. If an action keeps having NO effect, try a different one.\n"
    "  4. Be efficient — avoid unnecessary moves.\n\n"
    "Output format (strict JSON, nothing else):\n"
    "  {\"action\": \"ACTION_NAME\", \"data\": {\"x\": 10, \"y\": 20}, \"reasoning\": \"one sentence\"}\n"
    "  Omit \"data\" for non-click actions.\n"
    "CRITICAL: Output ONLY raw JSON. No markdown, no text before or after.\n"
)


def _history_text(history: List[Dict], available_actions: List[str]) -> str:
    if not history:
        return ""
    action_total: Dict[str, int] = {}
    action_no_effect: Dict[str, int] = {}
    for h in history:
        act = h["action"]
        action_total[act] = action_total.get(act, 0) + 1
        if not h.get("grid_changed", True):
            action_no_effect[act] = action_no_effect.get(act, 0) + 1

    txt = "\nAction summary so far:\n"
    for act in sorted(action_total):
        total = action_total[act]
        no_eff = action_no_effect.get(act, 0)
        txt += f"  {act}: tried {total}x, no-effect {no_eff}x\n"

    only_failing = [a for a in action_total if action_no_effect.get(a, 0) == action_total[a]]
    never_tried = [a for a in available_actions if a not in action_total]
    if only_failing:
        txt += f"  NOTE: {', '.join(only_failing)} had NO effect so far.\n"
    if never_tried:
        txt += f"  NOTE: Not yet tried: {', '.join(never_tried)}.\n"

    txt += "\nRecent steps (last 5):\n"
    for h in history[-5:]:
        act = h["action"]
        data = h.get("action_data", {})
        changed = h.get("grid_changed", "?")
        if act == "ACTION6" and data:
            txt += f"  Step {h['step']}: {act} at (col={data.get('x','?')},row={data.get('y','?')}) -> state={h['state']}, changed={changed}\n"
        else:
            txt += f"  Step {h['step']}: {act} -> state={h['state']}, changed={changed}\n"
    return txt


def build_naive_prompt(
    game_id: str,
    level: int,
    step: int,
    grid_text: str,
    available_actions: List[str],
    history: List[Dict],
) -> str:
    user = (
        f"Game: {game_id} | Level: {level} | Step: {step}\n"
        f"Grid (see attached image): {grid_text}\n\n"
        f"Available actions: {', '.join(available_actions)}\n"
        f"{_history_text(history, available_actions)}\n"
        "Choose your next action:"
    )
    return _SYSTEM_BASE + "\n---\n" + user


def build_withrule_prompt(
    game_id: str,
    level: int,
    step: int,
    grid_text: str,
    available_actions: List[str],
    history: List[Dict],
    rules: str,
) -> str:
    system = _SYSTEM_BASE + "\n=== KNOWN GAME RULES ===\n" + rules + "\n========================\n"
    user = (
        f"Game: {game_id} | Level: {level} | Step: {step}\n"
        f"Grid (see attached image): {grid_text}\n\n"
        f"Available actions: {', '.join(available_actions)}\n"
        f"{_history_text(history, available_actions)}\n"
        "Choose your next action:"
    )
    return system + "\n---\n" + user


def build_icl_prompt(
    game_id: str,
    level: int,
    step: int,
    grid_text: str,
    available_actions: List[str],
    history: List[Dict],
    examples: str,
) -> str:
    system = _SYSTEM_BASE + "\n=== ICL EXAMPLES ===\n" + examples + "\n====================\n"
    user = (
        f"Game: {game_id} | Level: {level} | Step: {step}\n"
        f"Grid (see attached image): {grid_text}\n\n"
        f"Available actions: {', '.join(available_actions)}\n"
        f"{_history_text(history, available_actions)}\n"
        "Choose your next action:"
    )
    return system + "\n---\n" + user


def build_rag_prompt(
    game_id: str,
    level: int,
    step: int,
    grid_text: str,
    available_actions: List[str],
    history: List[Dict],
    retrieved: List[Dict],
) -> str:
    ctx = ""
    if retrieved:
        lines = ["=== SIMILAR PAST EXPERIENCES (retrieved from this run) ==="]
        for i, r in enumerate(retrieved, 1):
            changed_str = "grid changed" if r["grid_changed"] else "no grid change"
            lines.append(
                f"  [{i}] Situation: {r['grid_desc']}\n"
                f"       Action taken: {r['action']} → outcome: {r['outcome']} ({changed_str})"
            )
        lines.append("==========================================================")
        ctx = "\n" + "\n".join(lines) + "\n"

    system = _SYSTEM_BASE + ctx
    user = (
        f"Game: {game_id} | Level: {level} | Step: {step}\n"
        f"Grid (see attached image): {grid_text}\n\n"
        f"Available actions: {', '.join(available_actions)}\n"
        f"{_history_text(history, available_actions)}\n"
        "Choose your next action:"
    )
    return system + "\n---\n" + user


# ---------------------------------------------------------------------------
# Action parser (shared with withrule)
# ---------------------------------------------------------------------------

def _parse_action(answer_text: str, available_actions: List) -> Tuple:
    action_name = None
    data: Dict = {}
    reasoning = ""

    text = answer_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(text[start:end + 1])
            action_name = parsed.get("action", "")
            data = parsed.get("data") or {}
            reasoning = parsed.get("reasoning", "")
        except Exception:
            pass

    if not action_name:
        for act in available_actions:
            if act.name in answer_text.upper():
                action_name = act.name
                break

    chosen = None
    if action_name:
        for act in available_actions:
            if act.name == action_name.upper():
                chosen = act
                break

    if chosen is None and available_actions:
        chosen = available_actions[0]

    if chosen and chosen.name != "ACTION6":
        data = {}

    return chosen, data or {}, reasoning


# ---------------------------------------------------------------------------
# Context file loader
# ---------------------------------------------------------------------------

def _load_context_file(game_id: str, suffix: str) -> str:
    """Load canonical ICL or human-rule context for an ARC game."""
    game_title = game_id.split("-")[0].lower()
    kind = "icl" if suffix == "fewshot" else "rules"
    path = os.path.join(_PROJECT_ROOT, "configs", "arc_agi3", kind, f"{game_title}.md")
    if not os.path.exists(path):
        print(f"[WARN] Context file not found: {path}. Falling back to naive context.")
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


# ---------------------------------------------------------------------------
# Main run loop
# ---------------------------------------------------------------------------

def run_baseline(
    *,
    mode: str,
    game_id: str,
    agent: OpenAICompatibleAgent,
    embedder=None,
    max_steps_per_level: int = 200,
    runs_dir: str = "runs",
    render_mode: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    assert mode in ("naive", "icl", "rag", "withrule"), f"Unknown mode: {mode}"

    # Load fixed context (ICL / withrule).
    static_ctx = ""
    if mode == "icl":
        static_ctx = _load_context_file(game_id, "fewshot")
    elif mode == "withrule":
        static_ctx = _load_context_file(game_id, "rules")

    # RAG buffer (only used in rag mode)
    rag_buffer: Optional[RAGBuffer] = None
    if mode == "rag":
        assert embedder is not None, "RAG mode requires an embedder"
        rag_buffer = RAGBuffer(embedder)

    model_str = model or "unknown_model"
    run_dir = make_run_dir(runs_dir, f"arc_agi3/{mode}", model_str)
    save_cmd(run_dir)
    print(f"[RUN] mode={mode} game={game_id} output={run_dir}")

    env_wrapper = ARCAGI3EnvWrapper(game_id, render_mode=render_mode)
    obs = env_wrapper.reset()
    win_levels = obs.win_levels if obs else 1
    print(f"[ARC] win_levels={win_levels}")

    current_level = 0
    prev_levels_completed = obs.levels_completed if obs else 0
    print(f"\n=== Level {current_level + 1}/{win_levels} ===")

    level_id = f"{game_id}_L{current_level + 1}"
    level_problem = Problem(
        id=level_id, prompt="", problem_type="arc_grid",
        meta={"game_id": game_id, "level": current_level + 1, "win_levels": win_levels},
    )
    step = 0
    history: List[Dict[str, Any]] = []
    trajectory: List[Dict[str, Any]] = []
    attempts: List[AttemptRecord] = []
    evals: List[EvalResult] = []
    stop_reason: Optional[str] = None

    def _save_level():
        episode = Episode(
            problem=level_problem,
            attempts=attempts,
            evals=evals,
            reasoning_trace=None,
            reasoning_confidence=0.0,
            flags=[f"baseline_{mode}"],
            stop_reason=stop_reason,
            model=model,
        )
        ep_path = write_episode(run_dir, episode)
        traj_path = write_trajectory(run_dir, level_id, trajectory)
        print(f"  Episode: {ep_path}")
        print(f"  Trajectory: {traj_path}")

    while True:
        if obs is None or obs.is_empty():
            print(f"[ARC] Empty observation at step {step}, stopping.")
            _save_level()
            break

        # Detect level transition
        cur_lc = obs.levels_completed if obs else 0
        if cur_lc > prev_levels_completed:
            print(f"  Level {current_level + 1} completed! (levels_completed={cur_lc})")
            evals.append(EvalResult(correct=True, pred="", gold="WIN", details="LEVEL_COMPLETE"))
            _save_level()
            if obs.state == env_wrapper.GameState.WIN:
                print("  All levels WON!")
                break
            prev_levels_completed = cur_lc
            current_level = cur_lc
            level_id = f"{game_id}_L{current_level + 1}"
            level_problem = Problem(
                id=level_id, prompt="", problem_type="arc_grid",
                meta={"game_id": game_id, "level": current_level + 1, "win_levels": win_levels},
            )
            step = 0
            history = []
            trajectory = []
            attempts = []
            evals = []
            stop_reason = None
            print(f"\n=== Level {current_level + 1}/{win_levels} ===")

        # Terminal states
        if obs.state == env_wrapper.GameState.WIN:
            print("  Game WON!")
            evals.append(EvalResult(correct=True, pred="", gold="WIN", details=""))
            _save_level()
            break
        if obs.state == env_wrapper.GameState.GAME_OVER:
            print(f"  GAME_OVER at level {current_level + 1}, step {step}")
            evals.append(EvalResult(correct=False, pred="", gold="WIN", details="GAME_OVER"))
            _save_level()
            break
        if step >= max_steps_per_level:
            print(f"  Level {current_level + 1} TIMEOUT after {step} steps")
            evals.append(EvalResult(correct=False, pred="", gold="WIN", details="TIMEOUT"))
            stop_reason = "max_iters"
            _save_level()
            break

        grid = obs.frame[-1] if obs.frame else None
        if grid is None:
            print(f"[ARC] Grid is None at step {step}, stopping.")
            _save_level()
            break

        available_actions = env_wrapper.get_available_actions(obs)
        h, w = grid.shape
        grid_summary = f"{h}x{w} full grid — see attached image"
        gd = _grid_desc(grid)
        print(f"[ARC] Step {step}: grid={h}x{w}, actions={[a.name for a in available_actions]}")

        # Render grid image
        grid_img = ARCAGI3EnvWrapper.grid_to_image(grid, cell_size=8)
        img_dir = os.path.join(run_dir, level_id)
        os.makedirs(img_dir, exist_ok=True)
        grid_img.save(os.path.join(img_dir, f"step_{step:03d}.png"))

        buf = io.BytesIO()
        grid_img.save(buf, format="PNG")
        image_data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

        # Build mode-specific prompt
        action_names = [a.name for a in available_actions]
        if mode == "naive":
            prompt = build_naive_prompt(game_id, current_level + 1, step, grid_summary, action_names, history)
        elif mode == "withrule":
            prompt = build_withrule_prompt(game_id, current_level + 1, step, grid_summary, action_names, history, static_ctx)
        elif mode == "icl":
            prompt = build_icl_prompt(game_id, current_level + 1, step, grid_summary, action_names, history, static_ctx)
        else:  # rag
            retrieved = rag_buffer.retrieve(gd, top_k=5)
            prompt = build_rag_prompt(game_id, current_level + 1, step, grid_summary, action_names, history, retrieved)

        if step == 0:
            level_problem.prompt = prompt

        # Call LLM
        print(f"[ARC] Step {step}: calling LLM ...")
        try:
            attempt = agent.answer(
                prompt=prompt,
                meta={"game_id": game_id, "level": current_level, "step": step, "image_data_url": image_data_url},
                response_format="json_object",
            )
        except Exception as e:
            print(f"[ARC] LLM error at step {step}: {e}")
            attempt = AgentAttempt(
                answer_text='{"action": "ACTION1", "reasoning": "API error fallback"}',
                reasoning_trace=ReasoningTrace(concepts=[], relations=[], explanation=str(e)),
                usage=Usage(input_tokens=0, output_tokens=0, total_tokens=0),
                raw={"error": str(e)},
            )

        chosen_action, action_data, reasoning = _parse_action(attempt.answer_text, available_actions)

        # Execute
        pre_grid = grid.copy()
        obs = env_wrapper.step(chosen_action, data=action_data) if chosen_action else env_wrapper.step(available_actions[0])

        step += 1
        state_str = str(obs.state) if obs else "UNKNOWN"
        post_grid = obs.frame[-1] if obs and obs.frame else None
        grid_changed, _ = compute_grid_diff(pre_grid, post_grid)

        print(f"  Step {step}: action={chosen_action.name if chosen_action else 'NONE'}, state={state_str}, changed={grid_changed}")
        if reasoning:
            print(f"  Reasoning: {reasoning[:120]}")

        # Update RAG buffer after action
        if mode == "rag":
            rag_buffer.add(gd, chosen_action.name if chosen_action else "NONE", state_str, grid_changed, step, current_level)

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
        })
        empty_trace = ReasoningTrace(concepts=[], relations=[], explanation="")
        attempts.append(AttemptRecord(
            input_prompt=prompt,
            answer_text=attempt.answer_text,
            reasoning_trace=empty_trace,
            usage=asdict(attempt.usage) if attempt.usage else {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        ))
        write_step(run_dir, level_id, step, {
            "prompt": prompt,
            "answer_text": attempt.answer_text,
            "usage": asdict(attempt.usage) if attempt.usage else {},
            "action": chosen_action.name if chosen_action else "NONE",
            "action_data": action_data,
            "reasoning": reasoning,
            "state": state_str,
            "grid_image": f"step_{step:03d}.png",
            "grid_changed": grid_changed,
            "mode": mode,
        })

    scorecard = env_wrapper.get_scorecard()
    if scorecard:
        print(f"\nFinal Score: {scorecard.score}, Total Actions: {scorecard.total_actions}")
    return run_dir
