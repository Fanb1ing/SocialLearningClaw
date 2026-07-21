#!/usr/bin/env python3
"""
ARC-AGI-3 × Memory Agent Baselines (Reflexion + ExpeL + A-MEM + TGM).

Baselines use online memory inside the same ARC-AGI-3 attempt.
Every N steps, memory is updated from the partial trajectory and immediately
injected into subsequent prompts. Failed games are not retried.

Reflexion:
  Every memory interval: LLM reflects on recent behavior → stored as verbal memory.
  Future steps receive all past reflections in the system prompt.

ExpeL:
  Every memory interval: recent trajectory is added to the experience pool.
  Periodically, LLM extracts generalizable game rules.
  Future steps receive extracted rules + recent successful examples.

A-MEM:
  Every memory interval: structured note is created, linked to related notes,
  and related notes may evolve. Future steps retrieve relevant notes.

TGM:
  Every memory interval: trajectory is abstracted into query/path/meta-cognition
  graph nodes. Reward-weighted meta-cognition strategies are retrieved for
  future steps. This reproduces the graph-memory architecture without training.

Run structure mirrors the original arc_runner.py:
  runs/arc_memory_{baseline}/{model}/{game}_{timestamp}/
    {game}_L{level}/
      step_000.png
      step_001.json
      step_001.png
      ...
      episode.json
      trajectory.json
    memory.json
    summary.json

Usage:
    cd /data5/fanbingbing/SocialLearningClaw
    .venv/bin/python scripts/run_arc_memory_baseline.py \\
        --game-id cd82-fb555c5d --baseline reflexion
    .venv/bin/python scripts/run_arc_memory_baseline.py \\
        --all-games --baseline both
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── env setup BEFORE arc_agi imports ─────────────────────────────────────────
_ARC_NO_PROXY = "three.arcprize.org,arcprize.org"
existing = os.environ.get("no_proxy") or os.environ.get("NO_PROXY") or ""
os.environ["no_proxy"] = f"{_ARC_NO_PROXY},{existing}" if existing else _ARC_NO_PROXY
os.environ["NO_PROXY"] = os.environ["no_proxy"]

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")
if os.environ.get("ARC_AGI_API_KEY") and not os.environ.get("ARC_API_KEY"):
    os.environ["ARC_API_KEY"] = os.environ["ARC_AGI_API_KEY"]

from socialclaw.agent.base import ReasoningTrace, Usage
from socialclaw.agent.openai_compatible import OpenAICompatibleAgent
from socialclaw.dataset.arc_agi3 import ARCAGI3EnvWrapper
from socialclaw.dataset.base import EvalResult, Problem
from socialclaw.logging import write_episode, write_step, write_trajectory
from socialclaw.memory_agents import AMemory, ExPeLMemory, ReflexionMemory, TrainableGraphMemory
from socialclaw.types import AttemptRecord, Episode
from socialclaw.utils import make_run_dir, save_cmd

DEFAULT_MODEL_NAME = "claude-opus-4.8"
DEFAULT_MODEL_ID   = "anthropic/claude-opus-4.8"
MAX_STEPS = 200
DEFAULT_MEMORY_UPDATE_INTERVAL = 10
DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"


# ── data types ────────────────────────────────────────────────────────────────

@dataclass
class LevelResult:
    level: int
    outcome: str        # WIN | GAME_OVER | TIMEOUT
    n_steps: int
    action_counts: Dict[str, int] = field(default_factory=dict)
    n_reflections_used: int = 0
    n_insights_used: int = 0
    n_notes_used: int = 0
    n_meta_used: int = 0


# ── ARC system prompt (no schema, with memory injection) ──────────────────────

ARC_BASE_SYSTEM = """\
You are an ARC-AGI-3 game-playing agent. You observe a grid world and choose actions to solve puzzles.

Grid coordinate system:
  - col (x): 0 = leftmost column, increases to the RIGHT
  - row (y): 0 = topmost row, increases DOWNWARD
  - All positions are written as (col, row) = (x, y)
  - The image shows sparse gridlines every 8 cells to help locate positions

Action definitions:
- ACTION1 = move up (row decreases by 1). No data needed.
- ACTION2 = move down (row increases by 1). No data needed.
- ACTION3 = move left (col decreases by 1). No data needed.
- ACTION4 = move right (col increases by 1). No data needed.
- ACTION5 = game-specific special action. No data needed unless the environment explicitly says otherwise.
- ACTION6 = click at specific (col, row). MUST provide: {"x": col_int, "y": row_int}.
- ACTION7 = special action. No data needed.

Rules:
1. Carefully examine the grid image to understand the current state.
2. Use your memory from past levels to form hypotheses about this level's rules early.
3. If an action has been tried many times with no grid change, try a DIFFERENT action.
4. Only provide x/y for ACTION6. Never attach data to move actions.

Output format (strict JSON, no markdown):
{
  "action": "ACTION_NAME",
  "data": {"x": 10, "y": 20},
  "reasoning": {
    "concepts_used": ["object or concept name"],
    "reasoning_path": ["A -> relation -> B"],
    "explanation": "Why I chose this action."
  }
}
Omit "data" or set to {} if action needs no coordinates.
Output ONLY raw JSON.\
"""


def _build_prompt(
    game_id: str,
    level: int,
    step: int,
    grid_summary: str,
    available_actions: List[str],
    history: List[Dict],
    memory_block: str,
) -> str:
    history_text = ""
    if history:
        action_total: Dict[str, int] = {}
        action_no_eff: Dict[str, int] = {}
        for h in history:
            act = h["action"]
            action_total[act] = action_total.get(act, 0) + 1
            if not h.get("grid_changed", True):
                action_no_eff[act] = action_no_eff.get(act, 0) + 1

        history_text = "\nAction summary so far:\n"
        for act in sorted(action_total):
            total = action_total[act]
            no_eff = action_no_eff.get(act, 0)
            history_text += f"  {act}: tried {total}x, no-effect {no_eff}x\n"

        only_failing = [a for a in action_total if action_no_eff.get(a, 0) == action_total[a]]
        never_tried  = [a for a in available_actions if a not in action_total]
        if only_failing:
            history_text += f"  NOTE: {', '.join(only_failing)} had NO effect in any attempt.\n"
        if never_tried:
            history_text += f"  NOTE: Not yet tried: {', '.join(never_tried)}.\n"

        history_text += "\nRecent steps (last 5):\n"
        for h in history[-5:]:
            act = h["action"]
            data = h.get("action_data", {})
            changed = h.get("grid_changed", "?")
            if act == "ACTION6" and data:
                history_text += (
                    f"  step {h['step']:03d}: {act}({data.get('x','?')},{data.get('y','?')})"
                    f" grid_changed={changed} state={h.get('state','?')}\n"
                )
            else:
                history_text += (
                    f"  step {h['step']:03d}: {act}"
                    f" grid_changed={changed} state={h.get('state','?')}\n"
                )

    system = ARC_BASE_SYSTEM
    if memory_block:
        system = ARC_BASE_SYSTEM + "\n\n" + memory_block

    user = (
        f"Game: {game_id} | Level: {level} | Step: {step}\n"
        f"Grid: {grid_summary}\n\n"
        f"Available actions: {', '.join(available_actions)}\n"
        f"{history_text}\n"
        "Choose your next action:"
    )
    return system, user


# ── action parsing ────────────────────────────────────────────────────────────

def _parse_action(text: str, available_actions):
    from arcengine import GameAction
    text = text.strip()
    text = re.sub(r"^```[a-z]*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n?```$", "", text)
    text = text.strip()

    action_data: Dict[str, Any] = {}
    reasoning = ReasoningTrace(concepts=[], relations=[], explanation="")
    chosen: Optional[GameAction] = None

    try:
        obj = json.loads(text)
        action_name = str(obj.get("action", "")).upper()
        action_data = obj.get("data", {}) or {}
        raw_reasoning = obj.get("reasoning", {}) or {}
        if isinstance(raw_reasoning, str):
            reasoning = ReasoningTrace(concepts=[], relations=[], explanation=raw_reasoning)
        elif isinstance(raw_reasoning, dict):
            reasoning = ReasoningTrace(
                concepts=raw_reasoning.get("concepts_used", []),
                relations=raw_reasoning.get("reasoning_path", []),
                explanation=raw_reasoning.get("explanation", ""),
            )
        for act in available_actions:
            if act.name.upper() == action_name:
                chosen = act
                break
    except Exception:
        pass

    if chosen is None and available_actions:
        chosen = available_actions[0]
    return chosen, action_data, reasoning


def _memory_item_count(memory: ReflexionMemory | ExPeLMemory | AMemory | TrainableGraphMemory) -> int:
    if isinstance(memory, ReflexionMemory):
        return len(memory.reflections)
    if isinstance(memory, ExPeLMemory):
        return len(memory.insights)
    if isinstance(memory, TrainableGraphMemory):
        return len(memory.meta)
    return len(memory.notes)


def _get_memory_block(
    memory: ReflexionMemory | ExPeLMemory | AMemory | TrainableGraphMemory,
    *,
    query: str,
) -> str:
    if isinstance(memory, (AMemory, TrainableGraphMemory)):
        return memory.get_memory_block(query)
    return memory.get_memory_block()


def _update_memory_online(
    *,
    memory: ReflexionMemory | ExPeLMemory | AMemory | TrainableGraphMemory,
    baseline: str,
    game_id: str,
    level: int,
    win_levels: int,
    outcome: str,
    action_counts: Dict[str, int],
    trajectory: List[Dict],
    step_count: int,
    final: bool,
) -> None:
    action_summary = ", ".join(f"{k}×{v}" for k, v in sorted(action_counts.items())) or "no actions"
    recent = trajectory[-10:]
    trajectory_summary = "\n".join(
        f"step {t.get('step')}: {t.get('action')} data={t.get('action_data')} "
        f"changed={t.get('grid_changed')} state={t.get('state', '?')}"
        for t in recent
    )
    phase = "final" if final else "partial"

    if baseline == "reflexion":
        task_ctx = (
            f"ARC-AGI-3 game '{game_id}', Level {level}/{win_levels}, {phase} trajectory.\n"
            f"Steps so far: {step_count}\n"
            f"Actions taken: {action_summary}\n"
            f"Recent trajectory:\n{trajectory_summary}"
        )
        failure_info = (
            f"Current outcome/status: {outcome}. "
            "Reflect on what the agent has learned so far and what strategy to use in the next steps. "
            "Do not assume the level has restarted; this memory will be used immediately in the same ongoing attempt."
        )
        reflection = memory.reflect(task_ctx, failure_info)
        if reflection:
            print(f"  [Reflexion online] {reflection[:120]}...")
        return

    if baseline == "expel":
        task = f"ARC-AGI-3 '{game_id}' Level {level} {phase} trajectory: {action_summary}"
        lesson = (
            f"Status={outcome} after {step_count} steps. Actions used: {action_summary}. "
            "Use this experience in the same ongoing attempt; do not restart the level."
        )
        memory.add_experience(task=task, outcome=(outcome == "WIN"), trajectory=trajectory_summary, lesson=lesson)
        return

    lesson = (
        f"Status={outcome} after {step_count} steps on level {level}. "
        f"Actions used: {action_summary}. "
        "This is online memory for the same ongoing ARC attempt; adapt next actions without resetting."
    )
    if baseline == "tgm":
        memory.add_experience(
            task=f"ARC-AGI-3 '{game_id}' Level {level} {phase} trajectory",
            outcome=(outcome == "WIN"),
            trajectory=trajectory_summary,
            lesson=(
                f"{lesson} Reward signal is {'positive' if outcome == 'WIN' else 'negative/partial'}; "
                "prefer strategies that cause grid changes, avoid repeated no-effect actions, and infer game-specific rules."
            ),
            context=(
                f"game_id={game_id}; level={level}; outcome={outcome}; win_levels={win_levels}; "
                f"step_count={step_count}; action_counts={action_summary}"
            ),
            domain="arc_agi3",
        )
        return

    memory.add_experience(
        task=f"ARC-AGI-3 '{game_id}' Level {level} {phase} trajectory",
        outcome=(outcome == "WIN"),
        trajectory=trajectory_summary,
        lesson=lesson,
        context=f"outcome={outcome}; win_levels={win_levels}; step_count={step_count}",
    )


# ── level runner ──────────────────────────────────────────────────────────────

def run_level(
    *,
    env_wrapper: ARCAGI3EnvWrapper,
    agent: OpenAICompatibleAgent,
    game_id: str,
    level: int,
    level_id: str,
    run_dir: str,
    memory: ReflexionMemory | ExPeLMemory | AMemory | TrainableGraphMemory,
    baseline: str,
    win_levels: int,
    max_steps: int = MAX_STEPS,
    memory_update_interval: int = DEFAULT_MEMORY_UPDATE_INTERVAL,
) -> tuple[str, LevelResult, List[Dict], List[AttemptRecord]]:
    """
    Run one ARC-AGI-3 level. Returns (outcome, level_result, trajectory, attempts).
    Mirrors the original arc_runner step loop but replaces schema with memory_block.
    """
    from arcengine import GameAction, GameState

    obs = env_wrapper.reset()
    history: List[Dict] = []
    trajectory: List[Dict] = []
    attempts: List[AttemptRecord] = []
    action_counts: Dict[str, int] = {}
    evals: List[EvalResult] = []
    stop_reason: Optional[str] = None
    level_dir = os.path.join(run_dir, level_id)
    os.makedirs(level_dir, exist_ok=True)

    consecutive_api_failures = 0
    MAX_CONSECUTIVE_API_FAILURES = 5
    memory_block = _get_memory_block(
        memory,
        query=f"ARC-AGI-3 game={game_id} level={level} step=0",
    )
    if memory_block:
        print(f"  [Memory] {_memory_item_count(memory)} items injected at level start")

    for step in range(max_steps):
        if obs is None or obs.is_empty():
            print(f"[ARC] Observation empty at step {step}, breaking.")
            stop_reason = "empty_obs"
            break

        # ARC observations may contain animation/transition frames. Use the
        # final frame so the LLM sees the fully settled state.
        grid = obs.frame[-1] if obs.frame else None
        if grid is None:
            print(f"[ARC] Grid is None at step {step}, breaking.")
            stop_reason = "empty_obs"
            break

        available = env_wrapper.get_available_actions(obs)
        available_names = [a.name for a in available]
        h, w = grid.shape
        grid_summary = f"{h}x{w} full grid — see attached image"

        print(f"[ARC] Step {step:03d}: available={available_names}")

        # Render current state PNG (mirrors original: step_{step:03d}.png = pre-action state)
        grid_img = ARCAGI3EnvWrapper.grid_to_image(grid, cell_size=8)
        img_path = os.path.join(level_dir, f"step_{step:03d}.png")
        grid_img.save(img_path)  # always save, not just step 0

        img_buffer = io.BytesIO()
        grid_img.save(img_buffer, format="PNG")
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode("ascii")
        image_data_url = f"data:image/png;base64,{img_base64}"

        # Build prompt
        system, user = _build_prompt(
            game_id=game_id,
            level=level,
            step=step,
            grid_summary=grid_summary,
            available_actions=available_names,
            history=history,
            memory_block=memory_block,
        )
        full_prompt = system + "\n---\n" + user

        # Call LLM (with image)
        try:
            attempt = agent.answer(
                prompt=full_prompt,
                meta={"image_data_url": image_data_url},
                response_format="json_object",
            )
        except Exception as e:
            print(f"[ARC] Step {step}: LLM FAILED: {e}")
            consecutive_api_failures += 1
            if consecutive_api_failures >= MAX_CONSECUTIVE_API_FAILURES:
                print(f"[ARC] {consecutive_api_failures} consecutive API failures — aborting run.")
                stop_reason = "api_error"
                break
            time.sleep(2)
            continue

        consecutive_api_failures = 0  # reset on success
        # Parse action
        chosen_action, action_data, reasoning_trace = _parse_action(attempt.answer_text, available)
        attempt.reasoning_trace = reasoning_trace

        attempt_rec = AttemptRecord(
            input_prompt=full_prompt,
            answer_text=attempt.answer_text,
            reasoning_trace=reasoning_trace,
            usage={
                "input_tokens": attempt.usage.input_tokens,
                "output_tokens": attempt.usage.output_tokens,
                "total_tokens": attempt.usage.total_tokens,
            },
        )
        attempts.append(attempt_rec)

        # Execute action
        if chosen_action is None:
            print(f"[ARC] Step {step}: no action parsed, skipping.")
            time.sleep(1)
            continue

        act_name = chosen_action.name
        action_counts[act_name] = action_counts.get(act_name, 0) + 1

        pre_grid_text = ARCAGI3EnvWrapper.grid_to_text(grid)
        obs_next = env_wrapper.step(chosen_action, data=action_data)

        post_grid = obs_next.frame[-1] if (obs_next and obs_next.frame) else None
        post_grid_text = ARCAGI3EnvWrapper.grid_to_text(post_grid) if post_grid is not None else ""
        grid_changed = (post_grid_text != pre_grid_text) and bool(post_grid_text)

        state_str = str(obs_next.state) if obs_next else "?"
        print(
            f"  => {act_name}"
            + (f"({action_data.get('x','?')},{action_data.get('y','?')})" if act_name == "ACTION6" and action_data else "")
            + f"  changed={grid_changed}  state={state_str}"
        )

        # Save step JSON — same index as the PNG (step_NNN.json ↔ step_NNN.png)
        step_data = {
            "prompt": full_prompt,
            "answer_text": attempt.answer_text,
            "reasoning_trace": asdict(reasoning_trace),
            "usage": attempt_rec.usage,
            "action": act_name,
            "action_data": action_data,
            "state": state_str,
            "grid_image": f"step_{step:03d}.png",
            "grid_changed": grid_changed,
        }
        write_step(run_dir, level_id, step, step_data)

        history.append({
            "step": step,
            "action": act_name,
            "action_data": action_data,
            "grid_changed": grid_changed,
            "state": state_str,
        })
        trajectory.append({
            "step": step,
            "action": act_name,
            "action_data": action_data,
            "grid_changed": grid_changed,
            "state": state_str,
        })

        obs = obs_next
        step_count = len(history)

        # Check terminal states
        if obs and obs.state == GameState.WIN:
            print(f"  Level {level} WON in {step + 1} steps!")
            evals.append(EvalResult(correct=True, pred=act_name, gold="WIN", details=""))
            stop_reason = "win"
            break

        if obs and obs.state == GameState.GAME_OVER:
            print(f"  Level {level} GAME_OVER at step {step}")
            evals.append(EvalResult(correct=False, pred=act_name, gold="WIN", details="GAME_OVER"))
            stop_reason = "game_over"
            break

        if memory_update_interval > 0 and step_count % memory_update_interval == 0:
            print(f"  [Memory] online update after {step_count} steps")
            _update_memory_online(
                memory=memory,
                baseline=baseline,
                game_id=game_id,
                level=level,
                win_levels=win_levels,
                outcome="IN_PROGRESS",
                action_counts=action_counts,
                trajectory=trajectory,
                step_count=step_count,
                final=False,
            )
            memory.save(os.path.join(run_dir, "memory.json"))
            memory_block = _get_memory_block(
                memory,
                query=f"ARC-AGI-3 game={game_id} level={level} step={step_count}",
            )
            if memory_block:
                print(f"  [Memory] refreshed for next step ({_memory_item_count(memory)} items)")

        time.sleep(0.3)
    else:
        evals.append(EvalResult(correct=False, pred="", gold="WIN", details="TIMEOUT"))
        stop_reason = "max_iters"

    n_steps = len(history)
    outcome = (
        "WIN" if stop_reason == "win"
        else "GAME_OVER" if stop_reason == "game_over"
        else "API_ERROR" if stop_reason == "api_error"
        else "TIMEOUT"
    )

    level_result = LevelResult(
        level=level,
        outcome=outcome,
        n_steps=n_steps,
        action_counts=action_counts,
    )

    return outcome, level_result, trajectory, attempts, evals


# ── game runner ───────────────────────────────────────────────────────────────

def run_game(
    game_id: str,
    baseline: str,
    agent: OpenAICompatibleAgent,
    model_name: str,
    runs_dir: str,
    embedder=None,
    max_steps: int = MAX_STEPS,
    memory_update_interval: int = DEFAULT_MEMORY_UPDATE_INTERVAL,
) -> str:
    print(f"\n{'='*65}")
    print(f"Game: {game_id}  |  Baseline: {baseline}  |  Model: {model_name}")
    print(f"{'='*65}")

    env_wrapper = ARCAGI3EnvWrapper(game_id)
    obs = env_wrapper.reset()
    win_levels = obs.win_levels if obs else 1
    print(f"[ARC] win_levels={win_levels}")

    # Memory
    if baseline == "reflexion":
        memory: ReflexionMemory | ExPeLMemory | AMemory | TrainableGraphMemory = ReflexionMemory(
            client=None,          # use agent directly
            model_id=agent.model,
            max_reflections=20,
        )
        # Monkey-patch: use agent for reflection calls
        _openai_client = _make_openai_client(agent)
        memory.client = _openai_client
    elif baseline == "expel":
        memory = ExPeLMemory(
            client=_make_openai_client(agent),
            model_id=agent.model,
            insight_interval=2,
            max_insights=10,
        )
    elif baseline == "amem":
        if embedder is None:
            raise ValueError("A-MEM baseline requires an embedder")
        memory = AMemory(
            client=_make_openai_client(agent),
            model_id=agent.model,
            embedder=embedder,
            max_notes=120,
            retrieve_k=5,
        )
    else:
        if embedder is None:
            raise ValueError("TGM baseline requires an embedder")
        memory = TrainableGraphMemory(
            client=_make_openai_client(agent),
            model_id=agent.model,
            embedder=embedder,
            max_meta=30,
            retrieve_k=3,
        )

    # Run directory (mirrors original: runs/arc_memory_{baseline}/{model}/{timestamp}/)
    run_dir = make_run_dir(runs_dir, f"arc_memory_{baseline}", model_name)
    save_cmd(run_dir)

    level_results: List[LevelResult] = []
    all_evals: List[EvalResult] = []

    # Mirror original arc_runner's level progression shape: only advance on WIN.
    # Failed levels stop the run. Memory evolves online inside the same attempt.
    current_level = 1
    while current_level <= win_levels:
        level = current_level
        print(f"\n=== Level {level}/{win_levels} ===")

        level_id = f"{game_id}_L{level}"
        outcome, level_result, trajectory, attempts, evals = run_level(
            env_wrapper=env_wrapper,
            agent=agent,
            game_id=game_id,
            level=level,
            level_id=level_id,
            run_dir=run_dir,
            memory=memory,
            baseline=baseline,
            win_levels=win_levels,
            max_steps=max_steps,
            memory_update_interval=memory_update_interval,
        )
        level_result.n_reflections_used = len(memory.reflections) if isinstance(memory, ReflexionMemory) else 0
        level_result.n_insights_used = len(memory.insights) if isinstance(memory, ExPeLMemory) else 0
        level_result.n_notes_used = len(memory.notes) if isinstance(memory, AMemory) else 0
        level_result.n_meta_used = len(memory.meta) if isinstance(memory, TrainableGraphMemory) else 0

        level_results.append(level_result)
        all_evals.extend(evals)

        # Save level episode + trajectory (mirrors original)
        problem = Problem(
            id=level_id,
            prompt=f"ARC-AGI-3 {game_id} Level {level}",
            problem_type="arc_grid",
            meta={"game_id": game_id, "level": level, "win_levels": win_levels},
        )
        episode = Episode(
            problem=problem,
            attempts=attempts,
            evals=evals,
            stop_reason=outcome.lower(),
            model=model_name,
        )
        ep_path = write_episode(run_dir, episode)
        traj_path = write_trajectory(run_dir, level_id, trajectory)
        print(f"  Episode: {ep_path}")
        print(f"  Trajectory: {traj_path} ({len(trajectory)} steps)")

        # Final memory update for this level attempt. No retry/restart follows.
        _update_memory_online(
            memory=memory,
            baseline=baseline,
            game_id=game_id,
            level=level,
            win_levels=win_levels,
            outcome=outcome,
            action_counts=level_result.action_counts,
            trajectory=trajectory,
            step_count=level_result.n_steps,
            final=True,
        )
        memory.save(os.path.join(run_dir, "memory.json"))

        # Level progression: WIN → next level; failure → stop run, no retry.
        if outcome == "WIN":
            current_level += 1
        else:
            print(f"  Level {level} {outcome} — stopping run (no retry).")
            break

    # Summary
    wins = sum(1 for r in level_results if r.outcome == "WIN")
    summary = {
        "game_id": game_id,
        "baseline": baseline,
        "model": model_name,
        "win_levels": win_levels,
        "levels_attempted": len(level_results),
        "wins": wins,
        "win_rate": wins / win_levels if win_levels else 0.0,
        "level_results": [
            {
                "level": r.level,
                "outcome": r.outcome,
                "n_steps": r.n_steps,
                "action_counts": r.action_counts,
                "n_reflections_used": r.n_reflections_used,
                "n_insights_used": r.n_insights_used,
                "n_notes_used": r.n_notes_used,
                "n_meta_used": r.n_meta_used,
            }
            for r in level_results
        ],
        "final_memory": (
            {"n_reflections": len(memory.reflections), "reflections": memory.reflections}
            if isinstance(memory, ReflexionMemory)
            else {"n_insights": len(memory.insights), "n_experiences": len(memory.experiences),
                  "insights": memory.insights}
            if isinstance(memory, ExPeLMemory)
            else {
                "n_meta": len(memory.meta),
                "meta_cognitions": [
                    {
                        "id": m.id,
                        "summary": m.summary,
                        "principles": m.principles,
                        "confidence": m.confidence,
                        "evidence_count": m.evidence_count,
                        "reward_sum": m.reward_sum,
                    }
                    for m in memory.meta.values()
                ],
            }
            if isinstance(memory, TrainableGraphMemory)
            else {"n_notes": len(memory.notes), "notes": [n.content for n in memory.notes]}
        ),
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*55}")
    print(f"Game {game_id}: {wins}/{win_levels} WON  ({100*wins/win_levels:.0f}%)")
    print(f"Run saved to: {run_dir}")
    return run_dir


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_openai_client(agent: OpenAICompatibleAgent):
    """Create openai.OpenAI client from agent credentials (for memory LLM calls)."""
    from openai import OpenAI
    return OpenAI(
        base_url=agent.base_url,
        api_key=agent.api_key,
    )


GAME_IDS = {
    "cd82": "cd82-fb555c5d",
    "sk48": "sk48-d8078629",
    "tu93": "tu93-0768757b",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game-id", help="Full game ID or short (cd82/sk48/tu93)")
    parser.add_argument("--all-games", action="store_true", help="Run cd82, sk48, tu93")
    parser.add_argument("--baseline", choices=["reflexion", "expel", "amem", "tgm", "both", "all"], default="both")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--runs-dir", default=str(project_root / "runs"))
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--memory-update-interval", type=int, default=DEFAULT_MEMORY_UPDATE_INTERVAL,
                        help="Update and inject memory every N steps inside the same ARC attempt")
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    args = parser.parse_args()

    # Build game list
    if args.all_games:
        games = list(GAME_IDS.values())
    elif args.game_id:
        gid = GAME_IDS.get(args.game_id, args.game_id)
        games = [gid]
    else:
        parser.error("Provide --game-id or --all-games")

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    agent = OpenAICompatibleAgent(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        model=args.model,
        max_tokens=8192,
        temperature=0.2,
    )

    max_steps = args.max_steps
    baselines = ["reflexion", "expel"] if args.baseline == "both" else ["reflexion", "expel", "amem", "tgm"] if args.baseline == "all" else [args.baseline]
    embedder = None
    if "amem" in baselines or "tgm" in baselines:
        from sentence_transformers import SentenceTransformer
        print(f"Loading embedding model for graph-memory baselines: {args.embed_model} ...")
        embedder = SentenceTransformer(args.embed_model)
    all_run_dirs = []

    for game_id in games:
        for baseline in baselines:
            run_dir = run_game(
                game_id=game_id,
                baseline=baseline,
                agent=agent,
                model_name=args.model_name,
                runs_dir=args.runs_dir,
                embedder=embedder,
                max_steps=max_steps,
                memory_update_interval=args.memory_update_interval,
            )
            all_run_dirs.append(run_dir)

    print("\n=== All runs completed ===")
    for d in all_run_dirs:
        print(f"  {d}")


if __name__ == "__main__":
    main()
