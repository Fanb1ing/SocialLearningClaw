#!/usr/bin/env python3
"""ARC-AGI-3 runner with manual game rules — no schema, no reasoning trace.

Schema retrieval is replaced with the hand-written rules in docs/sk48_rules.md.
The agent only needs to output a plain action JSON (no reasoning fields).
Results saved to: runs/arc_agi3_withrule/<model>/<timestamp>/
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
from dataclasses import asdict
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PROJECT_ROOT)

from socialclaw.agent.base import AgentAttempt, ReasoningTrace, Usage
from socialclaw.agent.openai_compatible import OpenAICompatibleAgent
from socialclaw.dataset.arc_agi3 import ARCAGI3EnvWrapper
from socialclaw.dataset.base import EvalResult, Problem
from socialclaw.logging import write_episode, write_step, write_trajectory
from socialclaw.schema.arc_agi3_parser import compute_grid_diff
from socialclaw.types import AttemptRecord, Episode
from socialclaw.utils import load_dotenv, make_run_dir, save_cmd

_RULES_PATH = os.path.join(_PROJECT_ROOT, "docs", "sk48_rules.md")


def _load_rules() -> str:
    with open(_RULES_PATH, "r", encoding="utf-8") as f:
        return f.read().strip()


def _build_prompt(
    *,
    game_id: str,
    level: int,
    step: int,
    grid_text: str,
    available_actions: List[str],
    rules: str,
    history: List[Dict],
) -> str:
    system = (
        "You are an ARC-AGI-3 game-playing agent.\n\n"
        "Grid coordinate system:\n"
        "  - col (x): 0 = leftmost column, increases to the RIGHT\n"
        "  - row (y): 0 = topmost row, increases DOWNWARD\n"
        "  - All positions are written as (col, row) = (x, y)\n"
        "  - The grid image has sparse gridlines every 8 cells to help locate positions\n\n"
        "Action definitions:\n"
        "  - ACTION1 = move the pink-black frame UP (row - 1). No data.\n"
        "  - ACTION2 = move the pink-black frame DOWN (row + 1). No data.\n"
        "  - ACTION3 = extend/retract the rod LEFTWARD. No data.\n"
        "  - ACTION4 = extend/retract the rod RIGHTWARD. No data.\n"
        "  - ACTION6 = click at grid position (col, row). "
        "MUST provide data: {\"x\": col_int, \"y\": row_int}.\n"
        "  - ACTION7 = special action. No data.\n\n"
        "Output format — strict JSON, nothing else:\n"
        "  {\n"
        "    \"action\": \"ACTION_NAME\",\n"
        "    \"data\": {\"x\": col, \"y\": row},\n"
        "    \"reasoning\": \"one sentence explaining which object you are targeting, where it is, and why this action moves toward threading it\"\n"
        "  }\n"
        "  Omit \"data\" for non-click actions.\n"
        "CRITICAL: Output ONLY the raw JSON object. No markdown, no text before or after.\n\n"
        "=== KNOWN GAME RULES ===\n"
        f"{rules}\n"
        "========================\n"
    )

    history_text = ""
    if history:
        action_total: Dict[str, int] = {}
        action_no_effect: Dict[str, int] = {}
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

        only_failing = [a for a in action_total if action_no_effect.get(a, 0) == action_total[a]]
        never_tried = [a for a in available_actions if a not in action_total]
        if only_failing:
            history_text += f"  NOTE: {', '.join(only_failing)} have had NO effect in any attempt.\n"
        if never_tried:
            history_text += f"  NOTE: Not yet tried: {', '.join(never_tried)}.\n"

        history_text += "\nRecent steps (last 5):\n"
        for h in history[-5:]:
            act = h["action"]
            data = h.get("action_data", {})
            changed = h.get("grid_changed", "?")
            if act == "ACTION6" and data:
                history_text += (
                    f"  Step {h['step']}: {act} at (col={data.get('x','?')},row={data.get('y','?')})"
                    f" -> state={h['state']}, grid_changed={changed}\n"
                )
            else:
                history_text += f"  Step {h['step']}: {act} -> state={h['state']}, grid_changed={changed}\n"

    user = (
        f"Game: {game_id} | Level: {level} | Step: {step}\n"
        f"Grid (see attached image): {grid_text}\n\n"
        f"Available actions: {', '.join(available_actions)}\n"
        f"{history_text}\n"
        "Choose your next action:"
    )

    return system + "\n---\n" + user


def _parse_action(answer_text: str, available_actions: List):
    """Parse LLM JSON output into (chosen_action, data_dict, reasoning_str)."""
    action_name = None
    data: Dict = {}
    reasoning = ""

    text = answer_text.strip()
    # Strip markdown fences
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
            parsed = json.loads(text[start : end + 1])
            action_name = parsed.get("action", "")
            data = parsed.get("data") or {}
            reasoning = parsed.get("reasoning", "")
        except Exception:
            pass

    # Fallback: scan for action name in raw text
    if not action_name:
        for act in available_actions:
            if act.name in answer_text.upper():
                action_name = act.name
                break

    chosen = None
    if action_name:
        upper = action_name.upper()
        for act in available_actions:
            if act.name == upper:
                chosen = act
                break

    if chosen is None and available_actions:
        chosen = available_actions[0]

    # Only ACTION6 carries coordinate data
    if chosen and chosen.name != "ACTION6":
        data = {}

    return chosen, data or {}, reasoning


def run_withrule(
    *,
    game_id: str,
    agent: OpenAICompatibleAgent,
    max_steps_per_level: int = 200,
    runs_dir: str = "runs",
    render_mode: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    rules = _load_rules()
    model_str = model or "unknown_model"
    run_dir = make_run_dir(runs_dir, "arc_agi3_withrule", model_str)
    save_cmd(run_dir)
    print(f"[RUN] Output directory: {run_dir}")

    print(f"[ARC] Creating environment for game={game_id} ...")
    env_wrapper = ARCAGI3EnvWrapper(game_id, render_mode=render_mode)
    # Single reset — game auto-advances levels internally; never reset between levels.
    obs = env_wrapper.reset()
    win_levels = obs.win_levels if obs else 1
    print(f"[ARC] win_levels={win_levels}, initial state={obs.state if obs else 'N/A'}")

    current_level = 0
    prev_levels_completed = obs.levels_completed if obs else 0
    print(f"\n=== Level {current_level + 1}/{win_levels} ===")

    level_id = f"{game_id}_L{current_level + 1}"
    level_problem = Problem(
        id=level_id,
        prompt="",
        problem_type="arc_grid",
        meta={"game_id": game_id, "level": current_level + 1, "win_levels": win_levels},
    )
    step = 0
    history: List[Dict[str, Any]] = []
    trajectory: List[Dict[str, Any]] = []
    attempts: List[AttemptRecord] = []
    evals: List[EvalResult] = []
    stop_reason: Optional[str] = None

    def _save_level() -> None:
        episode = Episode(
            problem=level_problem,
            attempts=attempts,
            evals=evals,
            reasoning_trace=None,
            reasoning_confidence=0.0,
            flags=["withrule_no_schema"],
            stop_reason=stop_reason,
            model=model,
        )
        ep_path = write_episode(run_dir, episode)
        traj_path = write_trajectory(run_dir, level_id, trajectory)
        print(f"  Episode saved: {ep_path}")
        print(f"  Trajectory saved: {traj_path}")

    while True:
        if obs is None or obs.is_empty():
            print(f"[ARC] Empty observation at step {step}, stopping.")
            _save_level()
            break

        # Detect automatic level transition (arcengine advances levels internally on level win).
        # After completing level N (not the last), obs.levels_completed increments and the game
        # is already on level N+1 — we must NOT call env_wrapper.reset() between levels.
        cur_lc = obs.levels_completed if obs else 0
        if cur_lc > prev_levels_completed:
            print(f"  Level {current_level + 1} completed in {step} steps! (levels_completed={cur_lc})")
            evals.append(EvalResult(correct=True, pred="", gold="WIN", details="LEVEL_COMPLETE"))
            _save_level()

            if obs.state == env_wrapper.GameState.WIN:
                print("  All levels WON!")
                break

            # Transition to next level without resetting — game is already there.
            prev_levels_completed = cur_lc
            current_level = cur_lc
            level_id = f"{game_id}_L{current_level + 1}"
            level_problem = Problem(
                id=level_id,
                prompt="",
                problem_type="arc_grid",
                meta={"game_id": game_id, "level": current_level + 1, "win_levels": win_levels},
            )
            step = 0
            history = []
            trajectory = []
            attempts = []
            evals = []
            stop_reason = None
            print(f"\n=== Level {current_level + 1}/{win_levels} ===")

        # Check terminal states
        if obs.state == env_wrapper.GameState.WIN:
            print(f"  Game WON!")
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

        # Use the last frame (fully settled after any sliding/animation),
        # not frame[0] which may be a mid-animation intermediate state.
        grid = obs.frame[-1] if obs.frame else None
        if grid is None:
            print(f"[ARC] Grid is None at step {step}, stopping.")
            _save_level()
            break

        available_actions = env_wrapper.get_available_actions(obs)
        h, w = grid.shape
        grid_summary = f"{h}x{w} full grid — see attached image"
        print(f"[ARC] Step {step}: grid={h}x{w}, actions={[a.name for a in available_actions]}")

        # Render grid image (with gridlines from the updated grid_to_image)
        grid_img = ARCAGI3EnvWrapper.grid_to_image(grid, cell_size=8)
        img_dir = os.path.join(run_dir, level_id)
        os.makedirs(img_dir, exist_ok=True)
        img_path = os.path.join(img_dir, f"step_{step:03d}.png")
        grid_img.save(img_path)

        img_buffer = io.BytesIO()
        grid_img.save(img_buffer, format="PNG")
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode("ascii")
        image_data_url = f"data:image/png;base64,{img_base64}"

        prompt = _build_prompt(
            game_id=game_id,
            level=current_level + 1,
            step=step,
            grid_text=grid_summary,
            available_actions=[a.name for a in available_actions],
            rules=rules,
            history=history,
        )

        if step == 0:
            level_problem.prompt = prompt

        # Call LLM
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
            print(f"[ARC] LLM call failed at step {step}: {e}")
            attempt = AgentAttempt(
                answer_text='{"action": "ACTION1"}',
                reasoning_trace=ReasoningTrace(concepts=[], relations=[], explanation=f"API error: {e}"),
                usage=Usage(input_tokens=0, output_tokens=0, total_tokens=0),
                raw={"error": str(e)},
            )

        chosen_action, action_data, reasoning = _parse_action(attempt.answer_text, available_actions)

        # Execute action
        pre_grid = grid.copy() if grid is not None else None
        obs = env_wrapper.step(chosen_action, data=action_data) if chosen_action else env_wrapper.step(available_actions[0])

        step += 1
        state_str = str(obs.state) if obs else "UNKNOWN"
        post_grid = obs.frame[-1] if obs and obs.frame else None
        grid_changed, _ = compute_grid_diff(pre_grid, post_grid)

        print(
            f"  Step {step}: action={chosen_action.name if chosen_action else 'NONE'}, "
            f"state={state_str}, grid_changed={grid_changed}"
        )
        if reasoning:
            print(f"  Reasoning: {reasoning}")

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
        })

    scorecard = env_wrapper.get_scorecard()
    if scorecard:
        print(f"\nFinal Score: {scorecard.score}, Total Actions: {scorecard.total_actions}")

    return run_dir


def main() -> None:
    p = argparse.ArgumentParser(
        description="ARC-AGI-3 with manual game rules (no schema, no reasoning)"
    )
    p.add_argument("--game-id", required=True, help="ARC-AGI-3 game ID (e.g. sk48-d8078629)")
    p.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    p.add_argument("--api-key", default="")
    p.add_argument("--model", default="google/gemini-2.5-pro", help="Vision LLM model")
    p.add_argument("--max-steps", type=int, default=200, help="Max steps per level")
    p.add_argument("--runs-dir", default="runs", help="Root runs directory (default: runs)")
    p.add_argument("--render", action="store_true", help="Enable terminal rendering")
    p.add_argument("--max-tokens", type=int, default=2048, help="Max tokens per LLM call")
    args = p.parse_args()

    load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

    if os.environ.get("ARC_AGI_API_KEY") and not os.environ.get("ARC_API_KEY"):
        os.environ["ARC_API_KEY"] = os.environ["ARC_AGI_API_KEY"]

    api_key = args.api_key.strip()
    if not api_key:
        for k in ["OPENROUTER_API_KEY", "OPENAI_API_KEY", "API_KEY"]:
            v = (os.environ.get(k) or "").strip()
            if v:
                api_key = v
                break
    if not api_key:
        raise SystemExit("Missing API key. Provide --api-key or set OPENROUTER_API_KEY in .env.")

    agent = OpenAICompatibleAgent(
        base_url=args.base_url,
        api_key=api_key,
        model=args.model,
        max_tokens=args.max_tokens,
    )

    run_dir = run_withrule(
        game_id=args.game_id,
        agent=agent,
        max_steps_per_level=args.max_steps,
        runs_dir=args.runs_dir,
        render_mode="terminal" if args.render else None,
        model=args.model,
    )
    print(f"\nRun completed: {run_dir}")


if __name__ == "__main__":
    main()
