"""Prompt-boundary checks for grid image delivery.

These tests avoid LLM calls but exercise the real workspace -> TychoAgent history
path. They pin the perception contract that the model-facing conversation contains
the current grid image whenever vision is enabled, across single/subagent/trigger
mode, level boundaries, and reset-after-GAME_OVER. Internally the actor-pull
subagent configuration is named TYCHO_MODE=orchestrator.
"""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from arcengine import GameAction


def _set_env(mode: str, *, vision: bool = True, image_cap: int = 8) -> dict:
    old = dict(os.environ)
    os.environ.update({
        "LLM_MODEL": "dummy",
        "LLM_BACKEND": "openai",
        "LLM_BASE_URL": "http://127.0.0.1:9",
        "LLM_API_KEY": "dummy",
        "MPLCONFIGDIR": "/private/tmp",
        "TYCHO_MODE": mode,
        "TYCHO_VISION": "1" if vision else "0",
        "TYCHO_RENDER_SCALE": "1",
        "TYCHO_TEXT_GRID": "full",
        "TYCHO_HISTORY": "tail_evict",
        "TYCHO_GRID_KEEP": "8",
        "TYCHO_IMAGE_KEEP": str(image_cap),
        "TYCHO_IMAGE_CAP": str(image_cap),
        "TYCHO_IMAGE_EVICT_MULT": "1",
        "TYCHO_MAX_TOOL_STEPS": "1",
        "TYCHO_MAX_LLM_CALLS": "20",
    })
    return old


def _restore_env(old: dict) -> None:
    os.environ.clear()
    os.environ.update(old)


def _grid(seed: int) -> list[list[int]]:
    """A distinct 64x64 grid that renders through the real ARC renderer."""
    return [[(seed + r * 3 + c * 5) % 16 for c in range(64)] for r in range(64)]


def _frame(grid_or_frames, levels_completed=0, state="NOT_FINISHED"):
    return SimpleNamespace(frame=grid_or_frames, levels_completed=levels_completed, state=state)


def _image_parts(history: list[dict]) -> list[bytes]:
    images = []
    for msg in history:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("image_png") is not None:
                images.append(part["image_png"])
    return images


def _text(history: list[dict]) -> str:
    chunks = []
    for msg in history:
        content = msg.get("content")
        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, list):
            chunks.extend(
                part.get("text", "")
                for part in content
                if isinstance(part, dict)
            )
    return "\n".join(chunks)


def _run_actor_sequence(
    mode: str, *, vision: bool = True, image_cap: int = 8, pre_death_turns: int = 0
) -> dict[str, bool]:
    old_env = _set_env(mode, vision=vision, image_cap=image_cap)
    try:
        from tycho.agent.agent import TychoAgent

        class StubTycho(TychoAgent):
            def __init__(self):
                super().__init__()
                self.chat_snapshots: list[dict] = []
                self.builder_reasons: list[str] = []
                self.summary_calls = 0

            def _summarize_level(self):
                self.summary_calls += 1
                if getattr(self, "ws", None) is not None:
                    self.ws.write_file(
                        f"notes/level_{self.level}_insights.md",
                        "boundary insight\n",
                    )

            def _invoke_builder(self, reason: str, *, reserve_calls: int = 0) -> str:
                self.builder_reasons.append(reason)
                return "confidence: high\nrecommended_action: ACTION1"

            def _chat(self, call_type, max_tokens=None):
                self.chat_snapshots.append({
                    "call_type": call_type,
                    "history_text": _text(self.history),
                    "images": list(_image_parts(self.history)),
                })
                return {"text": "act", "tool_calls": [
                    {"id": f"a{len(self.chat_snapshots)}", "name": "take_action",
                     "input": {"action": "ACTION1"}},
                ]}

        with TemporaryDirectory() as td:
            agent = StubTycho()
            agent.reset("image_delivery", [GameAction.ACTION1], ws_root=td)

            start = _grid(1)
            after = _grid(2)
            terminal = _grid(3)
            next_level = _grid(4)
            reset_start = _grid(5)
            game_over = _grid(6)

            first = agent.choose_action([], _frame([start], 0), [GameAction.ACTION1])
            img0 = agent.ws.current_png(0, 0)
            snap0 = agent.chat_snapshots[-1]

            second = agent.choose_action([], _frame([after], 0), [GameAction.ACTION1])
            img1 = agent.ws.current_png(0, 1)
            snap1 = agent.chat_snapshots[-1]

            agent.ws.record_solved(0, second.action.name)
            agent.ws.record_terminal(0, terminal, action=second.action.name)
            third = agent.choose_action([], _frame([terminal, next_level], 1), [GameAction.ACTION1])
            img2 = agent.ws.current_png(1, 0)
            snap2 = agent.chat_snapshots[-1]

            pre_death_grid = next_level
            for i in range(pre_death_turns):
                pre_death_grid = _grid(7 + i)
                agent.choose_action([], _frame([pre_death_grid], 1), [GameAction.ACTION1])
            death_turn = agent.turn_in_level

            game_over_event = {
                "level": 1,
                "turn": death_turn,
                "action": "ACTION2",
                "row": None,
                "col": None,
                "state": "GAME_OVER",
                "prev": pre_death_grid,
                "next": game_over,
            }
            agent.ws.record_game_over(
                level=1,
                turn_in_level=death_turn,
                action="ACTION2",
                prev_grid=pre_death_grid,
                game_over_grid=game_over,
            )
            agent.note_external_reset(game_over_event)
            fourth = agent.choose_action([], _frame([reset_start], 1), [GameAction.ACTION1])
            img3 = agent.ws.current_png(1, 0)
            snap3 = agent.chat_snapshots[-1]

            ws = Path(agent.ws.dir)
            terminal_png = (ws / "level_0" / "terminal.png").read_bytes()
            game_over_prev_png = (ws / "level_1" / f"death_{death_turn:03d}_prev.png").read_bytes()
            game_over_next_png = (ws / "level_1" / f"death_{death_turn:03d}_next.png").read_bytes()
            checks = {
                "actions_returned": all(
                    a.action.name == "ACTION1" for a in (first, second, third, fourth)
                ),
                "turn0_png_exists": bool(img0),
                "turn1_png_exists": bool(img1),
                "new_level_png_exists": bool(img2),
                "reset_png_exists": bool(img3),
            }
            if vision:
                checks.update({
                    "turn0_image_sent": snap0["images"][-1] == img0,
                    "turn1_image_sent": snap1["images"][-1] == img1,
                    "new_level_uses_next_playable_image": snap2["images"][-1] == img2,
                    "new_level_does_not_send_terminal_image": terminal_png not in snap2["images"],
                    "reset_frame_image_sent_after_game_over_note": (
                        bool(snap3["images"]) and snap3["images"][-1] == img3
                    ),
                    "game_over_evidence_text_sent": "[GAME_OVER evidence]" in snap3["history_text"]
                    and "caused API GAME_OVER" in snap3["history_text"],
                    "game_over_evidence_images_sent_when_cap_allows": (
                        image_cap < 3
                        or (game_over_prev_png in snap3["images"] and game_over_next_png in snap3["images"])
                    ),
                    "game_over_evidence_images_omitted_when_cap_tight": (
                        image_cap >= 3
                        or (game_over_prev_png not in snap3["images"] and game_over_next_png not in snap3["images"])
                    ),
                    "game_over_image_payload_cap_respected": len(snap3["images"]) <= image_cap,
                })
            else:
                checks.update({
                    "vision_off_sends_no_images": all(not s["images"] for s in agent.chat_snapshots),
                    "vision_off_has_text_grid_fallback": "Current grid" in snap0["history_text"]
                    and "y00:" in snap0["history_text"],
                })
            if mode == "trigger":
                checks.update({
                    "trigger_builder_ran_at_level_start": any(
                        "level 1 just started" in r for r in agent.builder_reasons
                    ),
                    "trigger_actor_still_received_new_level_image": (not vision) or snap2["images"][-1] == img2,
                    "trigger_death_prompt_does_not_assign_world_model_ownership": (
                        "init_state()/level constants before retrying" not in snap3["history_text"]
                    ),
                })
            else:
                checks["non_trigger_no_auto_builder"] = not agent.builder_reasons
                if mode == "single":
                    checks["single_death_prompt_nudges_init_state_update"] = (
                        "init_state()/level constants before retrying" in snap3["history_text"]
                        and "better initial state instead of rediscovering" in snap3["history_text"]
                    )
                elif mode == "orchestrator":
                    checks["orchestrator_death_prompt_does_not_assign_world_model_ownership"] = (
                        "init_state()/level constants before retrying" not in snap3["history_text"]
                    )
            return checks
    finally:
        _restore_env(old_env)


def _check_builder_divergence_images() -> dict[str, bool]:
    old_env = _set_env("orchestrator", vision=True)
    try:
        from tycho.agent import builder as builder_mod
        from tycho.agent.builder import WorldModelBuilder
        from tycho.serving.llm_client import LLMConfig
        from tycho.workspace.agent_tools import tool_specs
        from tycho.workspace.workspace import GameWorkspace

        with TemporaryDirectory() as td:
            ws = GameWorkspace(
                "builder_images",
                root=td,
                render=True,
                available_actions=[GameAction.ACTION1],
                render_scale=1,
            )
            ws.record(_grid(7), level=0, turn_in_level=0, state="NOT_FINISHED", available=["ACTION1"])
            ws.record(_grid(8), level=0, turn_in_level=1, action="ACTION1",
                      state="NOT_FINISHED", available=["ACTION1"])
            prev_png = ws.current_png(0, 0)
            next_png = ws.current_png(0, 1)

            class FakeExecutor:
                def __init__(self, ws):
                    self.ws = ws

                def _run_python(self, code, timeout=30):
                    return "[verify state] simulation_accuracy=0.00\n  first divergence: L0 turn 1 action ACTION1"

                def execute(self, name, args):
                    return "(unused)"

            captured = []
            orig_chat = builder_mod.chat_tools

            def fake_chat(history, tools_spec, cfg, system=None, max_tokens=None, effort=None,
                          call_type=None, **kwargs):
                captured.append(history)
                return {"text": "done", "tool_calls": []}

            builder_mod.chat_tools = fake_chat
            try:
                b = WorldModelBuilder(
                    LLMConfig.from_env(),
                    FakeExecutor(ws),
                    tool_specs("generic", include_builder=False),
                    effort="",
                    max_tokens=1024,
                    max_steps=1,
                )
                report, trace = b.build(0, hint="test", max_calls=1)
            finally:
                builder_mod.chat_tools = orig_chat

            images = _image_parts(captured[0]) if captured else []
            text = _text(captured[0]) if captured else ""
            return {
                "builder_called_once": len(captured) == 1,
                "builder_report_returned": "confidence:" in report,
                "builder_divergence_text_sent": "first divergence: L0 turn 1" in text,
                "builder_prev_image_sent": len(images) >= 2 and images[-2] == prev_png,
                "builder_next_image_sent": len(images) >= 2 and images[-1] == next_png,
                "builder_trace_empty_without_tools": trace == [],
            }
    finally:
        _restore_env(old_env)


def main() -> int:
    ok = True
    groups = []
    for mode in ("single", "orchestrator", "trigger"):
        label = "subagent_orchestrator" if mode == "orchestrator" else mode
        groups.append((f"{label}_vision_on", _run_actor_sequence(mode, vision=True, image_cap=8)))
    groups.append(("single_vision_off", _run_actor_sequence("single", vision=False)))
    groups.append(("trigger_image_cap_one", _run_actor_sequence("trigger", vision=True, image_cap=1)))
    groups.append(("single_game_over_image_cap_four", _run_actor_sequence(
        "single", vision=True, image_cap=4, pre_death_turns=3)))
    groups.append(("builder_divergence_images", _check_builder_divergence_images()))

    for group, checks in groups:
        print(f"=== {group} ===")
        for name, passed in checks.items():
            print(f"  {name}: {passed}")
            ok = ok and passed
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
