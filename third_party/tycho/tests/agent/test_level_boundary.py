"""Smoke-test level-boundary consolidation without calling an LLM.

The level-complete path is easy to regress because the engine has already advanced to the next
level by the time choose_action() re-enters. This test drives that exact shape with synthetic frames:
old level solved, terminal event already written by the harness, then the next level arrives.
"""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from arcengine import GameAction


def _set_env(mode: str) -> dict:
    old = dict(os.environ)
    os.environ.update({
        "LLM_MODEL": "dummy",
        "LLM_BACKEND": "openai",
        "LLM_BASE_URL": "http://127.0.0.1:9",
        "LLM_API_KEY": "dummy",
        "MPLCONFIGDIR": "/private/tmp",
        "TYCHO_MODE": mode,
        "TYCHO_VISION": "0",
        "TYCHO_TEXT_GRID": "full",
    })
    return old


def _restore_env(old: dict) -> None:
    os.environ.clear()
    os.environ.update(old)


def _frame(grid, levels_completed=0, state="NOT_FINISHED"):
    return SimpleNamespace(frame=grid, levels_completed=levels_completed, state=state)


def _run(mode: str) -> dict[str, bool]:
    old_env = _set_env(mode)
    try:
        from tycho.agent.agent import TychoAgent

        class StubTycho(TychoAgent):
            def __init__(self):
                super().__init__()
                self.summary_prompts: list[str] = []
                self.summary_calls = 0
                self.builder_reasons: list[str] = []
                self.action_prompts: list[str] = []

            def _latest_boundary_prompt(self) -> str:
                for msg in reversed(self.history):
                    content = msg.get("content")
                    if msg.get("role") == "user" and isinstance(content, str) and "COMPLETED level" in content:
                        return content
                return ""

            def _invoke_builder(self, reason: str, *, reserve_calls: int = 0) -> str:
                self.builder_reasons.append(reason)
                return "World-model builder report (stub): confidence: high\nrecommended_action: ACTION1"

            def _chat(self, call_type, max_tokens=None):
                if call_type == "level_summary":
                    self.summary_calls += 1
                    prompt = self._latest_boundary_prompt()
                    if prompt and not self.summary_prompts:
                        self.summary_prompts.append(prompt)
                    if self.summary_calls == 1:
                        tool_calls = [
                            {"id": "w1", "name": "write_file", "input": {
                                "path": "notes/level_0_insights.md",
                                "content": "terminal showed the goal state\n"}},
                            {"id": "w2", "name": "write_file", "input": {
                                "path": "notes/world_model.md",
                                "content": "terminal-derived model insight\n"}},
                        ]
                        if mode in ("orchestrator", "trigger"):
                            tool_calls.append({"id": "bad-wm", "name": "write_file", "input": {
                                "path": "world_model.py",
                                "content": "BAD = True\n",
                            }})
                        return {"text": "writing boundary notes", "tool_calls": tool_calls}
                    return {"text": "done", "tool_calls": []}

                content = self.history[-1].get("content") if self.history else ""
                if isinstance(content, str):
                    text = content
                elif isinstance(content, list):
                    text = "\n".join(p.get("text", "") for p in content if isinstance(p, dict))
                else:
                    text = ""
                self.action_prompts.append(text)
                return {"text": "act", "tool_calls": [
                    {"id": "a1", "name": "take_action", "input": {"action": "ACTION1"}},
                ]}

        with TemporaryDirectory() as td:
            agent = StubTycho()
            agent.reset("boundary_smoke", [GameAction.ACTION1], ws_root=td)
            initial = [[0, 0], [1, 1]]
            terminal = [[0, 0], [2, 2]]
            next_init = [[3, 3], [4, 4]]

            first = agent.choose_action([], _frame([initial], 0), [GameAction.ACTION1])
            agent.ws.record_solved(0, first.action.name)
            agent.ws.record_terminal(0, terminal, action=first.action.name)
            second = agent.choose_action([], _frame([terminal, next_init], 1), [GameAction.ACTION1])

            ws = Path(agent.ws.dir)
            history_text = "\n".join(
                msg.get("content", "") if isinstance(msg.get("content"), str) else
                "\n".join(p.get("text", "") for p in msg.get("content", []) if isinstance(p, dict))
                for msg in agent.history
            )
            boundary_prompt = agent.summary_prompts[0] if agent.summary_prompts else ""
            world_model_text = (ws / "world_model.py").read_text()
            checks = {
                "summary_prompt_mentions_terminal": (
                    "terminal.txt" in boundary_prompt
                    and "terminal.json" in boundary_prompt
                ),
                "summary_prompt_blocks_action": "Do not take an action" in boundary_prompt,
                "level_notes_written": (ws / "notes" / "level_0_insights.md").exists(),
                "world_model_notes_written": (
                    (ws / "notes" / "world_model.md").read_text().strip()
                    == "terminal-derived model insight"
                ),
                "new_level_prompt_no_previous_terminal_block": "[previous level WON]" not in history_text,
                "new_level_prompt_mentions_carry_files": (
                    "If notes/level_0_insights.md exists, read it." in history_text
                    and "If you have derived a reliable world model" in history_text
                    and "whether init_state() still holds" in history_text
                ),
                "agent_advanced_to_level_1": agent.level == 1 and agent.turn_in_level == 1,
                "new_level_grid_recorded": (ws / "level_1" / "turn_000.txt").exists(),
                "old_terminal_record_still_present": (ws / "level_0" / "terminal.txt").exists(),
                "actions_returned": first.action.name == "ACTION1" and second.action.name == "ACTION1",
            }
            if mode in ("orchestrator", "trigger"):
                checks["builder_owned_prompt_variant"] = (
                    "world_model.py is builder-owned in this mode" in boundary_prompt
                )
                checks["builder_owned_summary_write_blocked"] = "BAD = True" not in world_model_text
            if mode == "trigger":
                checks.update({
                    "level_start_builder_fired": (
                        len(agent.builder_reasons) == 1
                        and "level 1 just started" in agent.builder_reasons[0]
                    ),
                    "builder_reason_reads_boundary_note": (
                        "If notes/level_0_insights.md exists, read it." in agent.builder_reasons[0]
                        and "reliable world model" in agent.builder_reasons[0]
                        and "init_state/render/outcome" in agent.builder_reasons[0]
                    ),
                    "builder_report_precedes_actor_action": (
                        bool(agent.action_prompts)
                        and agent.action_prompts[-1].startswith("[auto world-model builder fired")
                    ),
                })
            elif mode == "orchestrator":
                checks["orchestrator_builder_not_auto_fired"] = len(agent.builder_reasons) == 0
            else:
                checks["single_prompt_can_update_model"] = (
                    "update world_model.py" in boundary_prompt
                )
            return checks
    finally:
        _restore_env(old_env)


def test_level_boundary_modes():
    for mode in ("single", "orchestrator", "trigger"):
        checks = _run(mode)
        assert all(checks.values()), {mode: checks}


def main() -> int:
    ok = True
    for mode in ("single", "orchestrator", "trigger"):
        print(f"=== {mode} ===")
        checks = _run(mode)
        for name, passed in checks.items():
            print(f"  {name}: {passed}")
            ok = ok and passed
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
