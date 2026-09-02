"""Same-batch tool deferral guards.

Native parallel-tool models can emit a cognitive tool and `take_action` in the same response. When
the cognitive tool is supposed to inform the action, committing the same-batch action is stale and
can spend a scored move before the actor reads the feedback.
"""

from __future__ import annotations

import os
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from arcengine import GameAction


def _set_env(mode: str = "single") -> dict:
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
        "TYCHO_SANDBOX_RUNTIME": "host",
    })
    return old


def _restore_env(old: dict) -> None:
    os.environ.clear()
    os.environ.update(old)


def _frame(grid, levels_completed=0, state="NOT_FINISHED"):
    return SimpleNamespace(frame=grid, levels_completed=levels_completed, state=state)


def _check_planner_same_batch_defers_action() -> dict[str, bool]:
    old_env = _set_env()
    try:
        from tycho.agent.agent import TychoAgent
        class StubTycho(TychoAgent):
            def __init__(self):
                super().__init__()
                self.freeform_calls = 0

            def _chat(self, call_type, max_tokens=None):
                self.freeform_calls += 1
                if self.freeform_calls == 1:
                    return {"text": "plan and stale action", "tool_calls": [
                        {"id": "p1", "name": "run_python", "input": {"code": "python plan.py astar"}},
                        {"id": "a1", "name": "take_action", "input": {"action": "ACTION1"}},
                    ]}
                return {"text": "commit after reading planner", "tool_calls": [
                    {"id": "a2", "name": "take_action", "input": {"action": "ACTION2"}},
                ]}

        with TemporaryDirectory() as td:
            agent = StubTycho()
            agent.reset("same_batch_deferral", [GameAction.ACTION1, GameAction.ACTION2], ws_root=td)
            choice = agent.choose_action(
                [],
                _frame([[[0, 0], [1, 1]]], 0),
                [GameAction.ACTION1, GameAction.ACTION2],
            )
            trace = choice.reasoning.get("tool_trace") or []
            tool_text = "\n".join(
                result.get("output", "")
                for msg in agent.history if msg.get("role") == "tool"
                for result in msg.get("results", [])
            )
            return {
                "first_action_deferred": any(
                    t.get("tool") == "take_action"
                    and t.get("args", {}).get("action") == "ACTION1"
                    and t.get("deferred") is True
                    for t in trace
                ),
                "second_action_committed": choice.action.name == "ACTION2",
                "planner_output_seen_before_commit": "PLANNER_COMMAND: python plan.py astar" in tool_text,
                "defer_message_explains_planner": "ran the planner in the SAME batch" in tool_text,
            }
    finally:
        _restore_env(old_env)


def _check_tool_cap_gets_final_commit_pass() -> dict[str, bool]:
    old_env = _set_env()
    os.environ["TYCHO_MAX_TOOL_STEPS"] = "1"
    try:
        from tycho.agent.agent import TychoAgent

        class StubTycho(TychoAgent):
            def __init__(self):
                super().__init__()
                self.call_types: list[str] = []

            def _chat(self, call_type, max_tokens=None):
                self.call_types.append(call_type)
                if call_type == "freeform_commit":
                    return {"text": "commit now", "tool_calls": [
                        {"id": "a2", "name": "take_action", "input": {"action": "ACTION2"}},
                    ]}
                return {"text": "inspect first", "tool_calls": [
                    {"id": "v1", "name": "set_verbosity", "input": {"grid": "off"}},
                ]}

        with TemporaryDirectory() as td:
            agent = StubTycho()
            agent.reset("tool_cap_final_commit", [GameAction.ACTION1, GameAction.ACTION2], ws_root=td)
            choice = agent.choose_action(
                [],
                _frame([[[0, 0], [1, 1]]], 0),
                [GameAction.ACTION1, GameAction.ACTION2],
            )
            trace = choice.reasoning.get("tool_trace") or []
            return {
                "final_commit_called": "freeform_commit" in agent.call_types,
                "chosen_from_final_commit": choice.action.name == "ACTION2",
                "did_not_default_to_first_action": choice.action.name != "ACTION1",
                "trace_marks_final_commit": any(
                    t.get("tool") == "take_action"
                    and t.get("args", {}).get("action") == "ACTION2"
                    and t.get("final_commit") is True
                    for t in trace
                ),
            }
    finally:
        _restore_env(old_env)


def test_tool_cap_gets_final_commit_pass():
    checks = _check_tool_cap_gets_final_commit_pass()
    assert all(checks.values()), checks


def _check_invalid_take_action_reprompts() -> dict[str, bool]:
    old_env = _set_env()
    os.environ["TYCHO_MAX_TOOL_STEPS"] = "3"
    try:
        from tycho.agent.agent import TychoAgent

        class StubTycho(TychoAgent):
            def __init__(self):
                super().__init__()
                self.calls_seen = 0

            def _chat(self, call_type, max_tokens=None):
                self.calls_seen += 1
                if self.calls_seen == 1:
                    return {"text": "try unavailable", "tool_calls": [
                        {"id": "bad", "name": "take_action", "input": {"action": "ACTION5"}},
                    ]}
                return {"text": "commit valid", "tool_calls": [
                    {"id": "good", "name": "take_action", "input": {"action": "ACTION2"}},
                ]}

        with TemporaryDirectory() as td:
            agent = StubTycho()
            agent.reset("invalid_action_retry", [GameAction.ACTION1, GameAction.ACTION2], ws_root=td)
            choice = agent.choose_action(
                [],
                _frame([[[0, 0], [1, 1]]], 0),
                [GameAction.ACTION1, GameAction.ACTION2],
            )
            trace = choice.reasoning.get("tool_trace") or []
            return {
                "invalid_not_committed": any(
                    t.get("tool") == "take_action"
                    and t.get("args", {}).get("action") == "ACTION5"
                    and t.get("invalid") is True
                    for t in trace
                ),
                "valid_retry_committed": choice.action.name == "ACTION2",
                "committed_trace_matches_choice": any(
                    t.get("tool") == "take_action"
                    and t.get("args") == {"action": "ACTION2"}
                    and t.get("committed") is True
                    for t in trace
                ),
            }
    finally:
        _restore_env(old_env)


def test_invalid_take_action_reprompts_instead_of_silent_fallback():
    checks = _check_invalid_take_action_reprompts()
    assert all(checks.values()), checks


def _check_invalid_action6_coords_reprompt() -> dict[str, bool]:
    old_env = _set_env()
    os.environ["TYCHO_MAX_TOOL_STEPS"] = "3"
    try:
        from tycho.agent.agent import TychoAgent

        class StubTycho(TychoAgent):
            def __init__(self):
                super().__init__()
                self.calls_seen = 0

            def _chat(self, call_type, max_tokens=None):
                self.calls_seen += 1
                if self.calls_seen == 1:
                    return {"text": "click without coordinates", "tool_calls": [
                        {"id": "bad", "name": "take_action", "input": {"action": "ACTION6"}},
                    ]}
                return {"text": "commit valid", "tool_calls": [
                    {"id": "good", "name": "take_action", "input": {"action": "ACTION2"}},
                ]}

        with TemporaryDirectory() as td:
            agent = StubTycho()
            agent.reset("invalid_action6_retry", [GameAction.ACTION2, GameAction.ACTION6], ws_root=td)
            choice = agent.choose_action(
                [],
                _frame([[[0, 0], [1, 1]]], 0),
                [GameAction.ACTION2, GameAction.ACTION6],
            )
            trace = choice.reasoning.get("tool_trace") or []
            tool_text = "\n".join(
                result.get("output", "")
                for msg in agent.history if msg.get("role") == "tool"
                for result in msg.get("results", [])
            )
            return {
                "invalid_click_not_committed": any(
                    t.get("tool") == "take_action"
                    and t.get("args", {}).get("action") == "ACTION6"
                    and t.get("invalid") is True
                    for t in trace
                ),
                "valid_retry_committed": choice.action.name == "ACTION2",
                "no_center_click_default": getattr(choice, "x", None) is None and getattr(choice, "y", None) is None,
                "message_names_required_coords": "ACTION6 requires integer row and col" in tool_text,
            }
    finally:
        _restore_env(old_env)


def test_invalid_action6_coords_reprompt_instead_of_center_click():
    checks = _check_invalid_action6_coords_reprompt()
    assert all(checks.values()), checks


def _check_builder_owned_world_model_write_blocked() -> dict[str, bool]:
    old_env = _set_env(mode="trigger")
    os.environ["TYCHO_MAX_TOOL_STEPS"] = "3"
    try:
        from tycho.agent.agent import TychoAgent

        class StubTycho(TychoAgent):
            def __init__(self):
                super().__init__()
                self.calls_seen = 0

            def _chat(self, call_type, max_tokens=None):
                self.calls_seen += 1
                if self.calls_seen == 1:
                    return {"text": "edit and stale action", "tool_calls": [
                        {"id": "w1", "name": "write_file",
                         "input": {"path": "world_model.py", "content": "BAD = True\n"}},
                        {"id": "a1", "name": "take_action", "input": {"action": "ACTION1"}},
                    ]}
                return {"text": "commit after block", "tool_calls": [
                    {"id": "a2", "name": "take_action", "input": {"action": "ACTION2"}},
                ]}

        with TemporaryDirectory() as td:
            agent = StubTycho()
            agent.reset("builder_owned_block", [GameAction.ACTION1, GameAction.ACTION2], ws_root=td)
            before = agent.ws.read_file("world_model.py")
            choice = agent.choose_action(
                [],
                _frame([[[0, 0], [1, 1]]], 0),
                [GameAction.ACTION1, GameAction.ACTION2],
            )
            after = agent.ws.read_file("world_model.py")
            trace = choice.reasoning.get("tool_trace") or []
            tool_text = "\n".join(
                result.get("output", "")
                for msg in agent.history if msg.get("role") == "tool"
                for result in msg.get("results", [])
            )
            return {
                "blocked_message_seen": "world_model.py is builder-owned" in tool_text,
                "world_model_not_overwritten": "BAD = True" not in after and after == before,
                "same_batch_action_deferred": any(
                    t.get("tool") == "take_action"
                    and t.get("args", {}).get("action") == "ACTION1"
                    and t.get("deferred") is True
                    for t in trace
                ),
                "retry_committed": choice.action.name == "ACTION2",
            }
    finally:
        _restore_env(old_env)


def test_builder_owned_world_model_write_blocked_before_action():
    checks = _check_builder_owned_world_model_write_blocked()
    assert all(checks.values()), checks


def _check_no_world_model_write_blocked() -> dict[str, bool]:
    old_env = _set_env(mode="no_world_model")
    os.environ["TYCHO_MAX_TOOL_STEPS"] = "3"
    try:
        from tycho.agent.agent import TychoAgent

        class StubTycho(TychoAgent):
            def __init__(self):
                super().__init__()
                self.calls_seen = 0

            def _chat(self, call_type, max_tokens=None):
                self.calls_seen += 1
                if self.calls_seen == 1:
                    return {"text": "invalid model edit and stale action", "tool_calls": [
                        {"id": "w1", "name": "write_file",
                         "input": {"path": "world_model.py", "content": "BAD = True\n"}},
                        {"id": "a1", "name": "take_action", "input": {"action": "ACTION1"}},
                    ]}
                return {"text": "commit after block", "tool_calls": [
                    {"id": "a2", "name": "take_action", "input": {"action": "ACTION2"}},
                ]}

        with TemporaryDirectory() as td:
            agent = StubTycho()
            agent.reset("no_wm_block", [GameAction.ACTION1, GameAction.ACTION2], ws_root=td)
            choice = agent.choose_action(
                [],
                _frame([[[0, 0], [1, 1]]], 0),
                [GameAction.ACTION1, GameAction.ACTION2],
            )
            trace = choice.reasoning.get("tool_trace") or []
            tool_text = "\n".join(
                result.get("output", "")
                for msg in agent.history if msg.get("role") == "tool"
                for result in msg.get("results", [])
            )
            return {
                "blocked_message_is_no_wm_specific": (
                    "does not use executable world-model files" in tool_text
                    and "builder-owned" not in tool_text
                ),
                "world_model_not_created": not (agent.ws.dir / "world_model.py").exists(),
                "trace_block_reason_is_no_wm_specific": any(
                    t.get("tool") == "write_file"
                    and t.get("blocked") == "world_model_unavailable"
                    for t in trace
                ),
                "same_batch_action_deferred": any(
                    t.get("tool") == "take_action"
                    and t.get("args", {}).get("action") == "ACTION1"
                    and t.get("deferred") is True
                    for t in trace
                ),
                "retry_committed": choice.action.name == "ACTION2",
            }
    finally:
        _restore_env(old_env)


def test_no_world_model_world_model_write_blocked_before_action():
    checks = _check_no_world_model_write_blocked()
    assert all(checks.values()), checks


def _check_invalid_final_commit_does_not_record_committed_unavailable_action() -> dict[str, bool]:
    old_env = _set_env()
    os.environ["TYCHO_MAX_TOOL_STEPS"] = "1"
    try:
        from tycho.agent.agent import TychoAgent

        class StubTycho(TychoAgent):
            def _chat(self, call_type, max_tokens=None):
                if call_type == "freeform_commit":
                    return {"text": "bad final", "tool_calls": [
                        {"id": "bad_final", "name": "take_action", "input": {"action": "ACTION5"}},
                    ]}
                return {"text": "inspect", "tool_calls": [
                    {"id": "v1", "name": "set_verbosity", "input": {"grid": "off"}},
                ]}

        with TemporaryDirectory() as td:
            agent = StubTycho()
            agent.reset("invalid_final_commit", [GameAction.ACTION1, GameAction.ACTION2], ws_root=td)
            choice = agent.choose_action(
                [],
                _frame([[[0, 0], [1, 1]]], 0),
                [GameAction.ACTION1, GameAction.ACTION2],
            )
            trace = choice.reasoning.get("tool_trace") or []
            committed = [
                t for t in trace
                if t.get("tool") == "take_action" and t.get("committed")
            ]
            return {
                "falls_back_to_default_after_invalid_final": choice.action.name == "ACTION1",
                "invalid_final_marked": any(
                    t.get("tool") == "take_action"
                    and t.get("args", {}).get("action") == "ACTION5"
                    and t.get("invalid") is True
                    and t.get("final_commit") is True
                    for t in trace
                ),
                "no_unavailable_action_committed": not committed,
            }
    finally:
        _restore_env(old_env)


def test_invalid_final_commit_does_not_record_committed_unavailable_action():
    checks = _check_invalid_final_commit_does_not_record_committed_unavailable_action()
    assert all(checks.values()), checks


def _check_invalid_final_action6_prefers_non_click_fallback() -> dict[str, bool]:
    old_env = _set_env()
    os.environ["TYCHO_MAX_TOOL_STEPS"] = "1"
    try:
        from tycho.agent.agent import TychoAgent

        class StubTycho(TychoAgent):
            def _chat(self, call_type, max_tokens=None):
                if call_type == "freeform_commit":
                    return {"text": "bad final click", "tool_calls": [
                        {"id": "bad_click", "name": "take_action", "input": {"action": "ACTION6"}},
                    ]}
                return {"text": "inspect", "tool_calls": [
                    {"id": "v1", "name": "set_verbosity", "input": {"grid": "off"}},
                ]}

        with TemporaryDirectory() as td:
            agent = StubTycho()
            agent.reset("invalid_final_click", [GameAction.ACTION6, GameAction.ACTION1], ws_root=td)
            choice = agent.choose_action(
                [],
                _frame([[[0, 0], [1, 1]]], 0),
                [GameAction.ACTION6, GameAction.ACTION1],
            )
            trace = choice.reasoning.get("tool_trace") or []
            return {
                "invalid_final_marked": any(
                    t.get("tool") == "take_action"
                    and t.get("args", {}).get("action") == "ACTION6"
                    and t.get("invalid") is True
                    and t.get("final_commit") is True
                    for t in trace
                ),
                "fallback_prefers_non_click": choice.action.name == "ACTION1",
                "no_center_click": getattr(choice, "x", None) is None and getattr(choice, "y", None) is None,
            }
    finally:
        _restore_env(old_env)


def test_invalid_final_action6_prefers_non_click_fallback():
    checks = _check_invalid_final_action6_prefers_non_click_fallback()
    assert all(checks.values()), checks


def main() -> int:
    ok = True
    checks = _check_planner_same_batch_defers_action()
    print("=== planner_same_batch ===")
    for name, passed in checks.items():
        print(f"  {name}: {passed}")
        ok = ok and passed
    checks = _check_tool_cap_gets_final_commit_pass()
    print("=== tool_cap_final_commit ===")
    for name, passed in checks.items():
        print(f"  {name}: {passed}")
        ok = ok and passed
    checks = _check_invalid_take_action_reprompts()
    print("=== invalid_take_action_retry ===")
    for name, passed in checks.items():
        print(f"  {name}: {passed}")
        ok = ok and passed
    checks = _check_invalid_action6_coords_reprompt()
    print("=== invalid_action6_retry ===")
    for name, passed in checks.items():
        print(f"  {name}: {passed}")
        ok = ok and passed
    checks = _check_builder_owned_world_model_write_blocked()
    print("=== builder_owned_world_model_block ===")
    for name, passed in checks.items():
        print(f"  {name}: {passed}")
        ok = ok and passed
    checks = _check_invalid_final_commit_does_not_record_committed_unavailable_action()
    print("=== invalid_final_commit ===")
    for name, passed in checks.items():
        print(f"  {name}: {passed}")
        ok = ok and passed
    checks = _check_invalid_final_action6_prefers_non_click_fallback()
    print("=== invalid_final_action6 ===")
    for name, passed in checks.items():
        print(f"  {name}: {passed}")
        ok = ok and passed
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
