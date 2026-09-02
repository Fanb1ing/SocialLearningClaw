from types import SimpleNamespace

from tycho.agent.agent import TychoAgent
from tycho.workspace.agent_tools import tool_specs


def test_take_action_schema_is_constrained_without_mutating_base_specs() -> None:
    agent = object.__new__(TychoAgent)
    agent._tools_spec = tool_specs("generic", include_builder=True)

    active = agent._tools_with_available_actions(["RESET", "ACTION1", "ACTION6"])
    active_take = next(tool for tool in active if tool["name"] == "take_action")
    base_take = next(tool for tool in agent._tools_spec if tool["name"] == "take_action")

    assert active_take["schema"]["properties"]["action"]["enum"] == [
        "RESET", "ACTION1", "ACTION6"
    ]
    assert "enum" not in base_take["schema"]["properties"]["action"]


def test_builder_tool_surface_has_function_editor_but_not_actor_verbosity() -> None:
    tools = tool_specs(
        "generic",
        include_edit_func=True,
        include_take_action=False,
        include_set_verbosity=False,
    )
    names = [tool["name"] for tool in tools]

    assert "edit_function" in names
    assert "set_verbosity" not in names
    assert "take_action" not in names


def test_second_same_turn_compaction_is_allowed_only_after_history_grows() -> None:
    agent = object.__new__(TychoAgent)
    agent.context_emergency_compaction = True
    agent.context_emergency_soft_tokens = 220_000
    agent._last_prompt_tokens = 230_000
    agent.level = 1
    agent.turn_in_level = 12
    agent._last_context_compaction_at = (1, 12)
    agent._context_compaction_streak = 1
    agent.history = [{"role": "user", "content": "compact seed"}]
    agent._last_context_compaction_history_len = len(agent.history)

    assert not agent._should_emergency_compact_before_call()

    agent.history.append({"role": "assistant", "content": "new reasoning/tool call"})
    assert agent._should_emergency_compact_before_call()
