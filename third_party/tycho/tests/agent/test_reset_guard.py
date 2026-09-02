from arcengine import GameAction

from tycho.agent.agent import TychoAgent


def test_consecutive_actor_reset_is_invalid():
    agent = TychoAgent.__new__(TychoAgent)
    agent._last_committed_control = "RESET"

    choice, trace_args, msg = agent._prepare_take_action(
        {"action": "RESET"},
        [GameAction.RESET, GameAction.ACTION1],
    )

    assert choice is None
    assert trace_args is None
    assert "previous committed environment control" in msg
    assert "No action was committed" in msg


def test_reset_is_valid_if_previous_control_was_not_reset():
    agent = TychoAgent.__new__(TychoAgent)
    agent._last_committed_control = "ACTION1"

    choice, trace_args, msg = agent._prepare_take_action(
        {"action": "RESET"},
        [GameAction.RESET, GameAction.ACTION1],
    )

    assert msg == ""
    assert choice is not None
    assert choice.action is GameAction.RESET
    assert trace_args == {"action": "RESET"}
