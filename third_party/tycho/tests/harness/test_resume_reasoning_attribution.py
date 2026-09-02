from __future__ import annotations

from arcengine import GameAction

from tycho.harness.agent import ActionChoice
from tycho.harness.harness import _reasoning_for_replayed_choice


def _reasoning(action: str, *, row=None, col=None) -> dict:
    args = {"action": action}
    if row is not None:
        args["row"] = row
    if col is not None:
        args["col"] = col
    return {
        "t": 7,
        "tool_trace": [
            {"tool": "take_action", "args": args, "committed": True},
        ],
    }


def test_resume_reuses_prior_reasoning_when_action_matches():
    prior = _reasoning("ACTION2")
    choice = ActionChoice(GameAction.ACTION2)

    assert _reasoning_for_replayed_choice(choice, prior, 7) is prior


def test_resume_reuses_prior_reasoning_when_click_coordinates_match():
    prior = _reasoning("ACTION6", row=11, col=22)
    choice = ActionChoice(GameAction.ACTION6, x=22, y=11)

    assert _reasoning_for_replayed_choice(choice, prior, 7) is prior


def test_resume_omits_stale_reasoning_when_action_mismatches():
    prior = _reasoning("ACTION1")
    choice = ActionChoice(GameAction.ACTION2)

    got = _reasoning_for_replayed_choice(choice, prior, 7)

    assert got is not prior
    assert got["resume_reasoning_omitted"] is True
    assert got["prior_committed_action"] == {"action": "ACTION1"}
    assert got["replay_action"] == {"action": "ACTION2"}
    assert "tool_trace" not in got


def test_resume_omits_stale_reasoning_when_click_coordinates_mismatch():
    prior = _reasoning("ACTION6", row=11, col=22)
    choice = ActionChoice(GameAction.ACTION6, x=23, y=11)

    got = _reasoning_for_replayed_choice(choice, prior, 7)

    assert got["resume_reasoning_omitted"] is True
    assert got["prior_committed_action"] == {"action": "ACTION6", "x": 22, "y": 11}
    assert got["replay_action"] == {"action": "ACTION6", "x": 23, "y": 11}
