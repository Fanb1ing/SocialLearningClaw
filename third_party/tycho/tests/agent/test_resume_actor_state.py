from __future__ import annotations

from types import SimpleNamespace

from tycho.agent.agent import TychoAgent


def _bare_agent() -> TychoAgent:
    agent = TychoAgent.__new__(TychoAgent)
    agent.history = [{"role": "user", "content": "frame"}]
    agent.calls = 3
    agent.t = 4
    agent.level = 1
    agent.turn_in_level = 2
    agent.prev_levels_done = 1
    agent.builder_invocations = 0
    agent.token_stats = {}
    agent._budget_level = 1
    agent._budget_level_start_usd = 0.0
    agent._done_reason = None
    agent._pending_frame_boundary = "attempt_restart"
    agent.show_grid = False
    agent.show_diff = True
    agent.diff_mode = "summary"
    agent.ws = SimpleNamespace(prev_grid=None, turns=[])
    agent.dispatcher = None
    agent.reset_reasoning_chain = lambda: None
    return agent


def test_resume_restores_actor_verbosity() -> None:
    original = _bare_agent()
    original.history.append({
        "role": "assistant",
        "content": "",
        "reasoning_items": [{
            "type": "reasoning",
            "id": "rs_resume",
            "encrypted_content": "opaque",
        }],
    })
    state = original.snapshot_state()
    resumed = _bare_agent()
    resumed.show_grid = True
    resumed.show_diff = False
    resumed.diff_mode = "off"

    resumed.restore_state(state)

    assert resumed.show_grid is False
    assert resumed.show_diff is True
    assert resumed.diff_mode == "summary"
    assert resumed._pending_frame_boundary == "attempt_restart"
    assert resumed.history[-1]["reasoning_items"][0]["id"] == "rs_resume"


def test_reset_callbacks_mark_next_frame_as_restarted_attempt() -> None:
    for callback in ("note_actor_reset", "note_external_reset"):
        agent = _bare_agent()
        agent._pending_frame_boundary = None

        getattr(agent, callback)()

        assert agent._pending_frame_boundary == "attempt_restart"
        assert agent.turn_in_level == 0
