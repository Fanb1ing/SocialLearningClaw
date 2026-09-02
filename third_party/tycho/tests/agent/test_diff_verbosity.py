from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from tycho.agent.agent import TychoAgent


def _agent(diff: str) -> TychoAgent:
    agent = TychoAgent.__new__(TychoAgent)
    agent.show_grid = False
    agent.show_diff = diff != "off"
    agent.diff_mode = diff
    agent.level = 0
    agent.turn_in_level = 1
    agent.vision = False
    agent._last_action = "ACTION1"
    agent._mode_spec = SimpleNamespace(wm_variant="none")
    agent.ws = SimpleNamespace(read_file=lambda *args, **kwargs: (
        "684 cells changed; region [rows 3-62, cols 0-63]\n"
        "  0->5 rows 3-8, cols 2-20\n"
        "  5->0 rows 9-14, cols 2-20"
    ))
    return agent


def test_diff_summary_inlines_only_existing_deterministic_header() -> None:
    agent = _agent("summary")
    message = agent._frame_message(
        np.zeros((2, 2), dtype=np.int16), "NOT_FINISHED", ["ACTION1"], None
    )

    assert "684 cells changed; region [rows 3-62, cols 0-63]" in message
    assert "0->5 rows" not in message
    assert "5->0 rows" not in message


def test_set_verbosity_supports_exact_summary_and_off() -> None:
    agent = _agent("on")

    assert "diff mode=summary" in agent._set_verbosity({"diff": "summary"})
    assert agent.show_diff is True
    assert agent.diff_mode == "summary"
    assert "diff mode=off" in agent._set_verbosity({"diff": "off"})
    assert agent.show_diff is False
    assert "must be 'on', 'summary', or 'off'" in agent._set_verbosity({"diff": "semantic"})
