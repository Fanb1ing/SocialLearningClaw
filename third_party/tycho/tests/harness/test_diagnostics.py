from __future__ import annotations

from types import SimpleNamespace

import pytest

from tycho.harness.diagnostics import classify_failure


@pytest.mark.parametrize("final_state", ["WIN", "GameState.WIN"])
def test_classify_failure_recognizes_serialized_win(final_state: str) -> None:
    record = SimpleNamespace(
        final_state=final_state,
        total_actions=10,
        total_actions_including_unfinished=10,
        noop_actions=0,
        distinct_frames=8,
        revisits=0,
        levels_completed=6,
    )

    assert classify_failure(record) == "solved"
