from tycho.harness.harness import _build_record
from tycho.harness.run_progress import _outcome
from tycho.harness.scoring import LevelResult


class _Frame:
    state = "GameState.NOT_FINISHED"


def test_unfinished_current_level_actions_are_reported_separately():
    rec = _build_record(
        "game-x",
        3,
        [5, 7, 9],
        [LevelResult(level_index=1, completed=True, actions_taken=10, baseline_actions=5)],
        _Frame(),
        resets=0,
        elapsed_s=1.0,
        truncated=[],
        noop_actions=0,
        distinct_frames=1,
        revisits=0,
        actions_before_first_level=10,
        run_error=None,
        trace=[],
        partial=False,
        current_level_index=2,
        current_level_actions=7,
    )

    assert rec.total_actions == 10
    assert rec.completed_level_actions == 10
    assert rec.unfinished_level_index == 2
    assert rec.unfinished_level_actions == 7
    assert rec.total_actions_including_unfinished == 17

    outcome = _outcome(rec.__dict__)
    assert outcome["status"] == "PARTIAL"
    assert outcome["stall_lvl"] == 2
    assert outcome["stall_actions"] == 7


def test_recorded_truncated_level_is_not_double_counted_as_unfinished():
    rec = _build_record(
        "game-x",
        3,
        [5, 7, 9],
        [
            LevelResult(level_index=1, completed=True, actions_taken=10, baseline_actions=5),
            LevelResult(level_index=2, completed=False, actions_taken=35, baseline_actions=7),
        ],
        _Frame(),
        resets=0,
        elapsed_s=1.0,
        truncated=[2],
        noop_actions=0,
        distinct_frames=1,
        revisits=0,
        actions_before_first_level=10,
        run_error=None,
        trace=[],
        partial=False,
        current_level_index=2,
        current_level_actions=35,
    )

    assert rec.total_actions == 45
    assert rec.unfinished_level_index is None
    assert rec.unfinished_level_actions == 0
    assert rec.total_actions_including_unfinished == 45
