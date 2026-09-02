from __future__ import annotations

import os
import json
import pytest

from tycho.harness import run_status
from tycho.harness.run_status import GameStatusStore, WorkerLeaseError, collect_run_status


def test_status_store_tracks_attempt_actions_and_llm_liveness(tmp_path) -> None:
    store = GameStatusStore(tmp_path / "resume" / "game01")
    store.claim(game="game01", attempt=1, resume=False)
    store.note_llm_started()
    store.note_llm_finished(response="a useful response")
    store.note_action(
        action_count=1,
        action="ACTION1",
        level=0,
        level_action_count=1,
        levels_completed=0,
        rhae=0.0,
    )

    status = store.read()
    assert status["state"] == "running"
    assert status["action_count"] == 1
    assert status["last_llm_response"] == "a useful response"
    assert [event["type"] for event in store.events()] == [
        "worker_started",
        "llm_started",
        "llm_finished",
        "action_committed",
    ]


def test_status_store_accumulates_call_cost_and_action_reconciles_total(tmp_path) -> None:
    store = GameStatusStore(tmp_path / "status" / "game01")
    store.claim(game="game01", attempt=1, resume=False)
    store.note_llm_finished(
        response="first",
        usage={"tokens_in": 100, "tokens_out": 20},
        model="test-model",
        cost_usd=1.25,
    )
    store.note_llm_finished(response="second", cost_usd=0.75)
    assert store.read()["inference_cost_usd"] == 2.0

    store.note_action(
        action_count=1,
        action="ACTION1",
        level=0,
        level_action_count=1,
        levels_completed=0,
        rhae=0.0,
        inference_budget={
            "cost_usd": 2.1,
            "level_cost_usd": 2.1,
            "max_cost_per_game_usd": 100.0,
            "max_cost_per_level_usd": 0.0,
        },
    )
    status = store.read()
    assert status["inference_cost_usd"] == 2.1
    assert status["max_inference_cost_per_game_usd"] == 100.0
    assert store.events()[1]["cost_usd"] == 1.25


def test_actionless_resume_count_increments_only_without_a_commit(tmp_path) -> None:
    store = GameStatusStore(tmp_path / "resume" / "game01")
    store.claim(game="game01", attempt=1, resume=False)
    store.update(state="error", pid=None)
    store.claim(game="game01", attempt=2, resume=True)
    assert store.read()["actionless_resume_count"] == 1

    store.note_action(
        action_count=1,
        action="ACTION2",
        level=0,
        level_action_count=1,
        levels_completed=0,
        rhae=0.0,
    )
    store.update(state="error", pid=None)
    store.claim(game="game01", attempt=3, resume=True)
    assert store.read()["actionless_resume_count"] == 1


def test_live_worker_lease_prevents_duplicate_process(tmp_path, monkeypatch) -> None:
    store = GameStatusStore(tmp_path / "resume" / "game01")
    store.update(state="running", pid=os.getpid(), host=run_status._node_token())
    other = GameStatusStore(store.dir)
    # The same process may reclaim its own lease (used by tests/worker setup).
    other.claim(game="game01", attempt=2, resume=True)

    other.update(state="running", pid=os.getpid() + 1, host=run_status._node_token())
    monkeypatch.setattr(run_status, "_pid_alive", lambda pid: True)
    with pytest.raises(WorkerLeaseError, match="already owned"):
        other.claim(game="game01", attempt=3, resume=True)


def test_recent_remote_worker_lease_is_not_stolen(tmp_path) -> None:
    store = GameStatusStore(tmp_path / "resume" / "game01")
    store.update(
        state="in_llm",
        pid=123,
        host="another-host",
        heartbeat_at=run_status.utc_now(),
    )
    with pytest.raises(WorkerLeaseError, match="already owned"):
        store.claim(game="game01", attempt=2, resume=True)


def test_collect_run_status_reports_floor_finished_mean_and_resume_health(tmp_path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "run_spec.json").write_text(json.dumps({
        "policy": {
            "games": {"game01-full": [10, 20], "bp35-full": [30]},
            "config": {"model": {"LLM_MODEL": "opus"},
                       "orchestration": {"TYCHO_MODE": "single"},
                       "reasoning": {"TYCHO_MAX_INFERENCE_COST_PER_GAME": "100"}},
        }
    }))
    (run / "game_game01.json").write_text(json.dumps({
        "env_score": 80.0, "levels_completed": 2, "n_levels": 2,
        "total_actions_including_unfinished": 12,
    }))
    (run / "manifest.json").write_text(json.dumps({
        "per_game": {
            "game01": {"rhae": 80.0, "levels": 2, "total_actions_including_unfinished": 12}
        }
    }))
    GameStatusStore(run / "status" / "game01").update(
        state="finished", current_rhae=80.0, action_count=12, levels_completed=2,
    )
    bp_store = GameStatusStore(run / "status" / "bp35")
    bp_store.claim(game="bp35", attempt=1, resume=False)
    bp_store.note_llm_finished(response="working", cost_usd=25.0)
    bp_store.note_action(
        action_count=7, action="ACTION2", level=0, level_action_count=7,
        levels_completed=0, rhae=20.0,
        inference_budget={"cost_usd": 25.0, "level_cost_usd": 25.0,
                          "max_cost_per_game_usd": 100.0,
                          "max_cost_per_level_usd": 0.0},
    )
    bp_store.update(actionless_resume_count=2)

    snapshot = collect_run_status(run)

    assert snapshot["floor_rhae"] == 50.0
    assert snapshot["mean_finished_rhae"] == 80.0
    assert snapshot["n_finished"] == 1
    game01 = next(game for game in snapshot["games"] if game["game"] == "game01")
    assert game01["eta_to_inference_game_cap_s"] is None
    assert game01["inference_game_cap_pct"] is None
    bp35 = next(game for game in snapshot["games"] if game["game"] == "bp35")
    assert bp35["actionless_resume_count"] == 2
    assert bp35["level_completion_pct"] == 0.0
    assert bp35["inference_cost_usd"] == 25.0
    assert bp35["inference_game_cap_usd"] == 100.0
    assert bp35["inference_game_cap_pct"] == 25.0
    assert bp35["inference_cost_per_hour"] > 0
    assert bp35["eta_to_inference_game_cap_s"] is not None
    assert bp35["actions_per_hour"] > 0
    assert snapshot["total_actions"] == 19
    assert snapshot["average_actions_per_hour"] > 0
    assert sum(point["actions"] for point in snapshot["action_rate_series"]) == 1
    assert snapshot["action_rate_bin_s"] >= 900
