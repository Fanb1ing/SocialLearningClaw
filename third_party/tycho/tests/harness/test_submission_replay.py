from __future__ import annotations

import hashlib
import json

import pytest
from arcengine import GameState

from tycho.harness import submission_replay as sr
from tycho.harness.inference_budget import cap_trace_by_inference_cost


def _grid(v: int) -> list[list[int]]:
    return [[v for _ in range(64)] for _ in range(64)]


def _record(game_id: str = "aa00-version") -> dict:
    return {
        "game_id": game_id,
        "n_levels": 1,
        "baselines": [1],
        "levels_completed": 1,
        "total_actions": 1,
        "resets": 0,
        "final_state": "GameState.WIN",
        "env_score": 100.0,
        "level_results": [{"level_index": 1, "completed": True, "actions_taken": 1, "baseline_actions": 1}],
        "wall_clock_s": 1.0,
        "truncated_levels": [],
        "error": None,
        "stop_reason": "win",
        "trace": [{
            "turn": 1,
            "action": "ACTION1",
            "x": None,
            "y": None,
            "state": "GameState.WIN",
            "levels_completed": 1,
            "frame_changed": True,
            "frame_key": 123,
            "frame": _grid(1),
            "reasoning": {"t": 0},
        }],
    }


def test_export_trace_from_run_dir(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({
        "seed": 0,
        "model": "test-model",
        "git_version": "abc123",
        "requested_games": ["aa00"],
    }))
    (run / "game_aa00.json").write_text(json.dumps(_record()))

    out = tmp_path / "trace.json"
    doc = sr.export_trace(run, out)

    assert out.exists()
    assert doc["schema"] == "tycho.arc_agi_3.submission_trace"
    assert doc["source_manifests"][0]["model"] == "test-model"
    assert doc["games"][0]["short_id"] == "aa00"
    action = doc["games"][0]["actions"][0]
    assert action["action"] == "ACTION1"
    assert action["expected_frame_sha256"] == sr.grid_sha256(_grid(1))


def test_export_trace_can_combine_disjoint_run_dirs(tmp_path):
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    run_a.mkdir()
    run_b.mkdir()
    (run_a / "manifest.json").write_text(json.dumps({"seed": 3, "model": "a"}))
    (run_b / "manifest.json").write_text(json.dumps({"seed": 3, "model": "b"}))
    (run_a / "game_aa00.json").write_text(json.dumps(_record("aa00-version")))
    (run_b / "game_bb00.json").write_text(json.dumps(_record("bb00-version")))

    doc = sr.export_trace([run_a, run_b], tmp_path / "trace.json")

    assert len(doc["source_run_ids"]) == 2
    assert doc["source_run_ids"][0] != doc["source_run_ids"][1]
    assert str(tmp_path) not in json.dumps(doc)
    assert doc["seed"] == 3
    assert [g["short_id"] for g in doc["games"]] == ["aa00", "bb00"]


def test_export_trace_rejects_duplicate_games(tmp_path):
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    run_a.mkdir()
    run_b.mkdir()
    (run_a / "game_aa00.json").write_text(json.dumps(_record("aa00-one")))
    (run_b / "game_aa00.json").write_text(json.dumps(_record("aa00-two")))

    with pytest.raises(ValueError, match="duplicate record"):
        sr.export_trace([run_a, run_b], tmp_path / "trace.json")


def test_export_trace_rejects_level_regression(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    rec = _record()
    rec["trace"] = [
        {
            **rec["trace"][0],
            "turn": 1,
            "action": "RESET",
            "state": "GameState.NOT_FINISHED",
            "levels_completed": 3,
        },
        {
            **rec["trace"][0],
            "turn": 2,
            "action": "RESET",
            "state": "GameState.NOT_FINISHED",
            "levels_completed": 0,
        },
    ]
    (run / "game_aa00.json").write_text(json.dumps(rec))

    with pytest.raises(ValueError, match="levels_completed decreases"):
        sr.export_trace(run, tmp_path / "trace.json")


def _costed_step(turn: int, levels_completed: int, fresh_tokens: int) -> dict:
    return {
        **_record()["trace"][0],
        "turn": turn,
        "state": "GameState.NOT_FINISHED",
        "levels_completed": levels_completed,
        "reasoning": {"llm_calls": [{
            "model": "claude-opus-4-8",
            "tokens_in": fresh_tokens,
            "tokens_out": 0,
            "cache_read": 0,
            "cache_write": 0,
        }]},
    }


def test_trace_game_cost_cap_keeps_crossing_action_then_stops() -> None:
    trace = [_costed_step(i, 0, 60_000_000) for i in range(1, 5)]  # $300 per action

    capped = cap_trace_by_inference_cost(trace, max_cost_per_game_usd=750)

    assert len(capped.trace) == 3
    assert capped.cost_usd == 900
    assert capped.cap_triggered
    assert capped.stop_reason == "inference_cost_game_limit"


def test_trace_level_cap_resets_on_completing_action() -> None:
    trace = [
        _costed_step(1, 0, 60_000_000),  # $300 on level 0
        _costed_step(2, 1, 60_000_000),  # completion crosses $500, but resets the level meter
        _costed_step(3, 1, 60_000_000),
        _costed_step(4, 1, 60_000_000),  # level 1 reaches $600 and stops
        _costed_step(5, 1, 60_000_000),
    ]

    capped = cap_trace_by_inference_cost(trace, max_cost_per_level_usd=500)

    assert len(capped.trace) == 4
    assert capped.stop_reason == "inference_cost_level_limit"


def test_export_trace_applies_cost_cap_and_recomputes_metadata(tmp_path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    rec = _record()
    rec["n_levels"] = 2
    rec["baselines"] = [2, 2]
    rec["trace"] = [
        _costed_step(1, 0, 60_000_000),
        _costed_step(2, 1, 60_000_000),
        _costed_step(3, 1, 60_000_000),
        _costed_step(4, 2, 60_000_000),
    ]
    (run / "game_aa00.json").write_text(json.dumps(rec))

    doc = sr.export_trace(
        run,
        tmp_path / "trace.json",
        max_inference_cost_per_game_usd=750,
    )

    game = doc["games"][0]
    assert len(game["actions"]) == 3
    assert game["levels_completed"] == 1
    assert game["stop_reason"] == "inference_cost_game_limit"
    assert game["inference_budget"]["cap_triggered"] is True


def test_materialize_budgeted_run_writes_coherent_record_and_manifest(tmp_path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({
        "seed": 0,
        "model": "claude-opus-4-8",
        "requested_games": ["aa00"],
    }))
    rec = _record()
    rec["n_levels"] = 2
    rec["baselines"] = [2, 2]
    rec["trace"] = [
        _costed_step(1, 0, 60_000_000),
        _costed_step(2, 1, 60_000_000),
        _costed_step(3, 1, 60_000_000),
        _costed_step(4, 2, 60_000_000),
    ]
    (run / "game_aa00.json").write_text(json.dumps(rec))

    out = tmp_path / "capped"
    manifest = sr.materialize_budgeted_run(
        run,
        out,
        max_inference_cost_per_game_usd=750,
    )

    capped = json.loads((out / "game_aa00.json").read_text())
    assert len(capped["trace"]) == 3
    assert capped["levels_completed"] == 1
    assert capped["completed_level_actions"] == 2
    assert capped["unfinished_level_actions"] == 1
    assert capped["total_actions_including_unfinished"] == 3
    assert capped["stop_reason"] == "inference_cost_game_limit"
    assert manifest["per_game"]["aa00"]["total_actions_including_unfinished"] == 3
    assert manifest["workspace"]["status"] == "not_materialized"
    assert len(manifest["source_run_id"]) == 64
    assert str(tmp_path) not in json.dumps(manifest)
    assert not (out / "ws").exists()


def test_materialize_recomputes_unchanged_record_accounting(tmp_path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({
        "model": "claude-opus-4-8",
        "per_game": {"aa00": {"wm_activity": {"wm_correct_levels": 1}}},
    }))
    rec = _record()
    rec["trace"][0]["action"] = "RESET"
    rec["resets"] = 0
    source = run / "game_aa00.json"
    source.write_text(json.dumps(rec))

    out = tmp_path / "capped"
    sr.materialize_budgeted_run(
        run,
        out,
        max_inference_cost_per_game_usd=750,
    )

    capped = json.loads((out / source.name).read_text())
    assert capped["resets"] == 1
    assert capped["action_accounting"] == "all_in_play_controls_including_reset"
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["per_game"]["aa00"]["wm_activity"]["wm_correct_levels"] == 1


class _Frame:
    def __init__(self, grid, *, state=GameState.NOT_FINISHED, levels_completed=0, frames=None):
        self.frame = frames if frames is not None else [grid]
        self.state = state
        self.levels_completed = levels_completed


class _Env:
    def __init__(self, frames):
        self.observation_space = frames[0]
        self._frames = list(frames[1:])
        self.steps = []

    def reset(self):
        raise AssertionError("replay should use observation_space and avoid an extra reset")

    def step(self, action, data=None, reasoning=None):
        self.steps.append((action.name, data, reasoning))
        return self._frames.pop(0)


class _Arcade:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.card_id = None
        self.made = []
        self.closed = []
        _Arcade.instances.append(self)

    def create_scorecard(self, **kwargs):
        self.card_id = "scorecard-test"
        self.scorecard_kwargs = kwargs
        return self.card_id

    def make(self, game_id, seed=0, scorecard_id=None):
        self.made.append((game_id, seed, scorecard_id))
        return _Env([
            _Frame(_grid(0)),
            _Frame(_grid(1), state=GameState.WIN, levels_completed=1),
        ])

    def close_scorecard(self, scorecard_id=None):
        self.closed.append(scorecard_id)

        class _Scorecard:
            score = 100.0

            def model_dump(self, mode="json"):
                return {"score": 100.0, "card_id": scorecard_id}

        return _Scorecard()


def _write_trace(tmp_path, *, expected_grid=None):
    action = _record()["trace"][0]
    if expected_grid is not None:
        action["frame"] = expected_grid
    doc = {
        "schema": "tycho.arc_agi_3.submission_trace",
        "schema_version": sr.TRACE_SCHEMA_VERSION,
        "created_at": "2026-01-01T00:00:00+00:00",
        "source_run_dir": "run",
        "source_manifest": {"model": "test"},
        "seed": 7,
        "games": [sr.GameTrace.from_record({**_record(), "trace": [action]}).to_json()],
    }
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(doc))
    return path


def test_replay_trace_validates_and_closes_scorecard(tmp_path, monkeypatch):
    _Arcade.instances.clear()
    monkeypatch.setattr(sr, "Arcade", _Arcade)
    trace = _write_trace(tmp_path)

    manifest = sr.replay_trace(trace, tmp_path / "replay", mode="offline")

    arc = _Arcade.instances[-1]
    assert arc.kwargs["operation_mode"].value == "offline"
    assert arc.made == [("aa00-version", 7, "scorecard-test")]
    assert arc.closed == ["scorecard-test"]
    assert manifest["closed"] is True
    assert manifest["scorecard_score"] == 100.0
    assert manifest["completed_games"] == ["aa00"]
    assert manifest["trace_sha256"] == hashlib.sha256(trace.read_bytes()).hexdigest()
    assert str(tmp_path) not in json.dumps(manifest)


def test_replay_trace_fails_closed_on_frame_mismatch(tmp_path, monkeypatch):
    _Arcade.instances.clear()
    monkeypatch.setattr(sr, "Arcade", _Arcade)
    trace = _write_trace(tmp_path, expected_grid=_grid(9))

    with pytest.raises(RuntimeError, match="frame hash mismatch"):
        sr.replay_trace(trace, tmp_path / "replay", mode="offline")

    manifest = json.loads((tmp_path / "replay" / "replay_manifest.json").read_text())
    assert manifest["failed"]["game"] == "aa00"
    assert manifest["failed"]["error"] == "RuntimeError"


def test_replay_accepts_game_over_trace_grid_from_any_returned_frame(tmp_path, monkeypatch):
    class _GameOverArcade(_Arcade):
        def make(self, game_id, seed=0, scorecard_id=None):
            self.made.append((game_id, seed, scorecard_id))
            return _Env([
                _Frame(_grid(0)),
                _Frame(
                    _grid(1),
                    state=GameState.GAME_OVER,
                    levels_completed=0,
                    frames=[_grid(2), _grid(3)],
                ),
            ])

    _GameOverArcade.instances.clear()
    monkeypatch.setattr(sr, "Arcade", _GameOverArcade)
    action = _record()["trace"][0]
    action["state"] = "GameState.GAME_OVER"
    action["levels_completed"] = 0
    action["frame"] = _grid(3)
    doc = {
        "schema": "tycho.arc_agi_3.submission_trace",
        "schema_version": sr.TRACE_SCHEMA_VERSION,
        "created_at": "2026-01-01T00:00:00+00:00",
        "source_manifests": [{"model": "test"}],
        "seed": 7,
        "games": [sr.GameTrace.from_record({**_record(), "trace": [action]}).to_json()],
    }
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps(doc))

    manifest = sr.replay_trace(trace, tmp_path / "replay", mode="offline")

    assert manifest["closed"] is True
    assert manifest["completed_games"] == ["aa00"]


def test_competition_replay_requires_api_key(tmp_path, monkeypatch):
    _Arcade.instances.clear()
    monkeypatch.setattr(sr, "Arcade", _Arcade)
    trace = _write_trace(tmp_path)

    with pytest.raises(ValueError, match="API key is required"):
        sr.replay_trace(trace, tmp_path / "replay", competition=True, api_key="")
