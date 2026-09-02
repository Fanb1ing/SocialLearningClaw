"""Regression checks for live viewer server data contracts."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from tycho.viewer import serve
from tycho.harness.run_status import GameStatusStore


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj))


def _minimal_env(game_id: str, trace: list[dict]) -> dict:
    return {
        "game_id": game_id,
        "levels_completed": 0,
        "env_score": 0.0,
        "wall_clock_s": 0.0,
        "trace": trace,
        "_slim": 1,
    }


def _check_path_escape_blocked(root: Path) -> dict[str, bool]:
    results = root / "results"
    results.mkdir(parents=True)
    _write_json(root / "game_leak.json", _minimal_env("leak-123", []))
    (root / "ws" / "leak").mkdir(parents=True)

    run = results / "run"
    (run / "ws" / "game01").mkdir(parents=True)
    _write_json(run / "game_game01.json", _minimal_env("game01-abc", []))

    return {
        "game_js_rejects_parent_run": serve._game_js(results, "..", "leak") is None,
        "ws_dir_rejects_parent_run": serve._ws_dir(results, "..", "leak") is None,
        "ws_dir_rejects_ws_parent": serve._ws_dir(results, "run", "..") is None,
        "ws_dir_rejects_parent_game": serve._ws_dir(results, "run", "../game01") is None,
        "valid_ws_still_resolves": serve._ws_dir(results, "run", "game01")
        == (run / "ws" / "game01").resolve(),
    }


def _check_terminal_cache_invalidation(root: Path) -> dict[str, bool]:
    serve._CACHE.clear()
    results = root / "results"
    run = results / "run"
    ws = run / "ws" / "game01" / "level_0"
    ws.mkdir(parents=True)
    trace = [
        {
            "turn": 1,
            "action": "ACTION1",
            "x": None,
            "y": None,
            "state": "GameState.NOT_FINISHED",
            "levels_completed": 1,
            "frame_changed": True,
            "frame": [[0]],
            "reasoning": {"level": 0, "turn_in_level": 1},
        }
    ]
    _write_json(run / "game_game01.json", _minimal_env("game01-abc", trace))

    first = serve._game_js(results, "run", "game01").decode()
    _write_json(ws / "terminal.json", {"terminal_grid": [[9]]})
    second = serve._game_js(results, "run", "game01").decode()

    return {
        "first_load_has_no_terminal": "boundary_terminal_grid" not in first,
        "second_load_sees_new_terminal": '"boundary_terminal_grid": [[9]]' in second,
    }


def _check_manifest_step_counts(root: Path) -> dict[str, bool]:
    serve._MANIFEST_CACHE.clear()
    results = root / "results"
    run = results / "run"
    run.mkdir(parents=True)
    _write_json(run / "manifest.json", {
        "per_game": {
            "game01": {"rhae": 12.5, "levels": 1, "wall_clock_s": 3.0, "total_actions": 2}
        },
    })
    _write_json(run / "game_game01.json", _minimal_env("game01-abc", [
        {"frame": [[0]], "reasoning": {}},
        {"frame": None, "reasoning": {}},
        {"frame": [[1]], "reasoning": {}},
    ]))

    manifest = serve._run_manifest(run)
    row = manifest["games"][0]
    return {
        "manifest_reports_real_steps": row["steps"] == 2,
        "manifest_keeps_rhae": row["rhae"] == 12.5,
    }


def _check_historical_blob_access(root: Path) -> dict[str, bool]:
    ws = root / "run" / "ws" / "game01"
    blobs = ws.parent / ".workspace_blobs"
    blobs.mkdir(parents=True)
    ws.mkdir()
    digest = "a" * 64
    (blobs / digest).write_bytes(b"exact historical bytes")

    return {
        "valid_blob_resolves": serve._ws_blob(ws, digest) == b"exact historical bytes",
        "short_digest_rejected": serve._ws_blob(ws, "a" * 63) is None,
        "traversal_digest_rejected": serve._ws_blob(ws, "../" + digest) is None,
        "uppercase_digest_rejected": serve._ws_blob(ws, digest.upper()) is None,
    }


def test_status_page_uses_structured_worker_telemetry(tmp_path) -> None:
    results = tmp_path / "results"
    run = results / "supervised"
    run.mkdir(parents=True)
    _write_json(run / "run_spec.json", {
        "policy": {
            "games": {"game01-full": [10, 20]},
            "config": {"model": {"LLM_MODEL": "opus"}},
        }
    })
    GameStatusStore(run / "status" / "game01").update(
        state="running",
        current_rhae=25.0,
        action_count=4,
        levels_completed=1,
        last_llm_response="planned ACTION1",
    )

    page = serve._status_page(results).decode()

    assert "Tycho benchmark status" in page
    assert "supervised" in page
    assert "planned ACTION1" in page
    assert "Action age" in page
    assert "LLM age" in page
    assert "Level completion" in page
    assert "ETA action cap" in page
    assert "Game inference cap" in page
    assert "ETA game cap" in page
    assert "data-sortable" in page
    assert "tycho-status-sort-v1" in page
    assert "Run action throughput" in page
    assert "Average actions/hour" in page
    assert "throughput-chart" in page
    assert "overflow-wrap:anywhere" in page
    assert "@media(max-width:800px)" in page


def main() -> int:
    ok = True
    with TemporaryDirectory() as td:
        root = Path(td)
        for group, checks in (
            ("path_escape", _check_path_escape_blocked(root / "a")),
            ("terminal_cache", _check_terminal_cache_invalidation(root / "b")),
            ("manifest_steps", _check_manifest_step_counts(root / "c")),
            ("historical_blob", _check_historical_blob_access(root / "d")),
        ):
            print(f"=== {group} ===")
            for name, passed in checks.items():
                print(f"  {name}: {passed}")
                ok = ok and passed
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
