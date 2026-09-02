from __future__ import annotations

from tycho.harness import run_parallel


def test_parallel_arcade_receives_arc_api_key(monkeypatch) -> None:
    captured = {}

    def fake_arcade(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setenv("ARC_API_KEY", "arc-test-key")
    monkeypatch.setattr(run_parallel, "Arcade", fake_arcade)

    mode = object()
    assert run_parallel._new_arcade(mode) is not None
    assert captured["arc_api_key"] == "arc-test-key"
    assert captured["operation_mode"] is mode
    assert captured["environments_dir"] == str(run_parallel.ENV_DIR)
