from __future__ import annotations

from socialclaw.run_arc_v3 import REQUIRED_PACKAGES, _prepare_tycho_args, check_runtime


def test_runtime_contract_accepts_exact_pinned_environment() -> None:
    contract = check_runtime(
        python_version=(3, 12, 7),
        package_version=lambda name: REQUIRED_PACKAGES[name],
    )
    assert contract.ok
    assert contract.errors == ()


def test_runtime_contract_rejects_current_class_of_version_drift() -> None:
    installed = {**REQUIRED_PACKAGES, "arc-agi": "0.9.8"}
    contract = check_runtime(
        python_version=(3, 11, 3),
        package_version=lambda name: installed[name],
    )
    assert not contract.ok
    assert any("Python 3.11" in error for error in contract.errors)
    assert any("arc-agi==0.9.9" in error for error in contract.errors)


def test_runner_defaults_to_efps_orchestrator(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("SC_V3_RUN_ID", raising=False)
    monkeypatch.delenv("TYCHO_MODE", raising=False)
    monkeypatch.delenv("TYCHO_ENVIRONMENTS_DIR", raising=False)

    args = _prepare_tycho_args(["--games", "cd82", "--out-dir", str(tmp_path / "run")])

    assert args[-2:] == ["--approach", "tycho_efps"]
    assert __import__("os").environ["TYCHO_MODE"] == "orchestrator"
    assert __import__("os").environ["SC_V3_RUN_ID"].startswith("v3_")
    assert __import__("os").environ["TYCHO_ENVIRONMENTS_DIR"].endswith(
        "third_party/arc_agi3_games"
    )


def test_runner_maps_v2_openrouter_options_without_forwarding_secrets(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BACKEND", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-for-test")

    args = _prepare_tycho_args([
        "--model", "provider/model",
        "--out-dir", str(tmp_path / "run"),
    ])

    environ = __import__("os").environ
    assert "--model" not in args
    assert "secret-for-test" not in args
    assert environ["LLM_MODEL"] == "provider/model"
    assert environ["LLM_BASE_URL"] == "https://openrouter.ai/api/v1"
    assert environ["LLM_API_KEY"] == "secret-for-test"
    assert environ["LLM_BACKEND"] == "openai"
