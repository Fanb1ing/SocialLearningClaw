from __future__ import annotations

import json
from pathlib import Path

import pytest

from tycho.workspace import sandbox as sandbox_module
from tycho.workspace.sandbox import PythonSandbox, SandboxError, SandboxResult


def test_auto_runtime_prefers_finch_on_macos(monkeypatch) -> None:
    monkeypatch.setattr(sandbox_module.sys, "platform", "darwin")
    monkeypatch.setattr(
        sandbox_module,
        "_runtime_usable",
        lambda name, timeout=8.0: name in {"docker", "finch"},
    )

    assert sandbox_module.resolve_runtime("auto") == "finch"


def test_auto_runtime_uses_healthy_backend(monkeypatch) -> None:
    monkeypatch.setattr(sandbox_module.sys, "platform", "linux")
    monkeypatch.setattr(
        sandbox_module,
        "_runtime_usable",
        lambda name, timeout=8.0: name == "finch",
    )

    assert sandbox_module.resolve_runtime("auto") == "finch"


def test_explicit_unavailable_runtime_fails(monkeypatch) -> None:
    monkeypatch.setattr(sandbox_module, "_runtime_usable", lambda *args, **kwargs: False)

    with pytest.raises(SandboxError, match="did not succeed"):
        sandbox_module.resolve_runtime("docker")


def test_container_command_has_fixed_isolation_policy(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(sandbox_module, "_runtime_usable", lambda *args, **kwargs: True)
    script = tmp_path / "probe.py"
    script.write_text("print('ok')\n")
    runner = PythonSandbox(runtime="docker", image="example/sandbox@sha256:123")

    command = runner.command(tmp_path, script, name="probe")

    assert command[:3] == ["docker", "run", "--rm"]
    assert "--init" in command
    assert ["--network", "none"] == command[
        command.index("--network") : command.index("--network") + 2
    ]
    assert ["--log-driver", "none"] == command[
        command.index("--log-driver") : command.index("--log-driver") + 2
    ]
    assert "--read-only" in command
    assert ["--cap-drop", "ALL"] == command[
        command.index("--cap-drop") : command.index("--cap-drop") + 2
    ]
    assert ["--security-opt", "no-new-privileges"] == command[
        command.index("--security-opt") : command.index("--security-opt") + 2
    ]
    assert "nofile=256:256" in command
    assert "fsize=16777216:16777216" in command
    assert not any("API_KEY" in value for value in command)
    assert command[-4:] == [
        "example/sandbox@sha256:123",
        "python",
        "-B",
        "/workspace/probe.py",
    ]


def test_host_runner_executes_workspace_source(tmp_path) -> None:
    runner = PythonSandbox(runtime="host")

    result = runner.run_source(tmp_path, "print(6 * 7)\n", timeout=5)

    assert result.returncode == 0
    assert result.stdout.strip() == "42"
    assert not result.timed_out
    assert not list(Path(tmp_path).glob(".tycho-run-*.py"))


def test_runner_bounds_captured_output(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(sandbox_module, "MAX_STDOUT_CHARS", 32)
    runner = PythonSandbox(runtime="host")

    result = runner.run_source(tmp_path, "print('x' * 1000)\n", timeout=5)

    assert result.returncode == 0
    assert result.stdout.startswith("x" * 32)
    assert "output truncated by sandbox" in result.stdout


def test_timeout_returns_partial_output_and_cleans_source(tmp_path) -> None:
    runner = PythonSandbox(runtime="host")

    result = runner.run_source(
        tmp_path,
        "import time\nprint('before', flush=True)\ntime.sleep(10)\n",
        timeout=0.1,
    )

    assert result.timed_out
    assert result.stdout.strip() == "before"
    assert not list(Path(tmp_path).glob(".tycho-run-*.py"))


def test_runner_normalizes_host_paths_in_output(tmp_path) -> None:
    runner = PythonSandbox(runtime="host")

    result = runner.run_source(
        tmp_path,
        f"print({str(tmp_path)!r})\nprint({str(Path.home())!r})\n",
        timeout=5,
    )

    assert result.stdout.splitlines() == ["/workspace", "<home>"]


def test_script_must_be_inside_workspace(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("print('no')\n")
    runner = PythonSandbox(runtime="host")

    with pytest.raises(SandboxError, match="outside workspace"):
        runner.command(workspace, outside, name="probe")


def test_benchmark_check_rejects_host_runtime() -> None:
    runner = PythonSandbox(runtime="host")

    with pytest.raises(SandboxError, match="requires a container runtime"):
        runner.check(require_isolation=True)


def test_container_check_verifies_live_policy(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(sandbox_module, "_runtime_usable", lambda *args, **kwargs: True)
    runner = PythonSandbox(runtime="docker", image="example/sandbox")
    payload = {
        "python": True,
        "numpy": "2.4.6",
        "pillow": "Image",
        "outside_readable": False,
        "network": False,
        "workspace_write": True,
        "cwd": "/workspace",
        "home": "/tmp",
        "root_read_only": True,
        "cap_eff": "0000000000000000",
        "no_new_privs": "1",
    }
    monkeypatch.setattr(
        runner,
        "run_source",
        lambda *args, **kwargs: SandboxResult(
            stdout=json.dumps(payload),
            stderr="",
            returncode=0,
        ),
    )

    result = runner.check(require_isolation=True)

    assert result == {"runtime": "docker", "image": "example/sandbox", **payload}


def test_container_check_fails_closed_on_policy_mismatch(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(sandbox_module, "_runtime_usable", lambda *args, **kwargs: True)
    runner = PythonSandbox(runtime="docker", image="example/sandbox")
    payload = {
        "outside_readable": True,
        "network": False,
        "workspace_write": True,
        "cwd": "/workspace",
        "home": "/tmp",
        "root_read_only": True,
        "cap_eff": "0000000000000000",
        "no_new_privs": "1",
    }
    monkeypatch.setattr(
        runner,
        "run_source",
        lambda *args, **kwargs: SandboxResult(
            stdout=json.dumps(payload),
            stderr="",
            returncode=0,
        ),
    )

    with pytest.raises(SandboxError, match="unexpected sandbox policy"):
        runner.check(require_isolation=True)
