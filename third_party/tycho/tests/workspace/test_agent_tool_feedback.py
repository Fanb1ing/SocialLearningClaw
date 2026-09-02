from __future__ import annotations

from pathlib import Path

from tycho.workspace.agent_tools import ToolExecutor, tool_specs
from tycho.workspace.sandbox import PythonSandbox, SandboxResult
from tycho.workspace.workspace import GameWorkspace


def test_comment_only_model_edit_skips_auto_verification(tmp_path, monkeypatch) -> None:
    ws = GameWorkspace("ast-skip", root=str(tmp_path), render=False)
    executor = ToolExecutor(ws)
    calls = []
    monkeypatch.setattr(executor, "_run_python", lambda code, timeout=15: calls.append(timeout) or "verified")

    executor.execute("write_file", {"path": "world_model.py", "content": "VALUE = 1\n"})
    assert calls == [20]
    result = executor.execute(
        "edit_file",
        {"path": "world_model.py", "old": "VALUE = 1", "new": "# explanatory comment\nVALUE = 1"},
    )
    assert "syntax tree is unchanged" in result
    assert calls == [20]

    executor.execute("edit_file", {"path": "world_model.py", "old": "VALUE = 1", "new": "VALUE = 2"})
    assert calls == [20, 20]


def test_run_python_timeout_reports_observation_without_diagnosis(tmp_path, monkeypatch) -> None:
    ws = GameWorkspace("timeout", root=str(tmp_path), render=False)
    sandbox = PythonSandbox(runtime="host")
    executor = ToolExecutor(ws, python_sandbox=sandbox)
    monkeypatch.setattr(
        sandbox,
        "run_source",
        lambda *args, **kwargs: SandboxResult("checkpoint 12\n", "still searching\n", -1, timed_out=True),
    )
    result = executor._run_python("print('slow')", timeout=7)
    assert result.startswith("(run_python did not finish within 7s; no conclusion about model correctness)")
    assert "[partial stdout before timeout]\ncheckpoint 12" in result
    assert "[partial stderr before timeout]\nstill searching" in result
    assert "infinite" not in result


def test_run_python_tool_accepts_bounded_explicit_timeout(tmp_path, monkeypatch) -> None:
    ws = GameWorkspace("timeout-arg", root=str(tmp_path), render=False)
    executor = ToolExecutor(ws)
    calls = []
    monkeypatch.setattr(
        executor,
        "_run_python",
        lambda code, timeout=15: calls.append((code, timeout)) or "ok",
    )

    assert executor.execute("run_python", {"code": "print(1)", "timeout_s": 120}) == "ok"
    assert calls == [("print(1)", 120)]
    assert "timeout_s must be an integer" in executor.execute(
        "run_python", {"code": "print(1)", "timeout_s": 2.5}
    )


def test_run_python_tool_documents_persistent_helper_modules() -> None:
    wm = next(s for s in tool_specs(world_model_enabled=True) if s["name"] == "run_python")["description"]
    no_wm = next(s for s in tool_specs(world_model_enabled=False) if s["name"] == "run_python")["description"]

    assert "workspace files do" in wm
    assert "reusable Python helper modules" in wm
    assert "`import helper`" in wm
    assert "analysis helper modules" in no_wm
    assert "world_model" not in no_wm
    spec = next(s for s in tool_specs(world_model_enabled=True) if s["name"] == "run_python")
    assert spec["schema"]["properties"]["timeout_s"]["maximum"] == 300


def test_tool_output_hides_host_paths(tmp_path, monkeypatch) -> None:
    ws = GameWorkspace("portable-output", root=str(tmp_path), render=False)
    executor = ToolExecutor(ws)
    workspace_file = ws.dir / "notes" / "evidence.txt"
    unrelated_host_file = Path.home() / "unrelated-checkout" / ".git" / "config"

    def fail_with_paths(_path: str) -> str:
        raise FileNotFoundError(f"{workspace_file}; {unrelated_host_file}")

    monkeypatch.setattr(ws, "ls", fail_with_paths)
    result = executor.execute("ls", {"path": "."})

    assert "/workspace/notes/evidence.txt" in result
    assert "<host-path>" in result
    assert str(tmp_path) not in result
    assert str(Path.home()) not in result
    assert "unrelated-checkout" not in result
