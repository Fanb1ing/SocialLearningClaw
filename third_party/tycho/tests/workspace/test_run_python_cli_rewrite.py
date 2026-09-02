"""run_python compatibility for CLI-style workspace helpers.

The actor prompt intentionally tells agents to run helpers as `python plan.py astar` inside
run_python. ToolExecutor rewrites that single-line shell-like form to runpy; this keeps the natural
prompt wording correct even though run_python is not a shell.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from tycho.workspace.agent_tools import ToolExecutor
from tycho.workspace.sandbox import PythonSandbox


def _check_shellish_plan_line() -> dict[str, bool]:
    out = ToolExecutor._shellish_to_python("python plan.py astar")
    return {
        "rewrites_to_runpy": "runpy.run_path('plan.py', run_name='__main__')" in out,
        "sets_script_and_arg": "sys.argv = ['plan.py', 'astar']" in out,
    }


def _check_python3_line() -> dict[str, bool]:
    out = ToolExecutor._shellish_to_python("python3 verify.py")
    return {
        "python3_supported": "runpy.run_path('verify.py', run_name='__main__')" in out,
        "sets_script_only": "sys.argv = ['verify.py']" in out,
    }


def _check_real_python_unchanged() -> dict[str, bool]:
    src = "print('hello')"
    multiline = "import sys\nprint(sys.argv)"
    return {
        "single_line_python_unchanged": ToolExecutor._shellish_to_python(src) == src,
        "multi_line_python_unchanged": ToolExecutor._shellish_to_python(multiline) == multiline,
    }


def _check_actual_run_python_execution() -> dict[str, bool]:
    with TemporaryDirectory() as td:
        root = Path(td)
        (root / "wmlib.py").write_text(
            "def current_grid(): return None\n"
            "def frames(): return {}\n"
            "def transitions(): return []\n"
        )
        (root / "cli_echo.py").write_text(
            "import sys\n"
            "print('ARGV', sys.argv)\n"
        )
        executor = ToolExecutor(
            SimpleNamespace(dir=root),
            python_sandbox=PythonSandbox(runtime="host"),
        )
        out = executor.execute("run_python", {"code": "python cli_echo.py astar"})
        return {
            "executes_rewritten_cli_line": "ARGV ['cli_echo.py', 'astar']" in out,
        }


def main() -> int:
    ok = True
    for group, checks in (
        ("shellish_plan_line", _check_shellish_plan_line()),
        ("python3_line", _check_python3_line()),
        ("real_python", _check_real_python_unchanged()),
        ("actual_run_python", _check_actual_run_python_execution()),
    ):
        print(f"=== {group} ===")
        for name, passed in checks.items():
            print(f"  {name}: {passed}")
            ok = ok and passed
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
