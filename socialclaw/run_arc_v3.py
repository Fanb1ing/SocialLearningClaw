"""Guarded entry point for the Tycho + executable-EFPS V3 runner."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import sys
from typing import Callable, Sequence

from .utils import load_dotenv


REQUIRED_PYTHON = (3, 12)
REQUIRED_PACKAGES = {
    "arc-agi": "0.9.9",
    "arcengine": "0.9.3",
    "jinja2": "3.1.6",
    "numpy": "2.4.6",
    "pillow": "12.3.0",
    "pyyaml": "6.0.3",
}


@dataclass(frozen=True)
class RuntimeContract:
    ok: bool
    python: str
    required_python: str
    packages: dict[str, str]
    required_packages: dict[str, str]
    errors: tuple[str, ...]


def check_runtime(
    *,
    python_version: tuple[int, int, int] | None = None,
    package_version: Callable[[str], str] = version,
) -> RuntimeContract:
    py = python_version or tuple(sys.version_info[:3])
    errors: list[str] = []
    if tuple(py[:2]) < REQUIRED_PYTHON:
        errors.append(
            f"Python {py[0]}.{py[1]} is unsupported; V3 requires Python "
            f"{REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]} or newer"
        )
    installed: dict[str, str] = {}
    for package, required in REQUIRED_PACKAGES.items():
        try:
            actual = package_version(package)
        except PackageNotFoundError:
            actual = "<missing>"
        installed[package] = actual
        if actual != required:
            errors.append(f"{package}=={required} required, found {actual}")
    return RuntimeContract(
        ok=not errors,
        python=".".join(map(str, py)),
        required_python=f">={REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}",
        packages=installed,
        required_packages=dict(REQUIRED_PACKAGES),
        errors=tuple(errors),
    )


def _argument_value(args: Sequence[str], name: str) -> str | None:
    for index, value in enumerate(args):
        if value == name and index + 1 < len(args):
            return args[index + 1]
        if value.startswith(name + "="):
            return value.split("=", 1)[1]
    return None


def _consume_option(args: list[str], name: str) -> str | None:
    for index, value in enumerate(args):
        if value == name:
            if index + 1 >= len(args):
                raise SystemExit(f"{name} requires a value")
            result = args[index + 1]
            del args[index:index + 2]
            return result
        if value.startswith(name + "="):
            result = value.split("=", 1)[1]
            del args[index]
            return result
    return None


def _configure_provider_compatibility(args: list[str]) -> None:
    """Map the useful V2 OpenAI-compatible CLI/env contract onto Tycho."""

    model = _consume_option(args, "--model")
    base_url = _consume_option(args, "--base-url")
    api_key = _consume_option(args, "--api-key")
    if model:
        os.environ["LLM_MODEL"] = model
    if base_url:
        os.environ["LLM_BASE_URL"] = base_url
    if api_key:
        os.environ["LLM_API_KEY"] = api_key
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if "LLM_API_KEY" not in os.environ and (openrouter_key or openai_key):
        os.environ["LLM_API_KEY"] = openrouter_key or openai_key
    if openrouter_key and "LLM_BASE_URL" not in os.environ:
        os.environ["LLM_BASE_URL"] = "https://openrouter.ai/api/v1"
    os.environ.setdefault("LLM_BACKEND", "openai")


def _prepare_tycho_args(args: Sequence[str]) -> list[str]:
    prepared = list(args)
    _configure_provider_compatibility(prepared)
    project_root = Path(__file__).resolve().parent.parent
    os.environ.setdefault(
        "TYCHO_ENVIRONMENTS_DIR",
        str(project_root / "third_party" / "arc_agi3_games"),
    )
    if _argument_value(prepared, "--approach") is None:
        prepared.extend(["--approach", "tycho_efps"])
    out_dir = _argument_value(prepared, "--out-dir")
    if out_dir and "SC_V3_RUN_ID" not in os.environ:
        identity = str(Path(out_dir).resolve()).encode("utf-8")
        os.environ["SC_V3_RUN_ID"] = "v3_" + hashlib.sha256(identity).hexdigest()[:16]
    os.environ.setdefault("TYCHO_MODE", "orchestrator")
    return prepared


def main(argv: Sequence[str] | None = None) -> int:
    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(str(project_root / ".env"))
    args = list(sys.argv[1:] if argv is None else argv)
    contract = check_runtime()
    if "--check-runtime" in args:
        print(json.dumps(asdict(contract), indent=2, sort_keys=True))
        return 0 if contract.ok else 2
    if not contract.ok:
        details = "\n  - ".join(contract.errors)
        raise SystemExit(
            "V3 runtime contract is not satisfied:\n  - " + details +
            "\nCreate/install the repository environment with Python 3.12, then run "
            "`sc-run-arc-v3 --check-runtime` before any experiment."
        )
    prepared = _prepare_tycho_args(args)
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *prepared]
        from tycho.harness.run_parallel import main as tycho_main
        tycho_main()
    finally:
        sys.argv = old_argv
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
