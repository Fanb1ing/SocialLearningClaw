"""Container-backed execution for Python authored inside a Tycho workspace."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid


DEFAULT_IMAGE = "tycho-python-sandbox:0.2"
RUNTIMES = ("docker", "finch", "bwrap")
MAX_STDOUT_CHARS = 16 * 1024 * 1024
MAX_STDERR_CHARS = 1024 * 1024


class SandboxError(RuntimeError):
    """The configured workspace execution runtime is unavailable or invalid."""


@dataclass(frozen=True)
class SandboxResult:
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool = False


@lru_cache(maxsize=len(RUNTIMES))
def _runtime_usable(name: str, *, timeout: float = 8.0) -> bool:
    if name == "bwrap":
        # A full live policy probe runs immediately in PythonSandbox.check().
        return shutil.which("bwrap") is not None and shutil.which("unshare") is not None
    executable = shutil.which(name)
    if executable is None:
        return False
    try:
        result = subprocess.run(
            [executable, "info"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def resolve_runtime(requested: str | None = None) -> str:
    value = (requested or os.environ.get("TYCHO_SANDBOX_RUNTIME", "auto")).strip().lower()
    if value == "host":
        return value
    if value in RUNTIMES:
        if not _runtime_usable(value):
            raise SandboxError(
                f"TYCHO_SANDBOX_RUNTIME={value!r}, but `{value} info` did not succeed"
            )
        return value
    if value != "auto":
        raise SandboxError(
            f"unknown TYCHO_SANDBOX_RUNTIME={value!r}; expected auto, docker, finch, or host"
        )
    candidates = ("finch", "docker") if sys.platform == "darwin" else RUNTIMES
    for candidate in candidates:
        if _runtime_usable(candidate):
            return candidate
    raise SandboxError(
        "no usable container runtime found; install Docker or Finch, or explicitly set "
        "TYCHO_SANDBOX_RUNTIME=host for trusted local development"
    )


class PythonSandbox:
    """Execute a fresh Python process with only one workspace mounted from the host."""

    def __init__(self, runtime: str | None = None, image: str | None = None):
        self.runtime = resolve_runtime(runtime)
        self.image = image or os.environ.get("TYCHO_SANDBOX_IMAGE", DEFAULT_IMAGE)

    @staticmethod
    def _workspace(path: str | Path) -> Path:
        workspace = Path(path).resolve(strict=True)
        if not workspace.is_dir():
            raise SandboxError(f"workspace is not a directory: {workspace}")
        return workspace

    @staticmethod
    def _script(workspace: Path, path: str | Path) -> Path:
        script = Path(path).resolve(strict=True)
        try:
            script.relative_to(workspace)
        except ValueError as exc:
            raise SandboxError(f"script is outside workspace: {script}") from exc
        if not script.is_file():
            raise SandboxError(f"script is not a file: {script}")
        return script

    @staticmethod
    def _portable_output(value: str, workspace: Path) -> str:
        text = value or ""
        replacements = (
            (str(workspace), "/workspace"),
            (str(Path.home()), "<home>"),
        )
        for source, target in replacements:
            if source:
                text = text.replace(source, target)
        return text

    def command(
        self,
        workspace: str | Path,
        script: str | Path,
        *,
        name: str,
        args: tuple[str, ...] = (),
    ) -> list[str]:
        ws = self._workspace(workspace)
        script_path = self._script(ws, script)
        if self.runtime == "host":
            return [sys.executable, "-B", str(script_path), *args]
        if self.runtime == "bwrap":
            # `unshare` creates the network namespace before bwrap. On this host bwrap can create
            # mount/pid namespaces but cannot configure loopback itself; nesting preserves the same
            # no-network guarantee while keeping the filesystem allowlist narrow. The model-authored
            # process receives no provider credentials or other inherited environment variables.
            python_root = Path(sys.prefix).resolve()
            relative_script = script_path.relative_to(ws).as_posix()
            return [
                shutil.which("unshare") or "unshare",
                "--user", "--map-root-user", "--net",
                shutil.which("bwrap") or "bwrap",
                "--die-with-parent",
                "--new-session",
                "--unshare-pid",
                "--unshare-ipc",
                "--unshare-uts",
                "--unshare-cgroup",
                "--cap-drop", "ALL",
                "--ro-bind", "/usr", "/usr",
                "--ro-bind-try", "/lib", "/lib",
                "--ro-bind-try", "/lib64", "/lib64",
                "--ro-bind-try", "/bin", "/bin",
                "--ro-bind", str(python_root), "/opt/venv",
                "--proc", "/proc",
                "--dev", "/dev",
                "--tmpfs", "/tmp",
                "--bind", str(ws), "/workspace",
                "--remount-ro", "/",
                "--chdir", "/workspace",
                "--clearenv",
                "--setenv", "HOME", "/tmp",
                "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
                "--setenv", "PYTHONNOUSERSITE", "1",
                "--setenv", "OMP_NUM_THREADS", "1",
                "--setenv", "OPENBLAS_NUM_THREADS", "1",
                "--setenv", "MKL_NUM_THREADS", "1",
                "/opt/venv/bin/python", "-B", f"/workspace/{relative_script}", *args,
            ]
        relative_script = script_path.relative_to(ws).as_posix()
        mount = f"type=bind,src={ws},dst=/workspace"
        command = [
            self.runtime,
            "run",
            "--rm",
            "--name",
            name,
            "--init",
            "--network",
            "none",
            "--log-driver",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "64",
            "--memory",
            "1g",
            "--cpus",
            "1",
            "--ulimit",
            "nofile=256:256",
            "--ulimit",
            "fsize=16777216:16777216",
            "--mount",
            mount,
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=128m",
            "--workdir",
            "/workspace",
            "--env",
            "HOME=/tmp",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--env",
            "PYTHONNOUSERSITE=1",
            "--env",
            "OMP_NUM_THREADS=1",
            "--env",
            "OPENBLAS_NUM_THREADS=1",
            "--env",
            "MKL_NUM_THREADS=1",
        ]
        if hasattr(os, "getuid") and hasattr(os, "getgid"):
            command.extend(["--user", f"{os.getuid()}:{os.getgid()}"])
        command.extend(
            [
                self.image,
                "python",
                "-B",
                f"/workspace/{relative_script}",
                *args,
            ]
        )
        return command

    def run_script(
        self,
        workspace: str | Path,
        script: str | Path,
        *,
        timeout: int | float,
        args: tuple[str, ...] = (),
    ) -> SandboxResult:
        ws = self._workspace(workspace)
        container_name = f"tycho-python-{uuid.uuid4().hex[:16]}"
        command = self.command(ws, script, name=container_name, args=args)
        process = subprocess.Popen(
            command,
            cwd=ws,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        captured: dict[str, list[str]] = {"stdout": [], "stderr": []}

        def drain(name: str, stream, limit: int) -> None:
            retained = 0
            truncated = False
            while True:
                chunk = stream.read(65536)
                if not chunk:
                    break
                remaining = max(0, limit - retained)
                if remaining:
                    captured[name].append(chunk[:remaining])
                    retained += min(len(chunk), remaining)
                if len(chunk) > remaining:
                    truncated = True
            if truncated:
                captured[name].append("\n[output truncated by sandbox]\n")

        readers = [
            threading.Thread(
                target=drain,
                args=("stdout", process.stdout, MAX_STDOUT_CHARS),
                daemon=True,
            ),
            threading.Thread(
                target=drain,
                args=("stderr", process.stderr, MAX_STDERR_CHARS),
                daemon=True,
            ),
        ]
        for reader in readers:
            reader.start()
        timed_out = False
        cleanup_error: BaseException | None = None
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            if self.runtime in ("docker", "finch"):
                try:
                    cleanup = subprocess.run(
                        [self.runtime, "rm", "-f", container_name],
                        capture_output=True,
                        text=True,
                        timeout=8,
                        check=False,
                    )
                    if cleanup.returncode:
                        cleanup_error = SandboxError(
                            f"{self.runtime} could not stop timed-out workspace container"
                        )
                except (OSError, subprocess.TimeoutExpired) as exc:
                    cleanup_error = exc
            if process.poll() is None:
                process.kill()
            process.wait()
        for reader in readers:
            reader.join(timeout=5)
        if cleanup_error is not None:
            raise SandboxError(
                f"{self.runtime} could not stop timed-out workspace container"
            ) from cleanup_error
        return SandboxResult(
            self._portable_output("".join(captured["stdout"]), ws),
            self._portable_output("".join(captured["stderr"]), ws),
            -1 if timed_out else (process.returncode or 0),
            timed_out=timed_out,
        )

    def run_source(
        self,
        workspace: str | Path,
        source: str,
        *,
        timeout: int | float,
        args: tuple[str, ...] = (),
    ) -> SandboxResult:
        ws = self._workspace(workspace)
        with tempfile.NamedTemporaryFile(
            "w",
            suffix=".py",
            prefix=".tycho-run-",
            dir=ws,
            delete=False,
        ) as handle:
            handle.write(source)
            script = Path(handle.name)
        try:
            return self.run_script(ws, script, timeout=timeout, args=args)
        finally:
            script.unlink(missing_ok=True)

    def check(self, *, require_isolation: bool = False) -> dict:
        """Verify the runtime before agents can execute workspace-authored Python."""
        if require_isolation and self.runtime == "host":
            raise SandboxError(
                "benchmark execution requires a container runtime; "
                "TYCHO_SANDBOX_RUNTIME=host is only for trusted local development"
            )
        with tempfile.TemporaryDirectory(prefix="tycho-sandbox-check-") as root:
            base = Path(root)
            workspace = base / "workspace"
            workspace.mkdir()
            outside = base / "outside.txt"
            outside.write_text(uuid.uuid4().hex)
            if self.runtime == "host":
                source = (
                    "import json, numpy\nfrom PIL import Image\n"
                    "print(json.dumps({'python': True, 'numpy': numpy.__version__, "
                    "'pillow': Image.__name__}))\n"
                )
            else:
                source = (
                    "import json, os, socket, numpy\n"
                    "from PIL import Image\n"
                    f"_outside = {str(outside)!r}\n"
                    "try:\n"
                    "    open(_outside).read(); outside_readable = True\n"
                    "except OSError:\n"
                    "    outside_readable = False\n"
                    "try:\n"
                    "    socket.create_connection(('1.1.1.1', 53), timeout=1).close()\n"
                    "    network = True\n"
                    "except OSError:\n"
                    "    network = False\n"
                    "open('workspace-write.txt', 'w').write('ok')\n"
                    "_status = {}\n"
                    "try:\n"
                    "    for line in open('/proc/self/status'):\n"
                    "        if ':' in line:\n"
                    "            key, value = line.split(':', 1); _status[key] = value.strip()\n"
                    "except OSError:\n"
                    "    pass\n"
                    "payload = {\n"
                    "    'python': True,\n"
                    "    'numpy': numpy.__version__,\n"
                    "    'pillow': Image.__name__,\n"
                    "    'outside_readable': outside_readable,\n"
                    "    'network': network,\n"
                    "    'workspace_write': os.path.exists('workspace-write.txt'),\n"
                    "    'cwd': os.getcwd(),\n"
                    "    'home': os.environ.get('HOME'),\n"
                    "    'root_read_only': bool(os.statvfs('/').f_flag & os.ST_RDONLY),\n"
                    "    'cap_eff': _status.get('CapEff'),\n"
                    "    'no_new_privs': _status.get('NoNewPrivs'),\n"
                    "}\n"
                    "print(json.dumps(payload))\n"
                )
            result = self.run_source(workspace, source, timeout=20)
        if result.timed_out:
            raise SandboxError(f"{self.runtime} sandbox check timed out")
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise SandboxError(
                f"{self.runtime} sandbox check failed"
                + (f": {detail}" if detail else "")
            )
        try:
            payload = json.loads(result.stdout.strip())
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SandboxError(
                f"{self.runtime} sandbox check returned invalid output"
            ) from exc
        if self.runtime != "host":
            expected = {
                "outside_readable": False,
                "network": False,
                "workspace_write": True,
                "cwd": "/workspace",
                "home": "/tmp",
                "root_read_only": True,
                "cap_eff": "0000000000000000",
                "no_new_privs": "1",
            }
            observed = {key: payload.get(key) for key in expected}
            if observed != expected:
                raise SandboxError(f"unexpected sandbox policy result: {observed}")
        return {"runtime": self.runtime, "image": self.image, **payload}


def _containerfile() -> Path:
    return Path(__file__).resolve().parent / "container" / "Containerfile"


def _build(runtime: str, image: str) -> None:
    containerfile = _containerfile()
    if not containerfile.is_file():
        raise SandboxError(f"container definition is missing: {containerfile}")
    subprocess.run(
        [
            runtime,
            "build",
            "--tag",
            image,
            "--file",
            str(containerfile),
            str(containerfile.parent),
        ],
        check=True,
    )


def _doctor(runtime: str | None, image: str | None) -> dict:
    sandbox = PythonSandbox(runtime=runtime, image=image)
    return sandbox.check(require_isolation=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=("auto", *RUNTIMES, "host"), default=None)
    parser.add_argument("--image", default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor")
    subparsers.add_parser("build")
    args = parser.parse_args()
    runtime = resolve_runtime(args.runtime)
    image = args.image or os.environ.get("TYCHO_SANDBOX_IMAGE", DEFAULT_IMAGE)
    if args.command == "build":
        if runtime == "host":
            raise SystemExit("the host runtime does not build container images")
        _build(runtime, image)
        print(f"built {image} with {runtime}")
        return
    try:
        payload = _doctor(runtime, image)
    except SandboxError as exc:
        raise SystemExit(f"sandbox check failed: {exc}") from exc
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
