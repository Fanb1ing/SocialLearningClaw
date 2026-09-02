"""Exact, content-addressed snapshots of agent-controlled workspace state."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
from pathlib import Path, PurePosixPath


SNAPSHOT_SCHEMA = 2
MAX_BLOB_BYTES = 16 * 1024 * 1024
_LEVEL_DIR = re.compile(r"^level_\d+$")
_IGNORED_PARTS = {"__pycache__", ".git", ".pytest_cache"}
_IGNORED_NAMES = {".current_frame.json", ".DS_Store"}
_IGNORED_SUFFIXES = {".pyc", ".pyo"}


def _is_harness_evidence_path(value: PurePosixPath) -> bool:
    if len(value.parts) >= 2 and value.parts[0] == "attempts" \
            and re.fullmatch(r"level_\d+_attempt_\d+", value.parts[1]):
        return True
    if len(value.parts) < 2 or _LEVEL_DIR.match(value.parts[0]) is None:
        return False
    name = value.parts[1]
    if name.startswith("animation_"):
        return True
    if re.fullmatch(r"turn_\d+\.(?:txt|json|png)", name):
        return True
    if re.fullmatch(r"diff_\d+_\d+\.txt", name):
        return True
    if re.fullmatch(r"scene_\d+\.json", name):
        return True
    if name in {"terminal.json", "terminal.txt", "terminal.png", "solved.json"}:
        return True
    if name.startswith("death_"):
        return True
    return False


def is_causal_workspace_path(path: str) -> bool:
    """Whether a workspace file can carry agent state rather than recorded evidence.

    Known turn/diff/terminal/death/animation paths are harness-owned evidence. Everything
    else, including an agent-created helper inside a level directory, is conservatively
    treated as causal agent state regardless of text or binary format.
    """
    value = PurePosixPath(path)
    if value.is_absolute() or ".." in value.parts or not value.parts:
        return False
    if value.parts[0] == ".workspace_blobs" or _is_harness_evidence_path(value):
        return False
    if any(part in _IGNORED_PARTS for part in value.parts):
        return False
    if value.name in _IGNORED_NAMES or value.suffix in _IGNORED_SUFFIXES:
        return False
    return True


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return digest.hexdigest(), size


class SnapshotMaterializationError(RuntimeError):
    """Raised when an exact workspace snapshot cannot be reconstructed."""


def materialize_workspace_snapshot(
    destination: Path,
    versions: dict[str, dict],
    *,
    blob_dir: Path,
) -> None:
    """Replace causal files in ``destination`` with one exact captured manifest.

    Harness-owned observation directories are left untouched. Reconstruction is
    all-or-nothing with respect to availability: every manifest entry must name a
    stored blob before any existing causal file is removed.
    """
    validated: list[tuple[str, Path]] = []
    for rel, descriptor in sorted(versions.items()):
        if not is_causal_workspace_path(rel):
            raise SnapshotMaterializationError(f"unsafe workspace snapshot path: {rel!r}")
        if not isinstance(descriptor, dict) or descriptor.get("stored") is not True:
            status = descriptor.get("status") if isinstance(descriptor, dict) else "invalid"
            raise SnapshotMaterializationError(f"{rel}: body unavailable ({status})")
        digest = descriptor.get("sha256")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise SnapshotMaterializationError(f"{rel}: invalid SHA-256 descriptor")
        blob = blob_dir / digest
        try:
            size = blob.stat().st_size
        except OSError as exc:
            raise SnapshotMaterializationError(f"{rel}: missing blob {digest}") from exc
        expected_size = descriptor.get("size")
        if expected_size is not None and size != expected_size:
            raise SnapshotMaterializationError(
                f"{rel}: blob size {size} does not match manifest {expected_size}"
            )
        validated.append((rel, blob))

    destination.mkdir(parents=True, exist_ok=True)
    for path in sorted(destination.rglob("*"), reverse=True):
        if not (path.is_file() or path.is_symlink()):
            continue
        rel = path.relative_to(destination).as_posix()
        if is_causal_workspace_path(rel):
            path.unlink()
    for rel, blob in validated:
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(blob, target)


class WorkspaceVersionStore:
    """Capture full workspace manifests and deduplicate immutable file bodies by hash."""

    def __init__(
        self,
        workspace_dir: Path,
        *,
        blob_dir: Path | None = None,
        max_blob_bytes: int = MAX_BLOB_BYTES,
    ) -> None:
        self.workspace_dir = workspace_dir
        self.blob_dir = blob_dir or workspace_dir.parent / ".workspace_blobs"
        self.max_blob_bytes = max(1, int(max_blob_bytes))
        self._text_pool: dict[str, str] = {}

    def _store_blob(self, digest: str, data: bytes) -> None:
        target = self.blob_dir / digest
        if target.exists():
            return
        self.blob_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.blob_dir, prefix=".tmp_blob_")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, target)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def capture(self) -> tuple[dict[str, dict], dict[str, str], list[str]]:
        versions: dict[str, dict] = {}
        contents: dict[str, str] = {}
        warnings: list[str] = []
        for path in sorted(self.workspace_dir.rglob("*")):
            if not (path.is_file() or path.is_symlink()):
                continue
            rel = path.relative_to(self.workspace_dir).as_posix()
            if not is_causal_workspace_path(rel):
                continue
            if path.is_symlink():
                versions[rel] = {
                    "kind": "symlink",
                    "size": 0,
                    "stored": False,
                    "status": "omitted_symlink",
                }
                warnings.append(f"{rel}: symlink omitted from workspace history")
                continue
            try:
                size = path.stat().st_size
                if size > self.max_blob_bytes:
                    digest, exact_size = _sha256_file(path)
                    versions[rel] = {
                        "sha256": digest,
                        "size": exact_size,
                        "kind": "unknown",
                        "stored": False,
                        "status": "omitted_large",
                    }
                    warnings.append(
                        f"{rel}: {exact_size} bytes exceeds workspace blob limit "
                        f"{self.max_blob_bytes}"
                    )
                    continue
                data = path.read_bytes()
            except OSError as exc:
                versions[rel] = {
                    "kind": "unknown",
                    "size": None,
                    "stored": False,
                    "status": f"unreadable:{type(exc).__name__}",
                }
                warnings.append(f"{rel}: unreadable ({type(exc).__name__})")
                continue

            digest = hashlib.sha256(data).hexdigest()
            self._store_blob(digest, data)
            try:
                text = data.decode("utf-8")
                kind = "text" if "\x00" not in text else "binary"
            except UnicodeDecodeError:
                text = ""
                kind = "binary"
            versions[rel] = {
                "sha256": digest,
                "size": len(data),
                "kind": kind,
                "stored": True,
            }
            if kind == "text":
                pooled = self._text_pool.setdefault(digest, text)
                contents[rel] = pooled
        return versions, contents, warnings
