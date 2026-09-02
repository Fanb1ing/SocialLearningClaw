"""Exact, bounded checkpoints for resuming one deterministic ARC game.

The engine state is reconstructed by replaying the committed action prefix.  The actor state and
the complete per-game workspace are restored from the checkpoint named by ``checkpoint/HEAD``.
Only ``HEAD`` and ``PREVIOUS`` are retained; immutable file bodies are deduplicated by SHA-256.

There is deliberately no warm-resume path.  If the checkpoint, workspace, execution contract, or
engine replay cannot be reproduced exactly, continuing would create a different experimental run.
The caller must mark that game blocked or restart it explicitly.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import pickle
import re
import shutil
import tempfile
import uuid
import zlib
from pathlib import Path, PurePosixPath
from typing import Optional


_JOURNAL_SCHEMA = 2
_IGNORED_WORKSPACE_PARTS = {"__pycache__", ".git", ".pytest_cache"}
_IGNORED_WORKSPACE_NAMES = {".DS_Store"}
_SHA256 = re.compile(r"[0-9a-f]{64}")


class ResumeError(RuntimeError):
    """Base class for failures that must not silently become a warm resume."""


class ResumeCompatibilityError(ResumeError):
    """The saved actor/checkpoint contract differs from the running code."""


class ResumeCheckpointError(ResumeError):
    """A committed checkpoint is missing, corrupt, or cannot be materialized exactly."""


def resume_fingerprint(agent=None) -> str:
    """Hash the actor serialization contract and journal schema."""
    parts = [f"schema={_JOURNAL_SCHEMA}"]
    for name in ("snapshot_state", "restore_state"):
        fn = getattr(agent, name, None) if agent is not None else None
        try:
            parts.append(inspect.getsource(fn)) if fn is not None else parts.append(f"{name}=<none>")
        except (OSError, TypeError):
            parts.append(f"{name}=<unsourceable>")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=path.suffix)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _atomic_write_json(path: Path, value: dict) -> None:
    _atomic_write_bytes(path, json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def _safe_relative_path(value: str) -> PurePosixPath:
    rel = PurePosixPath(value)
    if rel.is_absolute() or not rel.parts or ".." in rel.parts:
        raise ResumeCheckpointError(f"unsafe workspace checkpoint path: {value!r}")
    return rel


class GameJournal:
    """Durable action log plus two-generation exact checkpoint for one game."""

    def __init__(self, journal_dir: str | Path):
        self.dir = Path(journal_dir)
        self.checkpoint_dir = self.dir / "checkpoint"
        self.blob_dir = self.checkpoint_dir / "blobs"
        self.generation_dir = self.checkpoint_dir / "generations"
        self.head_path = self.checkpoint_dir / "HEAD.json"
        self.previous_path = self.checkpoint_dir / "PREVIOUS.json"
        self.actions_path = self.dir / "actions.jsonl"
        self.meta_path = self.dir / "meta.json"
        self.generation_dir.mkdir(parents=True, exist_ok=True)
        self.blob_dir.mkdir(parents=True, exist_ok=True)

    def init_meta(self, *, game_id: str, fingerprint: str, context_mode: str) -> None:
        wanted = {
            "schema": _JOURNAL_SCHEMA,
            "game_id": game_id,
            "fingerprint": fingerprint,
            "context_mode": context_mode,
        }
        if not self.meta_path.exists():
            _atomic_write_json(self.meta_path, wanted)

    def exists(self) -> bool:
        return self.meta_path.exists() and self.head_path.exists()

    def meta(self) -> dict:
        try:
            return json.loads(self.meta_path.read_text()) if self.meta_path.exists() else {}
        except (OSError, json.JSONDecodeError) as exc:
            raise ResumeCheckpointError(f"cannot read resume metadata: {exc}") from exc

    def compatible(self, *, fingerprint: str, context_mode: str) -> tuple[bool, str]:
        metadata = self.meta()
        if not metadata:
            return False, "no metadata"
        if metadata.get("schema") != _JOURNAL_SCHEMA:
            return False, f"journal schema changed ({metadata.get('schema')} != {_JOURNAL_SCHEMA})"
        if metadata.get("fingerprint") != fingerprint:
            return False, (
                f"resume contract changed ({metadata.get('fingerprint')} != {fingerprint})"
            )
        if metadata.get("context_mode") != context_mode:
            return False, (
                f"context mode changed ({metadata.get('context_mode')} != {context_mode})"
            )
        return True, "ok"

    def _read_pointer(self, path: Path, *, required: bool = True) -> Optional[dict]:
        if not path.exists():
            if required:
                raise ResumeCheckpointError(f"missing checkpoint pointer: {path.name}")
            return None
        try:
            pointer = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ResumeCheckpointError(f"invalid checkpoint pointer {path.name}: {exc}") from exc
        generation = pointer.get("generation")
        if not isinstance(generation, str) or Path(generation).name != generation:
            raise ResumeCheckpointError(f"invalid generation in {path.name}")
        return pointer

    def _read_generation(self, pointer: dict) -> dict:
        path = self.generation_dir / pointer["generation"]
        try:
            raw = path.read_bytes()
            expected = pointer.get("sha256")
            if expected and hashlib.sha256(raw).hexdigest() != expected:
                raise ResumeCheckpointError(f"generation digest mismatch: {path.name}")
            generation = json.loads(raw)
        except ResumeCheckpointError:
            raise
        except (OSError, json.JSONDecodeError) as exc:
            raise ResumeCheckpointError(f"cannot read generation {path.name}: {exc}") from exc
        if generation.get("schema") != _JOURNAL_SCHEMA:
            raise ResumeCheckpointError(f"unsupported checkpoint schema in {path.name}")
        if generation.get("action_count") != pointer.get("action_count"):
            raise ResumeCheckpointError(f"action count mismatch in {path.name}")
        return generation

    def load_checkpoint(self) -> dict:
        return self._read_generation(self._read_pointer(self.head_path))

    def _read_action_lines(self) -> list[dict]:
        if not self.actions_path.exists():
            return []
        out: list[dict] = []
        try:
            lines = self.actions_path.read_text().splitlines()
        except OSError as exc:
            raise ResumeCheckpointError(f"cannot read action log: {exc}") from exc
        for line in lines:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                break
            if record.get("k") != len(out):
                break
            out.append(record)
        return out

    def steps(self) -> list[dict]:
        """Return exactly the action prefix committed by ``HEAD``.

        An action line written just before a crash but not followed by a HEAD update is an
        uncommitted tail and is ignored.  A HEAD that references more actions than are present is
        corruption and blocks resume.
        """
        if not self.exists():
            return []
        count = int(self.load_checkpoint().get("action_count", -1))
        records = self._read_action_lines()
        if count < 0 or len(records) < count:
            raise ResumeCheckpointError(
                f"checkpoint commits {count} actions but log contains {len(records)}"
            )
        return records[:count]

    def _truncate_actions(self, count: int) -> None:
        records = self._read_action_lines()
        if len(records) == count:
            return
        if len(records) < count:
            raise ResumeCheckpointError(
                f"cannot truncate action log to {count}; only {len(records)} valid records"
            )
        body = "".join(json.dumps(row, sort_keys=True) + "\n" for row in records[:count])
        _atomic_write_bytes(self.actions_path, body.encode())

    def _store_blob(self, data: bytes) -> dict:
        digest = hashlib.sha256(data).hexdigest()
        target = self.blob_dir / digest
        if not target.exists():
            _atomic_write_bytes(target, data)
        return {"sha256": digest, "size": len(data)}

    def _capture_workspace(self, workspace_dir: Path) -> dict:
        if not workspace_dir.is_dir():
            raise ResumeCheckpointError(f"workspace does not exist: {workspace_dir}")
        files: dict[str, dict] = {}
        directories = ["."]
        for path in sorted(workspace_dir.rglob("*")):
            rel_path = path.relative_to(workspace_dir)
            if any(part in _IGNORED_WORKSPACE_PARTS for part in rel_path.parts):
                continue
            rel = rel_path.as_posix()
            if path.is_symlink():
                raise ResumeCheckpointError(f"workspace symlink cannot be checkpointed: {rel}")
            if path.is_dir():
                directories.append(rel)
                continue
            if not path.is_file() or path.name in _IGNORED_WORKSPACE_NAMES:
                continue
            try:
                descriptor = self._store_blob(path.read_bytes())
                descriptor["mode"] = path.stat().st_mode & 0o777
            except OSError as exc:
                raise ResumeCheckpointError(f"cannot checkpoint workspace file {rel}: {exc}") from exc
            files[rel] = descriptor
        return {"directories": directories, "files": files}

    def _blob_bytes(self, descriptor: dict, *, label: str) -> bytes:
        digest = descriptor.get("sha256")
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise ResumeCheckpointError(f"{label}: invalid blob digest")
        path = self.blob_dir / digest
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise ResumeCheckpointError(f"{label}: missing blob {digest}") from exc
        if len(data) != descriptor.get("size") or hashlib.sha256(data).hexdigest() != digest:
            raise ResumeCheckpointError(f"{label}: corrupt blob {digest}")
        return data

    def restore_workspace(self, workspace_dir: str | Path, checkpoint: Optional[dict] = None) -> None:
        """Replace the mutable workspace with the exact files committed by HEAD."""
        generation = checkpoint or self.load_checkpoint()
        workspace = generation.get("workspace")
        if not isinstance(workspace, dict):
            raise ResumeCheckpointError("checkpoint has no workspace manifest")
        files = workspace.get("files")
        directories = workspace.get("directories")
        if not isinstance(files, dict) or not isinstance(directories, list):
            raise ResumeCheckpointError("invalid workspace manifest")

        # Validate every body before changing the live workspace.
        materialized: dict[str, tuple[bytes, int]] = {}
        for rel, descriptor in sorted(files.items()):
            _safe_relative_path(rel)
            if not isinstance(descriptor, dict):
                raise ResumeCheckpointError(f"{rel}: invalid workspace descriptor")
            materialized[rel] = (
                self._blob_bytes(descriptor, label=rel),
                int(descriptor.get("mode", 0o644)),
            )
        clean_dirs = []
        for rel in directories:
            if rel == ".":
                continue
            clean_dirs.append(_safe_relative_path(rel).as_posix())

        destination = Path(workspace_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp = Path(tempfile.mkdtemp(prefix=f".{destination.name}.resume_", dir=destination.parent))
        backup = destination.parent / f".{destination.name}.resume_previous"
        try:
            for rel in clean_dirs:
                (temp / rel).mkdir(parents=True, exist_ok=True)
            for rel, (data, mode) in materialized.items():
                target = temp / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                target.chmod(mode)
            if backup.exists():
                if destination.exists():
                    shutil.rmtree(backup)
                else:
                    os.replace(backup, destination)
            if destination.exists():
                os.replace(destination, backup)
            os.replace(temp, destination)
            if backup.exists():
                shutil.rmtree(backup)
        finally:
            if temp.exists():
                shutil.rmtree(temp)

    def load_actor_state(self, checkpoint: Optional[dict] = None) -> object:
        generation = checkpoint or self.load_checkpoint()
        descriptor = generation.get("actor")
        if not isinstance(descriptor, dict):
            raise ResumeCheckpointError("checkpoint has no actor state")
        compressed = self._blob_bytes(descriptor, label="actor state")
        try:
            return pickle.loads(zlib.decompress(compressed))
        except Exception as exc:  # noqa: BLE001 - any decode failure invalidates exact resume
            raise ResumeCheckpointError(f"cannot decode actor state: {exc}") from exc

    def record_step(
        self,
        k: int,
        *,
        action: str,
        x: Optional[int],
        y: Optional[int],
        frame_hash: int,
        level: int,
        state: str,
        agent_snapshot: Optional[object],
        workspace_dir: str | Path,
    ) -> None:
        """Atomically advance the checkpoint to committed action ``k``."""
        current_pointer = self._read_pointer(self.head_path, required=False)
        committed = int(current_pointer.get("action_count", 0)) if current_pointer else 0
        if k != committed:
            raise ResumeCheckpointError(f"expected action k={committed}, got k={k}")
        self._truncate_actions(committed)
        if agent_snapshot is None:
            raise ResumeCheckpointError("actor does not provide snapshot_state(); exact resume unavailable")

        actor_bytes = zlib.compress(
            pickle.dumps(agent_snapshot, protocol=pickle.HIGHEST_PROTOCOL), level=6
        )
        generation = {
            "schema": _JOURNAL_SCHEMA,
            "action_count": k + 1,
            "actor": self._store_blob(actor_bytes),
            "workspace": self._capture_workspace(Path(workspace_dir)),
        }
        generation_name = f"{k + 1:08d}-{uuid.uuid4().hex}.json"
        generation_bytes = json.dumps(
            generation, sort_keys=True, separators=(",", ":")
        ).encode()
        generation_path = self.generation_dir / generation_name
        _atomic_write_bytes(generation_path, generation_bytes)
        new_pointer = {
            "generation": generation_name,
            "action_count": k + 1,
            "sha256": hashlib.sha256(generation_bytes).hexdigest(),
        }

        record = {
            "k": k,
            "action": action,
            "x": x,
            "y": y,
            "frame_hash": frame_hash,
            "level": level,
            "state": state,
        }
        with self.actions_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

        if current_pointer is not None:
            _atomic_write_json(self.previous_path, current_pointer)
        _atomic_write_json(self.head_path, new_pointer)
        try:
            self._collect_garbage()
        except Exception:  # noqa: BLE001 - commit point has passed; collection is best-effort
            # HEAD is already committed. Stale immutable blobs are a storage issue, not grounds to
            # report the action as failed and risk replaying it twice.
            pass

    def _collect_garbage(self) -> None:
        pointers = [
            pointer
            for pointer in (
                self._read_pointer(self.head_path, required=False),
                self._read_pointer(self.previous_path, required=False),
            )
            if pointer is not None
        ]
        keep_generations = {pointer["generation"] for pointer in pointers}
        keep_blobs: set[str] = set()
        for pointer in pointers:
            generation = self._read_generation(pointer)
            actor = generation.get("actor", {})
            if isinstance(actor.get("sha256"), str):
                keep_blobs.add(actor["sha256"])
            for descriptor in generation.get("workspace", {}).get("files", {}).values():
                if isinstance(descriptor, dict) and isinstance(descriptor.get("sha256"), str):
                    keep_blobs.add(descriptor["sha256"])
        for path in self.generation_dir.iterdir():
            if path.is_file() and path.name not in keep_generations:
                path.unlink()
        for path in self.blob_dir.iterdir():
            if path.is_file() and path.name not in keep_blobs:
                path.unlink()
