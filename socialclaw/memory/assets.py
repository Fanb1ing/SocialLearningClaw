from __future__ import annotations

import hashlib
import io
import json
import os
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Optional

import numpy as np


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _json_copy(value: Dict[str, Any]) -> Dict[str, Any]:
    """Return a detached JSON-compatible copy or raise a useful error."""
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))
    except (TypeError, ValueError) as error:
        raise ValueError("Artifact metadata must be JSON serializable") from error


@dataclass(frozen=True)
class MemoryArtifactRef:
    """Content-addressed evidence stored outside a Memory/Trajectory JSON file."""

    artifact_id: str
    role: str
    media_type: str
    relative_path: str
    sha256: str
    size_bytes: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.artifact_id.strip():
            raise ValueError("artifact_id must not be empty")
        if not self.role.strip():
            raise ValueError("artifact role must not be empty")
        if not self.media_type.strip():
            raise ValueError("artifact media_type must not be empty")
        relative = PurePosixPath(self.relative_path)
        if (
            not self.relative_path
            or relative.is_absolute()
            or ".." in relative.parts
            or "." in relative.parts
        ):
            raise ValueError("artifact relative_path must stay below the asset root")
        if not _SHA256_RE.fullmatch(self.sha256):
            raise ValueError("artifact sha256 must be 64 lowercase hexadecimal characters")
        if self.size_bytes < 0:
            raise ValueError("artifact size_bytes must be non-negative")
        object.__setattr__(self, "metadata", _json_copy(self.metadata))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "role": self.role,
            "media_type": self.media_type,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "metadata": _json_copy(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "MemoryArtifactRef":
        return cls(
            artifact_id=str(payload["artifact_id"]),
            role=str(payload["role"]),
            media_type=str(payload["media_type"]),
            relative_path=str(payload["relative_path"]),
            sha256=str(payload["sha256"]),
            size_bytes=int(payload["size_bytes"]),
            metadata=dict(payload.get("metadata") or {}),
        )


class ContentAddressedArtifactStore:
    """Persist lossless arrays and rendered images once, addressed by content.

    References always use POSIX paths relative to ``root`` so a trajectory
    corpus can be moved between machines without rewriting JSON records.
    """

    GRID_MEDIA_TYPE = "application/x.numpy.ndarray"
    PNG_MEDIA_TYPE = "image/png"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def put_grid(
        self,
        grid: np.ndarray,
        *,
        role: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryArtifactRef:
        array = np.asarray(grid)
        if array.ndim != 2:
            raise ValueError("A grid artifact must be a two-dimensional array")
        if array.dtype.hasobject:
            raise ValueError("Object arrays are not valid grid evidence")
        canonical = np.ascontiguousarray(array, dtype=np.dtype("<i2"))
        logical_sha256 = self._logical_array_sha256(canonical)
        buffer = io.BytesIO()
        np.save(buffer, canonical, allow_pickle=False)
        payload = buffer.getvalue()
        relative_path = f"grids/{logical_sha256}.npy"
        self._write_once(relative_path, payload)
        merged_metadata = {
            **dict(metadata or {}),
            "logical_sha256": logical_sha256,
            "shape": list(canonical.shape),
            "dtype": canonical.dtype.str,
        }
        return self._reference(
            payload=payload,
            role=role,
            media_type=self.GRID_MEDIA_TYPE,
            relative_path=relative_path,
            metadata=merged_metadata,
        )

    def put_png(
        self,
        image,
        *,
        role: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryArtifactRef:
        """Store a PIL-compatible image using deterministic PNG encoding."""
        if not hasattr(image, "save") or not hasattr(image, "size"):
            raise TypeError("put_png expects a PIL-compatible image")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=False)
        payload = buffer.getvalue()
        content_sha256 = hashlib.sha256(payload).hexdigest()
        relative_path = f"images/{content_sha256}.png"
        self._write_once(relative_path, payload)
        merged_metadata = {
            **dict(metadata or {}),
            "width": int(image.size[0]),
            "height": int(image.size[1]),
            "mode": str(getattr(image, "mode", "")),
        }
        return self._reference(
            payload=payload,
            role=role,
            media_type=self.PNG_MEDIA_TYPE,
            relative_path=relative_path,
            metadata=merged_metadata,
        )

    def verify(self, reference: MemoryArtifactRef) -> Path:
        path = self.resolve(reference)
        if not path.is_file():
            raise FileNotFoundError(f"Missing artifact: {path}")
        payload = path.read_bytes()
        actual_sha256 = hashlib.sha256(payload).hexdigest()
        if actual_sha256 != reference.sha256:
            raise ValueError(
                f"Artifact hash mismatch for {reference.relative_path}: "
                f"expected {reference.sha256}, got {actual_sha256}"
            )
        if len(payload) != reference.size_bytes:
            raise ValueError(
                f"Artifact size mismatch for {reference.relative_path}: "
                f"expected {reference.size_bytes}, got {len(payload)}"
            )
        return path

    def load_grid(self, reference: MemoryArtifactRef) -> np.ndarray:
        if reference.media_type != self.GRID_MEDIA_TYPE:
            raise ValueError("Artifact is not a lossless grid")
        path = self.verify(reference)
        with path.open("rb") as handle:
            value = np.load(handle, allow_pickle=False)
        array = np.asarray(value)
        expected_logical = str(reference.metadata.get("logical_sha256", ""))
        actual_logical = self._logical_array_sha256(array)
        if expected_logical != actual_logical:
            raise ValueError(
                f"Logical grid hash mismatch for {reference.relative_path}: "
                f"expected {expected_logical}, got {actual_logical}"
            )
        return array

    def resolve(self, reference: MemoryArtifactRef) -> Path:
        relative = PurePosixPath(reference.relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Artifact reference escapes the asset root")
        return self.root.joinpath(*relative.parts)

    def _reference(
        self,
        *,
        payload: bytes,
        role: str,
        media_type: str,
        relative_path: str,
        metadata: Dict[str, Any],
    ) -> MemoryArtifactRef:
        content_sha256 = hashlib.sha256(payload).hexdigest()
        return MemoryArtifactRef(
            artifact_id=f"artifact_{content_sha256[:20]}",
            role=role,
            media_type=media_type,
            relative_path=relative_path,
            sha256=content_sha256,
            size_bytes=len(payload),
            metadata=metadata,
        )

    def _write_once(self, relative_path: str, payload: bytes) -> None:
        path = self.root.joinpath(*PurePosixPath(relative_path).parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.read_bytes() != payload:
                raise ValueError(f"Content-address collision at {path}")
            return
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    @staticmethod
    def _logical_array_sha256(array: np.ndarray) -> str:
        canonical = np.ascontiguousarray(array)
        digest = hashlib.sha256()
        digest.update(b"socialclaw-array-v1\0")
        digest.update(canonical.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(canonical.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(canonical.tobytes(order="C"))
        return digest.hexdigest()
