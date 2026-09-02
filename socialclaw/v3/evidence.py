"""Stable, typed references to evidence recorded by a Tycho workspace.

Evidence IDs are derived only from portable run/game/interaction identity and
content hashes. Absolute paths, credentials, model responses, and simulated
world-model states never enter the identifier.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


EVIDENCE_INDEX_SCHEMA = 1


class EvidenceRole(str, Enum):
    DECISION = "decision"
    LEVEL_INITIALIZATION = "level_initialization"
    TRANSIENT_ANIMATION = "transient_animation"
    COMPLETION_TERMINAL = "completion_terminal"
    FATAL_TERMINAL = "fatal_terminal"
    RESET_ARCHIVE = "reset_archive"


class EvidenceError(ValueError):
    """Base error for malformed or inconsistent durable evidence."""


class EvidenceClosureError(EvidenceError):
    """A cognitive record cites evidence absent from the durable index."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _action_dict(value: Mapping[str, Any] | str | None) -> dict[str, Any]:
    if isinstance(value, str):
        return {"action": value, "row": None, "col": None}
    value = dict(value or {})
    return {
        "action": str(value.get("action") or ""),
        "row": value.get("row"),
        "col": value.get("col"),
    }


@dataclass(frozen=True)
class EvidenceRef:
    evidence_id: str
    run_id: str
    game_id: str
    role: str
    relative_path: str
    level: int | None
    attempt: int | None
    turn: int | None
    action: dict[str, Any]
    frame_sha256: str
    metadata: dict[str, Any]

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        game_id: str,
        role: EvidenceRole | str,
        relative_path: str,
        level: int | None,
        attempt: int | None,
        turn: int | None,
        action: Mapping[str, Any] | str | None,
        frame_sha256: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> "EvidenceRef":
        role_value = EvidenceRole(role).value
        if not run_id.strip() or not game_id.strip():
            raise EvidenceError("run_id and game_id must be non-empty")
        if not relative_path or Path(relative_path).is_absolute():
            raise EvidenceError("relative_path must be a non-empty relative path")
        if len(frame_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in frame_sha256):
            raise EvidenceError("frame_sha256 must be a lowercase SHA-256 digest")
        action_value = _action_dict(action)
        identity = {
            "run_id": run_id,
            "game_id": game_id,
            "role": role_value,
            "relative_path": Path(relative_path).as_posix(),
            "level": level,
            "attempt": attempt,
            "turn": turn,
            "action": action_value,
            "frame_sha256": frame_sha256,
        }
        evidence_id = "evi_" + _sha256(_canonical(identity))[:32]
        return cls(
            evidence_id=evidence_id,
            metadata=dict(metadata or {}),
            **identity,
        )

    def validate(self) -> None:
        rebuilt = EvidenceRef.create(
            run_id=self.run_id,
            game_id=self.game_id,
            role=self.role,
            relative_path=self.relative_path,
            level=self.level,
            attempt=self.attempt,
            turn=self.turn,
            action=self.action,
            frame_sha256=self.frame_sha256,
            metadata=self.metadata,
        )
        if rebuilt.evidence_id != self.evidence_id:
            raise EvidenceError(f"evidence ID does not match its identity: {self.evidence_id}")


@dataclass(frozen=True)
class EvidenceIndex:
    run_id: str
    game_id: str
    evidence: tuple[EvidenceRef, ...]
    schema: int = EVIDENCE_INDEX_SCHEMA

    @property
    def ids(self) -> frozenset[str]:
        return frozenset(item.evidence_id for item in self.evidence)

    def validate(self) -> None:
        if self.schema != EVIDENCE_INDEX_SCHEMA:
            raise EvidenceError(f"unsupported evidence index schema {self.schema}")
        seen: set[str] = set()
        for item in self.evidence:
            item.validate()
            if item.run_id != self.run_id or item.game_id != self.game_id:
                raise EvidenceError(f"evidence {item.evidence_id} belongs to another run or game")
            if item.evidence_id in seen:
                raise EvidenceError(f"duplicate evidence ID {item.evidence_id}")
            seen.add(item.evidence_id)

    def require(self, evidence_ids: Iterable[str]) -> None:
        requested = {str(value) for value in evidence_ids}
        missing = sorted(requested - self.ids)
        if missing:
            raise EvidenceClosureError("unknown Evidence IDs: " + ", ".join(missing))

    def to_json(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "game_id": self.game_id,
            "evidence": [asdict(item) for item in self.evidence],
        }

    def write(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(self.to_json(), indent=2, sort_keys=True) + "\n")
        temporary.replace(target)

    @classmethod
    def read(cls, path: str | Path) -> "EvidenceIndex":
        data = json.loads(Path(path).read_text())
        index = cls(
            schema=int(data.get("schema", 0)),
            run_id=str(data["run_id"]),
            game_id=str(data["game_id"]),
            evidence=tuple(EvidenceRef(**item) for item in data.get("evidence", [])),
        )
        index.validate()
        return index


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(errors="replace"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"cannot parse evidence metadata {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"evidence metadata must be an object: {path}")
    return value


def _content_hash(value: Any) -> str:
    return _sha256(_canonical(value))


def index_workspace(
    workspace: str | Path,
    *,
    run_id: str,
    game_id: str | None = None,
) -> EvidenceIndex:
    """Index observed Tycho evidence without importing or executing a world model."""

    root = Path(workspace).resolve(strict=True)
    game = game_id or root.name
    refs: list[EvidenceRef] = []

    def add(
        path: Path,
        *,
        role: EvidenceRole,
        level: int | None,
        attempt: int | None,
        turn: int | None,
        action: Mapping[str, Any] | str | None,
        frame_value: Any,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        refs.append(EvidenceRef.create(
            run_id=run_id,
            game_id=game,
            role=role,
            relative_path=path.relative_to(root).as_posix(),
            level=level,
            attempt=attempt,
            turn=turn,
            action=action,
            frame_sha256=_content_hash(frame_value),
            metadata=metadata,
        ))

    def scan_level(level_dir: Path, *, level: int, attempt: int) -> None:
        for meta_path in sorted(level_dir.glob("turn_*.json")):
            meta = _read_json(meta_path)
            turn = int(meta.get("turn", meta_path.stem.rsplit("_", 1)[-1]))
            frame_path = meta_path.with_suffix(".txt")
            frame = frame_path.read_text(errors="replace") if frame_path.exists() else meta
            role = EvidenceRole.LEVEL_INITIALIZATION if turn == 0 else EvidenceRole.DECISION
            add(
                meta_path,
                role=role,
                level=level,
                attempt=attempt,
                turn=turn,
                action=meta,
                frame_value=frame,
                metadata={"state": meta.get("state", ""), "available": meta.get("available", [])},
            )

        terminal_path = level_dir / "terminal.json"
        if terminal_path.exists():
            event = _read_json(terminal_path)
            add(
                terminal_path,
                role=EvidenceRole.COMPLETION_TERMINAL,
                level=level,
                attempt=attempt,
                turn=event.get("pre_turn"),
                action=event.get("action"),
                frame_value=event.get("terminal_grid", event),
                metadata={"outcome": event.get("outcome", "win")},
            )

        for death_path in sorted(level_dir.glob("death_*.json")):
            event = _read_json(death_path)
            add(
                death_path,
                role=EvidenceRole.FATAL_TERMINAL,
                level=level,
                attempt=int(event.get("efps_attempt", attempt)),
                turn=event.get("turn"),
                action=event,
                frame_value={"prev": event.get("prev"), "next": event.get("next")},
                metadata={"state": event.get("state", "GAME_OVER")},
            )

        for animation_meta in sorted(level_dir.glob("animation_*/meta.json")):
            event = _read_json(animation_meta)
            selected = []
            for item in event.get("selected_frame_files") or []:
                rel = item.get("txt") if isinstance(item, dict) else None
                candidate = root / rel if rel else None
                if candidate is not None and not candidate.exists():
                    candidate = animation_meta.parent / Path(rel).name
                if candidate is not None and candidate.exists():
                    selected.append(candidate.read_text(errors="replace"))
            add(
                animation_meta,
                role=EvidenceRole.TRANSIENT_ANIMATION,
                level=level,
                attempt=attempt,
                turn=event.get("turn"),
                action=event,
                frame_value=selected or event,
                metadata={
                    "terminal": event.get("terminal", "nonterminal"),
                    "selected_frame_indices": event.get("selected_frame_indices", []),
                },
            )

    archived_attempts: dict[int, int] = {}
    attempts_dir = root / "attempts"
    if attempts_dir.exists():
        for attempt_dir in sorted(attempts_dir.glob("level_*_attempt_*")):
            manifest_path = attempt_dir / "attempt.json"
            if not manifest_path.exists():
                continue
            manifest = _read_json(manifest_path)
            level = int(manifest["level"])
            attempt = int(manifest["attempt"])
            archived_attempts[level] = max(archived_attempts.get(level, -1), attempt)
            add(
                manifest_path,
                role=EvidenceRole.RESET_ARCHIVE,
                level=level,
                attempt=attempt,
                turn=None,
                action="RESET",
                frame_value=manifest,
                metadata={"reason": manifest.get("reason", "reset"), "n_frames": manifest.get("n_frames")},
            )
            archived_level = attempt_dir / f"level_{level}"
            if archived_level.exists():
                scan_level(archived_level, level=level, attempt=attempt)

    for level_dir in sorted(root.glob("level_*")):
        if not level_dir.is_dir():
            continue
        try:
            level = int(level_dir.name.split("_", 1)[1])
        except ValueError:
            continue
        active_attempt = archived_attempts.get(level, -1) + 1
        scan_level(level_dir, level=level, attempt=active_attempt)

    refs.sort(key=lambda item: (
        item.level if item.level is not None else -1,
        item.attempt if item.attempt is not None else -1,
        item.turn if item.turn is not None else -1,
        item.role,
        item.relative_path,
    ))
    index = EvidenceIndex(run_id=run_id, game_id=game, evidence=tuple(refs))
    index.validate()
    return index
