from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..memory import ContentAddressedArtifactStore
from .models import TrajectoryEpisode
from .recorder import TrajectoryRecorder


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json_atomic(path: str | Path, payload: Dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def load_corpus_episodes(root: str | Path) -> List[TrajectoryEpisode]:
    base = Path(root)
    return [
        TrajectoryRecorder.load(path)
        for path in sorted((base / "episodes").glob("*.json"))
    ]


def validate_corpus(root: str | Path) -> Dict[str, Any]:
    base = Path(root)
    store = ContentAddressedArtifactStore(base / "assets")
    episodes = load_corpus_episodes(base)
    errors: List[str] = []
    artifact_refs = {}
    for episode in episodes:
        try:
            episode.validate()
        except ValueError as error:
            errors.append(f"{episode.episode_id}: {error}")
        observations = [episode.initial_observation]
        observations.extend(step.result.observation for step in episode.steps)
        for observation in observations:
            for reference in observation.artifacts:
                artifact_refs[(reference.artifact_id, reference.relative_path)] = reference
    for reference in artifact_refs.values():
        try:
            store.verify(reference)
            if reference.media_type == store.GRID_MEDIA_TYPE:
                store.load_grid(reference)
        except (OSError, ValueError) as error:
            errors.append(f"{reference.artifact_id}: {error}")
    manifest_path = base / "manifest.json"
    split_membership: Dict[str, str] = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            recorded_ids = [episode.episode_id for episode in episodes]
            manifest_ids = [str(item) for item in manifest.get("episode_ids", [])]
            if len(manifest_ids) != len(set(manifest_ids)):
                errors.append("manifest episode_ids contain duplicates")
            if sorted(manifest_ids) != recorded_ids:
                errors.append("manifest episode_ids do not match episode files")
            for split_name, relative_path in (manifest.get("splits") or {}).items():
                split_path = (base / str(relative_path)).resolve()
                try:
                    split_path.relative_to(base.resolve())
                except ValueError:
                    errors.append(f"split {split_name}: path escapes corpus root")
                    continue
                payload = json.loads(split_path.read_text(encoding="utf-8"))
                for episode_id in payload.get("episode_ids", []):
                    if episode_id not in recorded_ids:
                        errors.append(f"split {split_name}: unknown episode {episode_id}")
                    if episode_id in split_membership:
                        errors.append(
                            f"episode {episode_id} appears in both "
                            f"{split_membership[episode_id]} and {split_name}"
                        )
                    split_membership[episode_id] = split_name
            missing_from_splits = sorted(set(recorded_ids) - set(split_membership))
            if missing_from_splits:
                errors.append(
                    "episodes missing from splits: " + ", ".join(missing_from_splits)
                )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"manifest/split validation failed: {error}")
    validation = {
        "format_version": 1,
        "status": "passed" if not errors else "failed",
        "episode_count": len(episodes),
        "step_count": sum(len(episode.steps) for episode in episodes),
        "unique_artifact_count": len(artifact_refs),
        "errors": errors,
    }
    write_json_atomic(base / "validation.json", validation)
    return validation


def write_corpus_metadata(
    root: str | Path,
    *,
    corpus_id: str,
    benchmark: str,
    game_id: str,
    environment_fingerprint: str,
    collector: Dict[str, Any],
    episodes: Iterable[TrajectoryEpisode],
    splits: Dict[str, List[str]],
    coverage: Optional[Dict[str, Any]] = None,
) -> Dict[str, Path]:
    base = Path(root)
    values = list(episodes)
    manifest = {
        "format_version": 1,
        "corpus_id": corpus_id,
        "created_at": _utc_now(),
        "benchmark": benchmark,
        "game_id": game_id,
        "environment_fingerprint": environment_fingerprint,
        "collector": collector,
        "episode_ids": [item.episode_id for item in values],
        "episode_count": len(values),
        "step_count": sum(len(item.steps) for item in values),
        "splits": {name: f"splits/{name}.json" for name in sorted(splits)},
    }
    paths = {
        "manifest": write_json_atomic(base / "manifest.json", manifest),
    }
    if coverage is not None:
        paths["coverage"] = write_json_atomic(base / "coverage.json", coverage)
    for name, ids in splits.items():
        paths[f"split_{name}"] = write_json_atomic(
            base / "splits" / f"{name}.json",
            {"format_version": 1, "split": name, "episode_ids": ids},
        )
    validate_corpus(base)
    paths["validation"] = base / "validation.json"
    return paths
