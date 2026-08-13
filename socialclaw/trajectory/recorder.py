from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from .models import FORMAT_VERSION, TrajectoryEpisode, TrajectoryOutcome, TrajectoryStep


class TrajectoryRecorder:
    """Atomically persist a normalized trajectory after every accepted step."""

    def __init__(
        self,
        root: str | Path,
        episode: TrajectoryEpisode,
        *,
        resume: bool = False,
    ) -> None:
        self.root = Path(root)
        self.path = self.root / "episodes" / f"{episode.episode_id}.json"
        if self.path.exists():
            if not resume:
                raise FileExistsError(f"Trajectory already exists: {self.path}")
            loaded = self.load(self.path)
            self._check_identity(episode, loaded)
            self.episode = loaded
        else:
            self.episode = episode
            self._persist(episode)

    def record_step(self, step: TrajectoryStep) -> TrajectoryEpisode:
        candidate = self.episode.with_step(step)
        self._persist(candidate)
        self.episode = candidate
        return candidate

    def finalize(self, outcome: TrajectoryOutcome) -> TrajectoryEpisode:
        candidate = self.episode.finalized(outcome)
        self._persist(candidate)
        self.episode = candidate
        return candidate

    @staticmethod
    def load(path: str | Path) -> TrajectoryEpisode:
        source = Path(path)
        with source.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"Unsupported trajectory format in {source}")
        episode_payload = payload.get("episode")
        if not isinstance(episode_payload, dict):
            raise ValueError(f"Missing trajectory episode in {source}")
        return TrajectoryEpisode.from_dict(episode_payload)

    def _persist(self, episode: TrajectoryEpisode) -> None:
        episode.validate()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        payload = episode.envelope()
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary.exists():
                temporary.unlink()

    @staticmethod
    def _check_identity(
        expected: TrajectoryEpisode, actual: TrajectoryEpisode
    ) -> None:
        fields = ("episode_id", "benchmark", "task_id", "split", "actor", "evidence_tier")
        mismatched = [
            field_name
            for field_name in fields
            if getattr(expected, field_name) != getattr(actual, field_name)
        ]
        if mismatched:
            raise ValueError(
                "Cannot resume a different trajectory; mismatched fields: "
                + ", ".join(mismatched)
            )
        if (
            expected.initial_observation.content_fingerprint()
            != actual.initial_observation.content_fingerprint()
        ):
            raise ValueError("Cannot resume with a different initial observation")
