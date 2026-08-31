from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Protocol, Sequence, Union

from .models import (
    Action,
    Observation,
    TrajectoryEpisode,
    TrajectoryOutcome,
    TrajectoryStep,
)


@dataclass(frozen=True)
class EpisodeStarted:
    episode: TrajectoryEpisode

    def __post_init__(self) -> None:
        if self.episode.steps or self.episode.terminal_outcome is not None:
            raise ValueError("EpisodeStarted requires an empty, unfinished episode")


@dataclass(frozen=True)
class StepObserved:
    episode_id: str
    step: TrajectoryStep


@dataclass(frozen=True)
class EpisodeFinished:
    episode_id: str
    outcome: TrajectoryOutcome


TrajectoryEvent = Union[EpisodeStarted, StepObserved, EpisodeFinished]


class TrajectorySource(Protocol):
    """A replaceable stream of normalized episode lifecycle events."""

    name: str

    def events(self) -> Iterator[TrajectoryEvent]:
        ...


class TrajectoryDomainAdapter(Protocol):
    """Benchmark-specific normalization behind the common trajectory contract."""

    name: str

    def normalize_observation(self, raw_observation: Any) -> Observation:
        ...

    def normalize_action(self, raw_action: Any) -> Action:
        ...

    def describe_transition(
        self,
        pre_observation: Observation,
        action: Action,
        post_observation: Observation,
        raw_result: Any,
    ) -> dict[str, Any]:
        ...

    def terminal_outcome(self, raw_result: Any) -> TrajectoryOutcome | None:
        ...

    def is_informative(self, transition_features: dict[str, Any]) -> bool:
        ...


class IterableTrajectorySource:
    """Replay complete episodes as the same event stream used by online actors."""

    name = "iterable"

    def __init__(self, episodes: Iterable[TrajectoryEpisode]) -> None:
        self._episodes = list(episodes)

    def episodes(self) -> Iterator[TrajectoryEpisode]:
        yield from self._episodes

    def events(self) -> Iterator[TrajectoryEvent]:
        for episode in self._episodes:
            start = TrajectoryEpisode(
                episode_id=episode.episode_id,
                benchmark=episode.benchmark,
                task_id=episode.task_id,
                split=episode.split,
                actor=episode.actor,
                evidence_tier=episode.evidence_tier,
                initial_observation=episode.initial_observation,
                provenance=episode.provenance,
                metadata=episode.metadata,
                created_at=episode.created_at,
            )
            yield EpisodeStarted(start)
            for step in episode.steps:
                yield StepObserved(episode.episode_id, step)
            if episode.terminal_outcome is not None:
                yield EpisodeFinished(episode.episode_id, episode.terminal_outcome)


class ActionPolicy(Protocol):
    """Optional interface for scripted, coverage, human, or Agent actors."""

    name: str

    def choose_action(
        self,
        observation: Observation,
        available_actions: Sequence[Action],
        *,
        step_index: int,
    ) -> Action:
        ...
