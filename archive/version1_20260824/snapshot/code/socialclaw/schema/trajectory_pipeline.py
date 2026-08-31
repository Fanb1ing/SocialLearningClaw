from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from ..memory import JsonMemoryStore, MemoryKind, MemoryRecord
from ..trajectory import TrajectoryEpisode, load_corpus_episodes
from .layered_graph import LayeredSchemaGraph
from .layered_storage import LayeredSchemaStorage
from .node import MemoryIndex, SchemaNode


def _stable_id(prefix: str, payload: Dict) -> str:
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _artifact_dicts(observation) -> List[Dict]:
    return [item.to_dict() for item in observation.artifacts]


@dataclass(frozen=True)
class ProjectionReport:
    episode_count: int
    transition_memory_count: int
    window_memory_count: int
    episode_memory_count: int

    def to_dict(self) -> Dict[str, int]:
        return dict(self.__dict__)


class TrajectoryMemoryProjector:
    """Project immutable trajectories into durable, deterministic MemoryRecords."""

    def __init__(self, store: JsonMemoryStore, *, window_size: int = 8) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.store = store
        self.window_size = window_size

    def project(
        self,
        episodes: Iterable[TrajectoryEpisode],
        *,
        corpus_root: str | Path | None = None,
    ) -> ProjectionReport:
        transition_count = window_count = episode_count = 0
        projected: List[MemoryRecord] = []
        for episode in episodes:
            transition_ids: List[str] = []
            for step in episode.steps:
                memory = self._transition(episode, step, corpus_root=corpus_root)
                projected.append(memory)
                transition_ids.append(memory.id)
                transition_count += 1
            window_ids = []
            for start in range(0, len(transition_ids), self.window_size):
                ids = transition_ids[start : start + self.window_size]
                memory = self._window(episode, start, ids, corpus_root=corpus_root)
                projected.append(memory)
                window_ids.append(memory.id)
                window_count += 1
            projected.append(
                self._episode(
                    episode, transition_ids, window_ids, corpus_root=corpus_root
                )
            )
            episode_count += 1
        self.store.put_many(projected)
        return ProjectionReport(
            episode_count=episode_count,
            transition_memory_count=transition_count,
            window_memory_count=window_count,
            episode_memory_count=episode_count,
        )

    @staticmethod
    def _transition(episode, step, *, corpus_root=None) -> MemoryRecord:
        delta = step.result.state_delta
        role = str(step.action.arguments.get("target_role", ""))
        payload = {
            "benchmark": episode.benchmark,
            "task_id": episode.task_id,
            "episode_id": episode.episode_id,
            "step_index": step.step_index,
        }
        record = MemoryRecord(
            id=_stable_id("memory_transition", payload),
            kind=MemoryKind.EPISODE,
            task=episode.task_id,
            context=(
                f"benchmark={episode.benchmark}; game={episode.task_id}; "
                f"level={step.observation.structured.get('level')}; "
                f"target_role={role or 'none'}"
            ),
            outcome=step.result.environment_status,
            success=(True if delta.get("level_delta", 0) > 0 else None),
            tags=["trajectory_transition", episode.benchmark, episode.evidence_tier.value],
            metadata={
                "memory_scope": "transition",
                "episode_id": episode.episode_id,
                "step_index": step.step_index,
                "benchmark": episode.benchmark,
                "game_id": episode.task_id,
                "level": step.observation.structured.get("level"),
                "action_name": step.action.name,
                "action_arguments": step.action.arguments,
                "target_role": role,
                "task_state_changed": bool(delta.get("task_state_changed")),
                "task_changed_cells": delta.get("task_changed_cells"),
                "changed_regions": list(delta.get("changed_regions") or []),
                "level_delta": int(delta.get("level_delta", 0) or 0),
                "environment_status": step.result.environment_status,
                "evidence_tier": episode.evidence_tier.value,
                "corpus_root": str(Path(corpus_root).resolve()) if corpus_root else "",
                "trajectory_path": f"episodes/{episode.episode_id}.json",
                "pre_artifacts": _artifact_dicts(step.observation),
                "post_artifacts": _artifact_dicts(step.result.observation),
                "pre_grid_shape": list(
                    step.observation.structured.get("grid_shape")
                    or step.observation.structured.get("shape")
                    or []
                ),
                "post_grid_shape": list(
                    step.result.observation.structured.get("grid_shape")
                    or step.result.observation.structured.get("shape")
                    or []
                ),
            },
        )
        record.add_event(
            observation=(
                f"level={record.metadata['level']}; action available; "
                f"pre_grid={step.observation.structured.get('logical_grid_sha256')}"
            ),
            action=json.dumps(step.action.to_dict(), ensure_ascii=False, sort_keys=True),
            result=(
                f"task_state_changed={record.metadata['task_state_changed']}; "
                f"task_changed_cells={record.metadata['task_changed_cells']}; "
                f"level_delta={record.metadata['level_delta']}; "
                f"status={record.metadata['environment_status']}"
            ),
            metadata={"trajectory_step_index": step.step_index},
        )
        return record

    @staticmethod
    def _window(
        episode, start: int, source_ids: Sequence[str], *, corpus_root=None
    ) -> MemoryRecord:
        payload = {
            "benchmark": episode.benchmark,
            "task_id": episode.task_id,
            "episode_id": episode.episode_id,
            "window_start": start,
        }
        return MemoryRecord(
            id=_stable_id("memory_window", payload),
            kind=MemoryKind.KNOWLEDGE,
            task=episode.task_id,
            context=f"Steps {start}..{start + len(source_ids) - 1} of {episode.episode_id}",
            outcome="trajectory window summary",
            tags=["trajectory_window", episode.benchmark],
            metadata={
                "memory_scope": "window_summary",
                "episode_id": episode.episode_id,
                "corpus_root": str(Path(corpus_root).resolve()) if corpus_root else "",
                "source_memory_ids": list(source_ids),
                "start_step": start,
                "end_step": start + len(source_ids) - 1,
            },
        )

    @staticmethod
    def _episode(
        episode, transition_ids, window_ids, *, corpus_root=None
    ) -> MemoryRecord:
        outcome = episode.terminal_outcome
        return MemoryRecord(
            id=_stable_id(
                "memory_episode",
                {
                    "benchmark": episode.benchmark,
                    "task_id": episode.task_id,
                    "episode_id": episode.episode_id,
                },
            ),
            kind=MemoryKind.SKILL,
            task=episode.task_id,
            context=f"Complete recorded episode {episode.episode_id}",
            outcome=outcome.status if outcome else "UNFINALIZED",
            success=outcome.success if outcome else None,
            tags=["trajectory_episode", episode.benchmark, episode.evidence_tier.value],
            metadata={
                "memory_scope": "level_episode",
                "episode_id": episode.episode_id,
                "corpus_root": str(Path(corpus_root).resolve()) if corpus_root else "",
                "source_memory_ids": list(transition_ids),
                "window_memory_ids": list(window_ids),
                "step_count": len(transition_ids),
                "evidence_tier": episode.evidence_tier.value,
                "terminal_outcome": outcome.to_dict() if outcome else None,
            },
        )


class PrototypeTrajectorySchemaInducer:
    """Deterministic baseline that groups transition evidence into atomic rules.

    This is deliberately a conservative prototype: it only emits groups with
    repeated evidence and never reads Gold schemas or environment source.
    """

    def __init__(self, *, min_support: int = 2) -> None:
        if min_support < 1:
            raise ValueError("min_support must be at least one")
        self.min_support = min_support

    def induce(self, memories: Iterable[MemoryRecord]) -> LayeredSchemaGraph:
        groups: Dict[Tuple, List[MemoryRecord]] = defaultdict(list)
        for record in memories:
            meta = record.metadata
            if meta.get("memory_scope") != "transition":
                continue
            effect = "effect" if meta.get("task_state_changed") else "no_effect"
            if int(meta.get("level_delta", 0)) > 0:
                effect = "level_completion"
            if meta.get("environment_status") in {"WIN", "GAME_OVER"}:
                effect = str(meta["environment_status"]).lower()
            key = (
                str(meta.get("game_id")),
                str(meta.get("action_name")),
                str(meta.get("target_role", "")),
                effect,
            )
            groups[key].append(record)

        graph = LayeredSchemaGraph()
        for key, records in sorted(groups.items()):
            if len(records) < self.min_support:
                continue
            game_id, action_name, target_role, effect = key
            role_text = f" on {target_role}" if target_role else ""
            levels = sorted({int(item.metadata.get("level", 0)) for item in records})
            changed = [
                int(item.metadata["task_changed_cells"])
                for item in records
                if item.metadata.get("task_changed_cells") is not None
            ]
            trigger = f"In {game_id} at levels {levels}, {action_name}{role_text} is available"
            if effect == "effect":
                expectation = (
                    "The task state changes"
                    + (f" (observed changed-cell range {min(changed)}..{max(changed)})" if changed else "")
                )
            elif effect == "no_effect":
                expectation = "The task state does not change; UI counters may still change"
            elif effect == "level_completion":
                expectation = "The action completes the current level"
            elif effect == "win":
                expectation = "The action completes the final level and produces WIN"
            else:
                expectation = "The action exhausts a terminal condition and produces GAME_OVER"
            node = SchemaNode.from_rule(
                level=2 if len(levels) > 1 else 3,
                trigger=trigger,
                action_sequence=[action_name + role_text],
                expectation=expectation,
                source_memory_id=records[0].id,
                reliability_weight=min(0.9, 0.5 + 0.04 * len(records)),
            )
            node.index = _stable_id(
                "schema", {"game": game_id, "action": action_name, "role": target_role, "effect": effect}
            )
            node.memory_index = MemoryIndex(source=sorted(item.id for item in records))
            node.metadata.update(
                {
                    "prototype_algorithm": "transition_bucket_v1",
                    "game_id": game_id,
                    "effect_class": effect,
                    "support_count": len(records),
                    "level_scope": levels,
                    "schema_kind": "state_transition" if effect in {"level_completion", "win", "game_over"} else "action_effect",
                }
            )
            graph.add(node)
        graph.validate(memory_ids={item.id for values in groups.values() for item in values})
        return graph


def run_prototype_pipeline(
    corpus_roots: Sequence[str | Path], output_root: str | Path, *, window_size: int = 8, min_support: int = 2
) -> Dict:
    output = Path(output_root)
    memory_store = JsonMemoryStore(output / "memory.json")
    projector = TrajectoryMemoryProjector(memory_store, window_size=window_size)
    reports = [
        projector.project(load_corpus_episodes(root), corpus_root=root)
        for root in corpus_roots
    ]
    report = ProjectionReport(
        episode_count=sum(item.episode_count for item in reports),
        transition_memory_count=sum(item.transition_memory_count for item in reports),
        window_memory_count=sum(item.window_memory_count for item in reports),
        episode_memory_count=sum(item.episode_memory_count for item in reports),
    )
    graph = PrototypeTrajectorySchemaInducer(min_support=min_support).induce(memory_store.list())
    LayeredSchemaStorage(output / "schema.json").save(graph)
    graph.validate(memory_ids={item.id for item in memory_store.list()})
    summary = {
        "format_version": 1,
        "algorithm": "transition_bucket_v1",
        "corpora": [str(Path(item).resolve()) for item in corpus_roots],
        "projection": report.to_dict(),
        "memory_count": len(memory_store),
        "schema_count": len(graph.list(include_inactive=True)),
        "schema_counts_by_game": dict(sorted(
            (game, sum(node.metadata.get("game_id") == game for node in graph.list(include_inactive=True)))
            for game in sorted({node.metadata.get("game_id") for node in graph.list(include_inactive=True)})
        )),
        "network_calls": 0,
        "gold_schema_reads": 0,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary
