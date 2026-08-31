from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ...dataset.arc_agi3 import ARCAGI3EnvWrapper
from ...memory import ContentAddressedArtifactStore
from ...trajectory import (
    Action,
    Decision,
    EvidenceTier,
    Observation,
    StepResult,
    TrajectoryEpisode,
    TrajectoryOutcome,
    TrajectoryRecorder,
    TrajectoryStep,
)


def _state_name(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw).removeprefix("GameState.")


class PublicARCSession:
    """Generic public ARC gateway with no game-semantic perception.

    The environment implementation necessarily executes behind this boundary;
    cognitive Agents receive only the normalized public observation returned by
    the environment and cannot access the environment object or its files.
    """

    def __init__(
        self,
        *,
        game_id: str,
        root: str | Path,
        episode_id: str,
        actor: str,
        provenance: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        episode_created_at: str | None = None,
        env: Optional[ARCAGI3EnvWrapper] = None,
    ) -> None:
        self.root = Path(root)
        self.game_id = game_id
        self.env = env or ARCAGI3EnvWrapper(game_id)
        if self.env.game_id != game_id:
            raise ValueError("Provided environment does not match the requested game")
        self.store = ContentAddressedArtifactStore(self.root / "assets")
        self.raw_observation = self.env.reset()
        self.observation = self._normalize(self.raw_observation)
        episode_values: Dict[str, Any] = {}
        if episode_created_at:
            episode_values["created_at"] = episode_created_at
        episode = TrajectoryEpisode(
            episode_id=episode_id,
            benchmark="arc_agi3",
            task_id=game_id,
            split="development",
            actor=actor,
            evidence_tier=EvidenceTier.NATURAL,
            initial_observation=self.observation,
            provenance=provenance,
            metadata=dict(metadata or {}),
            **episode_values,
        )
        self.recorder = TrajectoryRecorder(self.root, episode)

    @property
    def episode(self) -> TrajectoryEpisode:
        return self.recorder.episode

    @property
    def asset_root(self) -> Path:
        return self.root / "assets"

    def _normalize(self, raw: Any) -> Observation:
        if raw is None or not getattr(raw, "frame", None):
            raise ValueError("ARC observation does not contain a public frame")
        grid = np.asarray(raw.frame[-1])
        grid_ref = self.store.put_grid(grid, role="environment_state")
        agent_image = ARCAGI3EnvWrapper.grid_to_image(
            grid, cell_size=8, grid_step=None
        )
        agent_image_ref = self.store.put_png(
            agent_image,
            role="agent_view",
            metadata={
                "renderer": "public_grid_to_image",
                "cell_size": 8,
                "grid_overlay": False,
                "audience": "cognitive_agent",
            },
        )
        review_image = ARCAGI3EnvWrapper.grid_to_image(
            grid, cell_size=8, grid_step=8
        )
        review_image_ref = self.store.put_png(
            review_image,
            role="review_view",
            metadata={
                "renderer": "public_grid_to_image",
                "cell_size": 8,
                "grid_overlay": True,
                "grid_step": 8,
                "audience": "human_review_only",
            },
        )
        available_ids = [int(item) for item in getattr(raw, "available_actions", []) or []]
        return Observation(
            text=(
                "Public visual observation. No object labels, action meanings, "
                "goal description, or hidden state are supplied."
            ),
            structured={
                "environment_status": _state_name(raw.state),
                "levels_completed": int(getattr(raw, "levels_completed", 0) or 0),
                "available_action_ids": available_ids,
                "grid_shape": [int(grid.shape[0]), int(grid.shape[1])],
                "logical_grid_sha256": grid_ref.metadata["logical_sha256"],
            },
            artifacts=[grid_ref, agent_image_ref, review_image_ref],
        )

    def available_action_contracts(self) -> List[Dict[str, Any]]:
        contracts = []
        for action in self.env.get_available_actions(self.raw_observation):
            schema = action.action_type.model_json_schema()
            properties = {
                key: dict(value)
                for key, value in dict(schema.get("properties") or {}).items()
            }
            properties.pop("game_id", None)
            if "x" in properties:
                properties["x"]["description"] = (
                    "Public display x coordinate: column, increasing left to right."
                )
            if "y" in properties:
                properties["y"]["description"] = (
                    "Public display y coordinate: row, increasing top to bottom."
                )
            required = [
                item for item in schema.get("required") or [] if item != "game_id"
            ]
            contracts.append(
                {
                    "name": action.name,
                    "arguments_schema": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                        "additionalProperties": False,
                    },
                }
            )
        return contracts

    def execute(
        self,
        action: Action,
        *,
        rationale: str,
        schemas_used: List[str],
        decision_metadata: Dict[str, Any],
    ) -> TrajectoryStep:
        available = {
            item.name: item
            for item in self.env.get_available_actions(self.raw_observation)
        }
        if action.name not in available:
            raise ValueError(f"ARC action {action.name!r} is unavailable")
        before = self.observation
        before_grid = self._load_grid(before)
        next_raw = self.env.step(available[action.name], data=dict(action.arguments))
        after = self._normalize(next_raw)
        after_grid = self._load_grid(after)
        delta = self._public_delta(before, before_grid, after, after_grid, next_raw)
        generic_actions = [
            Action(item["name"]) for item in self.available_action_contracts()
        ]
        step = TrajectoryStep(
            step_index=len(self.episode.steps),
            observation=before,
            available_actions=generic_actions,
            action=action,
            result=StepResult(
                observation=after,
                environment_status=_state_name(next_raw.state),
                state_delta=delta,
            ),
            decision=Decision(
                response=json.dumps(action.to_dict(), ensure_ascii=False),
                rationale=rationale,
                retrieved_schema_ids=list(schemas_used),
                claimed_schema_ids=list(schemas_used),
                metadata=dict(decision_metadata),
            ),
        )
        self.recorder.record_step(step)
        self.raw_observation = next_raw
        self.observation = after
        return step

    def reset_after_game_over(self) -> TrajectoryStep:
        """Recover the current public level after GAME_OVER and audit the reset."""
        if self.observation.structured.get("environment_status") != "GAME_OVER":
            raise ValueError("Environment reset recovery requires public GAME_OVER state")
        before = self.observation
        before_levels = int(before.structured.get("levels_completed") or 0)
        next_raw = self.env.reset()
        after = self._normalize(next_raw)
        after_levels = int(after.structured.get("levels_completed") or 0)
        if after_levels != before_levels:
            raise ValueError(
                "Environment reset changed levels_completed; refusing ambiguous recovery"
            )
        before_grid = self._load_grid(before)
        after_grid = self._load_grid(after)
        delta = self._public_delta(before, before_grid, after, after_grid, next_raw)
        delta["runtime_reset"] = True
        reset_action = Action("ENV_RESET", metadata={"runtime_recovery": True})
        step = TrajectoryStep(
            step_index=len(self.episode.steps),
            observation=before,
            available_actions=[reset_action],
            action=reset_action,
            result=StepResult(
                observation=after,
                environment_status=_state_name(next_raw.state),
                state_delta=delta,
            ),
            decision=None,
            metadata={
                "runtime_recovery": True,
                "counts_toward_agent_step_budget": False,
            },
        )
        self.recorder.record_step(step)
        self.raw_observation = next_raw
        self.observation = after
        return step

    @staticmethod
    def _public_delta(
        before: Observation,
        before_grid: np.ndarray,
        after: Observation,
        after_grid: np.ndarray,
        raw_after: Any,
    ) -> Dict[str, Any]:
        same_shape = before_grid.shape == after_grid.shape
        if same_shape:
            changed = before_grid != after_grid
            count = int(np.count_nonzero(changed))
            if count:
                rows, columns = np.where(changed)
                bounds = [
                    int(columns.min()),
                    int(rows.min()),
                    int(columns.max()),
                    int(rows.max()),
                ]
            else:
                bounds = None
        else:
            count = None
            bounds = None
        return {
            "grid_changed": bool(not same_shape or count),
            "changed_cells": count,
            "changed_bounds": bounds,
            "level_delta": int(after.structured["levels_completed"])
            - int(before.structured["levels_completed"]),
            "environment_status": _state_name(raw_after.state),
        }

    def _load_grid(self, observation: Observation) -> np.ndarray:
        reference = next(
            item
            for item in observation.artifacts
            if item.media_type == ContentAddressedArtifactStore.GRID_MEDIA_TYPE
        )
        return self.store.load_grid(reference)

    def finish(
        self,
        *,
        status: str,
        success: bool,
        metadata: Dict[str, Any],
        details: str = "The evaluation harness stopped the public ARC session.",
    ) -> TrajectoryEpisode:
        return self.recorder.finalize(
            TrajectoryOutcome(
                status=status,
                success=success,
                details=details,
                metadata=metadata,
            )
        )


__all__ = ["PublicARCSession"]
