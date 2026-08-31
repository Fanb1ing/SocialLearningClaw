from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from ..dataset.arc_agi3 import ARCAGI3EnvWrapper
from ..memory import ContentAddressedArtifactStore
from ..schema.arc_agi3_parser import color_name, compute_grid_diff, extract_grid_objects
from .models import (
    Action,
    Decision,
    EvidenceTier,
    Observation,
    StepResult,
    TrajectoryEpisode,
    TrajectoryOutcome,
    TrajectoryStep,
)
from .recorder import TrajectoryRecorder


def _state_name(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw).removeprefix("GameState.")


def _grid_summary(grid: np.ndarray, *, max_objects: int = 20) -> str:
    objects = extract_grid_objects(grid)
    lines = [f"Grid shape: {grid.shape[0]} rows x {grid.shape[1]} columns."]
    for index, obj in enumerate(objects[:max_objects]):
        left, top = obj["top_left"]
        right, bottom = obj["bottom_right"]
        lines.append(
            f"object_{index}: color={color_name(obj['color'])}; "
            f"bounds=({left},{top})-({right},{bottom}); area={obj['area']}"
        )
    if len(objects) > max_objects:
        lines.append(f"{len(objects) - max_objects} additional objects omitted from text.")
    return "\n".join(lines)


class ARCAGI3TrajectoryAdapter:
    """Normalize public ARC observations/actions into the common contract."""

    name = "arc_agi3"

    def __init__(
        self,
        game_id: str,
        artifact_store: ContentAddressedArtifactStore,
        *,
        cell_size: int = 8,
    ) -> None:
        self.game_id = game_id
        self.artifact_store = artifact_store
        self.cell_size = cell_size

    def normalize_observation(self, raw_observation: Any) -> Observation:
        if raw_observation is None or not getattr(raw_observation, "frame", None):
            raise ValueError("ARC observation does not contain a frame")
        grid = np.asarray(raw_observation.frame[-1])
        grid_ref = self.artifact_store.put_grid(
            grid,
            role="environment_state",
            metadata={"game_id": self.game_id},
        )
        image = ARCAGI3EnvWrapper.grid_to_image(grid, cell_size=self.cell_size)
        image_ref = self.artifact_store.put_png(
            image,
            role="agent_view",
            metadata={
                "game_id": self.game_id,
                "renderer": "ARCAGI3EnvWrapper.grid_to_image",
                "cell_size": self.cell_size,
            },
        )
        available_ids = list(getattr(raw_observation, "available_actions", []) or [])
        structured = {
            "game_id": self.game_id,
            "environment_status": _state_name(raw_observation.state),
            "levels_completed": int(getattr(raw_observation, "levels_completed", 0) or 0),
            "win_levels": int(getattr(raw_observation, "win_levels", 1) or 1),
            "level": min(
                int(getattr(raw_observation, "levels_completed", 0) or 0) + 1,
                int(getattr(raw_observation, "win_levels", 1) or 1),
            ),
            "available_action_ids": [int(item) for item in available_ids],
            "grid_shape": list(grid.shape),
            "logical_grid_sha256": grid_ref.metadata["logical_sha256"],
        }
        return Observation(
            text=_grid_summary(grid),
            structured=structured,
            artifacts=[grid_ref, image_ref],
        )

    def available_actions(
        self, env: ARCAGI3EnvWrapper, raw_observation: Any
    ) -> list[Action]:
        return [Action(item.name) for item in env.get_available_actions(raw_observation)]

    def describe_transition(
        self,
        pre_observation: Observation,
        action: Action,
        post_observation: Observation,
        raw_result: Any,
    ) -> Dict[str, Any]:
        pre_grid = self.load_grid(pre_observation)
        post_grid = self.load_grid(post_observation)
        changed, changed_regions = compute_grid_diff(pre_grid, post_grid)
        same_shape = pre_grid.shape == post_grid.shape
        changed_cells = (
            int(np.count_nonzero(pre_grid != post_grid)) if same_shape else None
        )
        # CD82's bottom row is a visible action-budget UI. Preserve it in raw
        # evidence but separate it from task-state effects for coverage/no-op
        # classification.
        if self.game_id.split("-", 1)[0] in {"cd82", "sk48", "tu93"} and same_shape and pre_grid.shape[0] > 1:
            task_pre = pre_grid[:-1, :]
            task_post = post_grid[:-1, :]
        else:
            task_pre = pre_grid
            task_post = post_grid
        task_changed_cells = (
            int(np.count_nonzero(task_pre != task_post))
            if task_pre.shape == task_post.shape
            else None
        )
        pre_level = int(pre_observation.structured["levels_completed"])
        post_level = int(post_observation.structured["levels_completed"])
        return {
            "grid_changed": bool(changed),
            "changed_cells": changed_cells,
            "changed_regions": changed_regions,
            "task_state_changed": bool(task_changed_cells),
            "task_changed_cells": task_changed_cells,
            "level_delta": post_level - pre_level,
            "environment_status": _state_name(raw_result.state),
            "action_signature": action.name,
        }

    def terminal_outcome(self, raw_result: Any) -> Optional[TrajectoryOutcome]:
        state = _state_name(raw_result.state)
        if state == "WIN":
            return TrajectoryOutcome(status="WIN", success=True, reward=1.0)
        if state == "GAME_OVER":
            return TrajectoryOutcome(status="GAME_OVER", success=False, reward=0.0)
        return None

    def is_informative(self, transition_features: Dict[str, Any]) -> bool:
        return bool(
            transition_features.get("task_state_changed")
            or transition_features.get("level_delta")
            or transition_features.get("environment_status") in {"WIN", "GAME_OVER"}
        )

    def load_grid(self, observation: Observation) -> np.ndarray:
        reference = next(
            (
                item
                for item in observation.artifacts
                if item.media_type == ContentAddressedArtifactStore.GRID_MEDIA_TYPE
            ),
            None,
        )
        if reference is None:
            raise ValueError("ARC observation is missing its lossless grid artifact")
        return self.artifact_store.load_grid(reference)


class ARCRecordingSession:
    """Execute public ARC actions while durably recording every transition."""

    def __init__(
        self,
        *,
        game_id: str,
        root: str | Path,
        episode_id: str,
        actor: str,
        evidence_tier: EvidenceTier,
        split: str,
        provenance: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        render_mode: Optional[str] = None,
        env: Optional[ARCAGI3EnvWrapper] = None,
    ) -> None:
        self.root = Path(root)
        self.game_id = game_id
        self.env = env or ARCAGI3EnvWrapper(game_id, render_mode=render_mode)
        if self.env.game_id != game_id:
            raise ValueError(
                f"Provided ARC environment is for {self.env.game_id}, not {game_id}"
            )
        self.artifact_store = ContentAddressedArtifactStore(self.root / "assets")
        self.adapter = ARCAGI3TrajectoryAdapter(game_id, self.artifact_store)
        self.raw_observation = self.env.reset()
        self.observation = self.adapter.normalize_observation(self.raw_observation)
        initial = TrajectoryEpisode(
            episode_id=episode_id,
            benchmark="arc_agi3",
            task_id=game_id,
            split=split,
            actor=actor,
            evidence_tier=evidence_tier,
            initial_observation=self.observation,
            provenance=provenance,
            metadata=dict(metadata or {}),
        )
        self.recorder = TrajectoryRecorder(self.root, initial)

    @property
    def episode(self) -> TrajectoryEpisode:
        return self.recorder.episode

    def execute(
        self,
        action: Action,
        *,
        rationale: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TrajectoryStep:
        available_generic = self.adapter.available_actions(self.env, self.raw_observation)
        available_sdk = {
            item.name: item for item in self.env.get_available_actions(self.raw_observation)
        }
        if action.name not in available_sdk:
            raise ValueError(f"ARC action {action.name} is not currently available")
        environment_data = {
            key: value
            for key, value in action.arguments.items()
            if key not in {"target_role", "scenario_tag"}
        }
        next_raw = self.env.step(available_sdk[action.name], data=environment_data)
        next_observation = self.adapter.normalize_observation(next_raw)
        features = self.adapter.describe_transition(
            self.observation, action, next_observation, next_raw
        )
        step = TrajectoryStep(
            step_index=len(self.episode.steps),
            observation=self.observation,
            available_actions=available_generic,
            action=action,
            result=StepResult(
                observation=next_observation,
                environment_status=_state_name(next_raw.state),
                state_delta=features,
            ),
            decision=Decision(
                response=json.dumps(action.to_dict(), ensure_ascii=False, sort_keys=True),
                rationale=rationale,
                metadata={"network_calls": 0},
            ),
            metadata=dict(metadata or {}),
        )
        self.recorder.record_step(step)
        self.raw_observation = next_raw
        self.observation = next_observation
        return step

    def finish(
        self,
        *,
        status: str,
        success: Optional[bool],
        details: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TrajectoryEpisode:
        return self.recorder.finalize(
            TrajectoryOutcome(
                status=status,
                success=success,
                details=details,
                metadata=dict(metadata or {}),
            )
        )


def transition_signature(step: TrajectoryStep) -> str:
    payload = {
        "action": step.action.name,
        "target_role": step.action.arguments.get("target_role", ""),
        "task_state_changed": step.result.state_delta.get("task_state_changed"),
        "task_changed_cells": step.result.state_delta.get("task_changed_cells"),
        "level_delta": step.result.state_delta.get("level_delta"),
        "environment_status": step.result.environment_status,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
