#!/usr/bin/env python3
"""Generate a tiny inspectable trajectory without an Agent or network call."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from socialclaw.memory import ContentAddressedArtifactStore
from socialclaw.trajectory import (
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


PALETTE = {
    0: (0, 0, 0),
    1: (0, 0, 255),
    2: (255, 0, 0),
    3: (0, 255, 0),
}


def render_grid(grid: np.ndarray, *, cell_size: int = 24) -> Image.Image:
    height, width = grid.shape
    image = Image.new("RGB", (width * cell_size, height * cell_size))
    pixels = image.load()
    for row in range(height):
        for col in range(width):
            color = PALETTE.get(int(grid[row, col]), (128, 128, 128))
            for dy in range(cell_size):
                for dx in range(cell_size):
                    pixels[col * cell_size + dx, row * cell_size + dy] = color
    return image


def make_observation(
    store: ContentAddressedArtifactStore,
    grid: np.ndarray,
    *,
    label: str,
) -> Observation:
    grid_ref = store.put_grid(
        grid,
        role="environment_state",
        metadata={"label": label},
    )
    image_ref = store.put_png(
        render_grid(grid),
        role="agent_view",
        metadata={"label": label, "renderer": "trajectory-contract-demo-v1"},
    )
    return Observation(
        text=f"Synthetic grid state: {label}",
        structured={
            "grid_shape": list(grid.shape),
            "nonzero_cells": int(np.count_nonzero(grid)),
            "logical_grid_sha256": grid_ref.metadata["logical_sha256"],
        },
        artifacts=[grid_ref, image_ref],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="outputs/review/trajectory_contract_phase_a",
    )
    args = parser.parse_args()

    root = Path(args.output_dir)
    store = ContentAddressedArtifactStore(root / "assets")
    initial_grid = np.array([[0, 1], [0, 0]], dtype=np.int16)
    moved_grid = np.array([[0, 0], [0, 1]], dtype=np.int16)
    initial = make_observation(store, initial_grid, label="initial")
    moved = make_observation(store, moved_grid, label="moved_down")

    episode = TrajectoryEpisode(
        episode_id="review.arc.trajectory-contract-v1",
        benchmark="arc_agi3",
        task_id="synthetic-review-level-1",
        split="review",
        actor="scripted_policy",
        evidence_tier=EvidenceTier.SYNTHETIC,
        initial_observation=initial,
        provenance={
            "scenario_id": "move_then_block",
            "policy": "fixed_action_script",
            "network_calls": 0,
        },
    )
    recorder = TrajectoryRecorder(root, episode)
    available = [Action("ACTION1"), Action("ACTION4")]
    recorder.record_step(
        TrajectoryStep(
            step_index=0,
            observation=initial,
            available_actions=available,
            action=Action("ACTION4"),
            result=StepResult(
                observation=moved,
                environment_status="PLAYING",
                state_delta={"grid_changed": True, "changed_cells": 2},
            ),
            decision=Decision(
                response='{"action":"ACTION4"}',
                rationale="deterministic scripted move",
                metadata={"source": "demo_policy"},
            ),
        )
    )
    recorder.record_step(
        TrajectoryStep(
            step_index=1,
            observation=moved,
            available_actions=available,
            action=Action("ACTION1"),
            result=StepResult(
                observation=moved,
                environment_status="PLAYING",
                state_delta={"grid_changed": False, "reason": "blocked"},
            ),
            decision=Decision(
                response='{"action":"ACTION1"}',
                rationale="deterministic blocked-action probe",
                metadata={"source": "demo_policy"},
            ),
        )
    )
    final = recorder.finalize(
        TrajectoryOutcome(
            status="TIMEOUT",
            success=False,
            details="Synthetic two-step review episode ended by scenario budget.",
        )
    )
    summary = {
        "episode_path": str(recorder.path),
        "episode_id": final.episode_id,
        "evidence_tier": final.evidence_tier.value,
        "steps": len(final.steps),
        "terminal_status": final.terminal_outcome.status,
        "grid_asset_files": len(list((root / "assets" / "grids").glob("*.npy"))),
        "image_asset_files": len(list((root / "assets" / "images").glob("*.png"))),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
