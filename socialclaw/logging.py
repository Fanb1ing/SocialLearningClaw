from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from .types import Episode


def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_episode(run_dir: str, episode: Episode, subdir: Optional[str] = None) -> str:
    """Write episode.json and return file path.

    subdir: if provided, use as the subdirectory name instead of episode.problem.id.
    """
    _safe_mkdir(run_dir)
    ep_dir = os.path.join(run_dir, subdir if subdir is not None else episode.problem.id)
    _safe_mkdir(ep_dir)
    path = os.path.join(ep_dir, "episode.json")

    payload: Dict[str, Any] = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "episode": asdict(episode),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    return path


def write_trajectory(run_dir: str, problem_id: str, trajectory: List[Dict[str, Any]]) -> str:
    """Write a concise trajectory.json and return the file path."""
    _safe_mkdir(run_dir)
    ep_dir = os.path.join(run_dir, problem_id)
    _safe_mkdir(ep_dir)
    path = os.path.join(ep_dir, "trajectory.json")

    payload: Dict[str, Any] = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "trajectory": trajectory,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    return path


def write_step(run_dir: str, problem_id: str, step: int, data: Dict[str, Any]) -> str:
    """Write a per-step record and return file path."""
    _safe_mkdir(run_dir)
    ep_dir = os.path.join(run_dir, problem_id)
    _safe_mkdir(ep_dir)
    path = os.path.join(ep_dir, f"step_{step:03d}.json")
    payload: Dict[str, Any] = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "step": step,
        "data": data,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    return path
