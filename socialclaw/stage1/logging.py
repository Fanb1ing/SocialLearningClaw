from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict

from .types import Episode


def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_episode(run_dir: str, episode: Episode) -> str:
    """Write episode.json and return file path."""
    _safe_mkdir(run_dir)
    ep_dir = os.path.join(run_dir, episode.problem.id)
    _safe_mkdir(ep_dir)
    path = os.path.join(ep_dir, "episode.json")

    payload: Dict[str, Any] = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "episode": asdict(episode),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    return path
