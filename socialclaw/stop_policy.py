from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from .types import Episode


@dataclass
class StopConfig:
    conf_threshold: float = 0.85
    max_iters: int = 2
    max_total_tokens: int = 8000


def should_stop(episode: Episode, cfg: StopConfig) -> Tuple[bool, Optional[str]]:
    iters = len(episode.attempts)
    if iters >= cfg.max_iters:
        return True, "max_iters"

    if episode.attempts:
        conf = episode.reasoning_confidence
        if isinstance(conf, (int, float)) and conf >= cfg.conf_threshold:
            return True, "confidence"

    total = 0
    for a in episode.attempts:
        usage = a.usage or {}
        tok = usage.get("total_tokens")
        if isinstance(tok, int):
            total += tok
    if total >= cfg.max_total_tokens:
        return True, "max_tokens"

    return False, None
