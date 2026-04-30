from __future__ import annotations

from ..types import Episode


def should_summarize(episode: Episode) -> bool:
    # Stage1 minimal: wrong then later correct with human knowledge points.
    if not episode.evals:
        return False
    if not episode.knowledge_points:
        return False

    first_correct = episode.evals[0].correct
    any_correct = any(e.correct for e in episode.evals)
    return (not first_correct) and any_correct
