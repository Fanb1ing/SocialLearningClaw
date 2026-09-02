"""RHAE scoring, faithful to the shipped `arc_agi.EnvironmentScoreCalculator`.

The published technical report (Eq. 1) writes `S = min(1.15, h/a)^2`. The code
that actually ships in `arc-agi` and underlies official scorecards computes:

    level_score = min((h / a)^2 * 100, 115)          # cap is post-square, at 115
    env_score   = sum(w_l * S_l) / sum(w_l)           # w_l = 1-indexed level
    env_score   = min(env_score, completed_weight_fraction * 100)

We replicate the shipped behaviour exactly, because that is what the leaderboard
uses. One correctness requirement the calculator depends on: you must feed it
ALL n levels of the environment (uncompleted/unreached levels with completed=
False), not just the levels the agent finished — otherwise both the weighted
average denominator and the completion cap are wrong. ``actions_taken`` must
also include every in-play RESET; only the initialization reset that creates a
play is uncharged by the official scorecard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

LEVEL_SCORE_CAP = 115.0  # post-square cap, matches shipped code


@dataclass
class LevelResult:
    level_index: int  # 1-indexed
    completed: bool
    actions_taken: int
    baseline_actions: int

    def score(self) -> float:
        if not self.completed or self.actions_taken <= 0:
            return 0.0
        return min((self.baseline_actions / self.actions_taken) ** 2 * 100.0, LEVEL_SCORE_CAP)


@dataclass
class EnvResult:
    game_id: str
    n_levels: int
    levels: list[LevelResult] = field(default_factory=list)
    state: Optional[str] = None
    resets: int = 0

    @property
    def levels_completed(self) -> int:
        return sum(1 for l in self.levels if l.completed)

    @property
    def total_actions(self) -> int:
        return sum(l.actions_taken for l in self.levels)

    def env_score(self) -> float:
        """Weighted average of level scores, capped at the completed-weight fraction.

        Scores over ALL n levels; any level not present is treated as a 0-score,
        uncompleted level (weight still counts in the denominator).
        """
        present = {l.level_index: l for l in self.levels}
        total_weighted = 0.0
        total_weight = 0
        completed_weight = 0
        for idx in range(1, self.n_levels + 1):
            w = idx
            total_weight += w
            lr = present.get(idx)
            if lr is not None:
                total_weighted += w * lr.score()
                if lr.completed:
                    completed_weight += w
        if total_weight == 0:
            return 0.0
        score = total_weighted / total_weight
        cap = completed_weight / total_weight * 100.0
        return min(score, cap)


def benchmark_score(env_results: list[EnvResult]) -> float:
    """Total RHAE: mean of environment scores across the split, in [0, 100]."""
    if not env_results:
        return 0.0
    return sum(e.env_score() for e in env_results) / len(env_results)
