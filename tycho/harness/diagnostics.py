"""Goal-agnostic behavioural diagnostics derived from harness records.

We never know a game's goal, so these signals describe *how an approach behaves*,
not whether it found the intended solution. They are computed purely from the
agent's own trajectory plus the per-level human baselines we already hold — no
game-internal oracle, no online calls. The point is to answer "what is this
approach good at / where does it break?" across the eval pool.

Signals per environment:
- exploration_cost: actions before the first level completion (proxy for how
  long the agent flails before it converts environment info into progress).
- execution_efficiency: RHAE env_score conditioned on having completed >=1 level
  (how good it is once it gets going).
- noop_rate: fraction of scored actions that did not change the grid (wasted
  budget; high = the agent doesn't model which actions do anything).
- exploration_coverage: distinct frames / total actions (state-space breadth).
- revisit_rate: revisits / total actions (churn / cycling without progress).
- failure_mode: a single coarse label (see classify_failure).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class EnvDiagnostics:
    game_id: str
    env_score: float
    levels_completed: int
    n_levels: int
    total_actions: int
    noop_rate: float
    exploration_coverage: float
    revisit_rate: float
    exploration_cost: Optional[int]
    failure_mode: str


def classify_failure(rec) -> str:
    """One coarse label per env. Order matters: most-specific first."""
    total = max(getattr(rec, "total_actions_including_unfinished", rec.total_actions), 1)
    noop_rate = rec.noop_actions / total
    coverage = rec.distinct_frames / (total + 1)  # +1 for the initial frame

    if str(rec.final_state).rsplit(".", 1)[-1] == "WIN":
        return "solved"
    if rec.levels_completed >= 1:
        return "partial-progress"  # made real progress, then stalled / ran out
    # Zero levels completed — diagnose why.
    if noop_rate >= 0.6:
        return "inert"  # mostly no-ops: doesn't find actions that do anything
    if coverage <= 0.1:
        return "stuck-loop"  # lots of actions, almost no new states
    if rec.revisits / total >= 0.5:
        return "cycling"  # explores but keeps falling back into seen states
    return "exploring-no-progress"  # changes things, sees variety, never wins L1


def diagnose_env(rec) -> EnvDiagnostics:
    total_actions = getattr(rec, "total_actions_including_unfinished", rec.total_actions)
    total = max(total_actions, 1)
    return EnvDiagnostics(
        game_id=rec.game_id,
        env_score=rec.env_score,
        levels_completed=rec.levels_completed,
        n_levels=rec.n_levels,
        total_actions=total_actions,
        noop_rate=round(rec.noop_actions / total, 3),
        exploration_coverage=round(rec.distinct_frames / (total + 1), 3),
        revisit_rate=round(rec.revisits / total, 3),
        exploration_cost=rec.actions_before_first_level,
        failure_mode=classify_failure(rec),
    )


def summarize(records) -> dict:
    """Aggregate diagnostics across a split's env records."""
    diags = [diagnose_env(r) for r in records]
    modes: dict[str, int] = {}
    for d in diags:
        modes[d.failure_mode] = modes.get(d.failure_mode, 0) + 1
    costs = [d.exploration_cost for d in diags if d.exploration_cost is not None]
    return {
        "failure_modes": dict(sorted(modes.items(), key=lambda kv: -kv[1])),
        "n_with_progress": len(costs),
        "mean_noop_rate": round(sum(d.noop_rate for d in diags) / max(len(diags), 1), 3),
        "mean_coverage": round(sum(d.exploration_coverage for d in diags) / max(len(diags), 1), 3),
        "mean_exploration_cost": round(sum(costs) / len(costs), 1) if costs else None,
        "per_env": [d.__dict__ for d in diags],
    }
