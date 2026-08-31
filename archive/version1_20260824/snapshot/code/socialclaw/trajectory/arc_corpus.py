from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np

from ..dataset.arc_agi3 import ARCAGI3EnvWrapper
from ..memory import ContentAddressedArtifactStore
from .arc_agi3 import ARCAGI3TrajectoryAdapter, transition_signature
from .models import TrajectoryEpisode


def arc_corpus_coverage(episodes: Iterable[TrajectoryEpisode]) -> Dict[str, Any]:
    """Summarize ARC-specific behavioral coverage without changing evidence."""
    values = list(episodes)
    levels = set()
    action_effects: Dict[str, Counter] = defaultdict(Counter)
    click_roles = Counter()
    outcomes = Counter()
    tiers = Counter()
    signatures = set()
    scenario_categories = Counter()
    pair_groups: Dict[str, set[str]] = defaultdict(set)
    for episode in values:
        tiers[episode.evidence_tier.value] += 1
        if episode.terminal_outcome is not None:
            outcomes[episode.terminal_outcome.status] += 1
        category = str(episode.metadata.get("scenario_category", "unspecified"))
        scenario_categories[category] += 1
        pair_group = str(episode.metadata.get("pair_group", ""))
        if pair_group:
            pair_groups[pair_group].add(episode.episode_id)
        for step in episode.steps:
            levels.add(int(step.observation.structured.get("level", 0)))
            changed = bool(step.result.state_delta.get("task_state_changed"))
            action_effects[step.action.name]["effect" if changed else "no_effect"] += 1
            role = str(step.action.arguments.get("target_role", ""))
            if role:
                click_roles[role] += 1
            signatures.add(transition_signature(step))
    paired = {key: len(ids) for key, ids in pair_groups.items() if len(ids) >= 2}
    return {
        "format_version": 1,
        "episode_count": len(values),
        "step_count": sum(len(item.steps) for item in values),
        "levels": sorted(level for level in levels if level > 0),
        "action_effects": {
            action: dict(sorted(counts.items()))
            for action, counts in sorted(action_effects.items())
        },
        "click_target_roles": dict(sorted(click_roles.items())),
        "terminal_outcomes": dict(sorted(outcomes.items())),
        "evidence_tiers": dict(sorted(tiers.items())),
        "scenario_categories": dict(sorted(scenario_categories.items())),
        "paired_case_groups": paired,
        "paired_case_count": len(paired),
        "transition_signature_count": len(signatures),
    }


def replay_arc_episode(root: str | Path, episode: TrajectoryEpisode) -> Dict[str, Any]:
    """Replay one episode through the public ARC API and compare every frame."""
    base = Path(root)
    store = ContentAddressedArtifactStore(base / "assets")
    adapter = ARCAGI3TrajectoryAdapter(episode.task_id, store)
    env = ARCAGI3EnvWrapper(episode.task_id)
    raw = env.reset()
    expected_initial = adapter.load_grid(episode.initial_observation)
    actual_initial = np.asarray(raw.frame[-1])
    errors = []
    if not np.array_equal(expected_initial, actual_initial):
        errors.append("initial observation differs from a fresh reset")

    for step in episode.steps:
        available = {item.name: item for item in env.get_available_actions(raw)}
        if step.action.name not in available:
            errors.append(f"step {step.step_index}: action is unavailable")
            break
        data = {
            key: value
            for key, value in step.action.arguments.items()
            if key not in {"target_role", "scenario_tag"}
        }
        raw = env.step(available[step.action.name], data=data)
        actual = np.asarray(raw.frame[-1])
        expected = adapter.load_grid(step.result.observation)
        if not np.array_equal(actual, expected):
            errors.append(f"step {step.step_index}: replayed grid differs")
            break
        actual_status = str(getattr(raw.state, "value", raw.state)).removeprefix(
            "GameState."
        )
        if actual_status != step.result.environment_status:
            errors.append(
                f"step {step.step_index}: status {actual_status} != "
                f"{step.result.environment_status}"
            )
            break
    return {
        "episode_id": episode.episode_id,
        "status": "passed" if not errors else "failed",
        "steps_replayed": len(episode.steps) if not errors else step.step_index + 1,
        "errors": errors,
    }
