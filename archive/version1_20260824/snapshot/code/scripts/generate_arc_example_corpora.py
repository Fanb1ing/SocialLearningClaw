#!/usr/bin/env python3
"""Generate compact SK48/TU93 prototype corpora beside the CD82 v1 corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/socialclaw-mpl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from socialclaw.dataset.arc_agi3 import arc_environment_fingerprint  # noqa: E402
from socialclaw.trajectory import (  # noqa: E402
    ARCRecordingSession,
    Action,
    EvidenceTier,
    arc_corpus_coverage,
    load_corpus_episodes,
    replay_arc_episode,
    validate_corpus,
    write_corpus_metadata,
)
from socialclaw.trajectory.arc_policies import fixed_arc_level_actions  # noqa: E402
from socialclaw.trajectory.corpus import write_json_atomic  # noqa: E402


GAME_CONFIG = {
    "sk48-d8078629": {"slug": "sk48_v1", "verified_levels": 3, "actions": [1, 2, 3, 4, 6, 7]},
    "tu93-0768757b": {"slug": "tu93_v1", "verified_levels": 9, "actions": [1, 2, 3, 4]},
}


def state_name(raw) -> str:
    return str(getattr(raw.state, "value", raw.state)).removeprefix("GameState.")


def execute(session, actions, rationale):
    for action in actions:
        if state_name(session.raw_observation) in {"WIN", "GAME_OVER"}:
            break
        session.execute(action, rationale=rationale)


def new_session(root, game_id, episode_id, category, *, tier, seed=None, pair=""):
    metadata = {"scenario_category": category}
    if seed is not None:
        metadata["seed"] = seed
    if pair:
        metadata["pair_group"] = pair
    return ARCRecordingSession(
        game_id=game_id,
        root=root,
        episode_id=episode_id,
        actor="verified_fixed_policy" if tier == EvidenceTier.SOURCE_GUIDED_NATURAL else "seeded_explorer",
        evidence_tier=tier,
        split="train",
        provenance={"public_interface_only": True, "network_calls": 0, "generator": Path(__file__).name},
        metadata=metadata,
    )


def finish(session, status, success):
    actual = session.adapter.terminal_outcome(session.raw_observation)
    if actual:
        status, success = actual.status, actual.success
    return session.finish(
        status=status,
        success=success,
        metadata={
            "levels_completed": int(session.raw_observation.levels_completed),
            "environment_status": state_name(session.raw_observation),
        },
    )


def prefix_actions(game_id, target_level):
    result = []
    for level in range(1, target_level + 1):
        result.extend(fixed_arc_level_actions(game_id, level))
    return result


def build_game(root: Path, game_id: str) -> list:
    config = GAME_CONFIG[game_id]
    episodes = []
    success_targets = [1, config["verified_levels"]]
    if config["verified_levels"] > 3:
        success_targets.insert(1, (config["verified_levels"] + 1) // 2)
    else:
        success_targets.insert(1, 2)

    for target in success_targets:
        session = new_session(
            root, game_id, f"success_through_level_{target}", "verified_success_prefix",
            tier=EvidenceTier.SOURCE_GUIDED_NATURAL, pair=f"level_{target}_outcome",
        )
        execute(session, prefix_actions(game_id, target), "Replay a source-assisted path verified through public actions.")
        if int(session.raw_observation.levels_completed) < target:
            raise RuntimeError(f"Verified path did not complete {game_id} level {target}")
        episodes.append(finish(session, "SCENARIO_COMPLETE", True))

        near = new_session(
            root, game_id, f"near_miss_level_{target}", "near_miss",
            tier=EvidenceTier.SOURCE_GUIDED_NATURAL, pair=f"level_{target}_outcome",
        )
        actions = prefix_actions(game_id, target)
        execute(near, actions[:-1], "Omit the final action from a verified public-action prefix.")
        episodes.append(finish(near, "TIMEOUT", False))

    # One-step and repeat probes expose direction, click/undo, and blocked cases.
    for action_id in config["actions"]:
        for repeat in (1, 2):
            session = new_session(
                root, game_id, f"probe_action_{action_id}_repeat_{repeat}", "single_action_probe",
                tier=EvidenceTier.SOURCE_GUIDED_NATURAL, pair=f"action_{action_id}_repeat",
            )
            if action_id == 6:
                actions = [Action("ACTION6", {"x": 20, "y": 20, "target_role": "background"})] * repeat
            else:
                actions = [Action(f"ACTION{action_id}")] * repeat
            execute(session, actions, "Probe a legal public action from a fresh initial observation.")
            episodes.append(finish(session, "SCENARIO_COMPLETE", None))

    random_count = 24 - len(episodes)
    for index in range(random_count):
        seed = 9300 + index if game_id.startswith("tu93") else 4800 + index
        rng = random.Random(seed)
        session = new_session(
            root, game_id, f"exploration_seed_{index:02d}", "seeded_exploration",
            tier=EvidenceTier.NATURAL, seed=seed, pair="seeded_exploration",
        )
        actions = []
        for _ in range(10):
            aid = rng.choice(config["actions"])
            if aid == 6:
                actions.append(Action("ACTION6", {"x": rng.randrange(64), "y": rng.randrange(63), "target_role": "sampled_screen"}))
            else:
                actions.append(Action(f"ACTION{aid}"))
        execute(session, actions, "Fixed-seed explorer selected legal public actions without Gold guidance.")
        episodes.append(finish(session, "TIMEOUT", False))
    assert len(episodes) == 24
    return episodes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "data/trajectory_corpora/arc_agi3")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summaries = {}
    for game_id, config in GAME_CONFIG.items():
        output = args.root.resolve() / config["slug"]
        if output.exists():
            if not args.force:
                raise FileExistsError(f"{output} exists; use --force")
            shutil.rmtree(output)
        output.mkdir(parents=True)
        episodes = build_game(output, game_id)
        coverage = arc_corpus_coverage(episodes)
        coverage["verified_success_level_count"] = config["verified_levels"]
        coverage["total_game_level_count"] = 8 if game_id.startswith("sk48") else 9
        coverage["trajectory_version_follow_up"] = (
            "v2 should extend verified success beyond level 3" if game_id.startswith("sk48") else "v2 may add mechanism-targeted trajectories"
        )
        write_corpus_metadata(
            output,
            corpus_id=f"arc_agi3_{config['slug']}",
            benchmark="arc_agi3",
            game_id=game_id,
            environment_fingerprint=arc_environment_fingerprint(game_id),
            collector={"script": Path(__file__).name, "network_calls": 0, "public_interface_only": True},
            episodes=episodes,
            splits={"train": [item.episode_id for item in episodes]},
            coverage=coverage,
        )
        validation = validate_corpus(output)
        replay_cases = [replay_arc_episode(output, episode) for episode in load_corpus_episodes(output)]
        replay = {
            "format_version": 1,
            "status": "passed" if all(item["status"] == "passed" for item in replay_cases) else "failed",
            "episode_count": len(replay_cases),
            "step_count": sum(item["steps_replayed"] for item in replay_cases),
            "episodes": replay_cases,
        }
        write_json_atomic(output / "replay_validation.json", replay)
        first = episodes[0].initial_observation.artifacts[1]
        review = output / "review"
        review.mkdir()
        shutil.copyfile(output / "assets" / first.relative_path, review / "initial.png")
        summaries[game_id] = {
            "episodes": len(episodes), "steps": coverage["step_count"],
            "verified_levels": config["verified_levels"], "validation": validation["status"],
            "replay": replay["status"], "output": str(output),
        }
        print(json.dumps({game_id: summaries[game_id]}, ensure_ascii=False), flush=True)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 0 if all(v["validation"] == v["replay"] == "passed" for v in summaries.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
