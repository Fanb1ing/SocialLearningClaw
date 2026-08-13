#!/usr/bin/env python3
"""Generate a deterministic, public-interface ARC trajectory corpus.

The CD82 pilot deliberately does not call an LLM. Every recorded observation
is returned by reset/step, and every action is sent through the public ARC API.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Iterable, List, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/socialclaw-mpl")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from socialclaw.dataset.arc_agi3 import (  # noqa: E402
    arc_environment_fingerprint,
)
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
from socialclaw.trajectory.arc_policies import (  # noqa: E402
    cd82_navigation_actions,
    detect_cd82_palette,
    plan_cd82_level,
)
from socialclaw.trajectory.corpus import write_json_atomic  # noqa: E402


GAME_ID = "cd82-fb555c5d"
CORPUS_ID = "arc_agi3_cd82_v1"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/trajectory_corpora/arc_agi3/cd82_v1"
TERMINAL_STATES = {"WIN", "GAME_OVER"}


def _state_name(value) -> str:
    return str(getattr(value, "value", value)).removeprefix("GameState.")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class CD82CorpusBuilder:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.episodes = []

    def new_session(
        self,
        episode_id: str,
        category: str,
        *,
        tier: EvidenceTier = EvidenceTier.SOURCE_GUIDED_NATURAL,
        pair_group: str = "",
        seed: Optional[int] = None,
    ) -> ARCRecordingSession:
        metadata = {"scenario_category": category}
        if pair_group:
            metadata["pair_group"] = pair_group
        if seed is not None:
            metadata["seed"] = seed
        return ARCRecordingSession(
            game_id=GAME_ID,
            root=self.root,
            episode_id=episode_id,
            actor="cd82_deterministic_policy" if seed is None else "seeded_coverage_explorer",
            evidence_tier=tier,
            split="train",
            provenance={
                "collector": "scripts/generate_arc_trajectory_corpus.py",
                "policy": "visible_grid_cd82_v1" if seed is None else "seeded_public_action_v1",
                "public_interface_only": True,
                "network_calls": 0,
            },
            metadata=metadata,
        )

    @staticmethod
    def execute(
        session: ARCRecordingSession,
        actions: Iterable[Action],
        *,
        rationale: str,
        focal_index: Optional[int] = None,
    ) -> None:
        for index, action in enumerate(actions):
            if _state_name(session.raw_observation.state) in TERMINAL_STATES:
                break
            session.execute(
                action,
                rationale=rationale,
                metadata={"focal_action": index == focal_index}
                if focal_index is not None
                else {},
            )

    @staticmethod
    def finish(
        session: ARCRecordingSession,
        requested_status: str,
        requested_success: Optional[bool],
    ) -> None:
        actual = session.adapter.terminal_outcome(session.raw_observation)
        if actual is not None:
            status, success = actual.status, actual.success
        else:
            status, success = requested_status, requested_success
        session.finish(
            status=status,
            success=success,
            details="Outcome observed after executing the recorded public action prefix.",
            metadata={
                "environment_status": _state_name(session.raw_observation.state),
                "levels_completed": int(session.raw_observation.levels_completed),
            },
        )

    def solve_through(self, session: ARCRecordingSession, target_level: int) -> None:
        while int(session.raw_observation.levels_completed) < target_level:
            before = int(session.raw_observation.levels_completed)
            level = before + 1
            actions = plan_cd82_level(session.raw_observation.frame[-1], level=level)
            self.execute(
                session,
                actions,
                rationale=(
                    "Deterministic CD82 policy derived a paint program from the visible "
                    "target, palette, and tool layout."
                ),
            )
            after = int(session.raw_observation.levels_completed)
            if after <= before:
                raise RuntimeError(f"CD82 deterministic policy did not complete level {level}")

    def record_successes(self) -> None:
        for target_level in range(1, 7):
            session = self.new_session(
                f"success_through_level_{target_level}",
                "deterministic_success",
                pair_group=f"level_{target_level}_outcome",
            )
            self.solve_through(session, target_level)
            self.finish(session, "SCENARIO_COMPLETE", True)
            self.episodes.append(session.episode)

    def _reach_level(self, session: ARCRecordingSession, level: int) -> None:
        if level > 1:
            self.solve_through(session, level - 1)

    def record_near_misses(self) -> None:
        for level in range(1, 7):
            for variant in ("omit_last", "replace_last"):
                episode_id = f"near_miss_level_{level}_{variant}"
                session = self.new_session(
                    episode_id,
                    "near_miss",
                    pair_group=f"level_{level}_outcome",
                )
                self._reach_level(session, level)
                actions = plan_cd82_level(session.raw_observation.frame[-1], level=level)
                if variant == "omit_last":
                    candidate = actions[:-1]
                else:
                    replacement = Action("ACTION1")
                    candidate = [*actions[:-1], replacement]
                self.execute(
                    session,
                    candidate,
                    rationale=f"Deterministic {variant} perturbation of the visible-grid solution.",
                )
                self.finish(session, "TIMEOUT", False)
                self.episodes.append(session.episode)

        # Replace one redundant near-miss with an observed action-budget loss.
        old = self.episodes.pop()
        old_path = self.root / "episodes" / f"{old.episode_id}.json"
        old_path.unlink()
        session = self.new_session(
            "near_miss_action_budget_game_over",
            "terminal_failure",
            pair_group="level_6_outcome",
        )
        for _ in range(110):
            if _state_name(session.raw_observation.state) in TERMINAL_STATES:
                break
            session.execute(
                Action("ACTION1"),
                rationale="Repeat a legal blocked action until the visible action budget is exhausted.",
            )
        self.finish(session, "GAME_OVER", False)
        self.episodes.append(session.episode)

    def record_mechanisms(self) -> None:
        for position in range(6):
            for action_id in range(1, 5):
                session = self.new_session(
                    f"mechanism_nav_pos_{position}_action_{action_id}",
                    "single_mechanism",
                    pair_group=f"navigation_action_{action_id}",
                )
                prefix = cd82_navigation_actions(0, position)
                actions = [*prefix, Action(f"ACTION{action_id}")]
                self.execute(
                    session,
                    actions,
                    rationale="Navigate to a visible tool position, then probe one legal direction action.",
                    focal_index=len(actions) - 1,
                )
                self.finish(session, "SCENARIO_COMPLETE", None)
                self.episodes.append(session.episode)

        for position in range(8):
            session = self.new_session(
                f"mechanism_paint_pos_{position}",
                "single_mechanism",
                pair_group="paint_position_effects",
            )
            prefix = cd82_navigation_actions(0, position)
            if position == 7:
                palette = detect_cd82_palette(session.raw_observation.frame[-1])
                black_x, black_y = palette[0]
                actions = [
                    Action(
                        "ACTION6",
                        {"x": black_x, "y": black_y, "target_role": "palette_button"},
                    ),
                    *prefix,
                    Action("ACTION5"),
                ]
            else:
                actions = [*prefix, Action("ACTION5")]
            self.execute(
                session,
                actions,
                rationale="Probe the selected visible paint tool on the lower canvas.",
                focal_index=len(actions) - 1,
            )
            self.finish(session, "SCENARIO_COMPLETE", None)
            self.episodes.append(session.episode)

        click_cases = [
            ("palette_black", 0, 0, "palette_button"),
            ("palette_white", 15, 0, "palette_button"),
            ("background", None, 20, "background"),
            ("detail_location_before_unlock", None, 31, "edge_detail_tool"),
        ]
        for name, color, x, role in click_cases:
            session = self.new_session(
                f"mechanism_click_{name}",
                "single_mechanism",
                pair_group="action6_click_roles",
            )
            if color is not None:
                x, y = detect_cd82_palette(session.raw_observation.frame[-1])[color]
            elif role == "edge_detail_tool":
                y = 20
            else:
                y = 20
            self.execute(
                session,
                [Action("ACTION6", {"x": x, "y": y, "target_role": role})],
                rationale="Probe a labeled role using only visible public click coordinates.",
                focal_index=0,
            )
            self.finish(session, "SCENARIO_COMPLETE", None)
            self.episodes.append(session.episode)

    def record_perturbations(self) -> None:
        probe = self.new_session("perturbation_plan_probe", "temporary")
        base = plan_cd82_level(probe.raw_observation.frame[-1], level=1)
        (self.root / "episodes" / "perturbation_plan_probe.json").unlink()

        variants: List[tuple[str, List[Action]]] = []
        for index in range(len(base)):
            variants.append((f"omit_{index}", [*base[:index], *base[index + 1 :]]))
            variants.append(
                (f"duplicate_{index}", [*base[:index], base[index], *base[index:]])
            )
            replacement_id = (int(base[index].name.removeprefix("ACTION")) % 5) + 1
            variants.append(
                (
                    f"replace_{index}",
                    [*base[:index], Action(f"ACTION{replacement_id}"), *base[index + 1 :]],
                )
            )
            variants.append(
                (
                    f"background_before_{index}",
                    [
                        *base[:index],
                        Action(
                            "ACTION6",
                            {"x": 20, "y": 20, "target_role": "background"},
                        ),
                        *base[index:],
                    ],
                )
            )
        for index in range(len(base) - 1):
            swapped = list(base)
            swapped[index], swapped[index + 1] = swapped[index + 1], swapped[index]
            variants.append((f"swap_{index}_{index + 1}", swapped))
        if len(variants) != 24:
            raise AssertionError(f"Expected 24 perturbations, got {len(variants)}")

        for name, actions in variants:
            session = self.new_session(
                f"perturbation_{name}",
                "success_sequence_perturbation",
                pair_group="level_1_solution_boundary",
            )
            self.execute(
                session,
                actions,
                rationale="Execute a deterministic edit of the verified level-one action sequence.",
            )
            success = int(session.raw_observation.levels_completed) >= 1
            self.finish(
                session,
                "SCENARIO_COMPLETE" if success else "TIMEOUT",
                success,
            )
            self.episodes.append(session.episode)

    def record_exploration(self) -> None:
        for seed in range(18):
            rng = random.Random(8200 + seed)
            session = self.new_session(
                f"exploration_seed_{seed:02d}",
                "seeded_coverage_exploration",
                tier=EvidenceTier.NATURAL,
                pair_group="seeded_exploration",
                seed=8200 + seed,
            )
            for _ in range(8):
                if _state_name(session.raw_observation.state) in TERMINAL_STATES:
                    break
                action_id = rng.randint(1, 6)
                if action_id == 6:
                    choices = [(20, 20, "background"), (31, 20, "edge_detail_tool")]
                    palette = detect_cd82_palette(session.raw_observation.frame[-1])
                    choices.extend((x, y, "palette_button") for x, y in palette.values())
                    x, y, role = rng.choice(choices)
                    action = Action(
                        "ACTION6", {"x": x, "y": y, "target_role": role}
                    )
                else:
                    action = Action(f"ACTION{action_id}")
                session.execute(
                    action,
                    rationale="Seeded coverage explorer selected a legal public action.",
                )
            self.finish(session, "TIMEOUT", False)
            self.episodes.append(session.episode)

    def build(self) -> list:
        phases = [
            ("success", self.record_successes),
            ("near-miss", self.record_near_misses),
            ("mechanism", self.record_mechanisms),
            ("perturbation", self.record_perturbations),
            ("exploration", self.record_exploration),
        ]
        for label, method in phases:
            method()
            print(f"[{label}] {len(self.episodes)} episodes recorded", flush=True)
        if len(self.episodes) != 96:
            raise AssertionError(f"Expected 96 published episodes, got {len(self.episodes)}")
        return self.episodes


def coverage_with_gates(episodes) -> dict:
    coverage = arc_corpus_coverage(episodes)
    effects = coverage["action_effects"]
    gates = {
        "episode_budget_96": coverage["episode_count"] == 96,
        "all_six_levels": coverage["levels"] == [1, 2, 3, 4, 5, 6],
        "all_six_actions": set(effects) == {f"ACTION{index}" for index in range(1, 7)},
        "changed_and_no_effect_per_action": all(
            effects.get(f"ACTION{index}", {}).get("effect", 0) > 0
            and effects.get(f"ACTION{index}", {}).get("no_effect", 0) > 0
            for index in range(1, 7)
        ),
        "win_game_over_timeout": {"WIN", "GAME_OVER", "TIMEOUT"}.issubset(
            coverage["terminal_outcomes"]
        ),
        "click_role_coverage": {
            "palette_button",
            "background",
            "edge_detail_tool",
        }.issubset(coverage["click_target_roles"]),
        "paired_case_coverage": coverage["paired_case_count"] >= 10,
    }
    coverage["gates"] = gates
    coverage["all_gates_passed"] = all(gates.values())
    return coverage


def _agent_view(episode, observation_index: int):
    observations = [episode.initial_observation]
    observations.extend(step.result.observation for step in episode.steps)
    observation = observations[observation_index]
    return next(item for item in observation.artifacts if item.role == "agent_view")


def write_human_review(root: Path, episodes) -> None:
    by_id = {item.episode_id: item for item in episodes}
    review_dir = root / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    selected = [
        ("success_through_level_6", 0, "01_level1_initial.png"),
        ("mechanism_paint_pos_0", 0, "02_before_paint.png"),
        ("mechanism_paint_pos_0", -1, "03_after_paint.png"),
        ("mechanism_click_palette_black", 0, "04_before_palette_click.png"),
        ("mechanism_click_palette_black", -1, "05_after_palette_click.png"),
        ("success_through_level_6", -1, "06_final_win.png"),
    ]
    for episode_id, index, filename in selected:
        reference = _agent_view(by_id[episode_id], index)
        shutil.copyfile(root / "assets" / reference.relative_path, review_dir / filename)

    lines = [
        "# CD82 真实轨迹语料审查",
        "",
        "这些图来自真实 `cd82-fb555c5d` 环境，不是 2×2 假方块。画面左上角是 10×10 目标图，",
        "下方中央是 10×10 作画画布，画布周围是可移动工具，顶部黄色边框内是颜色按钮，",
        "最底部是剩余动作预算。",
        "",
        "## 1. 真实初始画面",
        "",
        "![Level 1 initial](01_level1_initial.png)",
        "",
        "## 2. ACTION5 作画前后",
        "",
        "![Before paint](02_before_paint.png)",
        "",
        "![After paint](03_after_paint.png)",
        "",
        "这条机制轨迹先把工具放到上方位置，再执行 ACTION5；下方画布对应区域发生变化。",
        "JSON 中同时保留无损 grid、渲染 PNG、动作、变化 cell 数和 task_state_changed。",
        "",
        "## 3. ACTION6 调色板点击前后",
        "",
        "![Before palette click](04_before_palette_click.png)",
        "",
        "![After palette click](05_after_palette_click.png)",
        "",
        "## 4. 六关完整通关后的真实终局帧",
        "",
        "![Final win](06_final_win.png)",
        "",
        "建议同时查看 `../coverage.json`、`../replay_validation.json` 和",
        "`../episodes/success_through_level_6.json`。",
    ]
    (review_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true", help="replace an existing corpus")
    parser.add_argument("--skip-replay", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        if not args.force:
            raise FileExistsError(f"Output exists; pass --force to replace it: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    builder = CD82CorpusBuilder(output)
    episodes = builder.build()
    coverage = coverage_with_gates(episodes)
    policy_path = PROJECT_ROOT / "socialclaw/trajectory/arc_policies.py"
    write_corpus_metadata(
        output,
        corpus_id=CORPUS_ID,
        benchmark="arc_agi3",
        game_id=GAME_ID,
        environment_fingerprint=arc_environment_fingerprint(GAME_ID),
        collector={
            "script": "scripts/generate_arc_trajectory_corpus.py",
            "script_sha256": _file_sha256(Path(__file__)),
            "policy": "socialclaw/trajectory/arc_policies.py",
            "policy_sha256": _file_sha256(policy_path),
            "network_calls": 0,
            "public_interface_only": True,
            "render_cell_size": 8,
        },
        episodes=episodes,
        splits={"train": [item.episode_id for item in episodes]},
        coverage=coverage,
    )
    validation = validate_corpus(output)

    if args.skip_replay:
        replay = {"status": "skipped", "episodes": []}
    else:
        replay_cases = []
        for index, episode in enumerate(load_corpus_episodes(output), start=1):
            replay_cases.append(replay_arc_episode(output, episode))
            if index % 12 == 0:
                print(f"[replay] {index}/96 episodes verified", flush=True)
        replay = {
            "format_version": 1,
            "status": "passed"
            if all(item["status"] == "passed" for item in replay_cases)
            else "failed",
            "episode_count": len(replay_cases),
            "step_count": sum(item["steps_replayed"] for item in replay_cases),
            "episodes": replay_cases,
        }
    write_json_atomic(output / "replay_validation.json", replay)
    write_human_review(output, episodes)

    summary = {
        "output": str(output),
        "validation": validation["status"],
        "replay_validation": replay["status"],
        "episode_count": coverage["episode_count"],
        "step_count": coverage["step_count"],
        "all_coverage_gates_passed": coverage["all_gates_passed"],
        "coverage_gates": coverage["gates"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if all(
        [
            validation["status"] == "passed",
            replay["status"] in {"passed", "skipped"},
            coverage["all_gates_passed"],
        ]
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
