#!/usr/bin/env python3
"""Summarize canonical ARC-AGI-3 outputs, with optional legacy support."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


METHOD_ORDER = [
    "naive",
    "icl",
    "rag",
    "withrule",
    "reflexion",
    "expel",
    "amem",
    "tgm",
    "schema",
]
LEGACY_NAMES = {
    "arc_zero_shot": "naive",
    "arc_few_shot": "icl",
    "arc_rag": "rag",
    "arc_withrule": "withrule",
    "arc_agi3_withrule": "withrule",
    "arc_memory_reflexion": "reflexion",
    "arc_memory_expel": "expel",
    "arc_memory_amem": "amem",
    "arc_memory_tgm": "tgm",
    "arc_agi3": "schema",
}


@dataclass(frozen=True)
class LevelResult:
    level: int
    outcome: str
    steps: int


@dataclass(frozen=True)
class RunResult:
    game: str
    method: str
    model: str
    run_dir: str
    levels: List[LevelResult]
    protocol_signature: str


def comparison_signature(manifest: dict) -> str:
    config = manifest.get("config") or {}
    payload = {
        "model": config.get("model"),
        "temperature": config.get("temperature"),
        "base_url": config.get("base_url"),
        "feedback": config.get("feedback"),
        "budget": config.get("budget"),
        "sample_ids": manifest.get("sample_ids"),
        "dataset_fingerprint": manifest.get("dataset_fingerprint"),
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def parse_episode(path: Path) -> LevelResult | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    episode = raw.get("episode", raw)
    problem = episode.get("problem") or {}
    problem_id = problem.get("id", "") if isinstance(problem, dict) else str(problem)
    match = re.search(r"_L(\d+)", problem_id)
    level = int(match.group(1)) if match else 0
    evaluations = episode.get("evals") or []
    outcome = "TIMEOUT"
    if evaluations:
        last = evaluations[-1]
        if last.get("correct"):
            outcome = "WIN"
        elif str(last.get("details", "")).upper() == "GAME_OVER":
            outcome = "GAME_OVER"
        elif last.get("details"):
            outcome = str(last["details"]).upper()
    return LevelResult(level=level, outcome=outcome, steps=len(episode.get("attempts") or []))


def scan_manifest_runs(output_root: Path) -> List[RunResult]:
    runs: List[RunResult] = []
    for manifest_path in output_root.glob("arc_agi3/*/*/*/manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        config = manifest.get("config") or {}
        if config.get("benchmark") != "arc_agi3":
            continue
        run_dir = manifest_path.parent
        levels = [
            result
            for result in (parse_episode(path) for path in run_dir.glob("*/episode.json"))
            if result is not None
        ]
        sample_ids = manifest.get("sample_ids") or []
        game = str(sample_ids[0] if sample_ids else config.get("split", "unknown"))
        runs.append(
            RunResult(
                game=game,
                method=str(config.get("method", "unknown")),
                model=str(config.get("model", "unknown")),
                run_dir=str(run_dir),
                levels=sorted(levels, key=lambda item: item.level),
                protocol_signature=comparison_signature(manifest),
            )
        )
    return runs


def scan_legacy_runs(root: Path) -> List[RunResult]:
    runs: List[RunResult] = []
    if not root.exists():
        return runs
    for folder, method in LEGACY_NAMES.items():
        base = root / folder
        if not base.exists():
            continue
        for model_dir in base.iterdir():
            if not model_dir.is_dir():
                continue
            for run_dir in model_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                by_game: dict[str, List[LevelResult]] = defaultdict(list)
                for episode_path in run_dir.glob("*/episode.json"):
                    game = episode_path.parent.name.split("_L")[0]
                    parsed = parse_episode(episode_path)
                    if parsed:
                        by_game[game].append(parsed)
                for game, levels in by_game.items():
                    runs.append(
                        RunResult(
                            game=game,
                            method=method,
                            model=model_dir.name.replace("--", "/"),
                            run_dir=str(run_dir),
                            levels=sorted(levels, key=lambda item: item.level),
                            protocol_signature=f"legacy:{run_dir}",
                        )
                    )
    return runs


def print_table(runs: Iterable[RunResult], *, allow_incomparable: bool = False) -> None:
    values = list(runs)
    grouped: dict[tuple[str, str], List[RunResult]] = defaultdict(list)
    for run in values:
        grouped[(run.game.split("-")[0].upper(), run.method)].append(run)
    games = sorted({key[0] for key in grouped})
    methods = [method for method in METHOD_ORDER if any(key[1] == method for key in grouped)]

    for game in games:
        selected = [
            sorted(grouped[(game, method)], key=lambda item: item.run_dir)[-1]
            for method in methods
            if grouped.get((game, method))
        ]
        if len({run.protocol_signature for run in selected}) > 1 and not allow_incomparable:
            print(
                f"Refusing to combine {game} runs with different protocol manifests.",
                file=sys.stderr,
            )
            for run in selected:
                print(f"- {run.method}: {run.run_dir}", file=sys.stderr)
            print("Use --allow-incomparable only for diagnostic viewing.", file=sys.stderr)
            raise SystemExit(2)

    print("| Game | " + " | ".join(methods) + " |")
    print("|---|" + "---:|" * len(methods))
    for game in games:
        cells = []
        for method in methods:
            candidates = grouped.get((game, method), [])
            if not candidates:
                cells.append("—")
                continue
            run = sorted(candidates, key=lambda item: item.run_dir)[-1]
            wins = sum(level.outcome == "WIN" for level in run.levels)
            steps = sum(level.steps for level in run.levels)
            cells.append(f"{wins}/{len(run.levels)}; {steps} steps")
        print(f"| {game} | " + " | ".join(cells) + " |")


def write_csv(path: Path, runs: Iterable[RunResult]) -> None:
    rows = []
    for run in runs:
        for level in run.levels:
            rows.append(
                {
                    "game": run.game,
                    "method": run.method,
                    "model": run.model,
                    "level": level.level,
                    "outcome": level.outcome,
                    "steps": level.steps,
                    "run_dir": run.run_dir,
                }
            )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["game"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="outputs")
    parser.add_argument("--model", required=True)
    parser.add_argument("--include-legacy", action="store_true")
    parser.add_argument("--allow-incomparable", action="store_true")
    parser.add_argument("--csv", default="")
    args = parser.parse_args()

    output_root = Path(args.runs_dir)
    runs = scan_manifest_runs(output_root)
    if args.include_legacy:
        legacy_root = output_root / "legacy" / "runs"
        runs.extend(scan_legacy_runs(legacy_root))
    runs = [run for run in runs if run.model == args.model]
    if not runs:
        print("No ARC-AGI-3 runs found.")
        return
    print_table(runs, allow_incomparable=args.allow_incomparable)
    if args.csv:
        write_csv(Path(args.csv), runs)


if __name__ == "__main__":
    main()
