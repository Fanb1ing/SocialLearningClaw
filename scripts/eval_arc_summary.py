#!/usr/bin/env python3
"""Summarize ARC-AGI-3 baseline results from the runs/ directory.

Scans all runs/<baseline>/<model>/<timestamp>/ directories, reads episode.json
files, and produces a markdown comparison table grouped by (game × baseline).

Usage:
  python scripts/eval_arc_summary.py
  python scripts/eval_arc_summary.py --runs-dir runs --games sk48 cd82 sc25
  python scripts/eval_arc_summary.py --csv results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# Known baseline folder names (maps runs/ subfolder to display name)
_BASELINE_MAP = {
    "arc_zero_shot":    "ZeroShot",
    "arc_few_shot":     "FewShot",
    "arc_rag":          "RAG",
    "arc_agi3":         "Schema",
    "arc_agi3_withrule":"WithRule",
    "arc_withrule":     "WithRule",
    "arc_memory_reflexion": "Reflexion",
    "arc_memory_expel":     "ExpeL",
    "arc_memory_amem":      "A-MEM",
    "arc_memory_tgm":       "TGM",
}

_BASELINE_ORDER = ["ZeroShot", "FewShot", "RAG", "WithRule", "Reflexion", "ExpeL", "A-MEM", "TGM", "Schema"]


@dataclass
class LevelResult:
    level: int
    outcome: str          # WIN / GAME_OVER / TIMEOUT
    steps: int
    flags: List[str] = field(default_factory=list)
    model: str = ""


@dataclass
class RunResult:
    baseline: str
    game: str
    model: str
    run_dir: str
    levels: List[LevelResult] = field(default_factory=list)

    @property
    def win_levels(self) -> int:
        return sum(1 for l in self.levels if l.outcome == "WIN")

    @property
    def total_levels(self) -> int:
        return len(self.levels)

    @property
    def total_steps(self) -> int:
        return sum(l.steps for l in self.levels)

    @property
    def win_rate(self) -> float:
        if not self.levels:
            return 0.0
        return self.win_levels / self.total_levels


def _parse_episode(ep_path: str) -> Optional[LevelResult]:
    try:
        with open(ep_path, "r") as f:
            raw = json.load(f)
    except Exception:
        return None

    # episode.json wraps content under {"created_at": ..., "episode": {...}}
    data = raw.get("episode", raw)

    problem_id = (data.get("problem") or {}).get("id", "") if isinstance(data.get("problem"), dict) else data.get("problem", "")
    # Extract level number from id like "sk48-d8078629_L3"
    level = 0
    m = re.search(r"_L(\d+)", problem_id)
    if m:
        level = int(m.group(1))

    evals = data.get("evals", [])
    outcome = "TIMEOUT"
    steps = 0
    if evals:
        last = evals[-1]
        if last.get("correct"):
            outcome = "WIN"
        elif last.get("details") == "GAME_OVER":
            outcome = "GAME_OVER"
        else:
            outcome = last.get("details", "TIMEOUT")

    attempts = data.get("attempts", [])
    steps = len(attempts)

    flags = data.get("flags", [])
    model = data.get("model", "")
    return LevelResult(level=level, outcome=outcome, steps=steps, flags=flags, model=model)


def _scan_runs(runs_dir: str, game_filter: Optional[List[str]] = None) -> List[RunResult]:
    results: List[RunResult] = []

    if not os.path.isdir(runs_dir):
        print(f"[WARN] runs_dir not found: {runs_dir}")
        return results

    for baseline_folder in sorted(os.listdir(runs_dir)):
        baseline_name = _BASELINE_MAP.get(baseline_folder)
        if baseline_name is None:
            continue
        baseline_dir = os.path.join(runs_dir, baseline_folder)
        if not os.path.isdir(baseline_dir):
            continue

        for model_folder in sorted(os.listdir(baseline_dir)):
            model_dir = os.path.join(baseline_dir, model_folder)
            if not os.path.isdir(model_dir):
                continue

            for ts_folder in sorted(os.listdir(model_dir)):
                run_dir = os.path.join(model_dir, ts_folder)
                if not os.path.isdir(run_dir):
                    continue

                # Collect all episode.json files in this run
                episodes: Dict[str, LevelResult] = {}
                games_seen: set = set()
                for item in os.listdir(run_dir):
                    item_path = os.path.join(run_dir, item)
                    if os.path.isdir(item_path):
                        ep_path = os.path.join(item_path, "episode.json")
                        if os.path.exists(ep_path):
                            # item name is like "sk48-d8078629_L1"
                            game_id = item.split("_L")[0] if "_L" in item else item
                            game_short = game_id.split("-")[0].lower()
                            if game_filter and game_short not in game_filter:
                                continue
                            games_seen.add(game_id)
                            lr = _parse_episode(ep_path)
                            if lr:
                                episodes[item] = lr

                # Group by game_id
                by_game: Dict[str, List[LevelResult]] = defaultdict(list)
                for ep_key, lr in episodes.items():
                    game_id = ep_key.split("_L")[0] if "_L" in ep_key else ep_key
                    by_game[game_id].append(lr)

                for game_id, lvls in by_game.items():
                    lvls_sorted = sorted(lvls, key=lambda l: l.level)
                    model_str = lvls_sorted[0].model if lvls_sorted else model_folder.replace("--", "/")
                    rr = RunResult(
                        baseline=baseline_name,
                        game=game_id.split("-")[0].upper(),
                        model=model_str or model_folder.replace("--", "/"),
                        run_dir=run_dir,
                        levels=lvls_sorted,
                    )
                    results.append(rr)

    return results


def _aggregate(results: List[RunResult]) -> Dict[Tuple[str, str], List[RunResult]]:
    table: Dict[Tuple[str, str], List[RunResult]] = defaultdict(list)
    for r in results:
        table[(r.game, r.baseline)].append(r)
    return table


def _format_cell(runs: List[RunResult]) -> str:
    if not runs:
        return "—"
    # Use the latest run if multiple exist
    r = runs[-1]
    outcome_counts: Dict[str, int] = defaultdict(int)
    for l in r.levels:
        outcome_counts[l.outcome] += 1
    parts = []
    for outcome in ["WIN", "GAME_OVER", "TIMEOUT"]:
        n = outcome_counts.get(outcome, 0)
        if n:
            parts.append(f"{outcome[0]}{n}")
    outcome_str = "/".join(parts) if parts else "?"
    win_str = f"{r.win_levels}/{r.total_levels}"
    return f"{win_str} ({outcome_str}) {r.total_steps}steps"


def print_table(results: List[RunResult]) -> None:
    table = _aggregate(results)
    games = sorted(set(r.game for r in results))
    baselines = [b for b in _BASELINE_ORDER if any(r.baseline == b for r in results)]

    # Header
    col_w = 30
    header = f"{'Game':<8}" + "".join(f"{b:<{col_w}}" for b in baselines)
    print("\n" + "=" * len(header))
    print("ARC-AGI-3 Baseline Comparison")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for game in games:
        row = f"{game:<8}"
        for bl in baselines:
            runs = table.get((game, bl), [])
            row += f"{_format_cell(runs):<{col_w}}"
        print(row)

    print("=" * len(header))
    print("\nFormat: wins/total_levels (W=WIN G=GAME_OVER T=TIMEOUT) total_steps")
    print()

    # Per-run detail
    print("Detailed runs:")
    for r in sorted(results, key=lambda x: (x.game, x.baseline, x.run_dir)):
        level_summary = " | ".join(
            f"L{l.level}:{l.outcome[0]}{l.steps}s" for l in r.levels
        )
        print(f"  [{r.game}] {r.baseline:<10} {r.model:<45} {level_summary}")
    print()


def save_csv(results: List[RunResult], csv_path: str) -> None:
    rows = []
    for r in results:
        for l in r.levels:
            rows.append({
                "game": r.game,
                "baseline": r.baseline,
                "model": r.model,
                "level": l.level,
                "outcome": l.outcome,
                "steps": l.steps,
                "run_dir": r.run_dir,
            })
    with open(csv_path, "w", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"CSV saved: {csv_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize ARC-AGI-3 baseline results")
    p.add_argument("--runs-dir", default=os.path.join(_PROJECT_ROOT, "runs"))
    p.add_argument("--games", nargs="*", help="Filter by game short name (e.g. sk48 cd82 sc25)")
    p.add_argument("--csv", default="", help="Also save results to CSV")
    args = p.parse_args()

    results = _scan_runs(args.runs_dir, game_filter=[g.lower() for g in args.games] if args.games else None)

    if not results:
        print("No results found.")
        return

    print_table(results)

    if args.csv:
        save_csv(results, args.csv)


if __name__ == "__main__":
    main()
