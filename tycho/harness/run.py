"""Agent registry and optional file-backed split runner.

Usage:
    python -m tycho.harness.run --approach tycho --split smoke

The primary benchmark entry point is `tycho.harness.run_parallel`. This smaller runner reads
`tycho/harness/splits/<name>.txt` when a project supplies named split files. A split named
`holdout` requires `--i-am-sure` and records its use.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .diagnostics import summarize
from .harness import RESULTS_DIR, RunSummary, _load_arcade, run_env
from .scoring import EnvResult, LevelResult, benchmark_score

SPLIT_DIR = Path(__file__).resolve().parent / "splits"
GUARDED = {"holdout"}
USAGE_FILE = RESULTS_DIR / "holdout_usage.json"

logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("tycho.run")


def load_split(split: str) -> list[str]:
    path = SPLIT_DIR / f"{split}.txt"
    if not path.exists():
        raise SystemExit(f"split file not found: {path}")
    games = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            games.append(line)
    if not games:
        raise SystemExit(f"split {split} is empty")
    return games


APPROACH_MODULES = {
    "tycho": "tycho.agent.agent",
}
try:
    from tycho.harness._run_extension import APPROACH_MODULES as _EXTENSION_APPROACH_MODULES
except (ImportError, AttributeError):
    _EXTENSION_APPROACH_MODULES = {}
APPROACH_MODULES.update(_EXTENSION_APPROACH_MODULES)


def _agent_module(approach: str = "tycho"):
    """Resolve an approach name to its module.

    Dotted module paths remain available for external baselines without adding them to Tycho's core.
    """
    mod = APPROACH_MODULES.get(approach)
    if mod is None and "." in approach:
        mod = approach
    if mod is None:
        known = ", ".join(sorted(APPROACH_MODULES))
        raise SystemExit(f"unknown approach {approach!r}; known approaches: {known}")
    return importlib.import_module(mod)


def load_agent(approach: str = "tycho"):
    """Build a fresh agent for the named approach."""
    return _agent_module(approach).build_agent()


def _agent_factory(approach: str = "tycho"):
    """Returns a zero-arg factory that builds a FRESH agent — one per game, so
    parallel games never share mutable agent state."""
    return _agent_module(approach).build_agent


def bump_usage(split: str, approach: str) -> int:
    RESULTS_DIR.mkdir(exist_ok=True)
    usage = json.loads(USAGE_FILE.read_text()) if USAGE_FILE.exists() else {}
    key = f"{split}:{approach}"
    usage[key] = usage.get(key, 0) + 1
    USAGE_FILE.write_text(json.dumps(usage, indent=2))
    return usage[key]


def resolve_games(arc, split_ids: list[str]) -> dict[str, list[int]]:
    """Map split entries (a short 4-char id, or the full id with its hash suffix) to
    {full_game_id: baseline_actions}, downloading/caching metadata as needed."""
    catalog = {e.game_id: e for e in arc.get_environments()}
    by_prefix = {gid.split("-", 1)[0]: gid for gid in catalog}
    resolved = {}
    for sid in split_ids:
        full = sid if sid in catalog else by_prefix.get(sid.split("-", 1)[0])
        if full is None:
            raise SystemExit(f"game '{sid}' not found in catalog of {len(catalog)} envs")
        info = catalog[full]
        if not info.baseline_actions:
            raise SystemExit(f"game '{full}' has no baseline_actions; cannot score RHAE")
        resolved[full] = info.baseline_actions
    return resolved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--approach", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--api-key", default="", help="ARC_API_KEY; blank uses anonymous key")
    ap.add_argument("--no-trace", action="store_true", help="omit per-turn trace (smaller records)")
    ap.add_argument("--i-am-sure", action="store_true", help="required to run a guarded holdout split")
    ap.add_argument("--hypothesis", default=None, help="optional experiment or hypothesis label")
    ap.add_argument("--viz", action="store_true", help="store raw frames and reasoning per step (large records)")
    ap.add_argument("--parallel", type=int, default=1, help="run N games concurrently (fresh agent+Arcade each); LLM calls are I/O-bound so threads parallelize well")
    args = ap.parse_args()

    if args.split in GUARDED and not args.i_am_sure:
        raise SystemExit(
            f"'{args.split}' is a guarded holdout. Re-run with --i-am-sure if you "
            f"really intend to spend a holdout evaluation (this is logged)."
        )

    split_ids = load_split(args.split)
    build = _agent_factory(args.approach)
    arc = _load_arcade(args.api_key)
    games = resolve_games(arc, split_ids)

    if args.split in GUARDED:
        n = bump_usage(args.split, args.approach)
        log.warning("HOLDOUT USE #%d for %s on %s", n, args.approach, args.split)

    t_start = datetime.now(timezone.utc)

    def play(full_id, baselines):
        """Run ONE game with its OWN fresh agent + Arcade (offline/local engine, so
        concurrent Arcades are safe; LLM calls are I/O-bound -> threads parallelize)."""
        a = build()
        local_arc = _load_arcade(args.api_key) if args.parallel > 1 else arc
        rec = run_env(a, local_arc, full_id, baselines, seed=args.seed,
                      scorecard_id=None, keep_trace=not args.no_trace, keep_frames=args.viz)
        return full_id, rec

    items = list(games.items())
    results: dict = {}
    if args.parallel > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        print(f"running {len(items)} games with parallelism {args.parallel}...")
        with ThreadPoolExecutor(max_workers=args.parallel) as ex:
            futs = {ex.submit(play, fid, bl): fid for fid, bl in items}
            for fut in as_completed(futs):
                fid, rec = fut.result()
                results[fid] = rec
                action_s = (
                    f"{rec.total_actions}+{rec.unfinished_level_actions}"
                    if rec.unfinished_level_actions else str(rec.total_actions)
                )
                print(f"  {fid:18} levels {rec.levels_completed}/{rec.n_levels:<2} "
                      f"actions {action_s:<5} score {rec.env_score:6.2f} "
                      f"{'(trunc '+str(rec.truncated_levels)+')' if rec.truncated_levels else ''}", flush=True)
    else:
        for fid, bl in items:
            _, rec = play(fid, bl)
            results[fid] = rec
            action_s = (
                f"{rec.total_actions}+{rec.unfinished_level_actions}"
                if rec.unfinished_level_actions else str(rec.total_actions)
            )
            print(f"  {fid:18} levels {rec.levels_completed}/{rec.n_levels:<2} "
                  f"actions {action_s:<5} score {rec.env_score:6.2f} "
                  f"{'(trunc '+str(rec.truncated_levels)+')' if rec.truncated_levels else ''}", flush=True)

    # preserve split order for the record
    records = [results[fid] for fid, _ in items if fid in results]
    env_results = [EnvResult(fid, results[fid].n_levels,
                             [LevelResult(**lr) for lr in results[fid].level_results],
                             results[fid].final_state, results[fid].resets)
                   for fid, _ in items if fid in results]

    score = benchmark_score(env_results)
    summary = RunSummary(
        approach=args.approach,
        split=args.split,
        benchmark_score=round(score, 4),
        n_envs=len(records),
        total_levels_completed=sum(r.levels_completed for r in records),
        total_actions=sum(r.total_actions for r in records),
        total_actions_including_unfinished=sum(r.total_actions_including_unfinished for r in records),
        wall_clock_s=round((datetime.now(timezone.utc) - t_start).total_seconds(), 2),
        timestamp=t_start.isoformat(),
        config={"seed": args.seed, "parallel": args.parallel, "budget_mult": 5},
        envs=[asdict(r) for r in records],
        diagnostics=summarize(records),
        hypothesis=args.hypothesis,
    )

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = t_start.strftime("%Y%m%dT%H%M%SZ")
    out = RESULTS_DIR / f"{args.approach}__{args.split}__{stamp}.json"
    out.write_text(json.dumps(asdict(summary), indent=2))

    print(f"\nRHAE benchmark score ({args.split}, {len(records)} envs): {score:.2f} / 100")
    d = summary.diagnostics
    print(f"  failure modes: {d['failure_modes']}")
    print(f"  mean noop-rate {d['mean_noop_rate']} | mean coverage {d['mean_coverage']} | "
          f"envs with progress {d['n_with_progress']}/{len(records)}")
    print(f"  {summary.generalization_note()}")
    if args.hypothesis:
        print(f"  hypothesis: {args.hypothesis}")
    print(f"  record -> {out.relative_to(Path.cwd()) if out.is_relative_to(Path.cwd()) else out}")


if __name__ == "__main__":
    main()
