"""Check whether agent actions follow planner recommendations in recorded runs.

This is a causal-ish trace diagnostic, not a success metric: it only counts a
planner recommendation when the tool result containing it appears before a
committed take_action in the same harness turn.

Usage:
  python -m tycho.harness.planner_follow_diagnostics results/opus48_single_s0_v2
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


_ACTION_RE = re.compile(
    r"\b(ACTION\d+)(?:\((?:(\d+),(\d+)|row=(\d+),col=(\d+))\))?"
)
_PLAN_FIRST_RE = re.compile(
    r"PLANNER_FIRST_ACTION:\s*(ACTION\d+(?:\((?:\d+,\d+|row=\d+,col=\d+)\))?)"
)
_OLD_FIRST_RE = re.compile(
    r"first_actions\s*=\s*(ACTION\d+(?:\((?:\d+,\d+|row=\d+,col=\d+)\))?)"
)
_NUMBERED_FIRST_RE = re.compile(r"^\s*1\.\s*(ACTION\d+)(?:\s+row=(\d+)\s+col=(\d+))?", re.M)
_FIRST_ACTIONS_LINE_RE = re.compile(r"^\s*first_actions\s*=\s*(.+)$", re.M)
_BUILDER_RECOMMENDATION_RE = re.compile(
    r"^\s*recommended_action:\s*(ACTION\d+(?:\((?:\d+,\d+|row=\d+,col=\d+)\))?|none)(?=\s|$)",
    re.M | re.I,
)


@dataclass(frozen=True)
class Action:
    action: str
    row: int | None = None
    col: int | None = None

    def compact(self) -> str:
        if self.row is None:
            return self.action
        return f"{self.action}({self.row},{self.col})"


def _parse_action(text: str | None) -> Action | None:
    if not text:
        return None
    m = _ACTION_RE.search(text)
    if not m:
        return None
    row_s = m.group(2) if m.group(2) is not None else m.group(4)
    col_s = m.group(3) if m.group(3) is not None else m.group(5)
    row = int(row_s) if row_s is not None else None
    col = int(col_s) if col_s is not None else None
    return Action(m.group(1), row, col)


def _args_action(args: dict | None) -> Action | None:
    if not isinstance(args, dict) or not args.get("action"):
        return None
    row = args.get("row")
    col = args.get("col")
    return Action(args["action"], int(row) if row is not None else None,
                  int(col) if col is not None else None)


def _matches(pred: Action, actual: Action) -> bool:
    if pred.action != actual.action:
        return False
    if pred.row is None:
        return pred.action != "ACTION6"
    return pred.row == actual.row and pred.col == actual.col


def _action_matches(pred: Action, actual: Action) -> bool:
    return pred.action == actual.action


def _extract_recommendation(text: str, source: str) -> Action | None:
    if source == "auto":
        if "[auto world-model feedback]" not in text or "PLANNER_PROBE:" not in text:
            return None
        if "plan_found" not in text:
            return None
        for rx in (_PLAN_FIRST_RE, _OLD_FIRST_RE):
            m = rx.search(text)
            if m:
                return _parse_action(m.group(1))
        return None
    if source == "manual":
        if "[auto world-model feedback]" in text:
            return None
        if "PLANNER_COMMAND: python plan.py" not in text and not re.search(r"^PLAN \(", text, re.M):
            return None
        if "plan_found" not in text and "PLAN (" not in text:
            return None
        m = _PLAN_FIRST_RE.search(text)
        if m:
            return _parse_action(m.group(1))
        m = _NUMBERED_FIRST_RE.search(text)
        if m:
            row = int(m.group(2)) if m.group(2) is not None else None
            col = int(m.group(3)) if m.group(3) is not None else None
            return Action(m.group(1), row, col)
    return None


def _extract_plan(text: str, source: str) -> list[Action]:
    """Return the bounded planner prefix actually surfaced before commitment."""
    first = _extract_recommendation(text, source)
    if first is None:
        return []
    line = _FIRST_ACTIONS_LINE_RE.search(text)
    if line:
        actions = [_parse_action(match.group(0)) for match in _ACTION_RE.finditer(line.group(1))]
        parsed = [action for action in actions if action is not None]
        if parsed:
            return parsed
    numbered = []
    for match in re.finditer(
        r"^\s*\d+\.\s*(ACTION\d+(?:\((?:\d+,\d+|row=\d+,col=\d+)\))?)",
        text,
        re.M,
    ):
        action = _parse_action(match.group(1))
        if action is not None:
            numbered.append(action)
    return numbered or [first]


def _extract_builder_guidance(report: str | None) -> Action | None:
    if not report:
        return None
    match = _BUILDER_RECOMMENDATION_RE.search(report)
    if not match or match.group(1).lower() == "none":
        return None
    return _parse_action(match.group(1))


def _tool_result_text(tool: dict) -> str:
    result = tool.get("result")
    if isinstance(result, str):
        return result
    if result is None:
        return ""
    return json.dumps(result, sort_keys=True)


def _step_action(step: dict) -> Action | None:
    """Prefer the committed tool call over denormalized top-level trace fields."""
    trace = ((step.get("reasoning") or {}).get("tool_trace") or [])
    for tool in reversed(trace):
        if tool.get("tool") == "take_action" and tool.get("committed"):
            action = _args_action(tool.get("args"))
            if action is not None:
                return action
    return _args_action({
        "action": step.get("action"),
        "row": step.get("y"),
        "col": step.get("x"),
    })
def _iter_game_files(run_dir: Path) -> Iterable[Path]:
    return sorted(p for p in run_dir.glob("game_*.json") if p.is_file())


def analyze_run(
    run_dir: Path,
    *,
    trace_limits: Mapping[str, int] | None = None,
) -> dict:
    events = []
    totals = {
        "auto_recommendations": 0,
        "auto_followed": 0,
        "auto_action_matched": 0,
        "manual_recommendations": 0,
        "manual_followed": 0,
        "manual_action_matched": 0,
        "builder_recommendations": 0,
        "builder_followed": 0,
        "builder_action_matched": 0,
        "planner_prefix_actions": 0,
        "planner_prefix_followed": 0,
        "planner_full_prefixes_followed": 0,
        "no_following_action": 0,
    }
    by_game: dict[str, dict[str, int]] = {}

    for path in _iter_game_files(run_dir):
        try:
            rec = json.loads(path.read_text())
        except Exception as e:  # noqa: BLE001
            events.append({"game": path.stem, "error": f"{type(e).__name__}: {e}"})
            continue
        game = rec.get("game_id") or path.stem.removeprefix("game_")
        short = str(game).split("-")[0]
        gtot = by_game.setdefault(short, {
            "auto_recommendations": 0,
            "auto_followed": 0,
            "auto_action_matched": 0,
            "manual_recommendations": 0,
            "manual_followed": 0,
            "manual_action_matched": 0,
            "builder_recommendations": 0,
            "builder_followed": 0,
            "builder_action_matched": 0,
            "planner_prefix_actions": 0,
            "planner_prefix_followed": 0,
            "planner_full_prefixes_followed": 0,
            "no_following_action": 0,
        })
        run_trace = rec.get("trace") or []
        if trace_limits is not None:
            if short not in trace_limits:
                raise ValueError(f"missing trace limit for {short}")
            run_trace = run_trace[: int(trace_limits[short])]
        for step_idx, step in enumerate(run_trace):
            trace = ((step.get("reasoning") or {}).get("tool_trace") or [])
            committed = []
            for i, tool in enumerate(trace):
                if tool.get("tool") == "take_action" and tool.get("committed"):
                    actual = _args_action(tool.get("args"))
                    if actual:
                        committed.append((i, actual))
            if not committed:
                fallback = _args_action({
                    "action": step.get("action"),
                    "row": step.get("y"),
                    "col": step.get("x"),
                })
                if fallback:
                    committed.append((10**9, fallback))
            for i, tool in enumerate(trace):
                text = _tool_result_text(tool)
                for source in ("auto", "manual"):
                    if source == "manual" and tool.get("tool") != "run_python":
                        continue
                    pred = _extract_recommendation(text, source)
                    if not pred:
                        continue
                    next_actual = next((a for j, a in committed if j > i), None)
                    key = f"{source}_recommendations"
                    totals[key] += 1
                    gtot[key] += 1
                    followed = bool(next_actual and _matches(pred, next_actual))
                    action_matched = bool(next_actual and _action_matches(pred, next_actual))
                    if action_matched:
                        totals[f"{source}_action_matched"] += 1
                        gtot[f"{source}_action_matched"] += 1
                    if followed:
                        totals[f"{source}_followed"] += 1
                        gtot[f"{source}_followed"] += 1
                    elif next_actual is None:
                        totals["no_following_action"] += 1
                        gtot["no_following_action"] += 1
                    events.append({
                        "game": short,
                        "step_index": step_idx,
                        "turn": step.get("turn"),
                        "source": source,
                        "recommended": pred.compact(),
                        "actual": next_actual.compact() if next_actual else None,
                        "followed": followed,
                        "action_matched": action_matched,
                    })

                    plan = _extract_plan(text, source)
                    if plan:
                        actual_prefix = []
                        start_level = (step.get("reasoning") or {}).get("level")
                        for future in run_trace[step_idx:]:
                            future_level = (future.get("reasoning") or {}).get("level")
                            if start_level is not None and future_level is not None and future_level != start_level:
                                break
                            actual_action = _step_action(future)
                            if actual_action is None or actual_action.action == "RESET":
                                break
                            actual_prefix.append(actual_action)
                            if len(actual_prefix) == len(plan):
                                break
                            state = str(future.get("state") or "")
                            if state.endswith("GAME_OVER"):
                                break
                        followed_prefix = 0
                        for predicted, actual_action in zip(plan, actual_prefix):
                            if not _matches(predicted, actual_action):
                                break
                            followed_prefix += 1
                        for target in (totals, gtot):
                            target["planner_prefix_actions"] += len(plan)
                            target["planner_prefix_followed"] += followed_prefix
                            target["planner_full_prefixes_followed"] += int(
                                followed_prefix == len(plan)
                            )

            builder_runs = (step.get("reasoning") or {}).get("builder_runs") or []
            guidance = next(
                (
                    parsed
                    for parsed in (
                        _extract_builder_guidance(run.get("report"))
                        for run in reversed(builder_runs)
                    )
                    if parsed is not None
                ),
                None,
            )
            if guidance is not None:
                actual = committed[-1][1] if committed else None
                for target in (totals, gtot):
                    target["builder_recommendations"] += 1
                    target["builder_followed"] += int(bool(actual and _matches(guidance, actual)))
                    target["builder_action_matched"] += int(
                        bool(actual and _action_matches(guidance, actual))
                    )
                events.append({
                    "game": short,
                    "step_index": step_idx,
                    "turn": step.get("turn"),
                    "source": "builder",
                    "recommended": guidance.compact(),
                    "actual": actual.compact() if actual else None,
                    "followed": bool(actual and _matches(guidance, actual)),
                    "action_matched": bool(actual and _action_matches(guidance, actual)),
                })

    def _rate(done: int, total: int) -> float | None:
        return round(done / total, 3) if total else None

    summary = dict(totals)
    summary["auto_follow_rate"] = _rate(totals["auto_followed"], totals["auto_recommendations"])
    summary["auto_action_match_rate"] = _rate(totals["auto_action_matched"], totals["auto_recommendations"])
    summary["manual_follow_rate"] = _rate(totals["manual_followed"], totals["manual_recommendations"])
    summary["manual_action_match_rate"] = _rate(totals["manual_action_matched"], totals["manual_recommendations"])
    summary["builder_follow_rate"] = _rate(totals["builder_followed"], totals["builder_recommendations"])
    summary["builder_action_match_rate"] = _rate(
        totals["builder_action_matched"], totals["builder_recommendations"]
    )
    summary["planner_prefix_action_follow_rate"] = _rate(
        totals["planner_prefix_followed"], totals["planner_prefix_actions"]
    )
    return {
        "run_dir": str(run_dir),
        "summary": summary,
        "by_game": by_game,
        "events": events,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--json", action="store_true", help="print machine-readable JSON")
    ap.add_argument("--output", type=Path, help="write machine-readable JSON to this path")
    args = ap.parse_args()
    results = [analyze_run(p) for p in args.run_dirs]
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))
        return 0
    for r in results:
        s = r["summary"]
        print(r["run_dir"])
        print(f"  auto:   {s['auto_followed']}/{s['auto_recommendations']} exact-followed "
              f"(rate={s['auto_follow_rate']}), action-match rate={s['auto_action_match_rate']}")
        print(f"  manual: {s['manual_followed']}/{s['manual_recommendations']} exact-followed "
              f"(rate={s['manual_follow_rate']}), action-match rate={s['manual_action_match_rate']}")
        print(f"  builder: {s['builder_followed']}/{s['builder_recommendations']} exact-followed "
              f"(rate={s['builder_follow_rate']}), action-match rate={s['builder_action_match_rate']}")
        print(f"  surfaced planner prefixes: {s['planner_prefix_followed']}/{s['planner_prefix_actions']} "
              f"actions followed (rate={s['planner_prefix_action_follow_rate']})")
        if s["no_following_action"]:
            print(f"  no following action in same turn: {s['no_following_action']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
