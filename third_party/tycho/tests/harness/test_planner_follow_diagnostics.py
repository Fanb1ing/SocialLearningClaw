"""Planner-follow diagnostic parser contract."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from tycho.harness.planner_follow_diagnostics import analyze_run


def _write_game(path: Path) -> None:
    trace = [
        {
            "turn": 1,
            "action": "ACTION4",
            "reasoning": {"tool_trace": [
                {"tool": "edit_file", "result": "\n\n[auto world-model feedback]\n"
                 "PLANNER_PROBE:\n  PLANNER_STATUS: plan_found\n"
                 "  PLANNER_FIRST_ACTION: ACTION4\n"
                 "  first_actions = ACTION4, ACTION6(row=10,col=20)\n"},
                {"tool": "take_action", "committed": True, "args": {"action": "ACTION4"}},
            ]},
        },
        {
            "turn": 2,
            "action": "ACTION6",
            "reasoning": {"builder_runs": [
                {"report": "confidence: high\nrecommended_action: ACTION6(row=10,col=20)\n"},
            ], "tool_trace": [
                {"tool": "edit_file", "result": "\n\n[auto world-model feedback]\n"
                 "PLANNER_PROBE:\n  PLANNER_STATUS: plan_found\n"
                 "  PLANNER_FIRST_ACTION: ACTION6\n"},
                {"tool": "take_action", "committed": True,
                 "args": {"action": "ACTION6", "row": 10, "col": 20}},
            ]},
        },
        {
            "turn": 3,
            "action": "ACTION6",
            "reasoning": {"tool_trace": [
                {"tool": "run_python", "result": "PLANNER_COMMAND: python plan.py astar\n"
                 "PLANNER_STATUS: plan_found (elapsed_s=0.01, length=2)\n"
                 "PLANNER_FIRST_ACTION: ACTION6(row=11,col=21)\n"
                 "first_actions = ACTION6(row=11,col=21), ACTION2\n"},
                {"tool": "take_action", "committed": True,
                 "args": {"action": "ACTION6", "row": 11, "col": 21}},
            ]},
        },
        {
            "turn": 4,
            "action": "ACTION2",
            "reasoning": {"builder_runs": [
                {"report": "confidence: medium\nrecommended_action: ACTION2\n"},
            ], "tool_trace": [
                {"tool": "Read", "result": "PLANNER_COMMAND: python plan.py astar\n"
                 "PLANNER_STATUS: plan_found\nPLANNER_FIRST_ACTION: ACTION1\n"},
                {"tool": "take_action", "committed": True, "args": {"action": "ACTION2"}},
            ]},
        },
    ]
    path.write_text(json.dumps({"game_id": "zz99-test", "trace": trace}))


def _check_planner_follow_diagnostics() -> dict[str, bool]:
    with TemporaryDirectory() as td:
        root = Path(td)
        _write_game(root / "game_zz99.json")
        res = analyze_run(root)
        s = res["summary"]
        return {
            "auto_counts_two_recommendations": s["auto_recommendations"] == 2,
            "auto_exact_counts_nonclick_only": s["auto_followed"] == 1,
            "auto_action_match_counts_coordinate_less_click": s["auto_action_matched"] == 2,
            "manual_counts_run_python_only": s["manual_recommendations"] == 1,
            "manual_exact_click_match": s["manual_followed"] == 1,
            "builder_counts_reports": s["builder_recommendations"] == 2,
            "builder_exact_matches": s["builder_followed"] == 2,
            "planner_prefix_actions_counted": s["planner_prefix_actions"] == 5,
            "planner_prefix_actions_followed": s["planner_prefix_followed"] == 4,
            "planner_full_prefixes_followed": s["planner_full_prefixes_followed"] == 2,
        }


def test_trace_limits_exclude_later_recommendations(tmp_path: Path) -> None:
    _write_game(tmp_path / "game_zz99.json")

    result = analyze_run(tmp_path, trace_limits={"zz99": 2})

    summary = result["summary"]
    assert summary["auto_recommendations"] == 2
    assert summary["manual_recommendations"] == 0
    assert summary["builder_recommendations"] == 1


def main() -> int:
    ok = True
    checks = _check_planner_follow_diagnostics()
    print("=== planner_follow_diagnostics ===")
    for name, passed in checks.items():
        print(f"  {name}: {passed}")
        ok = ok and passed
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
