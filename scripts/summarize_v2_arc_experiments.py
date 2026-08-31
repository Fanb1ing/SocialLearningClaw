from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(run_dir: Path) -> dict[str, Any]:
    timeline = run_dir / "timeline.json"
    if not timeline.is_file():
        raise FileNotFoundError(f"Finalized timeline is missing: {timeline}")
    return json.loads(timeline.read_text(encoding="utf-8"))["summary"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create one compact result table from finalized V2 ARC runs."
    )
    parser.add_argument("--run", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows = []
    records = []
    for run_dir in args.run:
        summary = _load(run_dir)
        levels_passed = int(summary.get("levels_passed") or 0)
        levels_attempted = int(summary.get("levels_attempted") or 0)
        usage = summary.get("usage") or {}
        failure = str(summary.get("failure_reason") or "无（游戏公开 WIN）")
        records.append({"run_dir": str(run_dir), **summary})
        rows.append(
            "| {game} | {passed}/{attempted} ({rate:.2%}) | {actions} | {tokens:,} | {reason} |".format(
                game=summary.get("game_id"),
                passed=levels_passed,
                attempted=levels_attempted,
                rate=float(summary.get("level_pass_rate") or 0.0),
                actions=int(summary.get("actions") or 0),
                tokens=int(usage.get("total_tokens") or 0),
                reason=failure.replace("|", "/"),
            )
        )

    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "results.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "results.md").write_text(
        "\n".join(
            [
                "# V2 ARC 多游戏正式实验结果",
                "",
                "统一设置：通用零先验 EFPS Agent；每关最多 30 个 Agent 动作；"
                "不限制关卡数，直到公开 WIN 或某关失败。TU93 的 GAME_OVER 可同关 reset，"
                "但 reset 不返还该关动作预算。",
                "",
                "| 游戏 | 通关率（通过/尝试） | Agent 步数 | 总 token | 简单失败原因 |",
                "|---|---:|---:|---:|---|",
                *rows,
                "",
                "每个 run 目录中的 `process.md` 是精简的人类可读逐步过程；"
                "`timeline.json` 保留完整机器审计输入，`token_usage.md` 给出逐 Agent/步骤/token 统计。",
                "",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
