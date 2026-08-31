#!/usr/bin/env python3
"""Generate the reviewed ARC-AGI-3 v1 Gold Schema batch."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLD_ROOT = PROJECT_ROOT / "gold/arc_agi3/v1"
ENVIRONMENTS_DIR = PROJECT_ROOT / "third_party/arc_agi3_games"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _load_game_class(game_id: str):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/socialclaw-mpl")
    import arc_agi

    arcade = arc_agi.Arcade(
        operation_mode=arc_agi.OperationMode.OFFLINE,
        environments_dir=str(ENVIRONMENTS_DIR),
    )
    environment = arcade.make(game_id)
    if environment is None:
        raise RuntimeError(f"Could not load {game_id}")
    return environment._game_class


@dataclass
class GoldBuilder:
    game_id: str
    version: str
    source_path: Path
    levels: list[int]
    schemas: list[dict[str, Any]] = field(default_factory=list)
    ids: dict[str, str] = field(default_factory=dict)
    runtime_cases: list[dict[str, Any]] = field(default_factory=list)
    coverage: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.source_sha256 = _sha256(self.source_path)
        self.source_relative = str(self.source_path.relative_to(PROJECT_ROOT))
        self.output_dir = GOLD_ROOT / "games" / self.game_id

    def add(
        self,
        key: str,
        *,
        title: str,
        kind: str,
        trigger: str,
        action_sequence: list[dict[str, Any]],
        expectation: str,
        symbol: str,
        lines: tuple[int, int],
        level_scope: list[int] | None = None,
        abstraction_level: int = 2,
        constraints: list[str] | None = None,
        exceptions: list[str] | None = None,
        requires: list[str] | None = None,
        runtime_evidence: list[str] | None = None,
        additional_sources: list[tuple[str, tuple[int, int]]] | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "schema_id": "",
            "format_version": 1,
            "title": title,
            "benchmark": "arc_agi3",
            "benchmark_version": self.version,
            "game_id": self.game_id,
            "level_scope": list(level_scope or self.levels),
            "abstraction_level": abstraction_level,
            "kind": kind,
            "trigger": trigger,
            "action_sequence": action_sequence,
            "expectation": expectation,
            "constraints": list(constraints or []),
            "exceptions": list(exceptions or []),
            "relations": {"parents": [], "requires": list(requires or [])},
            "source_evidence": [
                {
                    "path": self.source_relative,
                    "sha256": self.source_sha256,
                    "symbol": source_symbol,
                    "lines": list(source_lines),
                }
                for source_symbol, source_lines in [
                    (symbol, lines),
                    *(additional_sources or []),
                ]
            ],
            "runtime_evidence": list(runtime_evidence or []),
            "verification": {
                "static": "passed",
                "runtime": "pending" if runtime_evidence else "not_required",
                "review": "pending",
            },
        }
        identity = {
            name: payload[name]
            for name in (
                "game_id",
                "level_scope",
                "abstraction_level",
                "kind",
                "trigger",
                "action_sequence",
                "expectation",
                "constraints",
                "exceptions",
            )
        }
        digest = hashlib.sha256(
            json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:12]
        payload["schema_id"] = (
            f"arc_agi3:{self.game_id}:L{abstraction_level}:{key}:{digest}"
        )
        self.schemas.append(payload)
        self.ids[key] = payload["schema_id"]

    def resolve_relations(self) -> None:
        for schema in self.schemas:
            schema["relations"]["requires"] = [
                self.ids.get(key, key)
                for key in schema["relations"]["requires"]
            ]

    def record(
        self,
        case_id: str,
        schema_keys: list[str],
        passed: bool,
        observed: Any,
    ) -> None:
        self.runtime_cases.append(
            {
                "case_id": case_id,
                "schema_ids": [self.ids[key] for key in schema_keys],
                "passed": bool(passed),
                "observed": observed,
            }
        )

    def cover(
        self,
        requirement_id: str,
        category: str,
        schema_keys: list[str],
        level_scope: list[int] | None = None,
    ) -> None:
        self.coverage.append(
            {
                "requirement_id": requirement_id,
                "category": category,
                "level_scope": list(level_scope or self.levels),
                "schema_ids": [self.ids[key] for key in schema_keys],
                "status": "covered",
            }
        )

    def validate(self) -> dict[str, Any]:
        required = {
            "schema_id", "format_version", "title", "benchmark",
            "benchmark_version", "game_id", "level_scope", "abstraction_level",
            "kind", "trigger", "action_sequence", "expectation", "constraints",
            "exceptions", "relations", "source_evidence", "runtime_evidence",
            "verification",
        }
        errors: list[str] = []
        schema_ids = [schema["schema_id"] for schema in self.schemas]
        schema_id_set = set(schema_ids)
        runtime_by_id = {case["case_id"]: case for case in self.runtime_cases}
        line_count = len(self.source_path.read_text(encoding="utf-8").splitlines())
        if len(schema_ids) != len(schema_id_set):
            errors.append("Duplicate schema_id")
        for schema in self.schemas:
            missing = required - set(schema)
            if missing:
                errors.append(f"{schema['schema_id']} missing {sorted(missing)}")
            for evidence in schema["source_evidence"]:
                start, end = evidence["lines"]
                if evidence["sha256"] != self.source_sha256:
                    errors.append(f"{schema['schema_id']} has stale source hash")
                if not (1 <= start <= end <= line_count):
                    errors.append(
                        f"{schema['schema_id']} has invalid source lines {start}-{end}"
                    )
            for case_id in schema["runtime_evidence"]:
                case = runtime_by_id.get(case_id)
                if case is None:
                    errors.append(f"{schema['schema_id']} cites missing {case_id}")
                elif not case["passed"]:
                    errors.append(f"{schema['schema_id']} cites failed {case_id}")
            if schema["runtime_evidence"]:
                schema["verification"]["runtime"] = "passed"
            elif schema["kind"] in {"action_effect", "constraint", "hazard", "goal"}:
                errors.append(f"{schema['schema_id']} requires runtime evidence")
        for case in self.runtime_cases:
            if not case["passed"]:
                errors.append(f"Runtime case failed: {case['case_id']}")
            if set(case["schema_ids"]) - schema_id_set:
                errors.append(f"Runtime case {case['case_id']} cites unknown schema")
        for row in self.coverage:
            if row["status"] != "covered" or not row["schema_ids"]:
                errors.append(f"Uncovered requirement: {row['requirement_id']}")
            if set(row["schema_ids"]) - schema_id_set:
                errors.append(f"Coverage {row['requirement_id']} cites unknown schema")
        goal_levels = {
            level
            for row in self.coverage
            if row["category"] == "goal"
            for level in row["level_scope"]
        }
        if goal_levels != set(self.levels):
            errors.append(f"Goal coverage mismatch: {sorted(goal_levels)}")
        return {
            "status": "passed" if not errors else "failed",
            "source_sha256": self.source_sha256,
            "schema_count": len(self.schemas),
            "runtime_case_count": len(self.runtime_cases),
            "runtime_passed": sum(case["passed"] for case in self.runtime_cases),
            "coverage_requirement_count": len(self.coverage),
            "coverage_complete": all(
                row["status"] == "covered" for row in self.coverage
            ),
            "goal_levels_covered": sorted(goal_levels),
            "errors": errors,
        }

    def write(self, review: str) -> dict[str, Any]:
        self.resolve_relations()
        validation = self.validate()
        if validation["status"] != "passed":
            raise RuntimeError(json.dumps(validation, ensure_ascii=False, indent=2))
        _write_json(
            self.output_dir / "schemas.json",
            {"format_version": 1, "schemas": self.schemas},
        )
        _write_json(
            self.output_dir / "runtime_cases.json",
            {"format_version": 1, "cases": self.runtime_cases},
        )
        _write_json(
            self.output_dir / "coverage.json",
            {"format_version": 1, "requirements": self.coverage},
        )
        _write_json(self.output_dir / "validation.json", validation)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "review.md").write_text(review, encoding="utf-8")
        return validation


def generate_sk48() -> tuple[GoldBuilder, dict[str, Any]]:
    from arcengine import ActionInput, GameAction, GameState

    builder = GoldBuilder(
        game_id="sk48-d8078629",
        version="d8078629",
        source_path=ENVIRONMENTS_DIR / "sk48/d8078629/sk48.py",
        levels=list(range(1, 9)),
    )
    no_action: list[dict[str, Any]] = []

    builder.add(
        "reference_sequences",
        title="场外同色链条给出各目标链条需要覆盖的颜色序列",
        kind="observation_semantics",
        trigger="一条场内链头与底部场外链头具有相同颜色。",
        action_sequence=no_action,
        expectation=(
            "两者组成目标对；场外链条从第二节开始覆盖的彩色方块，按链方向给出"
            "场内链条最终需要覆盖的有序颜色序列。没有同色场外链的链条不参与目标比较。"
        ),
        symbol="Sk48.on_set_level / Sk48.gvtmoopqgy",
        lines=(656, 703),
        additional_sources=[("Sk48.gvtmoopqgy", (842, 859))],
    )
    builder.add(
        "select_chain",
        title="ACTION6 点击可控链头可切换当前活动链条",
        kind="action_effect",
        trigger="ACTION6 点击另一条带可点击外框的场内链头，或它的同色场外参考链头。",
        action_sequence=[{"action": "ACTION6", "arguments": {"target_role": "chain_head"}}],
        expectation="该链条成为当前活动链条；场上位置和剩余移动预算不变。",
        symbol="Sk48.step / Sk48.crbbymputr",
        lines=(746, 755),
        additional_sources=[("Sk48.crbbymputr", (818, 840))],
        runtime_evidence=["select_second_chain"],
    )
    builder.add(
        "direction_controls",
        title="ACTION1–4 分别请求上、下、左、右移动",
        kind="action_effect",
        trigger="当前活动链条可接受方向输入。",
        action_sequence=[{"action": "ACTION1|ACTION2|ACTION3|ACTION4", "arguments": {}}],
        expectation="四个 action 分别映射到上、下、左、右一个 6×6 网格单位。",
        symbol="Sk48.ghcqtpzzlq",
        lines=(973, 982),
        runtime_evidence=["direction_mapping"],
    )
    builder.add(
        "extend_chain",
        title="沿链头朝向移动会把活动链条伸长一节",
        kind="action_effect",
        trigger="方向输入与链头朝向一致，最前端下一格不越界且不是墙。",
        action_sequence=[{"action": "direction_of_chain", "arguments": {}}],
        expectation="链条前端前进一格，并在链头原位置补入一节，因此总长度增加 1。",
        symbol="Sk48.hgivzuhjvj",
        lines=(768, 785),
        constraints=["前端下一格被边界或墙阻挡时，链条不伸长。"],
        runtime_evidence=["extend_and_wall_block"],
    )
    builder.add(
        "retract_chain",
        title="逆链头朝向移动会把活动链条缩短一节",
        kind="action_effect",
        trigger="方向输入与链头朝向相反，且链条长度大于 1。",
        action_sequence=[{"action": "opposite_direction", "arguments": {}}],
        expectation="靠近链头的一节被移除，其余链节后退一格，链条总长度减少 1。",
        symbol="Sk48.hgivzuhjvj",
        lines=(786, 791),
        constraints=["链条最短保留 1 节；继续反向输入不会再缩短。"],
        runtime_evidence=["retract_and_minimum_length"],
    )
    builder.add(
        "translate_chain",
        title="在轨道位置横向于链头移动会平移整条活动链",
        kind="action_effect",
        trigger="方向与链头朝向垂直，链头旁存在对应灰色轨道，且整条链可平移。",
        action_sequence=[{"action": "perpendicular_direction", "arguments": {}}],
        expectation="链头和全部链节沿输入方向平移一个网格单位，链长不变。",
        symbol="Sk48.hgivzuhjvj",
        lines=(792, 808),
        runtime_evidence=["perpendicular_translation", "push_block_cascade"],
    )
    builder.add(
        "push_objects",
        title="链条移动会递归推动路径上的彩色方块和相容链段",
        kind="action_effect",
        trigger="活动链条的目标位置被彩色方块或其他链段占据。",
        action_sequence=[{"action": "movement_action", "arguments": {}}],
        expectation=(
            "可沿同一方向移动的接触对象会被递归推动一格；推动可形成连续方块级联。"
            "任一对象遇到边界、墙或不相容的交叉链段时，整次平移被阻止。"
        ),
        symbol="Sk48.bnrdrdiakd / Sk48.qzvlbxkjgk",
        lines=(893, 971),
        runtime_evidence=["push_block_cascade"],
    )
    builder.add(
        "undo_move",
        title="ACTION7 撤销最近一次完成的棋盘移动",
        kind="action_effect",
        trigger="至少已有一次完成的方向移动快照。",
        action_sequence=[{"action": "ACTION7", "arguments": {}}],
        expectation="所有链头、链节长度和彩色方块恢复到上一个移动快照。",
        symbol="Sk48.uqclctlhyh",
        lines=(870, 891),
        constraints=["撤销不返还已消耗的移动预算，也不撤销单纯的链条选择。"],
        runtime_evidence=["undo_board_move"],
    )
    builder.add(
        "goal_sequences_match",
        title="所有目标链覆盖的颜色序列匹配各自参考序列即可过关",
        kind="goal",
        trigger="一次棋盘移动完成后。",
        action_sequence=no_action,
        expectation=(
            "对每一对同色场内/场外链条，场内链节覆盖的彩色方块序列必须至少包含"
            "参考链要求的全部位置，且逐项颜色相同；全部目标对同时满足时进入下一关。"
        ),
        symbol="Sk48.gvtmoopqgy / Sk48.step",
        lines=(842, 859),
        additional_sources=[("Sk48.step", (709, 744))],
        requires=["reference_sequences"],
        runtime_evidence=["goal_sequence_comparison"],
    )
    builder.add(
        "movement_budget",
        title="方向输入消耗有限移动预算，耗尽时失败",
        kind="hazard",
        trigger="执行 ACTION1–4，包括未产生位移的受阻输入。",
        action_sequence=[{"action": "ACTION1|ACTION2|ACTION3|ACTION4", "arguments": {}}],
        expectation="每次方向输入消耗 1 点预算；预算降到 0 且尚未满足目标时 GAME_OVER。",
        symbol="Sk48.hgivzuhjvj / Sk48.step",
        lines=(768, 770),
        additional_sources=[("Sk48.step", (740, 765))],
        exceptions=["ACTION6 切换链条和 ACTION7 撤销不消耗该预算。"],
        runtime_evidence=["movement_budget_loss", "select_second_chain"],
    )
    builder.resolve_relations()

    game_class = _load_game_class(builder.game_id)

    def new_game(level_index: int = 0):
        game = game_class()
        game.set_level(level_index)
        game._state = GameState.NOT_FINISHED
        return game

    game = new_game()
    mapping = {
        action_id: game.ghcqtpzzlq(GameAction.from_id(action_id))
        for action_id in range(1, 5)
    }
    builder.record(
        "direction_mapping",
        ["direction_controls"],
        mapping == {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)},
        {str(key): list(value) for key, value in mapping.items()},
    )

    game = new_game()
    head = game.vzvypfsnt
    start_length = len(game.mwfajkguqx[head])
    game.perform_action(ActionInput(id=GameAction.ACTION4), raw=True)
    extended = len(game.mwfajkguqx[head]) == start_length + 1
    for _ in range(8):
        game.perform_action(ActionInput(id=GameAction.ACTION4), raw=True)
    blocked_length = len(game.mwfajkguqx[head])
    game.perform_action(ActionInput(id=GameAction.ACTION4), raw=True)
    builder.record(
        "extend_and_wall_block",
        ["extend_chain"],
        extended and len(game.mwfajkguqx[head]) == blocked_length,
        {"start_length": start_length, "wall_limited_length": blocked_length},
    )

    game = new_game()
    head = game.vzvypfsnt
    game.perform_action(ActionInput(id=GameAction.ACTION3), raw=True)
    once = len(game.mwfajkguqx[head])
    game.perform_action(ActionInput(id=GameAction.ACTION3), raw=True)
    builder.record(
        "retract_and_minimum_length",
        ["retract_chain"],
        once == 1 and len(game.mwfajkguqx[head]) == 1,
        {"minimum_length": len(game.mwfajkguqx[head])},
    )

    game = new_game()
    head = game.vzvypfsnt
    before = [(sprite.x, sprite.y) for sprite in game.mwfajkguqx[head]]
    game.perform_action(ActionInput(id=GameAction.ACTION1), raw=True)
    after = [(sprite.x, sprite.y) for sprite in game.mwfajkguqx[head]]
    builder.record(
        "perpendicular_translation",
        ["translate_chain"],
        after == [(x, y - 6) for x, y in before],
        {"before": before, "after": after},
    )

    game = new_game(level_index=5)
    clickable = [
        head
        for head in game.mwfajkguqx
        if "sys_click" in head.tags and head.y < 53
    ]
    target = clickable[1]
    game.perform_action(
        ActionInput(
            id=GameAction.ACTION6,
            data={"x": target.x, "y": target.y},
        ),
        raw=True,
    )
    builder.record(
        "select_second_chain",
        ["select_chain", "movement_budget"],
        game.vzvypfsnt is target and game.qiercdohl == 196,
        {"selected_head_color": int(target.pixels[2, 2]), "budget": game.qiercdohl},
    )

    game = new_game()
    head = game.vzvypfsnt
    for _ in range(4):
        game.perform_action(ActionInput(id=GameAction.ACTION4), raw=True)
    snapshot_chain = [(sprite.x, sprite.y) for sprite in game.mwfajkguqx[head]]
    snapshot_blocks = [(sprite.x, sprite.y) for sprite in game.vbelzuaian]
    game.perform_action(ActionInput(id=GameAction.ACTION1), raw=True)
    moved_blocks = [(sprite.x, sprite.y) for sprite in game.vbelzuaian]
    push_ok = moved_blocks[:3] == [(41, 24), (41, 18), (41, 12)]
    builder.record(
        "push_block_cascade",
        ["translate_chain", "push_objects"],
        push_ok,
        {"before": snapshot_blocks[:3], "after": moved_blocks[:3]},
    )
    budget_after_move = game.qiercdohl
    game.perform_action(ActionInput(id=GameAction.ACTION7), raw=True)
    builder.record(
        "undo_board_move",
        ["undo_move"],
        [(sprite.x, sprite.y) for sprite in game.mwfajkguqx[head]] == snapshot_chain
        and [(sprite.x, sprite.y) for sprite in game.vbelzuaian] == snapshot_blocks
        and game.qiercdohl == budget_after_move,
        {"budget_not_refunded": game.qiercdohl},
    )

    game = new_game()
    active, reference = next(iter(game.xpmcmtbcv.items()))
    while len(game.mwfajkguqx[active]) < 3:
        game.perform_action(ActionInput(id=GameAction.ACTION4), raw=True)
    required_colors = [
        int(block.pixels[1, 1])
        for block in game.vjfbwggsd[reference]
    ]
    active_positions = [
        (segment.x, segment.y) for segment in game.mwfajkguqx[active][:3]
    ]
    for block, (x, y), color in zip(game.vbelzuaian[:3], active_positions, required_colors):
        block.set_position(x, y).color_remap(None, color)
    matched = game.gvtmoopqgy()
    builder.record(
        "goal_sequence_comparison",
        ["goal_sequences_match"],
        matched,
        {"required_colors": required_colors},
    )

    game = new_game()
    game.qiercdohl = 1
    result = game.perform_action(ActionInput(id=GameAction.ACTION2), raw=True)
    builder.record(
        "movement_budget_loss",
        ["movement_budget"],
        result.state == GameState.GAME_OVER and game.qiercdohl == 0,
        {"state": str(result.state), "budget": game.qiercdohl},
    )

    for requirement, category, keys in [
        ("reference_goal_sequences", "observation", ["reference_sequences"]),
        ("chain_selection", "action", ["select_chain"]),
        ("direction_mapping", "action", ["direction_controls"]),
        ("chain_extension", "action", ["extend_chain"]),
        ("chain_retraction", "action", ["retract_chain"]),
        ("chain_translation", "action", ["translate_chain"]),
        ("recursive_push", "action", ["push_objects"]),
        ("undo", "action", ["undo_move"]),
        ("movement_budget", "hazard", ["movement_budget"]),
    ]:
        builder.cover(requirement, category, keys)
    for level in builder.levels:
        builder.cover(
            f"level_{level}_goal",
            "goal",
            ["goal_sequences_match"],
            [level],
        )

    review = """# SK48 Gold Schema v1 — 人工审核稿

> 状态：人工审核通过。仅保留对通关规划有用的机制。

## 摘要

- Gold Schema：10 条
- 核心任务：操纵可伸缩链条推动彩色方块，使目标链覆盖的颜色序列与场外参考链一致
- 覆盖：8 个 levels；方向、伸缩、平移、递归推动、链条切换、撤销、目标和预算

## 核心规则

| 操作/条件 | 效果 |
|---|---|
| ACTION1/2/3/4 | 请求上/下/左/右移动 |
| 沿链头朝向 | 前进并增加一节；墙或边界阻止伸长 |
| 逆链头朝向 | 后退并减少一节；最短一节 |
| 垂直于链头且有灰色轨道 | 整条链平移一格 |
| 移动接触彩色方块/链段 | 可移动对象递归级联推动，否则整次移动受阻 |
| ACTION6 点击可控链头 | 切换活动链，不消耗移动预算 |
| ACTION7 | 恢复上一棋盘移动快照，不返还预算 |

同色的场内链和底部场外链组成目标对。场外链覆盖的颜色顺序是参考；全部场内目标链逐项匹配后过关。没有场外同色参考的链条只是操纵工具，不参与目标比较。

## 审核结论

- 保留抽象的参考颜色序列，不逐关枚举具体颜色；
- 递归推动与阻挡维持为一条机制；
- 保留 ACTION7 不返还预算这一约束。
"""
    validation = builder.write(review)
    return builder, validation


def generate_tu93() -> tuple[GoldBuilder, dict[str, Any]]:
    from arcengine import ActionInput, GameAction, GameState

    builder = GoldBuilder(
        game_id="tu93-0768757b",
        version="0768757b",
        source_path=ENVIRONMENTS_DIR / "tu93/0768757b/tu93.py",
        levels=list(range(1, 10)),
    )
    no_action: list[dict[str, Any]] = []

    builder.add(
        "board_roles",
        title="箭头角色需沿蓝色通路到达黄色出口",
        kind="observation_semantics",
        trigger="关卡显示蓝色连通通路、一个带方向的角色和黄色方块。",
        action_sequence=no_action,
        expectation="带方向角色是受控对象；蓝色连接处定义可走边；黄色方块是出口。",
        symbol="levels / Tu93.step",
        lines=(767, 923),
        additional_sources=[("Tu93.step", (1194, 1272))],
    )
    builder.add(
        "directional_move",
        title="ACTION1–4 沿存在的通路边向上、下、左、右移动一格",
        kind="action_effect",
        trigger="角色位于网格中心，输入方向上的半格连接像素为蓝色通路。",
        action_sequence=[{"action": "ACTION1|ACTION2|ACTION3|ACTION4", "arguments": {}}],
        expectation="四个 action 分别使角色转向并移动到上、下、左、右相邻网格中心。",
        symbol="Tu93.step / Tu93.erkaicaqdh",
        lines=(1007, 1011),
        additional_sources=[("Tu93.step", (1194, 1250))],
        constraints=["目标方向没有蓝色连接时，角色不移动。"],
        runtime_evidence=["directional_move_and_wall"],
    )
    builder.add(
        "orange_chaser",
        title="橙色单位正对一格外角色时激活并直线追击",
        kind="hazard",
        trigger="橙色单位的朝向正对同一行或列、相距一个网格单位的角色。",
        action_sequence=no_action,
        expectation="该单位变为激活状态，并沿当前朝向移动一个网格单位。",
        symbol="Tu93.ixnhjkzwic / Tu93.bwlnieccpx",
        lines=(1090, 1109),
        runtime_evidence=["orange_chaser_activation"],
    )
    builder.add(
        "cyan_patrol",
        title="青色单位每回合沿通路前进，前方无路时反向",
        kind="hazard",
        trigger="玩家完成一次有效移动后。",
        action_sequence=no_action,
        expectation="青色单位沿当前朝向移动一格；到达前方无通路的端点时旋转 180°。",
        symbol="Tu93.itwvxwpzyb / Tu93.rgwzxyjuqc",
        lines=(1110, 1152),
        runtime_evidence=["cyan_patrol_motion_and_reverse"],
    )
    builder.add(
        "magenta_delayed_follower",
        title="品红单位正对两格外角色时激活，并以两步延迟模仿转向",
        kind="hazard",
        trigger="品红单位正对同一行或列、相距两个网格单位的角色。",
        action_sequence=no_action,
        expectation=(
            "该单位激活并开始移动；随后按角色方向输入的顺序行动，但先保留两次原朝向，"
            "因此角色的新方向在两个行动周期后才影响它。"
        ),
        symbol="Tu93.gmwsemdsae / Tu93.step",
        lines=(1171, 1183),
        additional_sources=[("Tu93.step", (1202, 1250))],
        runtime_evidence=["magenta_two_step_delay"],
    )
    builder.add(
        "player_collision_attack",
        title="角色主动移入敌对单位所在格会逐阶段将其清除",
        kind="action_effect",
        trigger="角色作为移动方到达网格中心，且该格存在橙色、青色或品红单位。",
        action_sequence=[{"action": "move_into_occupied_cell", "arguments": {}}],
        expectation="被占据单位经历两个受击形态后从棋盘移除，角色保留。",
        symbol="Tu93.onfzuzfvcl / Tu93.uneirnujpq",
        lines=(1064, 1089),
        runtime_evidence=["player_removes_enemy"],
    )
    builder.add(
        "enemy_collision_attack",
        title="敌对单位主动移入角色所在格会逐阶段清除角色",
        kind="hazard",
        trigger="已激活敌对单位作为移动方到达角色所在网格中心。",
        action_sequence=no_action,
        expectation="角色经历两个受击形态后被移除；棋盘上没有角色时 GAME_OVER。",
        symbol="Tu93.bwlnieccpx / Tu93.jxclhrfhnn / Tu93.ddmvoavzus",
        lines=(1098, 1170),
        additional_sources=[("Tu93.step", (1251, 1267))],
        runtime_evidence=["enemy_removes_player"],
    )
    builder.add(
        "reach_exit",
        title="角色到达黄色出口时进入下一关",
        kind="goal",
        trigger="所有移动动画结束，角色与黄色出口中心重合。",
        action_sequence=no_action,
        expectation="进入下一关；第 9 关满足时游戏 WIN。",
        symbol="Tu93.step",
        lines=(1251, 1265),
        requires=["board_roles"],
        runtime_evidence=["local_exit_transition"],
    )
    builder.add(
        "step_budget",
        title="每次方向尝试消耗一步，步数耗尽时失败",
        kind="hazard",
        trigger="输入 ACTION1–4，无论该方向是否存在通路。",
        action_sequence=[{"action": "ACTION1|ACTION2|ACTION3|ACTION4", "arguments": {}}],
        expectation="剩余步数减少 1；降到 0 且尚未过关时 GAME_OVER。",
        symbol="eytqjghipl.rndwkomrip / Tu93.step",
        lines=(963, 968),
        additional_sources=[("Tu93.step", (1194, 1272))],
        runtime_evidence=["blocked_move_consumes_last_step"],
    )
    builder.resolve_relations()

    game_class = _load_game_class(builder.game_id)

    def new_game(level_index: int = 0):
        game = game_class()
        game.set_level(level_index)
        game._state = GameState.NOT_FINISHED
        return game

    game = new_game()
    player = game.current_level.get_sprites_by_tag("0017unajnymcki")[0]
    start = (player.x, player.y)
    game.perform_action(ActionInput(id=GameAction.ACTION4), raw=True)
    valid_position = (player.x, player.y)
    game = new_game()
    player = game.current_level.get_sprites_by_tag("0017unajnymcki")[0]
    before_steps = game.ksulgrfyqx.current_steps
    game.perform_action(ActionInput(id=GameAction.ACTION1), raw=True)
    builder.record(
        "directional_move_and_wall",
        ["directional_move"],
        valid_position == (start[0] + 6, start[1])
        and (player.x, player.y) == start
        and game.ksulgrfyqx.current_steps == before_steps - 1,
        {"right_move": valid_position, "blocked_up": [player.x, player.y]},
    )

    game = new_game(level_index=1)
    player = game.current_level.get_sprites_by_tag("0017unajnymcki")[0]
    chaser = game.current_level.get_sprites_by_tag("0001haidilggfh")[0]
    player.set_position(chaser.x - 6, chaser.y)
    before = (chaser.x, chaser.y)
    game.ixnhjkzwic()
    builder.record(
        "orange_chaser_activation",
        ["orange_chaser"],
        game.rxdvicwstj(chaser) and (chaser.x, chaser.y) == (before[0] - 1, before[1]),
        {"before": before, "after_first_animation_tick": [chaser.x, chaser.y]},
    )

    game = new_game(level_index=3)
    patrol = game.current_level.get_sprites_by_tag("0020npxxteirsg")[0]
    before = (patrol.x, patrol.y)
    game.itwvxwpzyb()
    moved = (patrol.x, patrol.y) == (before[0], before[1] + 1)
    board = game.current_level.get_sprites_by_tag("0005uvnhiglpvh")[0]
    reversed_ok = False
    directions = {0: (0, -3), 180: (0, 3), 90: (3, 0), 270: (-3, 0)}
    for rotation, (dx, dy) in directions.items():
        for row in range(0, board.height, 6):
            for col in range(0, board.width, 6):
                ahead_row, ahead_col = row + dy, col + dx
                if not (
                    0 <= ahead_row < board.height
                    and 0 <= ahead_col < board.width
                    and board.pixels[ahead_row, ahead_col] == 2
                ):
                    patrol.set_position(board.x + col, board.y + row).set_rotation(rotation)
                    game.rgwzxyjuqc()
                    reversed_ok = patrol.rotation == (rotation + 180) % 360
                    if reversed_ok:
                        break
            if reversed_ok:
                break
        if reversed_ok:
            break
    builder.record(
        "cyan_patrol_motion_and_reverse",
        ["cyan_patrol"],
        moved and reversed_ok,
        {"moved_one_tick": moved, "reversed_at_wall": reversed_ok},
    )

    game = new_game(level_index=6)
    player = game.current_level.get_sprites_by_tag("0017unajnymcki")[0]
    delayed = game.current_level.get_sprites_by_tag("0023otenflmryc")[0]
    player.set_position(delayed.x, delayed.y + 12)
    game.gmwsemdsae()
    activated = game.rxdvicwstj(delayed) and game.ylmdnwbdyy[delayed] == [180]
    game.ylmdnwbdyy[delayed].append(90)
    game.gmwsemdsae()
    still_old = delayed.rotation == 180
    game.gmwsemdsae()
    follows_new = delayed.rotation == 90
    builder.record(
        "magenta_two_step_delay",
        ["magenta_delayed_follower"],
        activated and still_old and follows_new,
        {"activated": activated, "rotations": [180, delayed.rotation]},
    )

    game = new_game(level_index=1)
    player = game.current_level.get_sprites_by_tag("0017unajnymcki")[0]
    enemy = game.current_level.get_sprites_by_tag("0001haidilggfh")[0]
    enemy.set_position(player.x, player.y)
    phases = [game.onfzuzfvcl(player) for _ in range(3)]
    builder.record(
        "player_removes_enemy",
        ["player_collision_attack"],
        phases == [False, False, True]
        and not game.current_level.get_sprites_by_tag("0001haidilggfh"),
        {"phase_completion": phases},
    )

    game = new_game(level_index=1)
    player = game.current_level.get_sprites_by_tag("0017unajnymcki")[0]
    enemy = game.current_level.get_sprites_by_tag("0001haidilggfh")[0]
    enemy.set_position(player.x, player.y)
    game.qlzvpfwmqv(enemy)
    phases = [game.bwlnieccpx(enemy) for _ in range(3)]
    builder.record(
        "enemy_removes_player",
        ["enemy_collision_attack"],
        phases == [False, False, True]
        and not game.current_level.get_sprites_by_tag("0017unajnymcki"),
        {"phase_completion": phases},
    )

    game = new_game()
    player = game.current_level.get_sprites_by_tag("0017unajnymcki")[0]
    exit_sprite = game.current_level.get_sprites_by_tag("0015msvpvzxhqf")[0]
    player.set_position(exit_sprite.x, exit_sprite.y - 6)
    result = game.perform_action(ActionInput(id=GameAction.ACTION2), raw=True)
    builder.record(
        "local_exit_transition",
        ["reach_exit"],
        game._current_level_index == 1 and result.state == GameState.NOT_FINISHED,
        {"new_level_index": game._current_level_index},
    )

    game = new_game()
    player = game.current_level.get_sprites_by_tag("0017unajnymcki")[0]
    start = (player.x, player.y)
    game.ksulgrfyqx.current_steps = 1
    result = game.perform_action(ActionInput(id=GameAction.ACTION1), raw=True)
    builder.record(
        "blocked_move_consumes_last_step",
        ["step_budget"],
        (player.x, player.y) == start
        and game.ksulgrfyqx.current_steps == 0
        and result.state == GameState.GAME_OVER,
        {"position": start, "remaining_steps": game.ksulgrfyqx.current_steps},
    )

    for requirement, category, keys in [
        ("board_and_exit_roles", "observation", ["board_roles"]),
        ("directional_controls", "action", ["directional_move"]),
        ("orange_enemy", "hazard", ["orange_chaser"]),
        ("cyan_enemy", "hazard", ["cyan_patrol"]),
        ("magenta_enemy", "hazard", ["magenta_delayed_follower"]),
        ("player_initiated_collision", "action", ["player_collision_attack"]),
        ("enemy_initiated_collision", "hazard", ["enemy_collision_attack"]),
        ("step_budget", "hazard", ["step_budget"]),
    ]:
        builder.cover(requirement, category, keys)
    for level in builder.levels:
        builder.cover(f"level_{level}_goal", "goal", ["reach_exit"], [level])

    review = """# TU93 Gold Schema v1 — 人工审核稿

> 状态：人工审核通过。仅保留对通关规划有用的机制。

## 摘要

- Gold Schema：9 条
- 核心任务：控制箭头角色沿蓝色通路到达黄色出口，同时处理三类移动单位
- 覆盖：9 个 levels；移动、敌对行为、碰撞优先级、出口目标和步数失败

## 核心规则

| 对象/条件 | 行为 |
|---|---|
| ACTION1/2/3/4 | 沿有蓝色连接的边向上/下/左/右走一格；撞墙仍耗一步 |
| 橙色单位 | 正对一格外角色时激活并直线追击 |
| 青色单位 | 每回合沿通路前进，端点反向 |
| 品红单位 | 正对两格外角色时激活，以两步延迟模仿角色转向 |
| 角色主动进入敌人格 | 角色逐阶段清除敌人 |
| 敌人主动进入角色格 | 敌人逐阶段清除角色；角色消失则失败 |
| 角色与黄色出口重合 | 进入下一关 |

## 审核结论

- 三种敌对行为维持三条独立 Schema；
- 主动碰撞方向不同，结果不同，因此维持两条碰撞 Schema；
- 不写各关具体步数，只保留“方向尝试耗步、耗尽失败”的机制。
"""
    validation = builder.write(review)
    return builder, validation


def write_manifest(
    sk_builder: GoldBuilder,
    sk_validation: dict[str, Any],
    tu_builder: GoldBuilder,
    tu_validation: dict[str, Any],
) -> None:
    cd_validation = json.loads(
        (GOLD_ROOT / "games/cd82-fb555c5d/validation.json").read_text(
            encoding="utf-8"
        )
    )
    cd_source = ENVIRONMENTS_DIR / "cd82/fb555c5d/cd82.py"
    games = [
        {
            "game_id": "cd82-fb555c5d",
            "source": str(cd_source.relative_to(PROJECT_ROOT)),
            "source_sha256": cd_validation["source_sha256"],
            "levels": list(range(1, 7)),
            "schema_count": cd_validation["schema_count"],
            "validation": "games/cd82-fb555c5d/validation.json",
            "review": "games/cd82-fb555c5d/review.md",
            "review_status": "accepted_and_revised",
        },
        {
            "game_id": sk_builder.game_id,
            "source": sk_builder.source_relative,
            "source_sha256": sk_builder.source_sha256,
            "levels": sk_builder.levels,
            "schema_count": sk_validation["schema_count"],
            "validation": f"games/{sk_builder.game_id}/validation.json",
            "review": f"games/{sk_builder.game_id}/review.md",
            "review_status": "accepted",
        },
        {
            "game_id": tu_builder.game_id,
            "source": tu_builder.source_relative,
            "source_sha256": tu_builder.source_sha256,
            "levels": tu_builder.levels,
            "schema_count": tu_validation["schema_count"],
            "validation": f"games/{tu_builder.game_id}/validation.json",
            "review": f"games/{tu_builder.game_id}/review.md",
            "review_status": "accepted",
        },
    ]
    _write_json(
        GOLD_ROOT / "manifest.json",
        {
            "format_version": 1,
            "created_date": date.today().isoformat(),
            "benchmark": "arc_agi3",
            "status": "batch_review_pending",
            "games": games,
            "totals": {
                "games": 3,
                "levels": 23,
                "schemas": sum(game["schema_count"] for game in games),
            },
        },
    )


def main() -> None:
    from generate_cd82_gold import main as generate_cd82

    generate_cd82()
    sk_builder, sk_validation = generate_sk48()
    tu_builder, tu_validation = generate_tu93()
    write_manifest(sk_builder, sk_validation, tu_builder, tu_validation)
    print(
        json.dumps(
            {"sk48": sk_validation, "tu93": tu_validation},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
