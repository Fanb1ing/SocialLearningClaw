#!/usr/bin/env python3
"""Generate and validate the first reviewable ARC-AGI-3 CD82 Gold Schema set."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GAME_ID = "cd82-fb555c5d"
GAME_VERSION = "fb555c5d"
SOURCE_PATH = (
    PROJECT_ROOT / "third_party/arc_agi3_games/cd82/fb555c5d/cd82.py"
)
SOURCE_RELATIVE = str(SOURCE_PATH.relative_to(PROJECT_ROOT))
OUTPUT_DIR = PROJECT_ROOT / "gold/arc_agi3/v1/games" / GAME_ID
SPEC_PATH = PROJECT_ROOT / "gold/arc_agi3/v1/schema_spec.json"
MANIFEST_PATH = PROJECT_ROOT / "gold/arc_agi3/v1/manifest.json"
ALL_LEVELS = [1, 2, 3, 4, 5, 6]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def _schema_id(key: str, level: int, payload: dict[str, Any]) -> str:
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
    return f"arc_agi3:{GAME_ID}:L{level}:{key}:{digest}"


def build_schemas(source_sha256: str) -> tuple[list[dict[str, Any]], dict[str, str]]:
    schemas: list[dict[str, Any]] = []
    ids: dict[str, str] = {}

    def add(
        key: str,
        *,
        title: str,
        kind: str,
        trigger: str,
        action_sequence: list[dict[str, Any]],
        expectation: str,
        source_symbol: str,
        source_lines: tuple[int, int],
        additional_sources: list[tuple[str, tuple[int, int]]] | None = None,
        levels: list[int] = ALL_LEVELS,
        abstraction_level: int = 2,
        constraints: list[str] | None = None,
        exceptions: list[str] | None = None,
        requires: list[str] | None = None,
        runtime_evidence: list[str] | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "schema_id": "",
            "format_version": 1,
            "title": title,
            "benchmark": "arc_agi3",
            "benchmark_version": GAME_VERSION,
            "game_id": GAME_ID,
            "level_scope": levels,
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
                    "path": SOURCE_RELATIVE,
                    "sha256": source_sha256,
                    "symbol": symbol,
                    "lines": list(lines),
                }
                for symbol, lines in [
                    (source_symbol, source_lines), *(additional_sources or [])
                ]
            ],
            "runtime_evidence": list(runtime_evidence or []),
            "verification": {
                "static": "passed",
                "runtime": "pending" if runtime_evidence else "not_required",
                "review": "pending",
            },
        }
        payload["schema_id"] = _schema_id(key, abstraction_level, payload)
        schemas.append(payload)
        ids[key] = payload["schema_id"]

    no_action: list[dict[str, Any]] = []
    click = lambda role: [{"action": "ACTION6", "arguments": {"target_role": role}}]
    act = lambda name: [{"action": name, "arguments": {}}]

    add(
        "reference_pattern",
        title="左上 10×10 图案是当前关卡的目标图案",
        kind="observation_semantics",
        trigger="每个 level 的初始画面左上方出现一个独立的 10×10 彩色图案。",
        action_sequence=no_action,
        expectation="该图案保持不变，并作为中央偏下 10×10 画布的目标参考。",
        source_symbol="Cd82.wvrremwltt",
        source_lines=(740, 753),
        additional_sources=[("levels", (257, 361))],
    )
    add(
        "paint_canvas",
        title="中央偏下的黑色 10×10 方块是可绘制画布",
        kind="observation_semantics",
        trigger="画面中央偏下存在初始颜色全为 0 的 10×10 方块。",
        action_sequence=no_action,
        expectation="ACTION5 和后四关的边缘点击工具只修改这块画布；目标图案本身不被修改。",
        source_symbol="sprites['xytrjjbyib']",
        source_lines=(234, 249),
        additional_sources=[
            ("Cd82.rtjwayrycq", (709, 738)),
            ("Cd82.coublenfir", (613, 628)),
        ],
    )
    add(
        "palette_select",
        title="点击调色板按钮会切换当前绘制颜色",
        kind="action_effect",
        trigger="ACTION6 的坐标落在一个顶部调色板按钮内。",
        action_sequence=click("palette_button"),
        expectation="当前颜色变为该按钮中心的颜色，颜色指示条移动到该按钮下方，活动工具同步改色。",
        source_symbol="Cd82.qbiojckwxl",
        source_lines=(551, 572),
        runtime_evidence=["palette_select_color_0"],
    )

    movement = {
        "move_action1": (
            "ACTION1",
            "活动工具位于右、右下、左下或左侧。",
            "位置分别变为右上、右、左、左上；其他位置保持不变。",
        ),
        "move_action2": (
            "ACTION2",
            "活动工具位于右上、右、左或左上。",
            "位置分别变为右、右下、左下、左；其他位置保持不变。",
        ),
        "move_action3": (
            "ACTION3",
            "活动工具位于上、右上、右下或下方。",
            "位置分别变为左上、上、下、左下；其他位置保持不变。",
        ),
        "move_action4": (
            "ACTION4",
            "活动工具位于上、下、左下或左上。",
            "位置分别变为右上、右下、下、上；其他位置保持不变。",
        ),
    }
    for key, (action, trigger, expectation) in movement.items():
        add(
            key,
            title=f"{action} 在八位置环上移动活动工具",
            kind="action_effect",
            trigger=trigger,
            action_sequence=act(action),
            expectation=expectation,
            constraints=["工具不能移出八个外围位置，也不能进入中央位置。"],
            source_symbol="Cd82.nqhfiooufi",
            source_lines=(531, 549),
            runtime_evidence=["navigation_transition_table"],
        )

    paint_regions = [
        ("paint_top_half", "顶部", "画布第 0–4 行的全部 50 个单元格"),
        ("paint_top_right_triangle", "右上", "每行从主对角线单元格到最右侧的上/右三角区域"),
        ("paint_right_half", "右侧", "画布第 5–9 列的全部 50 个单元格"),
        ("paint_bottom_right_triangle", "右下", "每行从副对角线单元格到最右侧的下/右三角区域"),
        ("paint_bottom_half", "底部", "画布第 5–9 行的全部 50 个单元格"),
        ("paint_bottom_left_triangle", "左下", "每行从最左侧到主对角线单元格的下/左三角区域"),
        ("paint_left_half", "左侧", "画布第 0–4 列的全部 50 个单元格"),
        ("paint_top_left_triangle", "左上", "每行从最左侧到副对角线单元格的上/左三角区域"),
    ]
    for index, (key, position, region) in enumerate(paint_regions):
        add(
            key,
            title=f"ACTION5 使用{position}工具填充对应区域",
            kind="action_effect",
            trigger=f"活动绘制工具位于{position}位置，当前选中颜色为 C。",
            action_sequence=act("ACTION5"),
            expectation=f"{region}被设为颜色 C，区域外的画布单元格保持原值；工具动画后返回原位。",
            source_symbol="Cd82.nlvliaznao / Cd82.hhjooisvrs / Cd82.rtjwayrycq",
            source_lines=(673, 738),
            runtime_evidence=[f"action5_region_{index}"],
        )

    add(
        "edge_detail_paint",
        title="点击正方向小工具只填充相邻的 12 格区域",
        kind="action_effect",
        trigger="活动工具位于上、右、下或左侧，同色小工具可见，当前颜色为 C。",
        action_sequence=click("edge_detail_tool"),
        expectation=(
            "点击上、右、下、左小工具时，分别把 rows 0–2/cols 3–6、"
            "rows 3–6/cols 7–9、rows 7–9/cols 3–6、rows 3–6/cols 0–2 "
            "这四个相邻 12 格区域设为 C；其余画布单元格保持原值。"
        ),
        source_symbol="Cd82.gfjfyvajah / Cd82.jgclfnjrnk / Cd82.coublenfir",
        source_lines=(574, 628),
        levels=[3, 4, 5, 6],
        runtime_evidence=["edge_region_0", "edge_region_2", "edge_region_4", "edge_region_6"],
    )
    add(
        "goal_match",
        title="画布在忽略两条对角线后匹配目标即可过关",
        kind="goal",
        trigger="一次 ACTION5 或有效小工具绘制动画结束后。",
        action_sequence=no_action,
        expectation=(
            "比较画布与左上目标图案的全部非对角线单元格；若完全相同则进入下一关，"
            "第 6 关满足时游戏 WIN。主对角线和副对角线上的单元格不参与比较。"
        ),
        constraints=["仅忽略坐标满足 row=col 或 row+col=9 的单元格。"],
        source_symbol="Cd82.wvrremwltt",
        source_lines=(740, 753),
        requires=["reference_pattern", "paint_canvas"],
        runtime_evidence=["goal_ignores_diagonals", "goal_rejects_non_diagonal_mismatch"],
    )
    add(
        "action_budget_loss",
        title="达到 100 次 action 时游戏失败",
        kind="hazard",
        trigger="当前累计 action_count 达到 100。",
        action_sequence=no_action,
        expectation="环境调用 lose()，状态变为 GAME_OVER。",
        source_symbol="Cd82.step",
        source_lines=(630, 635),
        runtime_evidence=["action_budget_game_over"],
    )

    for schema in schemas:
        schema["relations"]["requires"] = [
            ids.get(key, key) for key in schema["relations"]["requires"]
        ]
    return schemas, ids


def _load_game_class():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/socialclaw-mpl")
    import arc_agi

    arcade = arc_agi.Arcade(
        operation_mode=arc_agi.OperationMode.OFFLINE,
        environments_dir=str(PROJECT_ROOT / "third_party/arc_agi3_games"),
    )
    environment = arcade.make(GAME_ID)
    if environment is None:
        raise RuntimeError(f"Could not load {GAME_ID}")
    return environment._game_class


def run_runtime_cases(schema_ids: dict[str, str]) -> list[dict[str, Any]]:
    from arcengine import ActionInput, GameAction, GameState

    game_class = _load_game_class()
    cases: list[dict[str, Any]] = []

    def new_game(level_index: int = 0):
        game = game_class()
        game.set_level(level_index)
        game._state = GameState.NOT_FINISHED
        return game

    def canvas(game):
        return game.current_level.get_sprites_by_name("xytrjjbyib")[0].pixels

    def record(case_id: str, schema_keys: list[str], passed: bool, observed: Any) -> None:
        cases.append(
            {
                "case_id": case_id,
                "schema_ids": [schema_ids[key] for key in schema_keys],
                "passed": bool(passed),
                "observed": observed,
            }
        )

    game = new_game(level_index=5)
    palette_input = game.yrfgxhebei()[0]
    game.perform_action(palette_input, raw=True)
    record(
        "palette_select_color_0",
        ["palette_select"],
        game.knqmgavuh == 0,
        {"selected_color": game.knqmgavuh},
    )

    transition_maps = {
        1: {0: 0, 1: 1, 2: 1, 3: 2, 4: 4, 5: 6, 6: 7, 7: 7},
        2: {0: 0, 1: 2, 2: 3, 3: 3, 4: 4, 5: 5, 6: 5, 7: 6},
        3: {0: 7, 1: 0, 2: 2, 3: 4, 4: 5, 5: 5, 6: 6, 7: 7},
        4: {0: 1, 1: 1, 2: 2, 3: 3, 4: 3, 5: 4, 6: 6, 7: 0},
    }
    navigation_observed: dict[str, dict[str, int]] = {}
    navigation_ok = True
    for action_id, expected in transition_maps.items():
        navigation_observed[str(action_id)] = {}
        for start, target in expected.items():
            game = new_game()
            game.xwmfgtlso = start
            game.azhynfjdiz()
            game.perform_action(ActionInput(id=GameAction.from_id(action_id)), raw=True)
            navigation_observed[str(action_id)][str(start)] = game.xwmfgtlso
            navigation_ok &= game.xwmfgtlso == target
    record(
        "navigation_transition_table",
        ["move_action1", "move_action2", "move_action3", "move_action4"],
        navigation_ok,
        navigation_observed,
    )

    masks = []
    top = np.zeros((10, 10), dtype=bool); top[0:5, :] = True; masks.append(top)
    tr = np.zeros((10, 10), dtype=bool)
    for row in range(10): tr[row, row:10] = True
    masks.append(tr)
    right = np.zeros((10, 10), dtype=bool); right[:, 5:10] = True; masks.append(right)
    br = np.zeros((10, 10), dtype=bool)
    for row in range(10): br[row, 9-row:10] = True
    masks.append(br)
    bottom = np.zeros((10, 10), dtype=bool); bottom[5:10, :] = True; masks.append(bottom)
    bl = np.zeros((10, 10), dtype=bool)
    for row in range(10): bl[row, 0:row+1] = True
    masks.append(bl)
    left = np.zeros((10, 10), dtype=bool); left[:, 0:5] = True; masks.append(left)
    tl = np.zeros((10, 10), dtype=bool)
    for row in range(10): tl[row, 0:10-row] = True
    masks.append(tl)
    paint_keys = [
        "paint_top_half", "paint_top_right_triangle", "paint_right_half",
        "paint_bottom_right_triangle", "paint_bottom_half",
        "paint_bottom_left_triangle", "paint_left_half", "paint_top_left_triangle",
    ]
    for index, (mask, key) in enumerate(zip(masks, paint_keys)):
        game = new_game(level_index=5)
        game.xwmfgtlso = index
        game.azhynfjdiz()
        game.perform_action(ActionInput(id=GameAction.ACTION5), raw=True)
        actual = canvas(game) == 15
        record(
            f"action5_region_{index}",
            [key],
            np.array_equal(actual, mask),
            {"position_index": index, "painted_cells": int(actual.sum())},
        )

    edge_masks: dict[int, np.ndarray] = {}
    mask = np.zeros((10, 10), dtype=bool); mask[0:3, 3:7] = True; edge_masks[0] = mask
    mask = np.zeros((10, 10), dtype=bool); mask[3:7, 7:10] = True; edge_masks[2] = mask
    mask = np.zeros((10, 10), dtype=bool); mask[7:10, 3:7] = True; edge_masks[4] = mask
    mask = np.zeros((10, 10), dtype=bool); mask[3:7, 0:3] = True; edge_masks[6] = mask
    for position, mask in edge_masks.items():
        game = new_game(level_index=5)
        game.xwmfgtlso = position
        game.azhynfjdiz()
        click_input = game.bmwcxxvjum()[0]
        game.perform_action(click_input, raw=True)
        actual = canvas(game) == 15
        record(
            f"edge_region_{position}",
            ["edge_detail_paint"],
            np.array_equal(actual, mask),
            {"position_index": position, "painted_cells": int(actual.sum())},
        )

    game = new_game(level_index=5)
    target = next(
        sprite for sprite in game.current_level.get_sprites()
        if sprite.name.startswith("eoqnvkspoa-")
    ).pixels.copy()
    target_with_diagonal_errors = target.copy()
    for index in range(10):
        target_with_diagonal_errors[index, index] = (int(target[index, index]) + 1) % 16
        target_with_diagonal_errors[index, 9-index] = (int(target[index, 9-index]) + 1) % 16
    canvas(game)[:, :] = target_with_diagonal_errors
    game.wvrremwltt()
    record(
        "goal_ignores_diagonals",
        ["goal_match"],
        game._state == GameState.WIN,
        {"state": str(game._state)},
    )

    game = new_game(level_index=5)
    canvas(game)[:, :] = target
    canvas(game)[0, 1] = (int(target[0, 1]) + 1) % 16
    game.wvrremwltt()
    record(
        "goal_rejects_non_diagonal_mismatch",
        ["goal_match"],
        game._state != GameState.WIN,
        {"state": str(game._state), "mismatch": [0, 1]},
    )

    game = new_game()
    game._action_count = 99
    result = game.perform_action(ActionInput(id=GameAction.ACTION1), raw=True)
    record(
        "action_budget_game_over",
        ["action_budget_loss"],
        result.state == GameState.GAME_OVER,
        {"state": str(result.state), "action_count": game._action_count},
    )
    return cases


def build_coverage(ids: dict[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(requirement_id: str, category: str, schema_keys: list[str], levels=ALL_LEVELS):
        rows.append(
            {
                "requirement_id": requirement_id,
                "category": category,
                "level_scope": levels,
                "schema_ids": [ids[key] for key in schema_keys],
                "status": "covered",
            }
        )

    add("observable_target_and_canvas", "observation", ["reference_pattern", "paint_canvas"])
    add("palette_controls", "control", ["palette_select"])
    for number in range(1, 5):
        add(f"ACTION{number}_navigation", "action", [f"move_action{number}"])
    add("ACTION5_all_eight_positions", "action", [
        "paint_top_half", "paint_top_right_triangle", "paint_right_half",
        "paint_bottom_right_triangle", "paint_bottom_half",
        "paint_bottom_left_triangle", "paint_left_half", "paint_top_left_triangle",
    ])
    add("ACTION6_palette_or_edge_tool", "action", ["palette_select", "edge_detail_paint"])
    add("level_1_goal", "goal", ["goal_match"], [1])
    add("level_2_goal", "goal", ["goal_match"], [2])
    add("level_3_goal", "goal", ["goal_match"], [3])
    add("level_4_goal", "goal", ["goal_match"], [4])
    add("level_5_goal", "goal", ["goal_match"], [5])
    add("level_6_goal", "goal", ["goal_match"], [6])
    add("action_budget", "hazard", ["action_budget_loss"])
    return rows


def build_spec() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "socialclaw.gold_schema.v1",
        "title": "SocialLearningClaw Gold Schema v1",
        "type": "object",
        "required": [
            "schema_id", "format_version", "title", "benchmark",
            "benchmark_version", "game_id", "level_scope", "abstraction_level",
            "kind", "trigger", "action_sequence", "expectation", "constraints",
            "exceptions", "relations", "source_evidence", "runtime_evidence",
            "verification",
        ],
        "properties": {
            "format_version": {"const": 1},
            "benchmark": {"const": "arc_agi3"},
            "abstraction_level": {"type": "integer", "minimum": 0},
            "kind": {"enum": [
                "observation_semantics", "action_precondition", "action_effect",
                "state_transition", "constraint", "hazard", "goal",
            ]},
            "level_scope": {"type": "array", "items": {"type": "integer", "minimum": 1}},
            "action_sequence": {"type": "array"},
            "constraints": {"type": "array"},
            "exceptions": {"type": "array"},
            "source_evidence": {"type": "array", "minItems": 1},
            "runtime_evidence": {"type": "array"},
        },
    }


def validate(
    schemas: list[dict[str, Any]],
    runtime_cases: list[dict[str, Any]],
    coverage: list[dict[str, Any]],
    source_sha256: str,
) -> dict[str, Any]:
    required = set(build_spec()["required"])
    source_line_count = len(SOURCE_PATH.read_text(encoding="utf-8").splitlines())
    schema_ids = [schema["schema_id"] for schema in schemas]
    runtime_by_id = {case["case_id"]: case for case in runtime_cases}
    errors: list[str] = []
    if len(schema_ids) != len(set(schema_ids)):
        errors.append("Duplicate schema_id")
    for schema in schemas:
        missing = required - set(schema)
        if missing:
            errors.append(f"{schema['schema_id']} missing {sorted(missing)}")
        if schema["game_id"] != GAME_ID:
            errors.append(f"{schema['schema_id']} has wrong game_id")
        if not schema["source_evidence"]:
            errors.append(f"{schema['schema_id']} has no source evidence")
        for evidence in schema["source_evidence"]:
            if evidence["sha256"] != source_sha256:
                errors.append(f"{schema['schema_id']} has stale source hash")
            start, end = evidence.get("lines", [0, 0])
            if not (1 <= start <= end <= source_line_count):
                errors.append(
                    f"{schema['schema_id']} has invalid source lines {start}-{end}"
                )
            if not str(evidence.get("symbol", "")).strip():
                errors.append(f"{schema['schema_id']} has an empty source symbol")
        for case_id in schema["runtime_evidence"]:
            if case_id not in runtime_by_id:
                errors.append(f"{schema['schema_id']} cites missing runtime case {case_id}")
            elif not runtime_by_id[case_id]["passed"]:
                errors.append(f"{schema['schema_id']} cites failed runtime case {case_id}")
        if schema["runtime_evidence"]:
            schema["verification"]["runtime"] = "passed"
        elif schema["kind"] in {"action_effect", "constraint", "hazard", "goal"}:
            errors.append(f"{schema['schema_id']} requires runtime evidence")
    for case in runtime_cases:
        if not case["passed"]:
            errors.append(f"Runtime case failed: {case['case_id']}")
        missing_ids = set(case["schema_ids"]) - set(schema_ids)
        if missing_ids:
            errors.append(f"Runtime case {case['case_id']} cites unknown schemas")
    for row in coverage:
        if row["status"] != "covered" or not row["schema_ids"]:
            errors.append(f"Uncovered requirement: {row['requirement_id']}")
        if set(row["schema_ids"]) - set(schema_ids):
            errors.append(f"Coverage {row['requirement_id']} cites unknown schemas")
    goal_levels = {
        level
        for row in coverage if row["category"] == "goal"
        for level in row["level_scope"]
    }
    if goal_levels != set(ALL_LEVELS):
        errors.append(f"Goal coverage mismatch: {sorted(goal_levels)}")
    return {
        "status": "passed" if not errors else "failed",
        "source_sha256": source_sha256,
        "schema_count": len(schemas),
        "runtime_case_count": len(runtime_cases),
        "runtime_passed": sum(case["passed"] for case in runtime_cases),
        "coverage_requirement_count": len(coverage),
        "coverage_complete": all(row["status"] == "covered" for row in coverage),
        "goal_levels_covered": sorted(goal_levels),
        "errors": errors,
    }


def write_review(schemas: list[dict[str, Any]], validation: dict[str, Any]) -> None:
    lines = [
        "# CD82 Gold Schema v1 — 人工审核稿",
        "",
        "> 状态：已按 2026-08-06 人工审核意见修订并通过自动验证。本文不包含完整获胜动作序列。",
        "",
        "## 摘要",
        "",
        f"- Gold Schema：{validation['schema_count']} 条",
        f"- 运行验证：{validation['runtime_passed']}/{validation['runtime_case_count']} 通过",
        f"- Coverage requirements：{validation['coverage_requirement_count']} 项，全部覆盖",
        "- 范围：CD82 六个 levels 中对通关规划有用的观察语义、操作机制、目标判定和 action 上限",
        "",
        "完整结构化节点见 [`schemas.json`](schemas.json)，执行证据见",
        "[`runtime_cases.json`](runtime_cases.json)，覆盖清单见 [`coverage.json`](coverage.json)。",
        "",
        "## 核心机制",
        "",
        "- 左上 10×10 图案是目标；中央偏下的黑色 10×10 方块是画布。",
        "- ACTION1–4 在画布周围八个位置之间移动工具；边界和中央位置不可进入。",
        "- ACTION5 按当前工具方向填充半平面或三角区域。",
        "- Level 3–6 在四个正方向额外提供只填充 12 格的小工具。",
        "- ACTION6 用于选择颜色或点击小工具。",
        "- 画布与目标的非对角线单元格完全相同时过关；两条对角线不参与比较。",
        "- 第 100 次 action 触发 GAME_OVER。",
        "",
        "## ACTION1–4 导航表",
        "",
        "| Action | 会发生移动的起点 → 终点 | 其他起点 |",
        "|---|---|---|",
        "| ACTION1 | 右→右上；右下→右；左下→左；左→左上 | 不动 |",
        "| ACTION2 | 右上→右；右→右下；左→左下；左上→左 | 不动 |",
        "| ACTION3 | 上→左上；右上→上；右下→下；下→左下 | 不动 |",
        "| ACTION4 | 上→右上；下→右下；左下→下；左上→上 | 不动 |",
        "",
        "## ACTION5 绘制表",
        "",
        "| 工具位置 | 被当前颜色覆盖的画布区域 |",
        "|---|---|",
        "| 上 | 第 0–4 行 |",
        "| 右上 | 每行从主对角线到最右侧 |",
        "| 右 | 第 5–9 列 |",
        "| 右下 | 每行从副对角线到最右侧 |",
        "| 下 | 第 5–9 行 |",
        "| 左下 | 每行从最左侧到主对角线 |",
        "| 左 | 第 0–4 列 |",
        "| 左上 | 每行从最左侧到副对角线 |",
        "",
        "## 小范围绘制工具",
        "",
        "| 位置 | ACTION6 点击后覆盖区域 |",
        "|---|---|",
        "| 上 | rows 0–2, cols 3–6（12 格） |",
        "| 右 | rows 3–6, cols 7–9（12 格） |",
        "| 下 | rows 7–9, cols 3–6（12 格） |",
        "| 左 | rows 3–6, cols 0–2（12 格） |",
        "",
        "## 审核意见落实",
        "",
        "- 保留八条 ACTION5 原子规则。",
        "- 保留对角线例外，因为它直接改变通关判定。",
        "- 删除调色板颜色枚举、初始 UI 状态、底部进度条和无效点击等非通关机制。",
        "- 四个同功能的 12 格小工具规则合并为一条方向参数化规则。",
        "- 删除‘小工具出现在哪些关卡’这一事实节点；适用关卡仅保留在 `level_scope` 元数据中。",
        "",
    ]
    (OUTPUT_DIR / "review.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    source_sha256 = _sha256(SOURCE_PATH)
    schemas, ids = build_schemas(source_sha256)
    runtime_cases = run_runtime_cases(ids)
    coverage = build_coverage(ids)
    validation = validate(schemas, runtime_cases, coverage, source_sha256)
    if validation["status"] != "passed":
        raise SystemExit(json.dumps(validation, ensure_ascii=False, indent=2))

    _write_json(SPEC_PATH, build_spec())
    _write_json(OUTPUT_DIR / "schemas.json", {"format_version": 1, "schemas": schemas})
    _write_json(OUTPUT_DIR / "runtime_cases.json", {"format_version": 1, "cases": runtime_cases})
    _write_json(OUTPUT_DIR / "coverage.json", {"format_version": 1, "requirements": coverage})
    _write_json(OUTPUT_DIR / "validation.json", validation)
    _write_json(
        MANIFEST_PATH,
        {
            "format_version": 1,
            "created_date": date.today().isoformat(),
            "benchmark": "arc_agi3",
            "status": "pilot_review_pending",
            "games": [
                {
                    "game_id": GAME_ID,
                    "source": SOURCE_RELATIVE,
                    "source_sha256": source_sha256,
                    "levels": ALL_LEVELS,
                    "schema_count": len(schemas),
                    "validation": "games/cd82-fb555c5d/validation.json",
                    "review": "games/cd82-fb555c5d/review.md",
                }
            ],
        },
    )
    write_review(schemas, validation)
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    print(f"Review: {OUTPUT_DIR / 'review.md'}")


if __name__ == "__main__":
    main()
