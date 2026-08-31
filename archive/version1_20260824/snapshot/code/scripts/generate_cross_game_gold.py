#!/usr/bin/env python3
"""Derive provisional ARC-AGI-3 cross-game Gold Schemas from game schemas."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLD_ROOT = PROJECT_ROOT / "gold/arc_agi3/v1"
CROSS_ROOT = GOLD_ROOT / "cross_game"
MANIFEST_PATH = GOLD_ROOT / "manifest.json"


@dataclass(frozen=True)
class Family:
    key: str
    title: str
    kind: str
    trigger: str
    expectation: str
    members: tuple[tuple[str, str], ...]
    constraints: tuple[str, ...] = ()
    exceptions: tuple[str, ...] = ()


def F(
    key: str,
    title: str,
    kind: str,
    trigger: str,
    expectation: str,
    members: list[tuple[str, str]],
    *constraints: str,
) -> Family:
    return Family(
        key, title, kind, trigger, expectation, tuple(members), constraints
    )


LEVEL1: tuple[Family, ...] = (
    F(
        "cardinal_motion",
        "四向输入提出受可通行条件约束的空间移动",
        "action_effect",
        "可控主体收到上、下、左或右的方向请求。",
        "若游戏特定的边界、通路和碰撞条件允许，主体沿该方向移动；否则主体不穿透阻挡。",
        [
            ("ar25-0c556536", "move"),
            ("cn04-2fe56bfb", "move"),
            ("dc22-fdcac232", "walk"),
            ("g50t-5849a774", "move"),
            ("ls20-9607627b", "move"),
            ("sc25-635fd71a", "move"),
            ("sp80-589a99af", "move"),
            ("tu93-0768757b", "directional_move"),
            ("wa30-ee6fef47", "move"),
        ],
        "阻挡只保证不发生非法穿透；输入仍可能消耗预算或推进自主单位。",
    ),
    F(
        "active_entity_selection",
        "点击选择决定后续操作作用的活动实体",
        "action_effect",
        "点击命中多个可控实体之一。",
        "被点击实体成为活动对象；后续移动、旋转、拖动或交互仅作用于它或它代表的关系组。",
        [
            ("ar25-0c556536", "select"),
            ("cn04-2fe56bfb", "select"),
            ("ka59-38d34dbb", "select"),
            ("lf52-271a04aa", "select"),
            ("r11l-495a7899", "select"),
            ("sk48-d8078629", "select_chain"),
            ("sp80-589a99af", "select"),
        ],
        "选择本身通常不等同于完成实体的后续操作。",
    ),
    F(
        "operator_selection",
        "先选择操作参数或程序，再执行后续状态变换",
        "state_transition",
        "界面提供颜色、法术或程序等多个可选操作模式。",
        "选择只更新当前参数/程序；随后的绘制、输入或运行使用该选择解释动作。",
        [
            ("cd82-fb555c5d", "palette_select"),
            ("sc25-635fd71a", "select_spell"),
            ("tn36-ef4dde99", "programs"),
        ],
    ),
    F(
        "relational_propagation",
        "对象变换沿推挤、附着、层级或分组关系传播",
        "state_transition",
        "被操作对象与其他对象存在显式关系边。",
        "根对象的位移或变换传播到关系可达对象；传播规则由推挤、携带、父子或同组关系的类型决定。",
        [
            ("dc22-fdcac232", "crane_move"),
            ("ka59-38d34dbb", "push"),
            ("lp85-305b61c3", "linked"),
            ("s5i5-18d95033", "hierarchy"),
            ("sk48-d8078629", "push_objects"),
            ("wa30-ee6fef47", "coupled"),
        ],
        "跨游戏节点只抽象传播结构，不假定各游戏的位移尺度相同。",
    ),
    F(
        "atomic_rejection",
        "非法复合变换不留下部分成功的中间状态",
        "constraint",
        "一次操作需要同时更新一个对象集合，且集合中至少一个结果非法。",
        "整次操作被阻止或回滚到快照，使对象集合不保留只完成一部分的位移/变换。",
        [
            ("ka59-38d34dbb", "push"),
            ("lp85-305b61c3", "collision"),
            ("r11l-495a7899", "wall"),
            ("s5i5-18d95033", "rollback"),
            ("sk48-d8078629", "push_objects"),
            ("sp80-589a99af", "move"),
        ],
    ),
    F(
        "snapshot_undo",
        "撤销从最近快照恢复可逆状态",
        "action_effect",
        "存在至少一个可撤销的历史快照。",
        "撤销恢复快照记录的对象位置、形态或世界状态；是否返还动作资源由具体游戏决定。",
        [
            ("ar25-0c556536", "undo"),
            ("bp35-0a0ad940", "undo"),
            ("lf52-271a04aa", "undo"),
            ("sb26-7fbdac44", "undo_energy"),
            ("sk48-d8078629", "undo_move"),
            ("su15-1944f8ab", "undo"),
        ],
        "不能从“棋盘恢复”推断“预算恢复”；例如 SK48 明确不返还预算。",
    ),
    F(
        "autonomous_after_action",
        "玩家操作会推进自主单位的策略更新",
        "state_transition",
        "玩家完成一个会推进世界时间的操作。",
        "自主单位根据朝向、目标、距离或记录历史选择并执行自己的状态转移。",
        [
            ("g50t-5849a774", "autonomous"),
            ("ka59-38d34dbb", "enemy"),
            ("su15-1944f8ab", "enemy"),
            ("tu93-0768757b", "cyan_patrol"),
            ("tu93-0768757b", "orange_chaser"),
            ("tu93-0768757b", "magenta_delayed_follower"),
            ("wa30-ee6fef47", "autonomous"),
        ],
        "不同自主单位的策略保持为具体成员 Schema，不在跨游戏层合并成同一策略。",
    ),
    F(
        "structured_interpretation",
        "可见结构被解释为图案、程序或重写序列",
        "state_transition",
        "玩家配置的离散结构达到可解释状态或主动请求求值。",
        "环境按精确图案匹配、递归序列、指令编码或示例翻译规则解释结构，并产生对应状态变化。",
        [
            ("sc25-635fd71a", "pattern"),
            ("sb26-7fbdac44", "evaluate"),
            ("tn36-ef4dde99", "run"),
            ("tr87-cd924810", "translate"),
        ],
        "跨游戏层不统一各自的编码表，只统一“结构先被解释，再执行结果”的过程。",
    ),
    F(
        "matching_contact_reduction",
        "满足匹配关系的接触会消除、合并或完成配对",
        "state_transition",
        "两个对象在同一位置或跳跃关系中接触，并满足具体游戏的同类/同色/同级条件。",
        "接触对象被消除、合并为高阶对象或退出未配对集合；不匹配接触不采用该归约。",
        [
            ("cn04-2fe56bfb", "cancel"),
            ("lf52-271a04aa", "type_match"),
            ("m0r0-492f87ba", "pair"),
            ("su15-1944f8ab", "merge"),
        ],
    ),
    F(
        "world_state_switch",
        "局部触发器切换影响后续可达性的世界状态",
        "state_transition",
        "角色、点击或对象落点激活开关、传感器或环境机关。",
        "门、平台、重力、桥段或空间边界改变状态，从而改变后续移动和碰撞的可行集合。",
        [
            ("bp35-0a0ad940", "gravity_switch"),
            ("bp35-0a0ad940", "world_switch"),
            ("dc22-fdcac232", "click_switch"),
            ("dc22-fdcac232", "key_gate"),
            ("lf52-271a04aa", "world_events"),
            ("m0r0-492f87ba", "gate"),
            ("vc33-5430563c", "transfer"),
        ],
    ),
    F(
        "conjunctive_goal",
        "过关要求一组目标谓词同时成立",
        "goal",
        "环境在动作或动画结算后检查目标集合。",
        "只有全部必需目标谓词都成立才进入下一关；满足部分目标不足以过关。",
        [
            ("ar25-0c556536", "goal"),
            ("cn04-2fe56bfb", "goal"),
            ("ft09-0d8bbf25", "goal"),
            ("lp85-305b61c3", "goal"),
            ("ls20-9607627b", "goal"),
            ("re86-8af5384d", "goal"),
            ("s5i5-18d95033", "goal"),
            ("su15-1944f8ab", "goal"),
            ("tn36-ef4dde99", "goal"),
            ("vc33-5430563c", "goal"),
            ("wa30-ee6fef47", "goal"),
        ],
        "每个成员保留自己的目标谓词、通配符和类型约束。",
    ),
    F(
        "finite_resource_failure",
        "有限动作资源在目标完成前耗尽会触发失败",
        "hazard",
        "受计数操作使剩余步数、能量、尝试次数或总动作额度到达失败边界。",
        "若目标尚未满足，环境进入 GAME_OVER 或执行具体游戏规定的失败/重置流程。",
        [
            ("ar25-0c556536", "budget"),
            ("cd82-fb555c5d", "action_budget_loss"),
            ("dc22-fdcac232", "budget"),
            ("ft09-0d8bbf25", "budget"),
            ("ka59-38d34dbb", "budget"),
            ("lf52-271a04aa", "budget"),
            ("lp85-305b61c3", "budget"),
            ("m0r0-492f87ba", "budget"),
            ("r11l-495a7899", "budget"),
            ("s5i5-18d95033", "budget"),
            ("sb26-7fbdac44", "undo_energy"),
            ("sc25-635fd71a", "budget"),
            ("sk48-d8078629", "movement_budget"),
            ("sp80-589a99af", "attempts"),
            ("tu93-0768757b", "step_budget"),
            ("vc33-5430563c", "budget"),
        ],
        "具体计数单位、是否包含选择/撤销以及失败是否先重置一条生命均由成员 Schema 决定。",
    ),
)


LEVEL0 = (
    {
        "key": "conditional_operator_system",
        "title": "交互环境由带前置条件的状态变换算子组成",
        "kind": "state_transition",
        "trigger": "Agent 选择对象、参数或动作并提交操作。",
        "expectation": "环境检查适用条件，执行算子并保持未被作用域覆盖的状态；不满足条件时拒绝、阻挡或采用成员定义的替代效果。",
        "members": (
            "cardinal_motion",
            "active_entity_selection",
            "operator_selection",
            "structured_interpretation",
            "world_state_switch",
        ),
    },
    {
        "key": "relational_state_system",
        "title": "局部操作可通过对象关系图产生非局部状态变化",
        "kind": "state_transition",
        "trigger": "被操作对象连接到其他对象，或世界含会随时间更新的自主对象。",
        "expectation": "变化沿关系边传播、被原子约束整体接受/拒绝，或触发匹配归约和自主更新。",
        "members": (
            "relational_propagation",
            "atomic_rejection",
            "autonomous_after_action",
            "matching_contact_reduction",
        ),
    },
    {
        "key": "bounded_goal_directed_system",
        "title": "规划是在有限资源内满足复合目标，并可利用有限恢复操作",
        "kind": "goal",
        "trigger": "当前状态尚未满足全部目标且仍有可用操作资源。",
        "expectation": "Agent 必须在失败边界前使全部必要目标谓词成立；撤销等恢复算子可修正状态，但不保证返还资源。",
        "members": (
            "conjunctive_goal",
            "finite_resource_failure",
            "snapshot_undo",
        ),
    },
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _schema_id(level: int, key: str, payload: dict[str, Any]) -> str:
    identity = {
        name: payload[name]
        for name in (
            "abstraction_level",
            "title",
            "kind",
            "trigger",
            "expectation",
            "game_scope",
            "member_schema_ids",
        )
    }
    digest = hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    return f"arc_agi3:cross_game:L{level}:{key}:{digest}"


def _dedupe_evidence(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for item in items:
        key = json.dumps(item, ensure_ascii=False, sort_keys=True)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _load_game_schemas() -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, str]]:
    manifest = _read_json(MANIFEST_PATH)
    review_status = {
        game["game_id"]: game["review_status"] for game in manifest["games"]
    }
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for game in manifest["games"]:
        schemas = _read_json(
            GOLD_ROOT / "games" / game["game_id"] / "schemas.json"
        )["schemas"]
        for schema in schemas:
            key = schema["schema_id"].split(":")[-2]
            index[(game["game_id"], key)] = schema
    return index, review_status


def _make_schema(
    *,
    level: int,
    key: str,
    title: str,
    kind: str,
    trigger: str,
    expectation: str,
    members: list[dict[str, Any]],
    review_status: dict[str, str],
    constraints: list[str] | None = None,
    exceptions: list[str] | None = None,
    parents: list[str] | None = None,
) -> dict[str, Any]:
    game_scope = sorted(
        {
            game_id
            for member in members
            for game_id in member.get("game_scope", [member["game_id"]])
            if game_id != "__cross_game__"
        }
    )
    member_ids = [member["schema_id"] for member in members]
    source_evidence = _dedupe_evidence(
        [evidence for member in members for evidence in member["source_evidence"]]
    )
    member_status = {
        game_id: review_status[game_id] for game_id in game_scope
    }
    provisional = any(status == "pending" for status in member_status.values())
    payload: dict[str, Any] = {
        "schema_id": "",
        "format_version": 1,
        "title": title,
        "benchmark": "arc_agi3",
        "benchmark_version": "inventory-v1",
        "game_id": "__cross_game__",
        "game_scope": game_scope,
        "level_scope": [],
        "abstraction_level": level,
        "kind": kind,
        "trigger": trigger,
        "action_sequence": [],
        "expectation": expectation,
        "constraints": list(constraints or []),
        "exceptions": list(exceptions or []),
        "relations": {
            "parents": list(parents or []),
            "requires": [],
            "members": member_ids,
        },
        "member_schema_ids": member_ids,
        "member_review_status": member_status,
        "source_evidence": source_evidence,
        "runtime_evidence": [],
        "derivation_status": "provisional" if provisional else "reviewable",
        "verification": {
            "static": "passed",
            "runtime": "derived_from_members",
            "review": "provisional" if provisional else "pending",
        },
    }
    payload["schema_id"] = _schema_id(level, key, payload)
    return payload


def build() -> list[dict[str, Any]]:
    game_index, review_status = _load_game_schemas()
    level1: list[dict[str, Any]] = []
    by_key: dict[str, dict[str, Any]] = {}
    for family in LEVEL1:
        members = [game_index[member] for member in family.members]
        schema = _make_schema(
            level=1,
            key=family.key,
            title=family.title,
            kind=family.kind,
            trigger=family.trigger,
            expectation=family.expectation,
            members=members,
            review_status=review_status,
            constraints=list(family.constraints),
            exceptions=list(family.exceptions),
        )
        level1.append(schema)
        by_key[family.key] = schema

    level0: list[dict[str, Any]] = []
    for definition in LEVEL0:
        members = [by_key[key] for key in definition["members"]]
        schema = _make_schema(
            level=0,
            key=definition["key"],
            title=definition["title"],
            kind=definition["kind"],
            trigger=definition["trigger"],
            expectation=definition["expectation"],
            members=members,
            review_status=review_status,
        )
        level0.append(schema)
        for member in members:
            member["relations"]["parents"].append(schema["schema_id"])
    return level0 + level1


def validate(schemas: list[dict[str, Any]]) -> dict[str, Any]:
    game_index, _ = _load_game_schemas()
    concrete_ids = {schema["schema_id"] for schema in game_index.values()}
    cross_ids = {schema["schema_id"] for schema in schemas}
    errors: list[str] = []
    for schema in schemas:
        members = set(schema["member_schema_ids"])
        expected_pool = concrete_ids if schema["abstraction_level"] == 1 else cross_ids
        if not members or not members <= expected_pool:
            errors.append(f"{schema['schema_id']} has invalid members")
        if schema["abstraction_level"] == 1 and len(schema["game_scope"]) < 3:
            errors.append(f"{schema['schema_id']} has fewer than three games")
        if schema["abstraction_level"] == 0 and len(members) < 2:
            errors.append(f"{schema['schema_id']} has fewer than two families")
        if not schema["source_evidence"]:
            errors.append(f"{schema['schema_id']} has no source evidence")
        for evidence in schema["source_evidence"]:
            path = PROJECT_ROOT / evidence["path"]
            if not path.exists():
                errors.append(f"{schema['schema_id']} missing {evidence['path']}")
                continue
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != evidence["sha256"]:
                errors.append(f"{schema['schema_id']} stale {evidence['path']}")
        if any(
            status == "pending"
            for status in schema["member_review_status"].values()
        ) and schema["derivation_status"] != "provisional":
            errors.append(f"{schema['schema_id']} hides pending members")
    level_counts = {
        str(level): sum(schema["abstraction_level"] == level for schema in schemas)
        for level in (0, 1)
    }
    covered_games = sorted(
        {game_id for schema in schemas for game_id in schema["game_scope"]}
    )
    return {
        "status": "passed" if not errors else "failed",
        "schema_count": len(schemas),
        "level_counts": level_counts,
        "covered_game_count": len(covered_games),
        "covered_games": covered_games,
        "all_pending_dependencies_exposed": not any(
            "hides pending" in error for error in errors
        ),
        "errors": errors,
    }


def write_review(schemas: list[dict[str, Any]], validation: dict[str, Any]) -> None:
    rows = "\n".join(
        f"| L{schema['abstraction_level']} | {schema['title']} | "
        f"{len(schema['game_scope'])} | {len(schema['member_schema_ids'])} | "
        f"{schema['derivation_status']} |"
        for schema in schemas
    )
    (CROSS_ROOT / "review.md").write_text(
        f"""# ARC-AGI-3 跨游戏 Gold Schema — 审核稿

> 状态：provisional。当前抽象覆盖全部 {validation['covered_game_count']} 个游戏，但成员中含 22 个尚未人工审核的游戏，因此不能发布为正式 Gold。

## 分层

- Level 2：单游戏机制，保存在 `games/<game-id>/schemas.json`；
- Level 1：至少三个游戏共同支持的机制族；
- Level 0：由多个 Level 1 机制族支持的任务系统结构。

跨游戏节点不靠关键词相似度自动合并。每条节点显式保存 `member_schema_ids`、`game_scope`、成员审核状态和全部源码证据；具体游戏的例外仍留在 Level 2。

## 节点

| 层级 | 抽象 | 游戏数 | 直接成员数 | 状态 |
|---|---|---:|---:|---|
{rows}

## 建议审核顺序

1. 先检查 `atomic_rejection`、`structured_interpretation` 是否抽象过宽；
2. 再检查 `finite_resource_failure` 是否需要拆成 GAME_OVER 与关卡内重置两类；
3. 单游戏成员审核完成后重新生成，只有成员状态不再 pending 的节点才可进入正式审核。
""",
        encoding="utf-8",
    )


def update_manifest(validation: dict[str, Any]) -> None:
    manifest = _read_json(MANIFEST_PATH)
    manifest["cross_game"] = {
        "status": "provisional",
        "schema_count": validation["schema_count"],
        "level_counts": validation["level_counts"],
        "covered_game_count": validation["covered_game_count"],
        "schemas": "cross_game/schemas.json",
        "validation": "cross_game/validation.json",
        "review": "cross_game/review.md",
    }
    _write_json(MANIFEST_PATH, manifest)

    readme = (GOLD_ROOT / "README.md").read_text(encoding="utf-8")
    marker = "\n## 跨游戏抽象\n"
    readme = readme.split(marker, 1)[0].rstrip() + "\n"
    readme += (
        marker
        + f"\n已生成 {validation['schema_count']} 条 provisional 跨游戏 Schema："
        + f"Level 0 为 {validation['level_counts']['0']} 条，"
        + f"Level 1 为 {validation['level_counts']['1']} 条，覆盖 "
        + f"{validation['covered_game_count']} 个游戏。详见 "
        + "[跨游戏审核稿](cross_game/review.md)。\n"
    )
    (GOLD_ROOT / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    schemas = build()
    validation = validate(schemas)
    if validation["status"] != "passed":
        raise RuntimeError(json.dumps(validation, ensure_ascii=False, indent=2))
    _write_json(
        CROSS_ROOT / "schemas.json",
        {"format_version": 1, "status": "provisional", "schemas": schemas},
    )
    _write_json(CROSS_ROOT / "validation.json", validation)
    _write_json(
        CROSS_ROOT / "schema_spec.json",
        {
            "format_version": 1,
            "description": "Cross-game extension of Gold Schema v1",
            "required_extensions": [
                "game_scope",
                "member_schema_ids",
                "member_review_status",
                "derivation_status",
            ],
            "allowed_abstraction_levels": [0, 1],
            "minimum_level_1_games": 3,
            "release_rule": "No provisional node is a formally accepted Gold Schema.",
        },
    )
    write_review(schemas, validation)
    update_manifest(validation)
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
