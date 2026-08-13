from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Set, Tuple

from .gold_loader import load_accepted_arc_gold


ACTION_RE = re.compile(r"ACTION\d+")
DIRECTION_ACTIONS = {"ACTION1", "ACTION2", "ACTION3", "ACTION4"}
ABSTRACT_DIRECTION_ACTIONS = {
    "direction_of_chain",
    "opposite_direction",
    "perpendicular_direction",
    "movement_action",
    "move_into_occupied_cell",
}


class MatchRelation(str, Enum):
    EQUIVALENT = "equivalent"
    LEARNED_NARROWER = "learned_narrower"
    LEARNED_BROADER = "learned_broader"
    PARTIAL = "partial"
    CONTRADICTION = "contradiction"
    UNRELATED = "unrelated"


@dataclass(frozen=True)
class CanonicalSchema:
    schema_id: str
    source: str
    game_id: str
    kind: str
    levels: Set[int]
    actions: Set[str]
    roles: Set[str]
    concepts: Set[str]
    effect_class: str
    trigger: str
    expectation: str
    text: str
    evidence_ids: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PairJudgment:
    learned_id: str
    gold_id: str
    score: float
    relation: MatchRelation
    reasons: List[str]
    component_scores: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["relation"] = self.relation.value
        return value


class SchemaPairJudge(Protocol):
    def judge(self, learned: CanonicalSchema, gold: CanonicalSchema) -> PairJudgment:
        ...


CONCEPT_PATTERNS: Dict[str, Tuple[str, ...]] = {
    "canvas": ("canvas", "画布", "10×10方块", "10×10 方块"),
    "palette": ("palette", "调色板", "当前颜色", "选中颜色"),
    "tool": ("tool", "工具"),
    "chain": ("chain", "链条", "链头", "链节"),
    "maze_path": ("maze", "通路", "蓝色连接", "可走边"),
    "direction": ("direction", "方向", "上、下、左、右", "转向"),
    "blocked": ("blocked", "unchanged", "保持不变", "不移动", "被阻止", "无路"),
    "level_completion": ("level completes", "next level", "下一关", "进入下一关", "过关"),
    "win": ("win", "游戏 win"),
    "game_over": ("game_over", "game over", "失败"),
    "budget": ("budget", "预算", "action_count", "步数", "次数"),
    "goal": ("goal", "目标", "出口", "匹配"),
    "select": ("select", "选择", "切换当前", "活动链条"),
    "extend": ("extend", "伸长", "长度增加"),
    "retract": ("retract", "缩短", "长度减少"),
    "translate": ("translate", "平移"),
    "push": ("push", "推动", "级联"),
    "undo": ("undo", "撤销", "恢复"),
    "collision": ("collision", "碰撞", "移入", "受击"),
    "enemy": ("enemy", "敌对", "橙色", "青色", "品红"),
}


def _concepts(text: str) -> Set[str]:
    normalized = text.lower().replace("_", " ")
    return {
        concept
        for concept, patterns in CONCEPT_PATTERNS.items()
        if any(pattern.lower() in normalized for pattern in patterns)
    }


def _gold_actions(value: Mapping[str, Any]) -> Set[str]:
    actions: Set[str] = set()
    for step in value.get("action_sequence", []):
        action = str(step.get("action", ""))
        for item in action.split("|"):
            if item in ABSTRACT_DIRECTION_ACTIONS:
                actions.update(DIRECTION_ACTIONS)
            elif item:
                actions.add(item)
    text = " ".join(
        str(value.get(key, "")) for key in ("title", "trigger", "expectation")
    )
    actions.update(ACTION_RE.findall(text))
    return actions


def canonicalize_gold(value: Mapping[str, Any]) -> CanonicalSchema:
    action_steps = list(value.get("action_sequence", []))
    roles = {
        str(step.get("arguments", {}).get("target_role"))
        for step in action_steps
        if step.get("arguments", {}).get("target_role")
    }
    text = " ".join(
        [
            str(value.get("title", "")),
            str(value.get("trigger", "")),
            str(value.get("expectation", "")),
            *[str(item) for item in value.get("constraints", [])],
            *[str(item) for item in value.get("exceptions", [])],
        ]
    )
    return CanonicalSchema(
        schema_id=str(value["schema_id"]),
        source="gold",
        game_id=str(value["game_id"]),
        kind=str(value.get("kind", "")),
        levels={int(item) for item in value.get("level_scope", [])},
        actions=_gold_actions(value),
        roles=roles,
        concepts=_concepts(text),
        effect_class="",
        trigger=str(value.get("trigger", "")),
        expectation=str(value.get("expectation", "")),
        text=text,
        metadata={"title": str(value.get("title", ""))},
    )


def canonicalize_learned(value: Mapping[str, Any]) -> CanonicalSchema:
    metadata = dict(value.get("metadata", {}))
    action = str(metadata.get("action_name", ""))
    text = " ".join(
        str(value.get(key, "")) for key in ("description", "trigger", "expectation")
    )
    concepts = _concepts(text)
    region = str(metadata.get("region", ""))
    if region == "canvas":
        concepts.add("canvas")
    elif region == "tool_area":
        concepts.add("tool")
    elif region == "chain_playfield":
        concepts.add("chain")
    elif region == "maze_board":
        concepts.add("maze_path")
    effect = str(metadata.get("effect_class", ""))
    if effect in {"level_completion", "win", "game_over", "no_effect"}:
        concepts.add(effect)
    role = str(metadata.get("target_role", ""))
    if role == "palette_button":
        concepts.add("palette")
    return CanonicalSchema(
        schema_id=str(value["index"]),
        source="learned",
        game_id=str(metadata.get("game_id", "")),
        kind=str(metadata.get("schema_kind", "")),
        levels={int(item) for item in metadata.get("level_scope", [])},
        actions={action} if action else set(ACTION_RE.findall(text)),
        roles={role} if role else set(),
        concepts=concepts,
        effect_class=effect,
        trigger=str(value.get("trigger", "")),
        expectation=str(value.get("expectation", "")),
        text=text,
        evidence_ids={
            str(item)
            for category in ("source", "positive", "negative")
            for item in value.get("memory_index", {}).get(category, [])
        },
        metadata=metadata,
    )


def _jaccard(left: Set[Any], right: Set[Any]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / max(1, len(left | right))


class StructuredARCJudge:
    """Conservative, zero-LLM proxy judge for cross-language ARC schemas.

    It gives credit for grounded structural overlap but caps generic
    ``grid changes`` statements below equivalence. The output is a diagnostic
    baseline, not a substitute for a calibrated semantic judge or human audit.
    """

    def judge(self, learned: CanonicalSchema, gold: CanonicalSchema) -> PairJudgment:
        if learned.game_id != gold.game_id:
            return self._unrelated(learned, gold, "game_scope_mismatch")
        level_score = _jaccard(learned.levels, gold.levels)
        if learned.levels and gold.levels and not learned.levels & gold.levels:
            return self._unrelated(learned, gold, "level_scope_disjoint")

        learned_condition = str(learned.metadata.get("condition_key", ""))
        gold_condition = str(gold.metadata.get("condition_key", ""))
        expected_effect = str(gold.metadata.get("expected_effect", ""))
        if (
            learned_condition
            and learned_condition == gold_condition
            and expected_effect
            and learned.effect_class
            and learned.effect_class != expected_effect
        ):
            return PairJudgment(
                learned_id=learned.schema_id,
                gold_id=gold.schema_id,
                score=1.0,
                relation=MatchRelation.CONTRADICTION,
                reasons=["same_condition_opposite_effect"],
                component_scores={"condition": 1.0, "effect": 0.0},
            )

        action_overlap = learned.actions & gold.actions
        terminal_goal = gold.kind == "goal" and learned.effect_class in {
            "level_completion",
            "win",
        }
        terminal_hazard = gold.kind == "hazard" and learned.effect_class == "game_over"
        if not action_overlap and not terminal_goal and not terminal_hazard:
            return self._unrelated(learned, gold, "no_action_or_terminal_bridge")

        action_score = 1.0 if action_overlap else 0.8
        if gold.roles:
            role_score = 1.0 if learned.roles & gold.roles else 0.0
        else:
            role_score = 1.0 if not learned.roles else 0.7
        if gold.kind == "action_effect":
            kind_score = 1.0 if learned.kind in {"action_effect", "constraint"} else 0.3
        elif gold.kind == "goal":
            kind_score = 1.0 if terminal_goal else 0.2
        elif gold.kind == "hazard":
            kind_score = 1.0 if terminal_hazard else 0.15
        else:
            kind_score = 0.2
        concept_score = _jaccard(learned.concepts, gold.concepts)
        score = (
            0.38 * action_score
            + 0.17 * role_score
            + 0.20 * kind_score
            + 0.20 * concept_score
            + 0.05 * level_score
        )
        reasons = [
            f"action_overlap={sorted(action_overlap)}" if action_overlap else "terminal_bridge",
            f"concept_overlap={sorted(learned.concepts & gold.concepts)}",
            f"level_jaccard={level_score:.3f}",
        ]
        generic = (
            "changes at" in learned.expectation.lower()
            or "task grid remains unchanged" in learned.expectation.lower()
            or "current level completes" in learned.expectation.lower()
        )
        if generic:
            score = min(score, 0.64)
            reasons.append("generic_transition_summary_caps_equivalence")
        if gold.roles and not learned.roles & gold.roles:
            score = min(score, 0.34)
            reasons.append("target_role_mismatch")
        if (
            learned.effect_class == "no_effect"
            and "blocked" not in gold.concepts
            and gold.kind == "action_effect"
        ):
            score = min(score, 0.34)
            reasons.append("gold_does_not_support_this_no_effect_condition")
        if gold.kind == "action_effect" and not learned.concepts & gold.concepts:
            score = min(score, 0.34)
            reasons.append("action_name_without_effect_concept_overlap")
        if gold.kind == "hazard" and not terminal_hazard:
            score = min(score, 0.34)
            reasons.append("action_overlap_does_not_capture_hazard_condition")
        if gold.kind == "observation_semantics":
            score = min(score, 0.29)
            reasons.append("action_rule_does_not_capture_observation_semantics")
        relation = (
            MatchRelation.EQUIVALENT
            if score >= 0.82
            else MatchRelation.PARTIAL
            if score >= 0.35
            else MatchRelation.UNRELATED
        )
        return PairJudgment(
            learned_id=learned.schema_id,
            gold_id=gold.schema_id,
            score=round(score, 6),
            relation=relation,
            reasons=reasons,
            component_scores={
                "action": action_score,
                "role": role_score,
                "kind": kind_score,
                "concept": round(concept_score, 6),
                "level_scope": round(level_score, 6),
            },
        )

    @staticmethod
    def _unrelated(
        learned: CanonicalSchema, gold: CanonicalSchema, reason: str
    ) -> PairJudgment:
        return PairJudgment(
            learned_id=learned.schema_id,
            gold_id=gold.schema_id,
            score=0.0,
            relation=MatchRelation.UNRELATED,
            reasons=[reason],
            component_scores={},
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LearnedGoldEvaluator:
    def __init__(self, judge: Optional[SchemaPairJudge] = None, *, partial_threshold: float = 0.35) -> None:
        self.judge = judge or StructuredARCJudge()
        self.partial_threshold = partial_threshold

    def evaluate(
        self,
        learned: Sequence[CanonicalSchema],
        gold: Sequence[CanonicalSchema],
        *,
        memory_ids: Optional[Set[str]] = None,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        judgments = [
            self.judge.judge(left, right)
            for left in learned
            for right in gold
            if left.game_id == right.game_id
        ]
        supporting = [
            item
            for item in judgments
            if item.score >= self.partial_threshold
            and item.relation not in {MatchRelation.CONTRADICTION, MatchRelation.UNRELATED}
        ]
        contradictions = [
            item for item in judgments if item.relation == MatchRelation.CONTRADICTION
        ]
        by_learned: Dict[str, List[PairJudgment]] = defaultdict(list)
        by_gold: Dict[str, List[PairJudgment]] = defaultdict(list)
        for item in supporting:
            by_learned[item.learned_id].append(item)
            by_gold[item.gold_id].append(item)
        alignments: List[Dict[str, Any]] = []
        for item in supporting:
            relation = item.relation
            if relation == MatchRelation.PARTIAL:
                if len(by_learned[item.learned_id]) > 1:
                    relation = MatchRelation.LEARNED_BROADER
                elif len(by_gold[item.gold_id]) > 1:
                    relation = MatchRelation.LEARNED_NARROWER
            value = item.to_dict()
            value["relation"] = relation.value
            alignments.append(value)
        alignments.extend(item.to_dict() for item in contradictions)
        learned_best = {
            item.schema_id: max(
                (value.score for value in supporting if value.learned_id == item.schema_id),
                default=0.0,
            )
            for item in learned
        }
        gold_best = {
            item.schema_id: max(
                (value.score for value in supporting if value.gold_id == item.schema_id),
                default=0.0,
            )
            for item in gold
        }
        graded_precision = sum(learned_best.values()) / max(1, len(learned))
        graded_recall = sum(gold_best.values()) / max(1, len(gold))
        graded_f1 = (
            2 * graded_precision * graded_recall / (graded_precision + graded_recall)
            if graded_precision + graded_recall
            else 0.0
        )
        exact_learned = {
            item.learned_id for item in judgments if item.relation == MatchRelation.EQUIVALENT
        }
        exact_gold = {item.gold_id for item in judgments if item.relation == MatchRelation.EQUIVALENT}
        evidence_total = sum(len(item.evidence_ids) for item in learned)
        evidence_resolved = (
            sum(len(item.evidence_ids & memory_ids) for item in learned)
            if memory_ids is not None
            else evidence_total
        )
        metrics = {
            "format_version": 1,
            "judge": "structured_arc_proxy_v1",
            "learned_count": len(learned),
            "gold_count": len(gold),
            "strict_learned_precision": round(len(exact_learned) / max(1, len(learned)), 6),
            "strict_gold_recall": round(len(exact_gold) / max(1, len(gold)), 6),
            "graded_learned_precision": round(graded_precision, 6),
            "graded_gold_recall": round(graded_recall, 6),
            "graded_semantic_f1": round(graded_f1, 6),
            "partially_covered_gold_count": sum(score >= self.partial_threshold for score in gold_best.values()),
            "supported_learned_count": sum(score >= self.partial_threshold for score in learned_best.values()),
            "unmatched_gold_count": sum(score < self.partial_threshold for score in gold_best.values()),
            "unmatched_learned_count": sum(score < self.partial_threshold for score in learned_best.values()),
            "split_gold_count": sum(len(values) > 1 for values in by_gold.values()),
            "overmerged_learned_count": sum(len(values) > 1 for values in by_learned.values()),
            "contradiction_count": len(contradictions),
            "contradiction_rate": round(len(contradictions) / max(1, len(judgments)), 6),
            "evidence_traceability": round(evidence_resolved / max(1, evidence_total), 6),
            "evidence_reference_count": evidence_total,
            "resolved_evidence_reference_count": evidence_resolved,
        }
        metrics["alignment_relation_counts"] = dict(
            sorted(Counter(item["relation"] for item in alignments).items())
        )
        action_gold = [item for item in gold if item.actions]
        metrics["action_signature_gold_count"] = len(action_gold)
        metrics["action_signature_recall"] = round(
            sum(
                any(
                    left.game_id == item.game_id and left.actions & item.actions
                    for left in learned
                )
                for item in action_gold
            )
            / max(1, len(action_gold)),
            6,
        )
        metrics["by_gold_kind"] = {}
        for kind in sorted({item.kind for item in gold}):
            ids = {item.schema_id for item in gold if item.kind == kind}
            metrics["by_gold_kind"][kind] = {
                "gold_count": len(ids),
                "graded_recall": round(
                    sum(gold_best[item] for item in ids) / max(1, len(ids)), 6
                ),
                "partial_coverage": round(
                    sum(gold_best[item] >= self.partial_threshold for item in ids)
                    / max(1, len(ids)),
                    6,
                ),
            }
        per_game = {}
        for game_id in sorted({item.game_id for item in [*learned, *gold]}):
            learned_ids = {item.schema_id for item in learned if item.game_id == game_id}
            gold_ids = {item.schema_id for item in gold if item.game_id == game_id}
            precision = sum(learned_best[item] for item in learned_ids) / max(1, len(learned_ids))
            recall = sum(gold_best[item] for item in gold_ids) / max(1, len(gold_ids))
            per_game[game_id] = {
                "learned_count": len(learned_ids),
                "gold_count": len(gold_ids),
                "graded_learned_precision": round(precision, 6),
                "graded_gold_recall": round(recall, 6),
                "partial_gold_coverage": round(
                    sum(gold_best[item] >= self.partial_threshold for item in gold_ids)
                    / max(1, len(gold_ids)),
                    6,
                ),
            }
        metrics["per_game"] = per_game
        return metrics, sorted(
            alignments,
            key=lambda item: (item["gold_id"], -item["score"], item["learned_id"]),
        )


def run_evaluation(
    learned_schema_path: str | Path,
    memory_path: str | Path,
    gold_root: str | Path,
    output_root: str | Path,
) -> Dict[str, Any]:
    learned_path = Path(learned_schema_path)
    memory_file = Path(memory_path)
    learned_payload = json.loads(learned_path.read_text(encoding="utf-8"))
    if int(learned_payload.get("format_version", 0)) != 1:
        raise ValueError("Unsupported learned Schema format")
    learned = [
        canonicalize_learned(value)
        for value in learned_payload.get("nodes", [])
        if value.get("status", "active") == "active"
    ]
    games = sorted({item.game_id for item in learned})
    gold_bundle = load_accepted_arc_gold(gold_root, game_ids=games)
    gold = [canonicalize_gold(value) for value in gold_bundle.schemas]
    memory_payload = json.loads(memory_file.read_text(encoding="utf-8"))
    memory_ids = {str(value["id"]) for value in memory_payload.get("records", [])}
    metrics, alignments = LearnedGoldEvaluator().evaluate(
        learned, gold, memory_ids=memory_ids
    )
    aligned_learned = {
        item["learned_id"]
        for item in alignments
        if item["relation"] != MatchRelation.CONTRADICTION.value
    }
    aligned_gold = {
        item["gold_id"]
        for item in alignments
        if item["relation"] != MatchRelation.CONTRADICTION.value
    }
    unmatched_learned = [asdict(item) for item in learned if item.schema_id not in aligned_learned]
    unmatched_gold = [asdict(item) for item in gold if item.schema_id not in aligned_gold]
    for values in (unmatched_learned, unmatched_gold):
        for value in values:
            for key in ("levels", "actions", "roles", "concepts", "evidence_ids"):
                value[key] = sorted(value[key])
    config = {
        "format_version": 1,
        "learned_schema_path": str(learned_path.resolve()),
        "learned_schema_sha256": _sha256(learned_path),
        "memory_path": str(memory_file.resolve()),
        "memory_sha256": _sha256(memory_file),
        "gold_root": str(Path(gold_root).resolve()),
        "gold_games": gold_bundle.games,
        "judge": "structured_arc_proxy_v1",
        "network_calls": 0,
        "writes_to_learned_state": 0,
    }
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    for name, value in (
        ("config.json", config),
        ("metrics.json", metrics),
        ("alignments.json", alignments),
        ("unmatched_learned.json", unmatched_learned),
        ("unmatched_gold.json", unmatched_gold),
    ):
        (output / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2, default=list) + "\n",
            encoding="utf-8",
        )
    with (output / "judge_cache.jsonl").open("w", encoding="utf-8") as handle:
        for value in alignments:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
    _write_markdown_report(
        output / "report.md", metrics, unmatched_gold, unmatched_learned
    )
    return metrics


def _write_markdown_report(
    path: Path,
    metrics: Mapping[str, Any],
    unmatched_gold: Sequence[Mapping[str, Any]],
    unmatched_learned: Sequence[Mapping[str, Any]],
) -> None:
    rows = "\n".join(
        f"| {game} | {value['learned_count']} | {value['gold_count']} | "
        f"{value['graded_learned_precision']:.3f} | {value['graded_gold_recall']:.3f} | "
        f"{value['partial_gold_coverage']:.3f} |"
        for game, value in metrics["per_game"].items()
    )
    conclusion = (
        "当前 learned Schema 能识别一部分动作/区域/终局相关性，但没有任何规则达到严格语义等价；"
        "分级分数只表示结构化代理相似度，不能作为正式论文主分数。"
        if metrics["strict_gold_recall"] == 0
        else "当前 learned Schema 已出现严格等价规则，但仍需人工抽查。"
    )
    kinds = "\n".join(
        f"| {kind} | {value['gold_count']} | {value['graded_recall']:.3f} | "
        f"{value['partial_coverage']:.3f} |"
        for kind, value in metrics["by_gold_kind"].items()
    )
    missing = "\n".join(
        f"- `{value['game_id']}` — {value['metadata'].get('title', value['schema_id'])}"
        for value in unmatched_gold
    ) or "- none"
    extra = "\n".join(
        f"- `{value['game_id']}` — `{value['schema_id']}` "
        f"({', '.join(value['actions']) or 'no action'}; {value['effect_class']})"
        for value in unmatched_learned
    ) or "- none"
    text = f"""# Learned Schema vs Gold Schema

## 结论

{conclusion}

- strict learned precision: {metrics['strict_learned_precision']:.3f}
- strict Gold recall: {metrics['strict_gold_recall']:.3f}
- graded learned precision: {metrics['graded_learned_precision']:.3f}
- graded Gold recall: {metrics['graded_gold_recall']:.3f}
- graded semantic F1: {metrics['graded_semantic_f1']:.3f}
- partially covered Gold: {metrics['partially_covered_gold_count']}/{metrics['gold_count']}
- action-signature recall: {metrics['action_signature_recall']:.3f}
- evidence traceability: {metrics['evidence_traceability']:.3f}

## 分游戏

| game | learned | Gold | graded precision | graded recall | partial coverage |
|---|---:|---:|---:|---:|---:|
{rows}

## 按 Gold 类型

| kind | Gold | graded recall | partial coverage |
|---|---:|---:|---:|
{kinds}

## 未覆盖 Gold

{missing}

## 未获 Gold 支持的 learned 节点

{extra}

## 解释边界

`structured_arc_proxy_v1` 只使用 game、level、action/role、Schema kind、视觉区域和双语概念标签。
凡是只写“grid changes/no change/level completes”的 learned Schema 都被限制在 partial 以下，不能因
action 名相同判为 equivalent。该报告适合定位生成算法缺什么；正式主指标还需要冻结人工 alignment
fixture，并用人工或独立语义 judge 校准。
"""
    path.write_text(text, encoding="utf-8")
