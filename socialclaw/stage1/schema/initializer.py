from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..agent.base import Agent
from ..dataset.base import Problem
from .graph import Concept, Relation


class SchemaInitializer:
    def __init__(self, agent: Agent):
        self.agent = agent

    def generate_schema(self, problem: Problem) -> Tuple[List[Concept], List[Relation]]:
        """Let Agent read the problem and output structured concepts + relations."""
        prompt = _build_generate_prompt(problem)
        attempt = self.agent.answer(prompt=prompt, meta={"task": "schema_init", "problem_id": problem.id})
        return _parse_schema_from_json(attempt.answer_text)

    def describe_missing(self, problem: Problem, concepts: List[Concept], missing: List[str] = None) -> Dict[str, str]:
        """Generate a human-facing question for missing concepts and relations."""
        concept_names = [c.name for c in concepts]
        missing_part = ""
        if missing:
            missing_part = f"系统判断缺少以下关键概念：{', '.join(missing)}。"
        question = (
            f"为了回答这道{problem.problem_type}题目，系统检索到了以下概念："
            f"{', '.join(concept_names) if concept_names else '（无）'}。"
            f"{missing_part}"
            "但判断认为这些概念及它们之间的关系不足以支撑答题。"
            "请你补充描述：\n"
            "1. 解题所需要的关键概念（名称+简要描述）；\n"
            "2. 这些概念之间的关系（如：概念A -> 关系类型 -> 概念B）。"
        )
        hint = (
            "格式示例：\n"
            "概念：\n"
            "1. 双黄线：道路中央用于分隔对向车流的黄色实线。\n"
            "2. 自车：搭载摄像头的主体车辆。\n"
            "关系：\n"
            "双黄线 -> 位于左侧 -> 自车\n"
            "双黄线 -> 属于 -> 交通标识"
        )
        return {"question": question, "context": problem.prompt[:500], "hint": hint}

    def parse_human_answer(self, answer: str, problem: Problem) -> Tuple[List[Concept], List[Relation]]:
        """Parse free-text human answer into structured Concepts + Relations."""
        concepts: List[Concept] = []
        relations: List[Relation] = []

        lines = answer.strip().split("\n")
        current_name = ""
        current_desc = ""
        in_relation_section = False

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Detect relation section headers
            lower = line.lower()
            if any(k in lower for k in ("关系", "relations", "边", "edges")):
                if current_name and current_desc:
                    concepts.append(_new_concept(current_name, current_desc))
                    current_name = ""
                    current_desc = ""
                in_relation_section = True
                continue

            if in_relation_section:
                rel = _try_parse_relation_line(line)
                if rel:
                    relations.append(rel)
                continue

            # Parse concept lines
            stripped = line.lstrip("-–•1234567890. ").strip()
            if ":" in stripped:
                name, desc = stripped.split(":", 1)
                name = name.strip()
                desc = desc.strip()
                if current_name and current_desc:
                    concepts.append(_new_concept(current_name, current_desc))
                current_name = name
                current_desc = desc
            else:
                if current_name:
                    current_desc += " " + stripped
                else:
                    current_name = stripped
                    current_desc = ""

        if current_name and current_desc:
            concepts.append(_new_concept(current_name, current_desc))

        # Fallback: if no concepts parsed, treat whole text as one concept
        if not concepts and answer.strip():
            concepts.append(_new_concept("human_concept", answer.strip()))

        return concepts, relations

    def parse_correction(self, correction: str, problem: Problem) -> Dict[str, Any]:
        """Parse human correction into schema update operations.

        Returns a dict with keys:
        - add_concepts: List[Concept]
        - add_relations: List[Relation]
        - update_concepts: List[dict]
        """
        result: Dict[str, Any] = {"add_concepts": [], "add_relations": [], "update_concepts": []}

        lines = correction.strip().split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            stripped = line.lstrip("-1234567890. ").strip()
            lower = stripped.lower()

            # Relation lines: "A -> type -> B"
            if "->" in stripped or "→" in stripped:
                rel = _try_parse_relation_line(stripped)
                if rel:
                    result["add_relations"].append(rel)
                continue

            # Update concept
            if any(k in lower for k in ("修改", "update", "修正")):
                if ":" in stripped:
                    parts = stripped.split(":", 1)
                    name_part = parts[0]
                    desc = parts[1].strip()
                    name = name_part
                    for kw in ("修改概念", "update concept", "修正概念"):
                        if kw in name.lower():
                            name = name.split(kw, 1)[-1].strip()
                            break
                    result["update_concepts"].append({"id": name, "description": desc})
                continue

            # Add concept
            if any(k in lower for k in ("添加", "add", "新增")):
                if ":" in stripped:
                    parts = stripped.split(":", 1)
                    name = parts[0]
                    for kw in ("添加概念", "add concept", "新增概念"):
                        if kw in name.lower():
                            name = name.split(kw, 1)[-1].strip()
                            break
                    desc = parts[1].strip()
                    result["add_concepts"].append(_new_concept(name, desc))
                continue

            # Default: treat as add_concept if contains colon
            if ":" in stripped:
                name, desc = stripped.split(":", 1)
                result["add_concepts"].append(_new_concept(name.strip(), desc.strip()))

        return result


def _build_generate_prompt(problem: Problem) -> str:
    return (
        "你是一个知识图谱构建助手。请阅读下面的题目，输出回答该题所需要的核心概念（concept）列表，"
        "以及概念之间的关系（relation）。\n\n"
        f"题目类型：{problem.problem_type}\n"
        f"题目内容：\n{problem.prompt}\n\n"
        "请以 JSON 格式输出，必须包含 concepts 和 relations 两个字段：\n"
        "{\n"
        '  "concepts": [\n'
        '    {"name": "概念名称", "description": "简要描述", "category": "类别"}\n'
        "  ],\n"
        '  "relations": [\n'
        '    {"source": "源概念名称", "target": "目标概念名称", "relation_type": "关系类型"}\n'
        "  ]\n"
        "}\n"
        "注意：\n"
        "1. concepts 中的 name 必须简洁，后续会被 LLM 在推理过程中直接引用。\n"
        "2. relations 中的 source/target 必须是 concepts 中已有的 name。\n"
        "3. 关系类型可选：prerequisite / causes / part_of / located_at / analogous / related。"
    )


def _parse_schema_from_json(text: str) -> Tuple[List[Concept], List[Relation]]:
    """Extract JSON object with concepts + relations from LLM output."""
    text = text.strip()
    # Try to extract JSON block
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return [], []
    text = text[start : end + 1]

    try:
        data = json.loads(text)
    except Exception:
        return [], []

    if not isinstance(data, dict):
        return [], []

    concepts: List[Concept] = []
    concept_names: set = set()
    for item in data.get("concepts", []):
        if not isinstance(item, dict):
            continue
        name = item.get("name", "").strip()
        desc = item.get("description", "").strip()
        if not name or not desc:
            continue
        concept_names.add(name)
        concepts.append(
            Concept(
                id=f"concept_{uuid.uuid4().hex[:8]}",
                name=name,
                description=desc,
                category=item.get("category", "general"),
                confidence=0.5,
                source="agent_init",
                created_at=datetime.utcnow().isoformat() + "Z",
            )
        )

    relations: List[Relation] = []
    for item in data.get("relations", []):
        if not isinstance(item, dict):
            continue
        src = item.get("source", "").strip()
        tgt = item.get("target", "").strip()
        rel_type = item.get("relation_type", "related").strip()
        if not src or not tgt:
            continue
        # Only add relation if both endpoints exist in generated concepts
        if src in concept_names and tgt in concept_names:
            relations.append(
                Relation(
                    source=src,  # Will be resolved to id by pipeline
                    target=tgt,
                    relation_type=rel_type,
                    weight=0.5,
                    evidence=[],
                )
            )

    return concepts, relations


def _new_concept(name: str, description: str) -> Concept:
    return Concept(
        id=f"concept_{uuid.uuid4().hex[:8]}",
        name=name,
        description=description,
        category="general",
        confidence=0.8,
        source="human_feedback",
        created_at=datetime.utcnow().isoformat() + "Z",
    )


def _try_parse_relation_line(line: str) -> Optional[Relation]:
    """Parse a relation line like 'A -> type -> B' or 'A → B (type)'."""
    line = line.strip().lstrip("-–•1234567890. ").strip()
    if "->" not in line and "→" not in line:
        return None

    normalized = line.replace("→", "->")
    parts = [p.strip() for p in normalized.split("->")]
    parts = [p for p in parts if p]

    if len(parts) >= 2:
        src = parts[0]
        tgt = parts[-1]
        rel_type = "related"
        if len(parts) >= 3:
            rel_type = parts[1]
        return Relation(source=src, target=tgt, relation_type=rel_type, weight=0.5, evidence=[])
    return None
