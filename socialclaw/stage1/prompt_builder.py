from __future__ import annotations

from typing import List

from .dataset.base import Problem
from .schema.graph import SchemaGraph


def build_prompt(
    *,
    problem: Problem,
    subgraph: SchemaGraph,
    attempt_index: int = 0,
) -> str:
    """Build a prompt that injects schema subgraph knowledge and asks for structured reasoning.

    Stage 1 uses "flat list" mode (方式 A) for simplicity.
    """
    concepts = subgraph.list_concepts()

    # Progressive disclosure
    concept_blocks: List[str] = []
    for i, c in enumerate(concepts, start=1):
        if attempt_index <= 0:
            blk = f"概念 {i}：{c.name}\n  描述：{c.description[:200]}"
        else:
            blk = f"概念 {i}：{c.name}\n  描述：{c.description}\n  类别：{c.category}\n  置信度：{c.confidence:.2f}"
            # Add related concepts from relations
            related = []
            for r in subgraph.list_relations():
                if r.source == c.id:
                    tgt = subgraph.get_concept(r.target)
                    if tgt:
                        related.append(f"→ {tgt.name}（{r.relation_type}）")
                elif r.target == c.id:
                    src = subgraph.get_concept(r.source)
                    if src:
                        related.append(f"← {src.name}（{r.relation_type}）")
            if related:
                blk += "\n  相关概念：\n    " + "\n    ".join(related)
        concept_blocks.append(blk)

    schema_text = (
        "【可用概念网络】\n\n" + "\n\n".join(concept_blocks) + "\n"
        if concept_blocks
        else "【可用概念网络】\n（暂无相关概念）\n"
    )

    # Include choices if present in meta (for MCQ)
    choices_text = ""
    choices = problem.meta.get("choices")
    if isinstance(choices, list) and choices:
        choices_text = "\n选项：\n" + "\n".join([f"{chr(65 + i)}. {c}" for i, c in enumerate(choices)]) + "\n"

    system = (
        "你是一个严谨的推理助手。请基于下面提供的概念网络进行推理，回答题目。\n\n"
        "你的回答必须包含以下两部分：\n\n"
        "[推理过程]\n"
        "- 使用的概念：{概念名称列表}\n"
        "- 推理路径：{概念 A -> 关系 -> 概念 B -> ...}\n"
        "- 解释：{你对推理过程的简要说明}\n\n"
        "[最终答案]\n"
        "{你的最终答案}\n\n"
        "注意：\n"
        "1. 在[推理过程]中务必列出你使用的概念名称。\n"
        "2. 若题目是选择题，[最终答案]只输出选项字母（如 A / B / C）。\n"
    )

    user = f"题目类型：{problem.problem_type}\n\n题目内容：\n{problem.prompt}\n{choices_text}\n"
    user += schema_text
    user += "\n请基于上述概念网络推理并作答。"

    return system + "\n---\n" + user
