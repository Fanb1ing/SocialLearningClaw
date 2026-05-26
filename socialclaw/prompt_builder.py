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

    Stage 1 uses "flat list" mode for simplicity.
    """
    concepts = subgraph.list_concepts()

    # Progressive disclosure
    concept_blocks: List[str] = []
    for i, c in enumerate(concepts, start=1):
        if attempt_index <= 0:
            blk = f"Concept {i}: {c.name}\n  Description: {c.description[:200]}"
        else:
            blk = f"Concept {i}: {c.name}\n  Description: {c.description}\n  Category: {c.category}\n  Confidence: {c.confidence:.2f}"
            # Add related concepts from relations
            related = []
            for r in subgraph.list_relations():
                if r.source == c.id:
                    tgt = subgraph.get_concept(r.target)
                    if tgt:
                        related.append(f"→ {tgt.name} ({r.relation_type})")
                elif r.target == c.id:
                    src = subgraph.get_concept(r.source)
                    if src:
                        related.append(f"← {src.name} ({r.relation_type})")
            if related:
                blk += "\n  Related concepts:\n    " + "\n    ".join(related)
        concept_blocks.append(blk)

    schema_text = (
        "[Available Concept Network]\n\n" + "\n\n".join(concept_blocks) + "\n"
        if concept_blocks
        else "[Available Concept Network]\n(No related concepts yet)\n"
    )

    # Include choices if present in meta (for MCQ)
    choices_text = ""
    choices = problem.meta.get("choices")
    if isinstance(choices, list) and choices:
        choices_text = "\nChoices:\n" + "\n".join([f"{chr(65 + i)}. {c}" for i, c in enumerate(choices)]) + "\n"

    if concept_blocks:
        concept_usage_rule = (
            "1. In [Reasoning Process], you MUST list the concept names you used.\n"
            "   Use EXACT concept names from the [Available Concept Network] above (e.g. 'Sales Enablement'), do not add explanations or invent new names.\n"
            "2. Each node in the reasoning path MUST be a concept name from the network, format: ConceptA -> relation_type -> ConceptB.\n"
            "   Do not use descriptive sentences as nodes, do not use Chinese explanations in place of concept names.\n"
        )
    else:
        concept_usage_rule = (
            "1. The concept network is currently empty. Please list the key term names you identify from the problem in [Reasoning Process].\n"
            "   Names should be concise (no more than 10 words), do not output explanatory sentences or parenthetical notes.\n"
            "2. The reasoning path may be omitted, or use only the term names you listed.\n"
        )

    system = (
        "You are a rigorous reasoning assistant. Please reason based on the concept network provided below and answer the question.\n\n"
        "Your response MUST contain the following two parts:\n\n"
        "[Reasoning Process]\n"
        "- Concepts used: {list of concept names}\n"
        "- Reasoning path: {Concept A -> relation -> Concept B -> ...}\n"
        "- Explanation: {brief explanation of your reasoning}\n\n"
        "[Final Answer]\n"
        "{your final answer}\n\n"
        "Notes:\n"
        f"{concept_usage_rule}"
        "3. If the question is multiple choice, [Final Answer] should output only the option letter (e.g. A / B / C).\n"
    )

    user = f"Problem type: {problem.problem_type}\n\nProblem content:\n{problem.prompt}{choices_text}\n"
    user += schema_text
    user += "\nPlease reason based on the above concept network and answer."

    return system + "\n---\n" + user
