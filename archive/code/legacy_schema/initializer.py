"""Archived initializer for the single-layer Concept/Relation schema."""

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
            missing_part = f"The system determined the following key concepts are missing: {', '.join(missing)}."
        question = (
            f"To answer this {problem.problem_type} problem, the system retrieved the following concepts: "
            f"{', '.join(concept_names) if concept_names else '(none)'}\n"
            f"{missing_part}\n"
            "However, these concepts and their relationships are insufficient to answer the question.\n"
            "Please supplement:\n"
            "1. Key concepts needed to solve the problem (name + brief description);\n"
            "2. Relationships between these concepts (e.g. ConceptA -> relation_type -> ConceptB)."
        )
        hint = (
            "Format example:\n"
            "Concepts:\n"
            "1. Double yellow line: A solid yellow line in the center of the road used to separate opposing traffic flows.\n"
            "2. Ego vehicle: The subject vehicle equipped with a camera.\n"
            "Relations:\n"
            "Double yellow line -> located_at_left -> Ego vehicle\n"
            "Double yellow line -> is_a -> Traffic sign"
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
            if any(k in lower for k in ("relations", "edges", "relation")):
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
            if any(k in lower for k in ("update", "correct", "fix")):
                if ":" in stripped:
                    parts = stripped.split(":", 1)
                    name_part = parts[0]
                    desc = parts[1].strip()
                    name = name_part
                    for kw in ("update concept", "correct concept", "fix concept"):
                        if kw in name.lower():
                            name = name.split(kw, 1)[-1].strip()
                            break
                    result["update_concepts"].append({"id": name, "description": desc})
                continue

            # Add concept
            if any(k in lower for k in ("add", "new")):
                if ":" in stripped:
                    parts = stripped.split(":", 1)
                    name = parts[0]
                    for kw in ("add concept", "new concept"):
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
        "You are a knowledge graph construction assistant. Please read the problem below and output the core concepts "
        "needed to answer it, and the relationships between those concepts.\n\n"
        f"Problem type: {problem.problem_type}\n"
        f"Problem content:\n{problem.prompt}\n\n"
        "Please output in JSON format, must include concepts and relations fields:\n"
        "{\n"
        '  "concepts": [\n'
        '    {"name": "Concept name", "description": "Brief description", "category": "Category"}\n'
        "  ],\n"
        '  "relations": [\n'
        '    {"source": "Source concept name", "target": "Target concept name", "relation_type": "Relation type"}\n'
        "  ]\n"
        "}\n"
        "Notes:\n"
        "1. The name in concepts must be concise; it will be directly referenced by the LLM during reasoning.\n"
        "2. The source/target in relations must be names that already exist in concepts.\n"
        "3. Relation types optional: prerequisite / causes / part_of / located_at / analogous / related."
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
