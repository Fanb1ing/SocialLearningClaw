from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Sequence

from ..prompt_builder import SkillSnippet
from ..types import MCQProblem


@dataclass
class SkillDoc:
    id: str
    title: str
    tags: List[str]


class SkillRetriever:
    def __init__(self, *, db_dir: str):
        self.db_dir = db_dir
        self.index_path = os.path.join(db_dir, "index.jsonl")

    def retrieve(self, problem: MCQProblem, *, top_k: int = 3) -> Sequence[SkillSnippet]:
        if not os.path.exists(self.index_path):
            return []

        text = (problem.prompt or "").lower()

        scored: List[tuple[int, SkillDoc]] = []
        with open(self.index_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                meta = json.loads(line)
                tags = meta.get("tags") or []
                title = meta.get("title") or ""
                score = 0
                for t in tags:
                    if isinstance(t, str) and t.lower() in text:
                        score += 1
                if title and any(w in text for w in title.lower().split()[:3]):
                    score += 1
                scored.append((score, SkillDoc(id=meta["id"], title=title, tags=tags)))

        scored.sort(key=lambda x: x[0], reverse=True)
        picked = [d for s, d in scored if s > 0][:top_k]

        # Stage1: we only have metadata in index; return minimal snippets.
        res: List[SkillSnippet] = []
        for d in picked:
            res.append(
                SkillSnippet(
                    skill_id=d.id,
                    title=d.title or d.id,
                    when_to_use="When the problem matches related tags.",
                    checklist=["Apply the rule described by this skill.", "Double-check choice mapping."],
                    details="",
                )
            )
        return res
