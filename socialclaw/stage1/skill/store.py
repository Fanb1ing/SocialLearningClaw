from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class SkillMeta:
    id: str
    created_at: str
    tags: List[str]
    trigger: List[str]
    title: str


class SkillStore:
    def __init__(self, *, db_dir: str):
        self.db_dir = db_dir
        self.skills_dir = os.path.join(db_dir, "skills")
        self.index_path = os.path.join(db_dir, "index.jsonl")
        os.makedirs(self.skills_dir, exist_ok=True)

    def write_markdown(self, *, meta: SkillMeta, markdown_body: str) -> str:
        path = os.path.join(self.skills_dir, f"{meta.id}.md")
        header = {
            "id": meta.id,
            "created_at": meta.created_at,
            "tags": meta.tags,
            "trigger": meta.trigger,
            "title": meta.title,
        }
        content = "---\n" + "\n".join([f"{k}: {json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v}" for k, v in header.items()]) + "\n---\n\n" + markdown_body
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

        with open(self.index_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(header, ensure_ascii=False) + "\n")

        return path


def new_skill_id(prefix: str = "skill") -> str:
    return f"{prefix}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
