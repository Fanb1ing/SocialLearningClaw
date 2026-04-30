from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..types import Episode
from .store import SkillMeta, new_skill_id


@dataclass
class SkillDraft:
    meta: SkillMeta
    body: str


def summarize_episode_template(episode: Episode) -> SkillDraft:
    skill_id = new_skill_id()
    title = f"MCQ: fix-after-human-feedback ({episode.problem.id})"
    tags = ["mcq", "cosmos_reason1"]
    trigger = ["wrong_then_human_help"]

    checklist = "\n".join([f"- {kp}" for kp in episode.knowledge_points[:10]])

    body = (
        f"# {title}\n\n"
        "## When to use\n"
        "When you answered a similar MCQ incorrectly; consult the key knowledge points / checklist.\n\n"
        "## Checklist\n"
        f"{checklist}\n\n"
        "## Common pitfalls\n"
        "- Misreading the options; mapping to wrong letter.\n"
        "- Overconfident guessing without using given knowledge points.\n\n"
        "## Minimal example\n"
        "(Omitted)\n"
    )

    meta = SkillMeta(
        id=skill_id,
        created_at="",
        tags=tags,
        trigger=trigger,
        title=title,
    )
    return SkillDraft(meta=meta, body=body)
