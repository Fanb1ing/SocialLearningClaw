from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from .types import MCQProblem


@dataclass
class SkillSnippet:
    skill_id: str
    title: str
    when_to_use: str
    checklist: List[str]
    details: str = ""


def build_prompt(
    *,
    problem: MCQProblem,
    skills: Sequence[SkillSnippet],
    knowledge_points: Sequence[str],
    attempt_index: int,
) -> str:
    """Build a system+user style prompt in a single string.

    Stage1: keep it provider-agnostic; the adapter decides how to map to chat messages.

    Note: Cosmos-Reason1 items may reference a video. Stage1 currently does NOT pass video bytes/frames.
    We only include the video path (if any) as metadata so the model is aware of the missing modality.
    """

    # Progressive disclosure
    skill_blocks: List[str] = []
    for s in skills:
        if attempt_index <= 0:
            blk = [f"[Skill {s.skill_id}] {s.title}", f"When to use: {s.when_to_use}", "Checklist:"]
            blk += [f"- {x}" for x in s.checklist]
        else:
            blk = [f"[Skill {s.skill_id}] {s.title}", f"When to use: {s.when_to_use}", "Checklist:"]
            blk += [f"- {x}" for x in s.checklist]
            if s.details:
                blk += ["Details:", s.details]
        skill_blocks.append("\n".join(blk))

    kp_block = "\n".join([f"- {x}" for x in knowledge_points]) if knowledge_points else ""

    choices = "\n".join(problem.choices)

    video_hint = ""
    video = (problem.meta or {}).get("video")
    if isinstance(video, str) and video:
        video_hint = (
            "This question references a video. In this Stage1 run, you will NOT receive the video content. "
            "Only the video path/id is provided for context.\n"
            f"video: {video}\n\n"
        )

    system = (
        "You are a careful agent solving a multiple-choice question. "
        "You must output the final answer as a single choice letter (A/B/C/...). "
        "You must also output confidence as a float in [0,1].\n\n"
        "Output format (exactly):\n"
        "choice: <LETTER>\n"
        "confidence: <FLOAT>\n"
    )

    if skill_blocks:
        system += "\nReference skills (use if relevant):\n\n" + "\n\n".join(skill_blocks) + "\n"

    user = ""
    if video_hint:
        user += video_hint
    user += f"Question:\n{problem.prompt}\n\nChoices:\n{choices}\n"
    if kp_block:
        user += "\nHuman key knowledge points (high priority):\n" + kp_block + "\n"

    user += "\nNow answer."

    return system + "\n---\n" + user
