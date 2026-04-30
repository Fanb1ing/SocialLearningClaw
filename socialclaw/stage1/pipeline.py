from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime

from .cc_agent.base import CCAgent
from .dataset_cosmos_reason1 import load_prepared_jsonl as load_cosmos_reason1
from .dataset_pbench import load_prepared_jsonl as load_pbench
from .evaluator import evaluate
from .human_io import ask_key_points_cli
from .logging import write_episode
from .prompt_builder import build_prompt
from .skill.gate import should_summarize
from .skill.retrieve import SkillRetriever
from .skill.store import SkillStore
from .skill.summarize import summarize_episode_template
from .stop_policy import StopConfig, should_stop
from .types import Episode


@dataclass
class RunConfig:
    prepared_dataset_path: str
    skills_db_dir: str = "skills_db"
    runs_dir: str = "runs"
    max_problems: int = 20
    top_k_skills: int = 3
    stop: StopConfig = field(default_factory=StopConfig)


def _load_dataset(path: str):
    name = os.path.basename(path).lower()
    # heuristic routing
    if "pbench" in path.lower():
        return load_pbench(path)
    if "cosmos" in path.lower() or "reason" in path.lower():
        return load_cosmos_reason1(path)
    # default
    return load_pbench(path)


def run_stage1(*, agent: CCAgent, cfg: RunConfig) -> str:
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg.runs_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    retriever = SkillRetriever(db_dir=cfg.skills_db_dir)
    store = SkillStore(db_dir=cfg.skills_db_dir)

    count = 0
    for problem in _load_dataset(cfg.prepared_dataset_path):
        ep = Episode(problem=problem)
        skills = retriever.retrieve(problem, top_k=cfg.top_k_skills)

        for attempt_index in range(cfg.stop.max_iters):
            prompt = build_prompt(
                problem=problem,
                skills=skills,
                knowledge_points=ep.knowledge_points,
                attempt_index=attempt_index,
            )
            result = agent.answer(
                prompt=prompt,
                meta={
                    "problem_id": problem.id,
                    "attempt": attempt_index,
                    "problem_meta": problem.meta or {},
                    "prepared_dataset_path": cfg.prepared_dataset_path,
                },
            )

            ep.attempts.append(
                {
                    "input_prompt": prompt,
                    "answer_text": result.answer_text,
                    "confidence": result.confidence,
                    "usage": {
                        "input_tokens": result.usage.input_tokens,
                        "output_tokens": result.usage.output_tokens,
                        "total_tokens": result.usage.total_tokens,
                    },
                    "tool_calls": {"count": result.tool_calls.count, "tools": result.tool_calls.tools},
                    "raw": result.raw,
                }
            )

            ev = evaluate(problem, result)
            ep.evals.append(ev)

            # Here 调试
            if ev.correct:
                break

            stop, reason = should_stop(ep, cfg.stop)
            if stop:
                ep.stop_reason = reason
                break

            kps = ask_key_points_cli(ep)
            ep.knowledge_points.extend(kps)

        if ep.stop_reason is None:
            stop, reason = should_stop(ep, cfg.stop)
            if stop:
                ep.stop_reason = reason

        if should_summarize(ep):
        # if True:
            draft = summarize_episode_template(ep)
            draft.meta.created_at = datetime.utcnow().isoformat() + "Z"
            store.write_markdown(meta=draft.meta, markdown_body=draft.body)
            ep.skill_written = draft.meta.id

        write_episode(run_dir, ep)

        count += 1
        if count >= cfg.max_problems:
            break

    return run_dir
