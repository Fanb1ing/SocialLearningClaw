from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .agent.base import Agent, AgentAttempt
from .dataset.base import EvalResult, Problem


_CHOICE_RE = re.compile(r"\b([A-Z])\b")


def normalize_choice(text: str) -> str:
    """Extract a choice letter from free-form text.

    Accepts formats like:
    - "A"
    - "Answer: B"
    - "I choose C."
    """
    t = (text or "").strip().upper()

    for pat in [
        r"ANSWER\s*[:：]\s*([A-Z])",
        r"CHOICE\s*[:：]\s*([A-Z])",
        r"OPTION\s*[:：]\s*([A-Z])",
    ]:
        m = re.search(pat, t)
        if m:
            return m.group(1)

    m = re.search(r'"CHOICE"\s*:\s*"([A-Z])"', t)
    if m:
        return m.group(1)

    m = _CHOICE_RE.search(t)
    if m:
        return m.group(1)

    return ""


def _llm_judge(problem: Problem, pred: str, gold: str, agent: Agent) -> bool:
    """Use LLM to judge whether pred correctly answers the problem.

    When gold answer is available: compares pred against gold.
    When gold is empty but rubrics exist: evaluates pred against rubrics criteria.
    """
    import json

    # Truncate long texts to keep prompt size reasonable
    prompt_problem = problem.prompt[:1200]
    prompt_pred = pred[:1200]

    rubrics = problem.meta.get("rubrics") if isinstance(problem.meta, dict) else None
    if isinstance(rubrics, str):
        try:
            rubrics = json.loads(rubrics)
        except Exception:
            rubrics = [rubrics] if rubrics else []

    if gold:
        prompt_gold = gold[:1200]
        prompt = (
            "You are a fair evaluation assistant. Please judge whether the model's response correctly and reasonably answers the problem.\n\n"
            f"Problem (first 1200 chars):\n{prompt_problem}\n\n"
            f"Ground truth (first 1200 chars):\n{prompt_gold}\n\n"
            f"Model response (first 1200 chars):\n{prompt_pred}\n\n"
            "Evaluation criteria:\n"
            "- The model response is considered correct if its core meaning matches the ground truth; exact wording is not required.\n"
            "- If the model response is correct or reasonable, output: correct\n"
            "- If the model response is wrong, off-topic, or misses key information, output: wrong\n\n"
            "Please output only one word (correct or wrong):"
        )
    elif rubrics:
        rubrics_text = "\n".join(f"- {r}" for r in rubrics[:15])
        prompt = (
            "You are a fair evaluation assistant. Please judge whether the model's response correctly and reasonably answers the problem.\n\n"
            f"Problem (first 1200 chars):\n{prompt_problem}\n\n"
            f"Model response (first 1200 chars):\n{prompt_pred}\n\n"
            f"Evaluation rubrics (the response should satisfy these criteria):\n{rubrics_text}\n\n"
            "Evaluation criteria:\n"
            "- The model response is considered correct if it satisfies all or most of the rubrics above.\n"
            "- If the model response satisfies the rubrics, output: correct\n"
            "- If the model response fails to satisfy key rubrics, is off-topic, or misses critical information, output: wrong\n\n"
            "Please output only one word (correct or wrong):"
        )
    else:
        # No gold and no rubrics: fall back to exact match
        return pred.strip() == gold.strip()

    try:
        attempt = agent.answer(
            prompt=prompt,
            meta={"task": "llm_judge", "problem_id": problem.id},
        )
        answer = attempt.answer_text.strip().lower()
        if "correct" in answer and "wrong" not in answer and "incorrect" not in answer:
            return True
        if "wrong" in answer or "incorrect" in answer:
            return False
        # Default to False on ambiguity
        return False
    except Exception:
        # Fallback to exact match on LLM failure
        return pred.strip() == gold.strip()


def evaluate(
    problem: Problem,
    attempt: AgentAttempt,
    agent: Optional[Agent] = None,
) -> EvalResult:
    """Evaluate an agent attempt against the problem answer key."""
    gold = str(problem.meta.get("answer_key", "")).strip()

    if problem.problem_type == "mcq":
        pred = normalize_choice(attempt.answer_text)
        correct = pred == gold and pred != ""
        details = "" if correct else f"pred={pred!r}, gold={gold!r}"
        return EvalResult(correct=correct, pred=pred, gold=gold, details=details)

    # For long_context: use LLM-as-judge if we have gold OR rubrics to evaluate against
    if problem.problem_type == "long_context" and agent:
        rubrics = problem.meta.get("rubrics") if isinstance(problem.meta, dict) else None
        has_rubrics = bool(rubrics)
        if gold or has_rubrics:
            pred = attempt.answer_text.strip()
            correct = _llm_judge(problem, pred, gold, agent)
            details = "" if correct else "llm_judge=wrong"
            return EvalResult(correct=correct, pred=pred, gold=gold, details=details)

    # Fallback: simple exact match
    pred = attempt.answer_text.strip()
    correct = pred == gold
    details = "" if correct else f"pred={pred!r}, gold={gold!r}"
    return EvalResult(correct=correct, pred=pred, gold=gold, details=details)
