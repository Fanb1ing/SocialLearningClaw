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
    - "答案：B"
    - "I choose C."
    """
    t = (text or "").strip().upper()

    for pat in [
        r"答案\s*[:：]\s*([A-Z])",
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
    """Use LLM to judge whether pred correctly answers the problem given gold."""
    # Truncate long texts to keep prompt size reasonable
    prompt_problem = problem.prompt[:1200]
    prompt_gold = gold[:1200]
    prompt_pred = pred[:1200]

    prompt = (
        "你是一个公正的评估助手。请判断下面的模型回答是否正确、合理地回应了题目要求。\n\n"
        f"题目（前1200字）：\n{prompt_problem}\n\n"
        f"标准答案（前1200字）：\n{prompt_gold}\n\n"
        f"模型回答（前1200字）：\n{prompt_pred}\n\n"
        "评估标准：\n"
        "- 模型回答只要核心语义与标准答案一致即可，不需要逐字相同。\n"
        "- 如果模型回答正确或合理，输出：correct\n"
        "- 如果模型回答错误、偏离主题或遗漏关键信息，输出：wrong\n\n"
        "请只输出一个字（correct 或 wrong）："
    )

    try:
        attempt = agent.answer(
            prompt=prompt,
            meta={"task": "llm_judge", "problem_id": problem.id},
        )
        answer = attempt.answer_text.strip().lower()
        if "correct" in answer and "wrong" not in answer and "incorrect" not in answer:
            return True
        if "wrong" in answer or "incorrect" in answer or "错误" in answer:
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

    # For long_context: use LLM-as-judge if agent is provided and gold is non-empty
    if problem.problem_type == "long_context" and agent and gold:
        pred = attempt.answer_text.strip()
        correct = _llm_judge(problem, pred, gold, agent)
        details = "" if correct else "llm_judge=wrong"
        return EvalResult(correct=correct, pred=pred, gold=gold, details=details)

    # Fallback: simple exact match (also covers empty gold)
    pred = attempt.answer_text.strip()
    correct = pred == gold
    details = "" if correct else f"pred={pred!r}, gold={gold!r}"
    return EvalResult(correct=correct, pred=pred, gold=gold, details=details)
