from __future__ import annotations

import re
from dataclasses import dataclass

from .types import CCAgentResult, EvalResult, MCQProblem


_CHOICE_RE = re.compile(r"\b([A-Z])\b")


def normalize_choice(text: str) -> str:
    """Extract a choice letter from free-form text.

    Accepts formats like:
    - "A"
    - "答案：B"
    - "I choose C."
    """

    t = (text or "").strip().upper()

    # Common explicit patterns
    for pat in [r"答案\s*[:：]\s*([A-Z])", r"CHOICE\s*[:：]\s*([A-Z])", r"OPTION\s*[:：]\s*([A-Z])"]:
        m = re.search(pat, t)
        if m:
            return m.group(1)

    # JSON-ish
    m = re.search(r"\"CHOICE\"\s*:\s*\"([A-Z])\"", t)
    if m:
        return m.group(1)

    # Fallback: first standalone capital letter
    m = _CHOICE_RE.search(t)
    if m:
        return m.group(1)

    return ""


def evaluate(problem: MCQProblem, result: CCAgentResult) -> EvalResult:
    pred = normalize_choice(result.answer_text)
    gold = (problem.answer_key or "").strip().upper()
    correct = pred == gold and pred != ""
    details = "" if correct else f"pred={pred!r}, gold={gold!r}"
    return EvalResult(correct=correct, pred=pred, gold=gold, details=details)
