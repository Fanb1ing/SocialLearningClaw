from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional


@dataclass
class Problem:
    id: str
    prompt: str
    problem_type: str  # "mcq" | "long_context" | "arc_grid"
    meta: Dict[str, Any] = field(default_factory=dict)
    retrieval_query: str = ""  # Optional shorter text for embedding retrieval (defaults to prompt)


@dataclass
class EvalResult:
    correct: bool
    pred: Any
    gold: Any
    details: str = ""
