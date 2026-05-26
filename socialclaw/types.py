from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .agent.base import ReasoningTrace
from .dataset.base import EvalResult, Problem


@dataclass
class AttemptRecord:
    input_prompt: str
    answer_text: str
    reasoning_trace: ReasoningTrace
    usage: Dict[str, int]


@dataclass
class Episode:
    problem: Problem
    attempts: List[AttemptRecord] = field(default_factory=list)
    evals: List[EvalResult] = field(default_factory=list)
    reasoning_trace: Optional[ReasoningTrace] = None
    reasoning_confidence: float = 0.0
    flags: List[str] = field(default_factory=list)
    stop_reason: Optional[str] = None
    model: Optional[str] = None
