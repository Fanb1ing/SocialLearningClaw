from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MCQProblem:
    """A multiple-choice (including binary) problem."""

    id: str
    prompt: str
    choices: List[str]
    answer_key: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Usage:
    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass
class ToolCallStats:
    count: int
    tools: List[str]


@dataclass
class CCAgentResult:
    answer_text: str
    confidence: Optional[float]
    usage: Usage
    tool_calls: ToolCallStats
    raw: Dict[str, Any]


@dataclass
class EvalResult:
    correct: bool
    pred: str
    gold: str
    details: str = ""


@dataclass
class Episode:
    problem: MCQProblem
    attempts: List[Dict[str, Any]] = field(default_factory=list)
    evals: List[EvalResult] = field(default_factory=list)
    knowledge_points: List[str] = field(default_factory=list)
    stop_reason: Optional[str] = None
    skill_written: Optional[str] = None  # skill_id if written
