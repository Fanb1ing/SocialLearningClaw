from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Protocol, Tuple


@dataclass
class Usage:
    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass
class ReasoningTrace:
    concepts: List[str]
    relations: List[Tuple[str, str, str]]
    explanation: str = ""


@dataclass
class AgentAttempt:
    answer_text: str
    reasoning_trace: ReasoningTrace
    usage: Usage
    raw: Dict[str, Any]


class Agent(Protocol):
    def answer(self, *, prompt: str, meta: Dict[str, Any]) -> AgentAttempt:
        ...
