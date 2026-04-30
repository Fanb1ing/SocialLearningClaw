from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol


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


class CCAgent(Protocol):
    """Abstract CC Agent interface.

    Hard constraints (Stage1):
    - usage must be present (at least total_tokens)
    - tool_calls must be present (at least count)
    """

    def answer(self, *, prompt: str, meta: Dict[str, Any]) -> CCAgentResult:  # pragma: no cover
        ...
