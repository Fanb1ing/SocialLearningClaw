from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Dict, List


@dataclass(frozen=True)
class ChatResponse:
    text: str
    usage: Dict[str, int]
    model: str


class OpenAIChatClient:
    """Small OpenAI-compatible client used by the unified experiment runner."""

    def __init__(self, *, base_url: str, api_key: str, model: str) -> None:
        from openai import OpenAI

        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def complete(
        self,
        messages: List[Dict[str, Any]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> ChatResponse:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                break
            except Exception as error:
                last_error = error
                if attempt == 2:
                    raise
                time.sleep(2**attempt)
        else:  # pragma: no cover - defensive; the loop either breaks or raises
            raise RuntimeError("LLM request failed") from last_error
        message = response.choices[0].message
        text = message.content or getattr(message, "reasoning_content", None) or ""
        usage = response.usage
        return ChatResponse(
            text=text,
            usage={
                "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
            },
            model=str(getattr(response, "model", self.model) or self.model),
        )
