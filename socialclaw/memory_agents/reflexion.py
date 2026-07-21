"""
Reflexion memory module.
Shinn et al., "Reflexion: Language Agents with Verbal Reinforcement Learning", NeurIPS 2023.

Maintains a list of verbal reflections generated after failures.
Reflections are injected into subsequent attempts via the system/user prompt.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List

from openai import OpenAI


class ReflexionMemory:
    """Accumulates verbal reflections on failures across episodes."""

    def __init__(
        self,
        client: OpenAI,
        model_id: str,
        max_reflections: int = 10,
    ) -> None:
        self.client = client
        self.model_id = model_id
        self.max_reflections = max_reflections
        self.reflections: List[str] = []

    # ── public API ────────────────────────────────────────────────────────────

    def get_memory_block(self) -> str:
        """Return the memory block to prepend to prompts (empty string if none)."""
        if not self.reflections:
            return ""
        lines = [f"{i + 1}. {r}" for i, r in enumerate(self.reflections)]
        return (
            "=== Reflections from past failures (apply these lessons) ===\n"
            + "\n".join(lines)
            + "\n============================================================\n"
        )

    def reflect(self, task_context: str, failure_info: str) -> str:
        """Call LLM to generate a reflection; store and return it."""
        prompt = (
            "You attempted a task and failed. Reflect concisely on what went wrong.\n\n"
            f"Task context:\n{task_context}\n\n"
            f"What happened (failure):\n{failure_info}\n\n"
            "Write 2–3 sentences identifying the mistake and what strategy to use next time. "
            "Be specific and actionable."
        )
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=256,
                    temperature=0.0,
                )
                reflection = (resp.choices[0].message.content or "").strip()
                if reflection:
                    self.reflections = (self.reflections + [reflection])[-self.max_reflections:]
                return reflection
            except Exception as e:
                print(f"    [Reflexion] reflect API error (attempt {attempt + 1}): {e}")
                if attempt < 2:
                    time.sleep(3)
        return ""

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"reflections": self.reflections}, f, indent=2, ensure_ascii=False)

    def load(self, path: str | Path) -> None:
        p = Path(path)
        if p.exists():
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            self.reflections = data.get("reflections", [])
