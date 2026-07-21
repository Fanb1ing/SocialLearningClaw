"""
ExpeL (Experience Pool) memory module.
Zhao et al., "ExpeL: LLM Agents Are Experiential Learners", AAAI 2024.

Maintains an experience pool of (task, trajectory, outcome, lesson) tuples.
Every `insight_interval` experiences, an LLM extracts generalizable insights.
New tasks receive the insight list + the most recent successful examples.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

from openai import OpenAI


class ExPeLMemory:
    """Cross-episode experience pool with periodic LLM insight extraction."""

    def __init__(
        self,
        client: OpenAI,
        model_id: str,
        insight_interval: int = 5,
        max_insights: int = 10,
        max_examples: int = 3,
    ) -> None:
        self.client = client
        self.model_id = model_id
        self.insight_interval = insight_interval
        self.max_insights = max_insights
        self.max_examples = max_examples
        self.experiences: List[Dict] = []
        self.insights: List[str] = []

    # ── public API ────────────────────────────────────────────────────────────

    def add_experience(
        self,
        task: str,
        outcome: bool,
        trajectory: str = "",
        lesson: str = "",
    ) -> None:
        """Add one episode; triggers insight extraction every `insight_interval` episodes."""
        self.experiences.append({
            "task": task,
            "trajectory": trajectory,
            "outcome": "success" if outcome else "failure",
            "lesson": lesson,
        })
        if len(self.experiences) % self.insight_interval == 0:
            self._extract_insights()

    def get_memory_block(self) -> str:
        """Return the memory block to prepend to prompts (empty string if none)."""
        parts: List[str] = []
        if self.insights:
            parts.append(
                "=== Learned strategies (distilled from past experience) ===\n"
                + "\n".join(self.insights)
                + "\n==========================================================="
            )
        successes = [e for e in self.experiences if e["outcome"] == "success"]
        recent_successes = successes[-self.max_examples:]
        if recent_successes:
            ex_lines = [
                f"- Task: {e['task'][:200]}...\n  Lesson: {e['lesson']}"
                for e in recent_successes
            ]
            parts.append(
                "=== Recent successful examples ===\n"
                + "\n".join(ex_lines)
                + "\n=================================="
            )
        return "\n\n".join(parts)

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"experiences": self.experiences, "insights": self.insights},
                f,
                indent=2,
                ensure_ascii=False,
            )

    def load(self, path: str | Path) -> None:
        p = Path(path)
        if p.exists():
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            self.experiences = data.get("experiences", [])
            self.insights = data.get("insights", [])

    # ── internal ──────────────────────────────────────────────────────────────

    def _extract_insights(self) -> None:
        recent = self.experiences[-20:]
        exp_text = "\n\n".join(
            f"[{i + 1}] Task: {e['task'][:300]}\n"
            f"Outcome: {e['outcome']}\n"
            f"Lesson: {e['lesson']}"
            for i, e in enumerate(recent)
        )
        prompt = (
            f"Review these {len(recent)} past experiences and extract up to "
            f"{self.max_insights} generalizable rules or strategies.\n\n"
            f"{exp_text}\n\n"
            "Output numbered rules (e.g. '1. ...'), one per line. "
            "Each rule must be concise (1–2 sentences) and directly actionable."
        )
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=512,
                    temperature=0.0,
                )
                raw = (resp.choices[0].message.content or "").strip()
                lines = [l.strip() for l in raw.split("\n") if l.strip()]
                self.insights = lines[: self.max_insights]
                print(f"    [ExpeL] Extracted {len(self.insights)} insights from {len(recent)} experiences.")
                return
            except Exception as e:
                print(f"    [ExpeL] insight extraction API error (attempt {attempt + 1}): {e}")
                if attempt < 2:
                    time.sleep(3)
