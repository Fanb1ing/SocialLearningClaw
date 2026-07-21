from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from ..memory_agents import AMemory, ExPeLMemory, ReflexionMemory, TrainableGraphMemory
from ..schema import SchemaManager
from .schema import SchemaMethodController


@dataclass
class RAGExperience:
    task: str
    response: str
    correct: bool
    embedding: np.ndarray


class ExperienceRAG:
    def __init__(self, embedder, top_k: int = 5) -> None:
        self.embedder = embedder
        self.top_k = top_k
        self.experiences: List[RAGExperience] = []

    def _encode(self, text: str) -> np.ndarray:
        value = self.embedder.encode(
            text, convert_to_numpy=True, normalize_embeddings=True
        )
        return value[0] if getattr(value, "ndim", 1) == 2 else value

    def add(self, task: str, response: str, correct: bool) -> None:
        self.experiences.append(
            RAGExperience(
                task=task,
                response=response[:1000],
                correct=correct,
                embedding=self._encode(task),
            )
        )

    def block(self, query: str) -> str:
        if not self.experiences:
            return ""
        query_embedding = self._encode(query)
        ranked = sorted(
            ((float(item.embedding @ query_embedding), item) for item in self.experiences),
            key=lambda pair: pair[0],
            reverse=True,
        )[: self.top_k]
        lines = []
        for score, item in ranked:
            outcome = "correct" if item.correct else "incorrect"
            lines.append(
                f"- Similarity {score:.3f}; previous outcome={outcome}\n"
                f"  Task: {item.task[:400]}\n  Response: {item.response[:400]}"
            )
        return "=== Retrieved prior experiences ===\n" + "\n".join(lines)


class MethodController:
    """Shared lifecycle for baselines and the layered schema method.

    Only binary correctness is passed to learning methods. Ground-truth answers
    are deliberately absent from this interface.
    """

    def __init__(
        self,
        *,
        method: str,
        openai_client,
        model: str,
        embedder=None,
        retrieve_k: int = 5,
        schema_manager: Optional[SchemaManager] = None,
    ) -> None:
        self.method = method
        self.memory: Any = None
        if method == "rag":
            if embedder is None:
                raise ValueError("RAG requires an embedding model")
            self.memory = ExperienceRAG(embedder, top_k=retrieve_k)
        elif method == "reflexion":
            self.memory = ReflexionMemory(openai_client, model, max_reflections=20)
        elif method == "expel":
            self.memory = ExPeLMemory(openai_client, model, insight_interval=5, max_insights=10)
        elif method == "amem":
            if embedder is None:
                raise ValueError("A-MEM requires an embedding model")
            self.memory = AMemory(openai_client, model, embedder, max_notes=120, retrieve_k=retrieve_k)
        elif method == "tgm":
            if embedder is None:
                raise ValueError("TGM requires an embedding model")
            self.memory = TrainableGraphMemory(
                openai_client, model, embedder, max_meta=30, retrieve_k=min(3, retrieve_k)
            )
        elif method == "schema":
            if schema_manager is None:
                raise ValueError("Schema method requires a SchemaManager")
            self.memory = SchemaMethodController(
                schema_manager, retrieve_k=retrieve_k
            )

    def context(self, query: str) -> str:
        if self.method == "rag":
            return self.memory.block(query)
        if self.method in {"reflexion", "expel"}:
            return self.memory.get_memory_block()
        if self.method in {"amem", "tgm"}:
            return self.memory.get_memory_block(query)
        if self.method == "schema":
            return self.memory.context(query)
        return ""

    def after_failed_attempt(self, *, task: str, response: str) -> None:
        if self.method == "reflexion":
            self.memory.reflect(
                task_context=task,
                failure_info=(
                    "The preceding response was evaluated as incorrect. The evaluator supplied "
                    "no ground-truth answer. Diagnose the strategy, not the unknown target.\n"
                    f"Response: {response[:1200]}"
                ),
            )

    def after_sample(self, *, task: str, response: str, correct: bool, domain: str) -> None:
        outcome = "correct" if correct else "incorrect"
        lesson = (
            "The response was correct; retain only generally useful reasoning steps."
            if correct
            else "The response was incorrect; revise the approach without assuming the gold answer."
        )
        if self.method == "rag":
            self.memory.add(task, response, correct)
        elif self.method == "expel":
            self.memory.add_experience(
                task=task, outcome=correct, trajectory=response[:1200], lesson=lesson
            )
        elif self.method == "amem":
            self.memory.add_experience(
                task=task,
                outcome=correct,
                trajectory=response[:1200],
                lesson=lesson,
                context=f"domain={domain}; binary_outcome={outcome}",
            )
        elif self.method == "tgm":
            self.memory.add_experience(
                task=task,
                outcome=correct,
                trajectory=response[:1200],
                lesson=lesson,
                context=f"domain={domain}; binary_outcome={outcome}",
                domain=domain,
            )
        elif self.method == "schema":
            self.memory.after_sample(
                task=task,
                response=response,
                correct=correct,
                domain=domain,
            )
