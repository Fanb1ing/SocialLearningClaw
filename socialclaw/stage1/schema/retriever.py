from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ..agent.base import Agent
from ..dataset.base import Problem
from .graph import Concept, SchemaGraph


@dataclass
class RetrieveResult:
    """Result of schema retrieval for a problem."""

    matched: List[Concept]  # Concepts from schema that match the required ones
    missing: List[str]  # Required concept names that have no match in schema


class SchemaRetriever:
    def __init__(
        self,
        graph: SchemaGraph,
        embeddings: Dict[str, np.ndarray],
        embedder,  # SentenceTransformer or any callable with .encode()
        agent: Optional[Agent] = None,
    ):
        self.graph = graph
        self.embeddings = embeddings
        self.embedder = embedder
        self.agent = agent

    def retrieve(
        self, problem: Problem, top_k: int = 5, threshold: float = 0.6
    ) -> RetrieveResult:
        """Retrieve schema concepts by:

        1. Using LLM to extract required concepts from the problem.
        2. Matching each required concept against the schema via embedding similarity.
        3. Returning matched schema concepts + missing concept names.
        """
        if not self.embeddings or not self.agent:
            # Fallback: if no agent or empty schema, return empty matched + no missing
            if not self.embeddings:
                # Schema is empty: all required concepts are missing (but we don't know them yet)
                return RetrieveResult(matched=[], missing=[])
            # No agent but schema exists: fallback to old query-based retrieval
            return self._legacy_retrieve(problem, top_k, threshold)

        required_concepts = self._extract_required_concepts(problem)
        if not required_concepts:
            return RetrieveResult(matched=[], missing=[])

        matched: List[Concept] = []
        missing: List[str] = []
        matched_ids: set = set()

        for req_name in required_concepts:
            best_cid, best_score = self._find_best_match(req_name)
            if best_cid and best_score >= threshold:
                c = self.graph.get_concept(best_cid)
                if c and c.id not in matched_ids:
                    matched.append(c)
                    matched_ids.add(c.id)
            else:
                missing.append(req_name)

        return RetrieveResult(matched=matched, missing=missing)

    def is_sufficient(self, result: RetrieveResult) -> bool:
        """Check whether retrieved concepts are sufficient to answer.

        Sufficiency = no missing concepts AND at least one matched concept.
        The 'missing' list is produced by LLM extraction + embedding matching,
        so this is already an LLM-based judgment without hard thresholds.
        """
        if result.missing:
            return False
        return bool(result.matched)

    def _extract_required_concepts(self, problem: Problem) -> List[str]:
        """Ask LLM to extract the key concepts required to answer this problem."""
        query_text = problem.retrieval_query if problem.retrieval_query else problem.prompt
        prompt = (
            "你是一个概念提取助手。请阅读下面的题目，列出回答这道题所需要的关键概念名称。\n"
            "只需要输出概念名称列表，每行一个，不要输出解释。\n\n"
            f"题目类型：{problem.problem_type}\n"
            f"题目内容（前800字）：\n{query_text[:800]}...\n\n"
            "输出格式（每行一个）：\n"
            "- 概念名称1\n"
            "- 概念名称2\n"
            "..."
        )

        try:
            attempt = self.agent.answer(
                prompt=prompt,
                meta={"task": "extract_concepts", "problem_id": problem.id},
            )
            return self._parse_concept_list(attempt.answer_text)
        except Exception:
            return []

    def _parse_concept_list(self, text: str) -> List[str]:
        """Parse LLM output into a list of concept names."""
        concepts: List[str] = []
        for line in text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            # Remove bullet markers
            stripped = line.lstrip("-–•*1234567890. ").strip()
            if stripped:
                concepts.append(stripped)
        return concepts

    def _find_best_match(self, concept_name: str) -> tuple[Optional[str], float]:
        """Find the best matching concept in schema by embedding similarity."""
        if not self.embeddings:
            return None, 0.0

        req_emb = self._encode_text(concept_name)
        ids = list(self.embeddings.keys())
        mat = np.stack([self.embeddings[cid] for cid in ids], axis=0)

        # Cosine similarity
        req_norm = req_emb / (np.linalg.norm(req_emb) + 1e-8)
        mat_norm = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)
        sims = mat_norm @ req_norm

        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        return ids[best_idx], best_score

    def _legacy_retrieve(
        self, problem: Problem, top_k: int = 5, threshold: float = 0.6
    ) -> RetrieveResult:
        """Fallback: direct query embedding retrieval (old behavior)."""
        query_text = problem.retrieval_query if problem.retrieval_query else problem.prompt
        q_emb = self._encode_text(query_text)
        ids = list(self.embeddings.keys())
        mat = np.stack([self.embeddings[cid] for cid in ids], axis=0)

        q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-8)
        mat_norm = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)
        sims = mat_norm @ q_norm

        scored = [(float(sims[i]), ids[i]) for i in range(len(ids))]
        scored.sort(key=lambda x: x[0], reverse=True)

        matched: List[Concept] = []
        for score, cid in scored[:top_k]:
            if score >= threshold:
                c = self.graph.get_concept(cid)
                if c:
                    matched.append(c)

        return RetrieveResult(matched=matched, missing=[])

    def _llm_is_sufficient(self, concepts: List[Concept]) -> bool:
        """Ask LLM whether the matched concepts are enough to answer.
        Only used when no missing concepts exist, for second validation."""
        concept_text = "\n".join(
            [f"- {c.name} ({c.category}): {c.description[:120]}" for c in concepts]
        ) if concepts else "（无概念被检索到）"

        prompt = (
            "你是一个概念充足度判断助手。请判断下面检索到的概念是否足够回答给定题目。\n\n"
            f"检索到的概念：\n{concept_text}\n\n"
            "请只输出一个字：\n"
            "- 如果概念足够答题，输出：sufficient\n"
            "- 如果概念不足或缺失关键概念，输出：insufficient"
        )

        try:
            attempt = self.agent.answer(
                prompt=prompt,
                meta={"task": "is_sufficient"},
            )
            answer = attempt.answer_text.strip().lower()
            if "sufficient" in answer and "insufficient" not in answer:
                return True
            if "insufficient" in answer or "不足" in answer or "不够" in answer:
                return False
            return False
        except Exception:
            return bool(concepts)

    def _encode_text(self, text: str) -> np.ndarray:
        emb = self.embedder.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        if emb.ndim == 2:
            emb = emb[0]
        return emb
