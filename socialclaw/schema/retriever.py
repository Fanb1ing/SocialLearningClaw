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
        import json
        query_text = problem.retrieval_query if problem.retrieval_query else problem.prompt
        prompt = (
            "You are a concept extraction assistant. Please read the problem below and list the key concept names needed to answer it.\n"
            "Output ONLY a JSON array of strings. No explanations.\n\n"
            f"Problem type: {problem.problem_type}\n"
            f"Problem content (first 800 chars):\n{query_text[:800]}...\n\n"
            'Example: ["Concept1", "Concept2", "Concept3"]'
        )

        try:
            attempt = self.agent.answer(
                prompt=prompt,
                meta={"task": "extract_concepts", "problem_id": problem.id},
                response_format="json_object",
            )
            return self._parse_concept_list(attempt.answer_text)
        except Exception:
            return []

    def _parse_concept_list(self, text: str) -> List[str]:
        """Parse LLM output into a list of concept names."""
        import json
        text = text.strip()
        # Try JSON array first (agent may be in json_object mode)
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
            if isinstance(parsed, dict):
                # Some models wrap the array in an object
                for key in ("concepts", "concept_names", "required_concepts", "names"):
                    if key in parsed and isinstance(parsed[key], list):
                        return [str(item).strip() for item in parsed[key] if str(item).strip()]
        except Exception:
            pass

        # Fallback: line-by-line parsing
        concepts: List[str] = []
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Remove bullet markers and brackets
            stripped = line.lstrip("-–•*1234567890. ").strip()
            for prefix in ["[", "{", "'", '"']:
                if stripped.startswith(prefix):
                    stripped = stripped[1:]
            for suffix in ["]", "}", "'", '"', ","]:
                if stripped.endswith(suffix):
                    stripped = stripped[:-1]
            stripped = stripped.strip()
            if stripped and len(stripped) < 80:
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
        ) if concepts else "(No concepts retrieved)"

        prompt = (
            "You are a concept sufficiency judge. Please judge whether the retrieved concepts below are sufficient to answer the given problem.\n\n"
            f"Retrieved concepts:\n{concept_text}\n\n"
            "Please output only one word:\n"
            "- If concepts are sufficient, output: sufficient\n"
            "- If concepts are insufficient or missing key concepts, output: insufficient"
        )

        try:
            attempt = self.agent.answer(
                prompt=prompt,
                meta={"task": "is_sufficient"},
            )
            answer = attempt.answer_text.strip().lower()
            if "sufficient" in answer and "insufficient" not in answer:
                return True
            if "insufficient" in answer:
                return False
            return False
        except Exception:
            return bool(concepts)

    def _encode_text(self, text: str) -> np.ndarray:
        emb = self.embedder.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        if emb.ndim == 2:
            emb = emb[0]
        return emb
