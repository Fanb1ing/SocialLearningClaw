from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Sequence

import numpy as np

from .models import MemoryRecord
from .store import MemoryStore


class Embedder(Protocol):
    def encode(self, sentences, **kwargs):
        ...


@dataclass(frozen=True)
class MemoryMatch:
    record: MemoryRecord
    score: float


class MemoryBank:
    """High-level memory API with optional semantic retrieval."""

    def __init__(self, store: MemoryStore, embedder: Optional[Embedder] = None) -> None:
        self.store = store
        self.embedder = embedder
        self._embeddings: Dict[str, np.ndarray] = {}

    def remember(self, record: MemoryRecord) -> MemoryRecord:
        stored = self.store.put(record)
        self._embeddings.pop(record.id, None)
        return stored

    def recall(self, memory_id: str) -> Optional[MemoryRecord]:
        return self.store.get(memory_id)

    def forget(self, memory_id: str) -> bool:
        self._embeddings.pop(memory_id, None)
        return self.store.delete(memory_id)

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        tags: Optional[Sequence[str]] = None,
    ) -> List[MemoryMatch]:
        if top_k <= 0:
            return []
        required_tags = set(tags or ())
        records = [
            record
            for record in self.store.list()
            if not required_tags or required_tags.issubset(record.tags)
        ]
        if not records:
            return []

        lexical = {record.id: self._lexical_score(query, record.text_for_retrieval()) for record in records}
        semantic: Dict[str, float] = {}
        if self.embedder is not None and query.strip():
            query_vector = self._encode(query)
            for record in records:
                vector = self._embeddings.get(record.id)
                if vector is None:
                    vector = self._encode(record.text_for_retrieval())
                    self._embeddings[record.id] = vector
                semantic[record.id] = float(vector @ query_vector)

        matches = [
            MemoryMatch(
                record=record,
                score=(0.85 * semantic[record.id] + 0.15 * lexical[record.id])
                if semantic
                else lexical[record.id],
            )
            for record in records
        ]
        matches.sort(key=lambda match: (match.score, match.record.updated_at), reverse=True)
        return matches[:top_k]

    def _encode(self, text: str) -> np.ndarray:
        assert self.embedder is not None
        value = self.embedder.encode(
            text, convert_to_numpy=True, normalize_embeddings=True
        )
        vector = np.asarray(value[0] if getattr(value, "ndim", 1) == 2 else value, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector

    @staticmethod
    def _lexical_score(query: str, document: str) -> float:
        tokenize = lambda text: set(
            re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", text.lower())
        )
        query_tokens = tokenize(query)
        if not query_tokens:
            return 0.0
        document_tokens = tokenize(document)
        return len(query_tokens & document_tokens) / len(query_tokens)
