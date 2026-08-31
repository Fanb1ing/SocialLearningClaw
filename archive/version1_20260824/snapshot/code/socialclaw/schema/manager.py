from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..memory import Embedder, MemoryBank, MemoryRecord
from ..memory.models import utc_now
from .induction import SchemaGenerator, SchemaProposal
from .layered_graph import LayeredSchemaGraph
from .layered_storage import LayeredSchemaStorage
from .node import SchemaNode, SchemaStatus


@dataclass(frozen=True)
class SchemaMatch:
    node: SchemaNode
    score: float


@dataclass(frozen=True)
class SchemaManagementConfig:
    positive_delta: float = 0.12
    negative_delta: float = 0.18
    level_sensitivity: float = 0.15
    daily_decay: float = 0.002
    isolation_penalty: float = 1.5
    mask_threshold: float = 0.20
    deprecate_threshold: float = 0.06
    duplicate_threshold: float = 0.90


class SchemaManager:
    """Coordinates memory-grounded generation, retrieval and maintenance."""

    def __init__(
        self,
        *,
        memory: MemoryBank,
        graph: Optional[LayeredSchemaGraph] = None,
        generator: Optional[SchemaGenerator] = None,
        embedder: Optional[Embedder] = None,
        storage: Optional[LayeredSchemaStorage] = None,
        config: SchemaManagementConfig = SchemaManagementConfig(),
    ) -> None:
        self.memory = memory
        self.graph = graph or LayeredSchemaGraph()
        self.generator = generator
        self.embedder = embedder
        self.storage = storage
        self.config = config
        self._embeddings: Dict[str, np.ndarray] = {}

    def learn(self, memory_id: str, *, candidate_k: int = 8) -> Optional[SchemaNode]:
        record = self.memory.recall(memory_id)
        if record is None:
            raise KeyError(f"Unknown memory record: {memory_id}")
        candidates = [match.node for match in self.retrieve(record.text_for_retrieval(), top_k=candidate_k)]
        proposal = (
            self.generator.propose(record, candidates)
            if self.generator is not None
            else self._fallback_proposal(record)
        )
        node = self._apply_proposal(proposal, record)
        self._persist()
        return node

    def remember_and_learn(self, record: MemoryRecord) -> Optional[SchemaNode]:
        self.memory.remember(record)
        return self.learn(record.id)

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        expand_neighbors: bool = True,
    ) -> List[SchemaMatch]:
        active = self.graph.list()
        if top_k <= 0 or not active:
            return []
        semantic: Dict[str, float] = {}
        if self.embedder is not None and query.strip():
            query_vector = self._encode(query)
            for node in active:
                vector = self._embeddings.get(node.index)
                if vector is None:
                    vector = self._encode(node.text_for_retrieval())
                    self._embeddings[node.index] = vector
                semantic[node.index] = float(vector @ query_vector)

        lexical = {
            node.index: self._lexical_similarity(query, node.text_for_retrieval())
            for node in active
        }
        scores = {
            node.index: (
                (0.70 * semantic[node.index] + 0.20 * lexical.get(node.index, 0.0))
                if semantic
                else 0.90 * lexical.get(node.index, 0.0)
            )
            + 0.10 * node.reliability_weight
            for node in active
        }
        ranked = sorted(active, key=lambda node: scores[node.index], reverse=True)[:top_k]

        if expand_neighbors:
            selected = {node.index: node for node in ranked}
            for node in list(ranked):
                for neighbor in self.graph.neighbors(node.index):
                    if neighbor.status == SchemaStatus.ACTIVE and len(selected) < top_k:
                        selected[neighbor.index] = neighbor
            ranked = sorted(selected.values(), key=lambda node: scores.get(node.index, 0.0), reverse=True)

        now = utc_now()
        for node in ranked:
            node.access_count += 1
            node.last_accessed_at = now
        result = [SchemaMatch(node=node, score=scores.get(node.index, 0.0)) for node in ranked[:top_k]]
        self._persist()
        return result

    def context_block(self, query: str, *, top_k: int = 5) -> str:
        """Render retrieved rules for injection into an agent/LLM prompt."""
        return self.format_context(self.retrieve(query, top_k=top_k))

    @staticmethod
    def format_context(matches: Sequence[SchemaMatch]) -> str:
        """Render already-retrieved matches without performing a second lookup."""
        if not matches:
            return ""
        lines = ["=== Retrieved world schemas ==="]
        for match in matches:
            node = match.node
            lines.append(
                f"- [{node.index}] level={node.level} "
                f"reliability={node.reliability_weight:.3f}: {node.description}"
            )
        lines.extend(
            [
                "Use a schema only when its trigger matches current evidence.",
                "Prefer current observations when a schema conflicts with the environment.",
                "=================================",
            ]
        )
        return "\n".join(lines)

    def apply_feedback(
        self,
        schema_ids: Sequence[str],
        *,
        memory_id: str,
        positive: bool,
    ) -> None:
        if self.memory.recall(memory_id) is None:
            raise KeyError(f"Feedback references unknown memory: {memory_id}")
        for schema_id in schema_ids:
            node = self.graph.get(schema_id)
            if node is None:
                raise KeyError(f"Unknown schema node: {schema_id}")
            base = self.config.positive_delta if positive else self.config.negative_delta
            amount = min(0.95, base * (1.0 + self.config.level_sensitivity * node.level))
            if positive:
                node.reliability_weight += amount * (1.0 - node.reliability_weight)
                self._append_unique(node.memory_index.positive, memory_id)
                node.memory_index.negative = [item for item in node.memory_index.negative if item != memory_id]
            else:
                node.reliability_weight -= amount * node.reliability_weight
                self._append_unique(node.memory_index.negative, memory_id)
                node.memory_index.positive = [item for item in node.memory_index.positive if item != memory_id]
            node.reliability_weight = min(1.0, max(0.0, node.reliability_weight))
            node.updated_at = utc_now()
            if node.status == SchemaStatus.MASKED and node.reliability_weight >= self.config.mask_threshold:
                node.status = SchemaStatus.ACTIVE
        self._persist()

    def apply_forgetting(self, *, now: Optional[datetime] = None) -> List[str]:
        """Decay stale rules and mask/deprecate weak ones; raw memory is retained."""
        current = now or datetime.now(timezone.utc)
        changed: List[str] = []
        for node in self.graph.list(include_inactive=True):
            if node.status == SchemaStatus.DEPRECATED:
                continue
            last_used = self._parse_time(node.last_accessed_at)
            age_days = max(0.0, (current - last_used).total_seconds() / 86400.0)
            if age_days <= 0:
                continue
            neighbor_factor = self.config.isolation_penalty if not node.related_schema_index.all() else 1.0
            evidence_factor = 1.0 / math.sqrt(max(1, len(node.memory_index.all())))
            decay = math.exp(-self.config.daily_decay * age_days * neighbor_factor * evidence_factor)
            old_weight = node.reliability_weight
            node.reliability_weight *= decay
            if node.reliability_weight < self.config.deprecate_threshold:
                node.status = SchemaStatus.DEPRECATED
            elif node.reliability_weight < self.config.mask_threshold:
                node.status = SchemaStatus.MASKED
            if node.reliability_weight != old_weight:
                node.updated_at = utc_now()
                changed.append(node.index)
        self._persist()
        return changed

    def update_task_mask(self, query: str, *, min_relevance: float = 0.15) -> List[str]:
        """Mask rules unrelated to the current task and reactivate relevant ones.

        Deprecation is never reversed here; it requires new supporting feedback.
        This operation is explicit because task masks are policy state, unlike a
        read-only retrieval call.
        """
        if not 0.0 <= min_relevance <= 1.0:
            raise ValueError("min_relevance must be between 0 and 1")
        query_vector = self._encode(query) if self.embedder is not None and query.strip() else None
        masked: List[str] = []
        for node in self.graph.list(include_inactive=True):
            if node.status == SchemaStatus.DEPRECATED:
                continue
            if query_vector is not None:
                relevance = max(0.0, float(self._encode(node.text_for_retrieval()) @ query_vector))
            else:
                relevance = self._lexical_similarity(query, node.text_for_retrieval())
            if relevance < min_relevance:
                node.status = SchemaStatus.MASKED
                masked.append(node.index)
            elif node.reliability_weight >= self.config.mask_threshold:
                node.status = SchemaStatus.ACTIVE
        self._persist()
        return masked

    def run_maintenance(self, *, now: Optional[datetime] = None) -> Dict[str, object]:
        """Entry point suitable for an external periodic scheduler."""
        decayed = self.apply_forgetting(now=now)
        merged = self.consolidate()
        return {"decayed": decayed, "merged": merged}

    def consolidate(self) -> List[tuple[str, str]]:
        """Merge highly similar same-level nodes and preserve all evidence links."""
        active = self.graph.list()
        merged: List[tuple[str, str]] = []
        consumed: set[str] = set()
        for index, left in enumerate(active):
            if left.index in consumed:
                continue
            for right in active[index + 1 :]:
                if right.index in consumed or left.level != right.level:
                    continue
                similarity = self._similarity(left, right)
                if similarity < self.config.duplicate_threshold:
                    continue
                self._merge_nodes(left, right)
                consumed.add(right.index)
                merged.append((left.index, right.index))
        self._persist()
        return merged

    def _apply_proposal(self, proposal: SchemaProposal, record: MemoryRecord) -> Optional[SchemaNode]:
        if proposal.operation == "skip":
            return None
        if proposal.operation == "merge":
            node = self.graph.get(proposal.target_schema_id or "")
            if node is None:
                return None
            self._append_unique(node.memory_index.source, record.id)
            if proposal.trigger:
                node.trigger = proposal.trigger
            if proposal.action_sequence:
                node.action_sequence = proposal.action_sequence
            if proposal.expectation:
                node.expectation = proposal.expectation
            node.description = self._format_description(node)
            node.updated_at = utc_now()
        else:
            if not (proposal.trigger and proposal.action_sequence and proposal.expectation):
                return None
            node = SchemaNode.from_rule(
                level=proposal.level,
                trigger=proposal.trigger,
                action_sequence=proposal.action_sequence,
                expectation=proposal.expectation,
                source_memory_id=record.id,
            )
            node.metadata["induction_rationale"] = proposal.rationale
            self.graph.add(node)

        for parent_id in proposal.parent_ids:
            parent = self.graph.get(parent_id)
            if parent and parent.level < node.level:
                self.graph.connect_parent_child(
                    parent_id, node.index, evidence_memory_id=record.id
                )
        for similar_id in proposal.similar_ids:
            similar = self.graph.get(similar_id)
            if similar and similar_id != node.index and similar.level == node.level:
                self.graph.connect_similar(
                    similar_id, node.index, evidence_memory_id=record.id
                )
        self._embeddings.pop(node.index, None)
        return node

    def _merge_nodes(self, survivor: SchemaNode, duplicate: SchemaNode) -> None:
        if self.generator is not None:
            try:
                survivor.description = self.generator.merge_description(
                    survivor, duplicate
                )
            except Exception:
                if len(duplicate.description) > len(survivor.description):
                    survivor.description = duplicate.description
        elif len(duplicate.description) > len(survivor.description):
            survivor.description = duplicate.description
        left_evidence = max(1, len(survivor.memory_index.all()))
        right_evidence = max(1, len(duplicate.memory_index.all()))
        survivor.reliability_weight = (
            survivor.reliability_weight * left_evidence
            + duplicate.reliability_weight * right_evidence
        ) / (left_evidence + right_evidence)
        for field_name in ("source", "positive", "negative"):
            combined = sorted(
                {
                    *getattr(survivor.memory_index, field_name),
                    *getattr(duplicate.memory_index, field_name),
                }
            )
            setattr(survivor.memory_index, field_name, combined)
        for relation in ("parents", "children", "similar"):
            combined = sorted(
                {
                    *getattr(survivor.related_schema_index, relation),
                    *getattr(duplicate.related_schema_index, relation),
                }
                - {survivor.index, duplicate.index}
            )
            setattr(survivor.related_schema_index, relation, combined)
        for neighbor_id, memory_ids in duplicate.related_schema_index.evidence.items():
            combined_evidence = survivor.related_schema_index.evidence.setdefault(
                neighbor_id, []
            )
            for memory_id in memory_ids:
                self._append_unique(combined_evidence, memory_id)
        self.graph.replace_references(duplicate.index, survivor.index)
        self.graph.remove(duplicate.index)
        survivor.updated_at = utc_now()
        self._embeddings.pop(survivor.index, None)
        self._embeddings.pop(duplicate.index, None)
        self.graph.validate()

    def _similarity(self, left: SchemaNode, right: SchemaNode) -> float:
        if self.embedder is not None:
            return float(self._encode(left.text_for_retrieval()) @ self._encode(right.text_for_retrieval()))
        return SequenceMatcher(None, left.description.lower(), right.description.lower()).ratio()

    def _encode(self, text: str) -> np.ndarray:
        assert self.embedder is not None
        value = self.embedder.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        vector = np.asarray(value[0] if getattr(value, "ndim", 1) == 2 else value, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector

    @staticmethod
    def _fallback_proposal(record: MemoryRecord) -> SchemaProposal:
        if not record.events:
            return SchemaProposal(operation="skip", rationale="No transition evidence")
        actions = [event.action for event in record.events if event.action]
        expectation = record.outcome or record.events[-1].result
        if not actions or not expectation:
            return SchemaProposal(operation="skip", rationale="Incomplete transition evidence")
        try:
            level = max(0, int(record.metadata.get("schema_level", 2)))
        except (TypeError, ValueError):
            level = 2
        return SchemaProposal(
            operation="create",
            level=level,
            trigger=record.context or record.events[0].observation or record.task,
            action_sequence=actions,
            expectation=expectation,
            rationale="Deterministic fallback; configure LLMSchemaGenerator for semantic induction",
        )

    @staticmethod
    def _format_description(node: SchemaNode) -> str:
        return (
            f"[Context/Perception: {node.trigger}] "
            f"[Action/Execution: {' -> '.join(node.action_sequence)}] "
            f"[Expectation: {node.expectation}]"
        )

    @staticmethod
    def _append_unique(values: List[str], item: str) -> None:
        if item not in values:
            values.append(item)

    @staticmethod
    def _lexical_similarity(query: str, document: str) -> float:
        tokens = lambda value: set(
            re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", value.lower())
        )
        query_tokens = tokens(query)
        if not query_tokens:
            return 0.0
        return len(query_tokens & tokens(document)) / len(query_tokens)

    @staticmethod
    def _parse_time(value: str) -> datetime:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)

    def _persist(self) -> None:
        self.graph.validate(
            memory_ids={record.id for record in self.memory.store.list()}
        )
        if self.storage is not None:
            self.storage.save(self.graph)
