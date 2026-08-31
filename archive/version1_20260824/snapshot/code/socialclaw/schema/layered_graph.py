from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional

from ..memory.models import utc_now
from .node import SchemaNode, SchemaStatus


class LayeredSchemaGraph:
    """Invariant-preserving graph for the new multi-level schema model."""

    def __init__(self, nodes: Iterable[SchemaNode] = ()) -> None:
        self.nodes: Dict[str, SchemaNode] = {}
        for node in nodes:
            self.add(node)

    def add(self, node: SchemaNode) -> SchemaNode:
        if node.index in self.nodes:
            raise ValueError(f"Schema node already exists: {node.index}")
        self.nodes[node.index] = node
        return node

    def get(self, schema_id: str) -> Optional[SchemaNode]:
        return self.nodes.get(schema_id)

    def list(self, *, include_inactive: bool = False) -> List[SchemaNode]:
        values = list(self.nodes.values())
        if include_inactive:
            return values
        return [node for node in values if node.status == SchemaStatus.ACTIVE]

    def connect_parent_child(
        self,
        parent_id: str,
        child_id: str,
        *,
        evidence_memory_id: Optional[str] = None,
    ) -> None:
        parent = self._require(parent_id)
        child = self._require(child_id)
        if parent_id == child_id:
            raise ValueError("A schema node cannot be its own parent")
        if parent.level >= child.level:
            raise ValueError("Parent schema must be more general (have a smaller level)")
        self._append_unique(parent.related_schema_index.children, child_id)
        self._append_unique(child.related_schema_index.parents, parent_id)
        self._add_link_evidence(parent, child_id, evidence_memory_id)
        self._add_link_evidence(child, parent_id, evidence_memory_id)
        parent.updated_at = child.updated_at = utc_now()

    def connect_similar(
        self,
        left_id: str,
        right_id: str,
        *,
        evidence_memory_id: Optional[str] = None,
    ) -> None:
        left = self._require(left_id)
        right = self._require(right_id)
        if left_id == right_id:
            return
        if left.level != right.level:
            raise ValueError("Similar schema nodes must be on the same level")
        self._append_unique(left.related_schema_index.similar, right_id)
        self._append_unique(right.related_schema_index.similar, left_id)
        self._add_link_evidence(left, right_id, evidence_memory_id)
        self._add_link_evidence(right, left_id, evidence_memory_id)
        left.updated_at = right.updated_at = utc_now()

    def neighbors(self, schema_id: str) -> List[SchemaNode]:
        node = self._require(schema_id)
        return [self.nodes[item] for item in node.related_schema_index.all() if item in self.nodes]

    def remove(self, schema_id: str) -> bool:
        if schema_id not in self.nodes:
            return False
        del self.nodes[schema_id]
        for node in self.nodes.values():
            links = node.related_schema_index
            links.parents = [item for item in links.parents if item != schema_id]
            links.children = [item for item in links.children if item != schema_id]
            links.similar = [item for item in links.similar if item != schema_id]
            links.evidence.pop(schema_id, None)
        return True

    def replace_references(self, old_id: str, new_id: str) -> None:
        """Redirect all graph links during node consolidation."""
        self._require(new_id)
        for node in self.nodes.values():
            for relation in ("parents", "children", "similar"):
                values = getattr(node.related_schema_index, relation)
                redirected = [new_id if item == old_id else item for item in values]
                setattr(
                    node.related_schema_index,
                    relation,
                    sorted({item for item in redirected if item != node.index}),
                )
            old_evidence = node.related_schema_index.evidence.pop(old_id, [])
            if old_evidence and new_id != node.index:
                existing = node.related_schema_index.evidence.setdefault(new_id, [])
                for memory_id in old_evidence:
                    self._append_unique(existing, memory_id)

    def validate(self, *, memory_ids: Optional[set[str]] = None) -> None:
        for node in self.nodes.values():
            links = node.related_schema_index
            for neighbor_id in links.all():
                if neighbor_id not in self.nodes:
                    raise ValueError(f"Schema {node.index} links to missing node {neighbor_id}")
            stale_evidence = set(links.evidence) - set(links.all())
            if stale_evidence:
                raise ValueError(
                    f"Schema {node.index} has evidence for non-neighbors: "
                    f"{sorted(stale_evidence)}"
                )
            for parent_id in links.parents:
                parent = self.nodes[parent_id]
                if node.index not in parent.related_schema_index.children:
                    raise ValueError(f"Asymmetric parent link: {parent_id} -> {node.index}")
                if parent.level >= node.level:
                    raise ValueError(f"Invalid layer order: {parent_id} -> {node.index}")
            for child_id in links.children:
                child = self.nodes[child_id]
                if node.index not in child.related_schema_index.parents:
                    raise ValueError(f"Asymmetric child link: {node.index} -> {child_id}")
            for similar_id in links.similar:
                if node.index not in self.nodes[similar_id].related_schema_index.similar:
                    raise ValueError(f"Asymmetric similarity link: {node.index} <-> {similar_id}")
                if node.level != self.nodes[similar_id].level:
                    raise ValueError(f"Cross-level similarity link: {node.index} <-> {similar_id}")
            if memory_ids is not None:
                missing = set(node.memory_index.all()) - memory_ids
                if missing:
                    raise ValueError(f"Schema {node.index} cites missing memories: {sorted(missing)}")
                link_memories = {
                    memory_id
                    for values in links.evidence.values()
                    for memory_id in values
                }
                missing_link_memories = link_memories - memory_ids
                if missing_link_memories:
                    raise ValueError(
                        f"Schema {node.index} link cites missing memories: "
                        f"{sorted(missing_link_memories)}"
                    )

    def lexical_candidates(self, query: str, *, top_k: int = 5) -> List[SchemaNode]:
        query_tokens = self._tokens(query)
        scored = []
        for node in self.list():
            node_tokens = self._tokens(node.text_for_retrieval())
            overlap = len(query_tokens & node_tokens) / max(1, len(query_tokens))
            score = 0.75 * overlap + 0.25 * node.reliability_weight
            scored.append((score, node))
        scored.sort(key=lambda item: (item[0], -item[1].level), reverse=True)
        return [node for _, node in scored[:top_k]]

    def _require(self, schema_id: str) -> SchemaNode:
        node = self.nodes.get(schema_id)
        if node is None:
            raise KeyError(f"Unknown schema node: {schema_id}")
        return node

    @staticmethod
    def _append_unique(values: List[str], item: str) -> None:
        if item not in values:
            values.append(item)

    @classmethod
    def _add_link_evidence(
        cls,
        node: SchemaNode,
        neighbor_id: str,
        memory_id: Optional[str],
    ) -> None:
        if not memory_id:
            return
        cls._append_unique(
            node.related_schema_index.evidence.setdefault(neighbor_id, []),
            memory_id,
        )

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", text.lower()))
