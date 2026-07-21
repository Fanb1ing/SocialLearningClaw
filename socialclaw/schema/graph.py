from __future__ import annotations

import difflib
import math
import uuid
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Predefined relation type aliases for fuzzy matching
_RELATION_TYPE_ALIASES: Dict[str, str] = {
    "continuously_run_along": "located_at",
    "runs_along": "located_at",
    "situated_at": "located_at",
    "found_at": "located_at",
    "of": "part_of",
    "belongs_to": "part_of",
    "is_a": "part_of",
    "type_of": "part_of",
    "leads_to": "causes",
    "results_in": "causes",
    "produces": "causes",
    "requires": "prerequisite",
    "needs": "prerequisite",
    "depends_on": "prerequisite",
    "similar_to": "analogous",
    "like": "analogous",
    "equivalent_to": "analogous",
}

_SIMILARITY_THRESHOLD = 0.75


def _normalize_relation_type(rel_type: str) -> str:
    """Normalize relation type to lowercase with underscores."""
    return rel_type.strip().lower().replace(" ", "_").replace("-", "_")


def _match_relation_type(query: str, candidate: str) -> bool:
    """Fuzzy match relation types: exact → alias → similarity."""
    q = _normalize_relation_type(query)
    c = _normalize_relation_type(candidate)

    if q == c:
        return True

    # Check alias mapping (both directions)
    if _RELATION_TYPE_ALIASES.get(q) == c or _RELATION_TYPE_ALIASES.get(c) == q:
        return True

    # String similarity fallback
    if difflib.SequenceMatcher(None, q, c).ratio() >= _SIMILARITY_THRESHOLD:
        return True

    # Substring containment (e.g. "located_at" contains "locate")
    if len(q) >= 4 and len(c) >= 4:
        if q in c or c in q:
            return True

    return False


@dataclass
class Concept:
    id: str
    name: str
    description: str
    category: str = "general"
    confidence: float = 0.5
    source: str = "agent_init"
    created_at: str = ""
    neighbors: List[str] = field(default_factory=list)


@dataclass
class Relation:
    source: str
    target: str
    relation_type: str = "related"
    weight: float = 0.5
    evidence: List[dict] = field(default_factory=list)


@dataclass
class ReasoningTrace:
    concepts: List[str]
    relations: List[Tuple[str, str, str]]
    explanation: str = ""


class SchemaGraph:
    def __init__(
        self,
        concepts: Optional[Dict[str, Concept]] = None,
        relations: Optional[List[Relation]] = None,
    ):
        self.concepts: Dict[str, Concept] = concepts or {}
        self.relations: List[Relation] = relations or []

    def add_concept(self, c: Concept) -> None:
        if not c.id:
            c.id = f"concept_{uuid.uuid4().hex[:8]}"
        existing = self.concepts.get(c.id)
        if existing is not None:
            # Observed ARC objects intentionally reuse stable ids across frames.
            # Preserve learned state while refreshing perceptual attributes.
            c.confidence = existing.confidence
            c.created_at = existing.created_at or c.created_at
            c.neighbors = sorted({*existing.neighbors, *c.neighbors})
        self.concepts[c.id] = c

    def add_relation(self, r: Relation) -> None:
        for existing in self.relations:
            if (
                existing.source == r.source
                and existing.target == r.target
                and _normalize_relation_type(existing.relation_type)
                == _normalize_relation_type(r.relation_type)
            ):
                existing.evidence.extend(
                    item for item in r.evidence if item not in existing.evidence
                )
                return
        self.relations.append(r)

    def get_concept(self, cid: str) -> Optional[Concept]:
        return self.concepts.get(cid)

    def get_concept_by_name(self, name: str) -> Optional[Concept]:
        name = name.strip()
        if not name:
            return None
        # Exact match first
        for c in self.concepts.values():
            if c.name == name:
                return c
        # Case-insensitive match
        name_lower = name.lower()
        for c in self.concepts.values():
            if c.name.lower() == name_lower:
                return c
        # Substring containment (if one is significantly shorter, the longer may contain the shorter)
        for c in self.concepts.values():
            c_lower = c.name.lower()
            if c_lower in name_lower or name_lower in c_lower:
                return c
        # Fuzzy similarity fallback
        for c in self.concepts.values():
            if difflib.SequenceMatcher(None, c.name.lower(), name_lower).ratio() >= _SIMILARITY_THRESHOLD:
                return c
        return None

    def get_relation(
        self, source: str, target: str, relation_type: str
    ) -> Optional[Relation]:
        for r in self.relations:
            if r.source == source and r.target == target and _match_relation_type(r.relation_type, relation_type):
                return r
        return None

    def find_relation(
        self, source_name: str, target_name: str, relation_type: str
    ) -> Optional[Relation]:
        """Find relation by concept names (more flexible than id match)."""
        src = self.get_concept_by_name(source_name)
        tgt = self.get_concept_by_name(target_name)
        if src and tgt:
            return self.get_relation(src.id, tgt.id, relation_type)
        return None

    def remove_concept(self, cid: str) -> bool:
        if cid not in self.concepts:
            return False
        del self.concepts[cid]
        self.relations = [
            r for r in self.relations if r.source != cid and r.target != cid
        ]
        return True

    def update_concept(self, cid: str, **kwargs) -> bool:
        c = self.concepts.get(cid)
        if not c:
            return False
        for k, v in kwargs.items():
            if hasattr(c, k):
                setattr(c, k, v)
        return True

    def update_relation(
        self, source: str, target: str, relation_type: str, **kwargs
    ) -> bool:
        r = self.get_relation(source, target, relation_type)
        if not r:
            return False
        for k, v in kwargs.items():
            if hasattr(r, k):
                setattr(r, k, v)
        return True

    def subgraph(self, concept_ids: List[str], depth: int = 1) -> "SchemaGraph":
        """Extract a subgraph centered around given concept ids."""
        ids = set(concept_ids)
        frontier = set(ids)
        for _ in range(depth):
            new_frontier = set(frontier)
            for r in self.relations:
                if r.source in frontier:
                    new_frontier.add(r.target)
                if r.target in frontier:
                    new_frontier.add(r.source)
            frontier = new_frontier

        concepts = {cid: c for cid, c in self.concepts.items() if cid in frontier}
        relations = [
            r for r in self.relations if r.source in frontier and r.target in frontier
        ]
        return SchemaGraph(concepts=concepts, relations=relations)

    def compute_confidence(self, trace: ReasoningTrace) -> float:
        """Compute schema-based reasoning confidence from a trace.

        Uses geometric mean of concept confidences and relation weights.
        Matches by concept id first, then falls back to name matching.
        """
        concept_scores = []
        for cid in trace.concepts:
            c = self.get_concept(cid)
            if not c:
                c = self.get_concept_by_name(cid)
            if c:
                concept_scores.append(c.confidence)
            else:
                warnings.warn(f"Concept {cid!r} referenced in trace but not in schema.")

        relation_scores = []
        for src, tgt, rel_type in trace.relations:
            r = self.get_relation(src, tgt, rel_type)
            if not r:
                r = self.find_relation(src, tgt, rel_type)
            if r:
                relation_scores.append(r.weight)
            else:
                warnings.warn(
                    f"Relation ({src!r}, {tgt!r}, {rel_type!r}) referenced in trace but not in schema."
                )

        if not concept_scores:
            return 0.0

        concept_geom = math.prod(concept_scores) ** (1 / len(concept_scores))
        relation_geom = (
            math.prod(relation_scores) ** (1 / len(relation_scores))
            if relation_scores
            else 1.0
        )

        return concept_geom * relation_geom

    def get_neighbors(self, cid: str) -> List[str]:
        """Return neighbor concept ids for a given concept id (dynamic from relations)."""
        nbrs: set = set()
        for r in self.relations:
            if r.source == cid:
                nbrs.add(r.target)
            if r.target == cid:
                nbrs.add(r.source)
        return sorted(nbrs)

    def list_concepts(self) -> List[Concept]:
        return list(self.concepts.values())

    def list_relations(self) -> List[Relation]:
        return list(self.relations)
