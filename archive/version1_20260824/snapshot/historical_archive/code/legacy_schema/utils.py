"""Utility helpers used only by the archived single-layer schema runner."""

from __future__ import annotations

from typing import Dict, List, Optional

from .graph import Concept, Relation, SchemaGraph


def add_concepts_with_embeddings(
    graph: SchemaGraph,
    embeddings: Dict,
    embedder,
    concepts: List[Concept],
) -> None:
    for concept in concepts:
        graph.add_concept(concept)
        try:
            embedding = embedder.encode(
                f"{concept.name}: {concept.description}",
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            if embedding.ndim == 2:
                embedding = embedding[0]
            embeddings[concept.id] = embedding
        except Exception:
            pass


def resolve_relation_names(
    graph: SchemaGraph,
    relation: Relation,
) -> Optional[Relation]:
    source = graph.get_concept_by_name(relation.source)
    target = graph.get_concept_by_name(relation.target)
    if source and target:
        return Relation(
            source=source.id,
            target=target.id,
            relation_type=relation.relation_type,
            weight=relation.weight,
            evidence=relation.evidence,
        )
    return None


def add_relations_resolved(
    graph: SchemaGraph,
    relations: List[Relation],
) -> None:
    for relation in relations:
        resolved = resolve_relation_names(graph, relation)
        if resolved:
            graph.add_relation(resolved)
