from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np

from .graph import Concept, Relation, SchemaGraph


class SchemaStorage:
    def __init__(
        self,
        concepts_path: str,
        relations_path: str,
        embeddings_path: str,
        concept_ids_path: str,
    ):
        self.concepts_path = concepts_path
        self.relations_path = relations_path
        self.embeddings_path = embeddings_path
        self.concept_ids_path = concept_ids_path

    def save(self, graph: SchemaGraph, embeddings: Optional[Dict[str, np.ndarray]] = None) -> None:
        os.makedirs(os.path.dirname(self.concepts_path), exist_ok=True)

        # Concepts (with dynamic neighbor info from relations)
        with open(self.concepts_path, "w", encoding="utf-8") as f:
            for c in graph.list_concepts():
                d = _concept_to_dict(c)
                d["neighbors"] = graph.get_neighbors(c.id)
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

        # Relations
        with open(self.relations_path, "w", encoding="utf-8") as f:
            for r in graph.list_relations():
                f.write(json.dumps(_relation_to_dict(r), ensure_ascii=False) + "\n")

        # Embeddings
        if embeddings:
            ids = list(embeddings.keys())
            mat = np.stack([embeddings[cid] for cid in ids], axis=0)
            np.save(self.embeddings_path, mat)
            with open(self.concept_ids_path, "w", encoding="utf-8") as f:
                json.dump(ids, f)

    def load(self) -> tuple[SchemaGraph, Dict[str, np.ndarray]]:
        graph = SchemaGraph()

        if os.path.exists(self.concepts_path):
            with open(self.concepts_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    c = _dict_to_concept(json.loads(line))
                    graph.add_concept(c)

        if os.path.exists(self.relations_path):
            with open(self.relations_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = _dict_to_relation(json.loads(line))
                    graph.add_relation(r)

        embeddings: Dict[str, np.ndarray] = {}
        if os.path.exists(self.embeddings_path) and os.path.exists(self.concept_ids_path):
            mat = np.load(self.embeddings_path)
            with open(self.concept_ids_path, "r", encoding="utf-8") as f:
                ids = json.load(f)
            if mat.ndim != 2:
                raise ValueError(f"Expected a 2D embedding matrix, got shape={mat.shape}")
            if len(ids) != len(mat):
                raise ValueError(
                    f"Schema embedding index mismatch: {len(ids)} ids for {len(mat)} vectors"
                )
            for i, cid in enumerate(ids):
                if graph.get_concept(cid) is not None:
                    embeddings[cid] = mat[i]

        return graph, embeddings

    def exists(self) -> bool:
        return os.path.exists(self.concepts_path)


def _concept_to_dict(c: Concept) -> dict:
    return {
        "id": c.id,
        "name": c.name,
        "description": c.description,
        "category": c.category,
        "confidence": c.confidence,
        "source": c.source,
        "created_at": c.created_at,
        "neighbors": c.neighbors,
    }


def _dict_to_concept(d: dict) -> Concept:
    return Concept(
        id=d["id"],
        name=d["name"],
        description=d["description"],
        category=d.get("category", "general"),
        confidence=d.get("confidence", 0.5),
        source=d.get("source", "agent_init"),
        created_at=d.get("created_at", ""),
        neighbors=d.get("neighbors", []),
    )


def _relation_to_dict(r: Relation) -> dict:
    return {
        "source": r.source,
        "target": r.target,
        "relation_type": r.relation_type,
        "weight": r.weight,
        "evidence": r.evidence,
    }


def _dict_to_relation(d: dict) -> Relation:
    return Relation(
        source=d["source"],
        target=d["target"],
        relation_type=d.get("relation_type", "related"),
        weight=d.get("weight", 0.5),
        evidence=d.get("evidence", []),
    )
