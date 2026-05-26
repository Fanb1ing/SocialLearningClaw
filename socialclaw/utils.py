from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from .schema.graph import Concept, Relation, SchemaGraph

_CST = timezone(timedelta(hours=8))


def load_dotenv(dotenv_path: str) -> None:
    """Load KEY=VALUE pairs from a .env file into os.environ."""
    if not os.path.exists(dotenv_path):
        return
    with open(dotenv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


def make_run_dir(runs_dir: str, benchmark: str, model: str) -> str:
    """Create and return a timestamped run directory.

    Path: {runs_dir}/{benchmark}/{model_sanitized}/{YYYYMMDD_HHMMSS}
    Timestamp is China Standard Time (CST, UTC+8).
    """
    ts = datetime.now(tz=_CST).strftime("%Y%m%d_%H%M%S")
    model_sanitized = model.replace("/", "--").replace(":", "-").replace(" ", "_")
    run_dir = os.path.join(runs_dir, benchmark, model_sanitized, ts)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_cmd(run_dir: str) -> None:
    """Write the full command line to {run_dir}/cmd.txt."""
    cmd_path = os.path.join(run_dir, "cmd.txt")
    with open(cmd_path, "w", encoding="utf-8") as f:
        f.write(" ".join(sys.argv) + "\n")


def add_concepts_with_embeddings(
    graph: SchemaGraph,
    embeddings: Dict,
    embedder,
    concepts: List[Concept],
) -> None:
    """Add concepts to graph and compute their embeddings."""
    for c in concepts:
        graph.add_concept(c)
        try:
            emb = embedder.encode(
                f"{c.name}: {c.description}",
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            if emb.ndim == 2:
                emb = emb[0]
            embeddings[c.id] = emb
        except Exception:
            pass


def resolve_relation_names(
    graph: SchemaGraph, relation: Relation
) -> Optional[Relation]:
    """Convert relation source/target from concept names to concept ids."""
    src = graph.get_concept_by_name(relation.source)
    tgt = graph.get_concept_by_name(relation.target)
    if src and tgt:
        return Relation(
            source=src.id,
            target=tgt.id,
            relation_type=relation.relation_type,
            weight=relation.weight,
            evidence=relation.evidence,
        )
    return None


def add_relations_resolved(graph: SchemaGraph, relations: List[Relation]) -> None:
    """Add relations to graph, resolving concept names to ids."""
    for r in relations:
        resolved = resolve_relation_names(graph, r)
        if resolved:
            graph.add_relation(resolved)
