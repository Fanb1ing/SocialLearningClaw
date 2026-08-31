from __future__ import annotations

import json
import os
from pathlib import Path

from .layered_graph import LayeredSchemaGraph
from .node import SchemaNode


class LayeredSchemaStorage:
    """Atomic JSON persistence for a layered schema graph."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def save(self, graph: LayeredSchemaGraph) -> None:
        graph.validate()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        payload = {
            "format_version": 1,
            "nodes": [node.to_dict() for node in graph.list(include_inactive=True)],
        }
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)

    def load(self) -> LayeredSchemaGraph:
        if not self.path.exists():
            return LayeredSchemaGraph()
        with self.path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("format_version") != 1:
            raise ValueError(f"Unsupported layered schema format in {self.path}")
        graph = LayeredSchemaGraph(
            SchemaNode.from_dict(item) for item in payload.get("nodes", [])
        )
        graph.validate()
        return graph
