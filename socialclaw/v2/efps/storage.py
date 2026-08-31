from __future__ import annotations

import json
from pathlib import Path

from ...trajectory.corpus import write_json_atomic
from .graph import EFPSGraph


class EFPSGraphStorage:
    """Persist current EFPS state plus an immutable per-revision review snapshot."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.graph_path = self.root / "graph.json"

    def save(self, graph: EFPSGraph, *, label: str = "") -> Path:
        graph.validate()
        payload = graph.to_dict()
        write_json_atomic(self.graph_path, payload)
        suffix = f"_{label}" if label else ""
        snapshot = self.root / "snapshots" / f"revision_{graph.revision:03d}{suffix}.json"
        write_json_atomic(snapshot, payload)
        return snapshot

    def load(self) -> EFPSGraph:
        payload = json.loads(self.graph_path.read_text(encoding="utf-8"))
        graph = EFPSGraph.from_dict(payload)
        graph.validate()
        return graph


__all__ = ["EFPSGraphStorage"]
