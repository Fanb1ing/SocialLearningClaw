from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .models import MemoryRecord


class MemoryStore:
    """In-memory record store with explicit CRUD semantics."""

    def __init__(self, records: Iterable[MemoryRecord] = ()) -> None:
        self._records: Dict[str, MemoryRecord] = {record.id: record for record in records}

    def put(self, record: MemoryRecord) -> MemoryRecord:
        self._records[record.id] = record
        return record

    def get(self, memory_id: str) -> Optional[MemoryRecord]:
        return self._records.get(memory_id)

    def delete(self, memory_id: str) -> bool:
        return self._records.pop(memory_id, None) is not None

    def list(self) -> List[MemoryRecord]:
        return list(self._records.values())

    def __len__(self) -> int:
        return len(self._records)


class JsonMemoryStore(MemoryStore):
    """Snapshot-backed memory store.

    Writes use ``os.replace`` so an interrupted save cannot leave a partially
    written memory file behind.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        super().__init__()
        self.load()

    def put(self, record: MemoryRecord) -> MemoryRecord:
        result = super().put(record)
        self.save()
        return result

    def delete(self, memory_id: str) -> bool:
        deleted = super().delete(memory_id)
        if deleted:
            self.save()
        return deleted

    def load(self) -> None:
        self._records = {}
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("format_version") != 1:
            raise ValueError(f"Unsupported memory format in {self.path}")
        for item in payload.get("records", []):
            record = MemoryRecord.from_dict(item)
            self._records[record.id] = record

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        payload = {
            "format_version": 1,
            "records": [record.to_dict() for record in self.list()],
        }
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)
