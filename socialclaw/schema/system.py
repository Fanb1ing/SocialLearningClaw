from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..llm import OpenAIChatClient
from ..memory import Embedder, JsonMemoryStore, MemoryBank
from .induction import LLMSchemaGenerator
from .layered_storage import LayeredSchemaStorage
from .manager import SchemaManagementConfig, SchemaManager


def build_schema_system(
    root: str | Path,
    *,
    llm: Optional[OpenAIChatClient] = None,
    embedder: Optional[Embedder] = None,
    config: SchemaManagementConfig = SchemaManagementConfig(),
) -> SchemaManager:
    """Build the complete memory -> schema stack from replaceable components."""
    root_path = Path(root)
    memory_store = JsonMemoryStore(root_path / "memory.json")
    memory = MemoryBank(memory_store, embedder=embedder)
    schema_storage = LayeredSchemaStorage(root_path / "schema.json")
    graph = schema_storage.load()
    graph.validate(memory_ids={record.id for record in memory_store.list()})
    generator = LLMSchemaGenerator(llm) if llm is not None else None
    return SchemaManager(
        memory=memory,
        graph=graph,
        generator=generator,
        embedder=embedder,
        storage=schema_storage,
        config=config,
    )
