from __future__ import annotations

from typing import Optional

from ..memory import MemoryRecord
from ..schema import SchemaManager, SchemaNode


class SchemaMethodController:
    """Benchmark-neutral lifecycle adapter for the layered schema system."""

    def __init__(self, manager: SchemaManager, *, retrieve_k: int = 5) -> None:
        self.manager = manager
        self.retrieve_k = retrieve_k

    def context(self, query: str) -> str:
        return self.manager.context_block(query, top_k=self.retrieve_k)

    def after_sample(
        self,
        *,
        task: str,
        response: str,
        correct: bool,
        domain: str,
    ) -> Optional[SchemaNode]:
        """Store one evaluated attempt without exposing its gold answer."""
        outcome = "correct" if correct else "incorrect"
        record = MemoryRecord(
            task=task,
            context=f"benchmark={domain}",
            outcome=f"The model response was evaluated as {outcome}.",
            success=correct,
            feedback=f"binary_outcome={outcome}; no gold answer was provided",
            tags=[domain, outcome],
            metadata={"schema_level": 1, "feedback_kind": "binary"},
        )
        record.add_event(
            observation=task,
            action=response[:8000],
            result=f"binary evaluation: {outcome}",
        )
        self.manager.memory.remember(record)
        try:
            node = self.manager.learn(record.id)
        except Exception as error:
            # The evaluated task result must survive even if an auxiliary
            # induction call fails. A later maintenance/replay pass can retry.
            record.metadata["induction_error"] = type(error).__name__
            self.manager.memory.remember(record)
            return None
        if node is not None:
            self.manager.apply_feedback(
                [node.index], memory_id=record.id, positive=correct
            )
        return node
