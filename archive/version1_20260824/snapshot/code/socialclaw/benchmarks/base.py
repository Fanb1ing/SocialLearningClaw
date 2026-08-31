from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence


@dataclass(frozen=True)
class BenchmarkSample:
    id: str
    prompt: str
    gold: Any
    media: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Evaluation:
    correct: bool
    prediction: Any
    details: str = ""


class BenchmarkAdapter(ABC):
    name: str
    default_split: str

    @abstractmethod
    def load(self, split: str, max_samples: int = 0) -> List[BenchmarkSample]:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, response: str, sample: BenchmarkSample) -> Evaluation:
        raise NotImplementedError

    @abstractmethod
    def system_prompt(self) -> str:
        raise NotImplementedError

    def user_prompt(self, sample: BenchmarkSample) -> str:
        return sample.prompt

    def rule_context(self) -> str:
        return ""

    def schema_query(self, sample: BenchmarkSample) -> str:
        """Gold-free text used for schema retrieval and episode memory."""
        return self.user_prompt(sample)

    def format_demonstrations(self, samples: Sequence[BenchmarkSample]) -> str:
        if not samples:
            return ""
        blocks = []
        for i, sample in enumerate(samples, 1):
            blocks.append(f"Example {i}\nProblem: {sample.prompt}\nAnswer: {sample.gold}")
        return "\n\n".join(blocks)

    def fingerprint(self, split: str) -> str:
        """Return a stable fingerprint for the adapter's local source files."""
        h = hashlib.sha256()
        for path in self.source_files(split):
            p = Path(path)
            h.update(str(p).encode())
            if p.exists():
                with p.open("rb") as f:
                    for chunk in iter(lambda: f.read(1024 * 1024), b""):
                        h.update(chunk)
        return h.hexdigest()

    def source_files(self, split: str) -> List[str]:
        return []
