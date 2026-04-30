from __future__ import annotations

import json
from typing import Iterable

from .types import MCQProblem


def load_prepared_jsonl(path: str) -> Iterable[MCQProblem]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            yield MCQProblem(
                id=str(obj["id"]),
                prompt=obj["prompt"],
                choices=list(obj["choices"]),
                answer_key=str(obj["answer_key"]),
                meta=obj.get("meta") or {},
            )
