from __future__ import annotations

import json
from typing import Iterable

from .base import Problem


def load_prepared_jsonl(path: str) -> Iterable[Problem]:
    """Load ARC-AGI prepared jsonl (static interface only for Stage 1).

    Stage 1 only uses the static grid prompt; interactive environment is Stage 2.
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            yield Problem(
                id=str(obj["id"]),
                prompt=obj["prompt"],
                problem_type="arc_grid",
                meta={
                    "answer_key": obj.get("answer_key"),
                    **(obj.get("meta") or {}),
                },
            )
