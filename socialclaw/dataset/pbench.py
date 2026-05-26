from __future__ import annotations

import json
from typing import Iterable

from .base import Problem


def load_prepared_jsonl(path: str) -> Iterable[Problem]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            yield Problem(
                id=str(obj["id"]),
                prompt=obj["prompt"],
                problem_type="mcq",
                meta={
                    "choices": list(obj.get("choices", [])),
                    "answer_key": str(obj.get("answer_key", "")),
                    **(obj.get("meta") or {}),
                },
            )
