from __future__ import annotations

import json
from typing import Iterable

from .base import Problem


def load_prepared_jsonl(path: str) -> Iterable[Problem]:
    """Load CL-Bench (long-context reading comprehension) prepared jsonl.

    Keeps problems that have either a gold answer or rubrics (or both).
    Only skips problems with neither (true noise, should not exist in practice).
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            context = obj.get("context", "")
            question = obj.get("question", "")
            answer = obj.get("answer", "")
            rubrics = obj.get("rubrics") or []

            # Skip only if neither answer nor rubrics exist
            if not str(answer).strip() and not rubrics:
                continue

            # Use question (or first 2000 chars of context+question) for retrieval
            # because the full context can be very long and dilute the embedding.
            retrieval_query = question[:2000] if question else (context + " " + question)[:2000]

            # Full prompt for LLM answering: context + question
            prompt = f"{context}\n\n{question}" if context else question

            yield Problem(
                id=str(obj["id"]),
                prompt=prompt,
                problem_type="long_context",
                retrieval_query=retrieval_query,
                meta={
                    "answer_key": str(answer).strip(),
                    "context": context,
                    "question": question,
                    "rubrics": rubrics,
                    "context_id": (obj.get("meta") or {}).get("context_id", ""),
                    **(obj.get("meta") or {}),
                },
            )
