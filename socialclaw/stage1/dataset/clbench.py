from __future__ import annotations

import json
from typing import Iterable

from .base import Problem


def load_prepared_jsonl(path: str) -> Iterable[Problem]:
    """Load CL-Bench (long-context reading comprehension) prepared jsonl."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            context = obj.get("context", "")
            question = obj.get("question", "")
            answer = obj.get("answer", "")

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
                    "answer_key": str(answer) if answer else "",
                    "context": context,
                    "question": question,
                    "rubrics": obj.get("rubrics", ""),
                    **(obj.get("meta") or {}),
                },
            )
