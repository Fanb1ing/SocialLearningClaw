from __future__ import annotations

import re
from pathlib import Path
from typing import List

from .base import BenchmarkAdapter, BenchmarkSample, Evaluation


SPLITS = (
    "aime_2024_sg",
    "aime_2024_cs",
    "aime_2025_sg",
    "aime_2025_cs",
    "math_500_sg",
)


def extract_boxed(text: str) -> str | None:
    matches = re.findall(r"\\boxed\{([^}]+)\}", text or "")
    if not matches:
        return None
    raw = matches[-1].strip()
    match = re.match(r"^(-?\d[\d,]*)", raw.replace(" ", ""))
    return match.group(1).replace(",", "") if match else raw


def answers_match(prediction: str | None, gold: str) -> bool:
    if prediction is None:
        return False
    try:
        return int(prediction) == int(str(gold).strip())
    except ValueError:
        pred = prediction.strip().replace(",", "").replace(" ", "").lstrip("0") or "0"
        target = str(gold).strip().replace(",", "").replace(" ", "").lstrip("0") or "0"
        return pred == target


class ContextMathBenchmark(BenchmarkAdapter):
    name = "contextmath"
    default_split = "aime_2024_sg"

    def __init__(self, data_dir: str | Path = "data/contextmath") -> None:
        self.data_dir = Path(data_dir)

    def _path(self, split: str) -> Path:
        if split not in SPLITS:
            raise ValueError(f"Unknown ContextMATH split {split!r}; choose from {SPLITS}")
        direct = self.data_dir / f"{split}.parquet"
        hf_name = self.data_dir / f"{split}-00000-of-00001.parquet"
        if direct.exists():
            return direct
        if hf_name.exists():
            return hf_name
        raise FileNotFoundError(f"ContextMATH split not found: {direct} or {hf_name}")

    def load(self, split: str, max_samples: int = 0) -> List[BenchmarkSample]:
        import pandas as pd

        path = self._path(split)
        rows = pd.read_parquet(path).to_dict("records")
        if max_samples:
            rows = rows[:max_samples]
        return [
            BenchmarkSample(
                id=str(row["id"]),
                prompt=str(row["question"]),
                gold=str(row["answer"]),
                metadata={"split": split, "ori_question": row.get("ori_question", "")},
            )
            for row in rows
        ]

    def load_icl_demonstrations(self, count: int = 3) -> List[BenchmarkSample]:
        """Use a disjoint MATH-500 pool rather than leaking AIME test answers."""
        return self.load("math_500_sg", max_samples=count)

    def evaluate(self, response: str, sample: BenchmarkSample) -> Evaluation:
        prediction = extract_boxed(response)
        return Evaluation(
            correct=answers_match(prediction, str(sample.gold)),
            prediction=prediction,
            details="" if prediction is not None else "missing_boxed_answer",
        )

    def system_prompt(self) -> str:
        return (
            "You are a mathematical reasoning expert. Solve the problem carefully. "
            "Put the final numerical answer in \\boxed{} notation."
        )

    def user_prompt(self, sample: BenchmarkSample) -> str:
        return f"{sample.prompt}\n\nPut your final numerical answer in \\boxed{{}}."

    def rule_context(self) -> str:
        return (
            "Verify that every piece of provided context is applied consistently. "
            "Separate contextual substitutions from the underlying mathematics, perform the "
            "calculation, and independently check the final integer before answering."
        )

    def source_files(self, split: str) -> List[str]:
        return [str(self._path(split))]

