from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


BASELINES = (
    "naive",
    "icl",
    "rag",
    "withrule",
    "reflexion",
    "expel",
    "amem",
    "tgm",
)
METHODS = BASELINES + ("schema",)
BENCHMARKS = ("arc_agi3", "contextmath", "intphys2")


@dataclass(frozen=True)
class ExperimentBudget:
    """Comparable resource limits shared by every method in an experiment."""

    max_samples: int = 0
    max_attempts: int = 1
    max_steps: int = 200
    max_tokens_per_call: int = 8192

    def validate(self) -> None:
        if self.max_samples < 0:
            raise ValueError("max_samples must be >= 0 (0 means all samples)")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if self.max_tokens_per_call < 1:
            raise ValueError("max_tokens_per_call must be >= 1")


@dataclass(frozen=True)
class ExperimentConfig:
    benchmark: str
    method: str
    model: str
    split: str = ""
    seed: int = 0
    temperature: float = 0.0
    base_url: str = "https://openrouter.ai/api/v1"
    output_root: str = "outputs"
    feedback: str = "binary"
    budget: ExperimentBudget = field(default_factory=ExperimentBudget)
    extra: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.benchmark not in BENCHMARKS:
            raise ValueError(f"Unknown benchmark {self.benchmark!r}; choose from {BENCHMARKS}")
        if self.method not in METHODS:
            raise ValueError(f"Unknown method {self.method!r}; choose from {METHODS}")
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if self.feedback not in {"none", "binary"}:
            raise ValueError("feedback must be 'none' or 'binary'; gold-answer feedback is forbidden")
        self.budget.validate()


@dataclass
class AttemptResult:
    attempt: int
    prediction: Any
    correct: bool
    response: str
    usage: Dict[str, int] = field(default_factory=dict)


@dataclass
class SampleResult:
    sample_id: str
    correct: bool
    prediction: Any
    attempts: List[AttemptResult]
    metadata: Dict[str, Any] = field(default_factory=dict)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def make_output_dir(config: ExperimentConfig) -> Path:
    config.validate()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model = config.model.replace("/", "--").replace(":", "-").replace(" ", "_")
    split = config.split or "default"
    path = Path(config.output_root) / config.benchmark / config.method / model / split / timestamp
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_manifest(
    output_dir: Path,
    config: ExperimentConfig,
    *,
    sample_ids: Iterable[str],
    dataset_fingerprint: str = "",
    demonstration_ids: Iterable[str] = (),
) -> Path:
    payload = {
        "format_version": 1,
        "created_at": utc_now(),
        "git_commit": git_commit(),
        "config": asdict(config),
        "sample_ids": list(sample_ids),
        "demonstration_ids": list(demonstration_ids),
        "dataset_fingerprint": dataset_fingerprint,
    }
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def write_results(output_dir: Path, results: List[SampleResult]) -> Path:
    total = len(results)
    correct = sum(bool(r.correct) for r in results)
    first_attempt_correct = sum(bool(r.attempts and r.attempts[0].correct) for r in results)
    payload = {
        "format_version": 1,
        "created_at": utc_now(),
        "metrics": {
            "accuracy": correct / total if total else 0.0,
            "first_attempt_accuracy": first_attempt_correct / total if total else 0.0,
            "correct": correct,
            "total": total,
        },
        "samples": [asdict(r) for r in results],
    }
    path = output_dir / "results.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
