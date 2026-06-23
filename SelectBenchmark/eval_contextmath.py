#!/usr/bin/env python3
"""
Evaluate ContextMATH benchmark.
- claude-opus-4-7: current strongest Claude (not in original paper)
- deepseek/deepseek-r1: paper baseline (paper reports 70.0/66.7/73.3/53.3 on AIME splits)

Usage:
    cd /data5/fanbingbing/SocialLearningClaw
    .venv/bin/python SelectBenchmark/eval_contextmath.py
"""

import os
import re
import json
import time
from pathlib import Path
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

# Models to evaluate
MODELS = {
    "claude-opus-4.7": "anthropic/claude-opus-4.7",
    "deepseek-r1": "deepseek/deepseek-r1",
}

# Paper-reported scores for DeepSeek-R1 (Table 1 in paper, on full 30-problem sets)
PAPER_SCORES = {
    "deepseek-r1": {
        "aime_2024_sg": 70.0,
        "aime_2024_cs": 66.7,
        "aime_2025_sg": 73.3,
        "aime_2025_cs": 53.3,
    }
}

# Local parquet file mapping
DATA_DIR = Path(__file__).parent / "data" / "ContextMATH"
SPLIT_FILES = {
    "aime_2024_sg": DATA_DIR / "aime_2024_sg-00000-of-00001.parquet",
    "aime_2024_cs": DATA_DIR / "aime_2024_cs-00000-of-00001.parquet",
    "aime_2025_sg": DATA_DIR / "aime_2025_sg-00000-of-00001.parquet",
    "aime_2025_cs": DATA_DIR / "aime_2025_cs-00000-of-00001.parquet",
    "math_500_sg":  DATA_DIR / "math_500_sg-00000-of-00001.parquet",
}

SPLITS = ["aime_2024_sg", "aime_2024_cs", "aime_2025_sg", "aime_2025_cs"]
MAX_SAMPLES = 10  # per split (full split = 30 problems each)

SYSTEM_PROMPT = (
    "You are a mathematical reasoning expert. "
    "Solve problems step by step. Always box your final answer using \\boxed{} notation."
)

USER_TEMPLATE = """{question}

Solve this problem carefully. Put your final numerical answer in \\boxed{{}}."""


def extract_boxed(text: str) -> str | None:
    """Extract the last \\boxed{...} answer, pulling out just the leading integer."""
    matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    if not matches:
        return None
    raw = matches[-1].strip()
    # Try to pull leading integer (AIME answers are integers 0-999)
    m = re.match(r"^(-?\d[\d,]*)", raw.replace(" ", ""))
    if m:
        return m.group(1).replace(",", "")
    return raw


def answers_match(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    try:
        return int(pred) == int(gold.strip())
    except ValueError:
        pass
    pred = pred.strip().replace(",", "").replace(" ", "").lstrip("0") or "0"
    gold = gold.strip().replace(",", "").replace(" ", "").lstrip("0") or "0"
    return pred == gold


def call_model(model_id: str, question: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_TEMPLATE.format(question=question)},
                ],
                max_tokens=8192,
                temperature=0.0,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            print(f"    API error (attempt {attempt+1}): {e}")
            if attempt < retries - 1:
                time.sleep(5)
    return ""


def eval_split(model_name: str, model_id: str, split: str) -> dict:
    df = pd.read_parquet(SPLIT_FILES[split])
    samples = df.head(MAX_SAMPLES).to_dict("records")

    records = []
    n_correct = 0

    print(f"\n  [{split}] {len(samples)} samples")
    for i, row in enumerate(samples):
        question = row["question"]
        gold = row["answer"]

        output = call_model(model_id, question)
        pred = extract_boxed(output)
        correct = answers_match(pred, gold)
        n_correct += correct

        records.append({
            "id": row["id"],
            "gold": gold,
            "predicted": pred,
            "correct": correct,
            "output_snippet": output[:300],
        })
        print(f"    [{i+1:02d}/{len(samples)}] {'✓' if correct else '✗'}  pred={pred!r}  gold={gold!r}")
        time.sleep(1)

    accuracy = 100.0 * n_correct / len(samples)
    print(f"  => Accuracy: {accuracy:.1f}%  ({n_correct}/{len(samples)})")
    return {"accuracy": accuracy, "n_correct": n_correct, "n_total": len(samples), "records": records}


def main():
    out_dir = Path(__file__).parent / "results" / "contextmath"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict] = {}

    for model_name, model_id in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_name}  ({model_id})")
        print("="*60)

        model_results: dict[str, dict] = {}
        for split in SPLITS:
            model_results[split] = eval_split(model_name, model_id, split)

        all_results[model_name] = model_results

        with open(out_dir / f"{model_name}.json", "w", encoding="utf-8") as f:
            json.dump(model_results, f, indent=2, ensure_ascii=False)

    # ── Summary table ──────────────────────────────────────────
    print("\n\n" + "="*70)
    print("RESULTS SUMMARY vs PAPER")
    print("="*70)
    header = f"{'Split':<22} {'DeepSeek-R1(paper)':>20} {'DeepSeek-R1(ours)':>20} {'Claude-Opus-4.7(ours)':>22}"
    print(header)
    print("-"*70)
    for split in SPLITS:
        paper = PAPER_SCORES.get("deepseek-r1", {}).get(split)
        paper_str = f"{paper:.1f}%" if paper is not None else "N/A"
        ours_ds = all_results.get("deepseek-r1", {}).get(split, {}).get("accuracy")
        ours_ds_str = f"{ours_ds:.1f}%" if ours_ds is not None else "N/A"
        ours_cl = all_results.get("claude-opus-4.7", {}).get(split, {}).get("accuracy")
        ours_cl_str = f"{ours_cl:.1f}%" if ours_cl is not None else "N/A"
        print(f"{split:<22} {paper_str:>20} {ours_ds_str:>20} {ours_cl_str:>22}")

    summary = {
        "paper_scores": PAPER_SCORES,
        "our_scores": {
            m: {s: all_results[m][s]["accuracy"] for s in SPLITS if s in all_results.get(m, {})}
            for m in MODELS
        },
        "note": (
            "Claude-opus-4.7 not in original paper. "
            "DeepSeek-R1 used to validate setup matches paper numbers (paper: 70.0/66.7/73.3/53.3). "
            f"Each split sampled {MAX_SAMPLES}/30 problems."
        ),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll results saved to {out_dir}/")


if __name__ == "__main__":
    main()
