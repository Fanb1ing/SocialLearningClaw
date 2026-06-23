#!/usr/bin/env python3
"""
ContextMATH evaluation with claude-opus-4.8 (current strongest Claude).
Mirrors eval_contextmath.py but targets claude-opus-4.8 only.
"""

import os, re, json, time
from pathlib import Path
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

MODEL_NAME = "claude-opus-4.8"
MODEL_ID   = "anthropic/claude-opus-4.8"

PAPER_SCORES_DEEPSEEK = {
    "aime_2024_sg": 70.0,
    "aime_2024_cs": 66.7,
    "aime_2025_sg": 73.3,
    "aime_2025_cs": 53.3,
}
CLAUDE_47_SCORES = {
    "aime_2024_sg": 80.0,
    "aime_2024_cs": 60.0,
    "aime_2025_sg": 90.0,
    "aime_2025_cs": 60.0,
}

SPLITS   = ["aime_2024_sg", "aime_2024_cs", "aime_2025_sg", "aime_2025_cs"]
DATA_DIR = Path(__file__).parent / "data" / "ContextMATH"
MAX_SAMPLES = 10

SYSTEM_PROMPT = (
    "You are a mathematical reasoning expert. "
    "Solve problems step by step. Always box your final answer using \\boxed{} notation."
)
USER_TEMPLATE = """{question}

Solve carefully. Put your final numerical answer in \\boxed{{}}."""


def extract_boxed(text: str):
    matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    if not matches:
        return None
    raw = matches[-1].strip()
    m = re.match(r"^(-?\d[\d,]*)", raw.replace(" ", ""))
    return m.group(1).replace(",", "") if m else raw


def answers_match(pred, gold: str) -> bool:
    if pred is None:
        return False
    try:
        return int(pred) == int(gold.strip())
    except ValueError:
        pass
    return pred.strip() == gold.strip()


def call_model(question: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL_ID,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": USER_TEMPLATE.format(question=question)},
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


def eval_split(split: str) -> dict:
    parquet = DATA_DIR / f"{split}-00000-of-00001.parquet"
    df = pd.read_parquet(parquet)
    samples = df.head(MAX_SAMPLES).to_dict("records")

    records, n_correct = [], 0
    print(f"\n  [{split}] {len(samples)} samples")
    for i, row in enumerate(samples):
        output = call_model(row["question"])
        pred   = extract_boxed(output)
        ok     = answers_match(pred, row["answer"])
        n_correct += ok
        records.append({
            "id": row["id"], "gold": row["answer"],
            "predicted": pred, "correct": ok,
            "output_snippet": output[:300],
        })
        print(f"    [{i+1:02d}/{len(samples)}] {'✓' if ok else '✗'}  pred={pred!r}  gold={row['answer']!r}")
        time.sleep(1)

    acc = 100.0 * n_correct / len(samples)
    print(f"  => Accuracy: {acc:.1f}%  ({n_correct}/{len(samples)})")
    return {"accuracy": acc, "n_correct": n_correct, "n_total": len(samples), "records": records}


def main():
    out_dir = Path(__file__).parent / "results" / "contextmath"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}\nModel: {MODEL_NAME}  ({MODEL_ID})\n{'='*60}")
    results = {}
    for split in SPLITS:
        results[split] = eval_split(split)

    with open(out_dir / f"{MODEL_NAME}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n\n" + "="*75)
    print("RESULTS SUMMARY")
    print("="*75)
    print(f"{'Split':<22} {'DeepSeek-R1(paper)':>20} {'Claude-4.7(ours)':>18} {'Claude-4.8(ours)':>18}")
    print("-"*75)
    for split in SPLITS:
        paper  = PAPER_SCORES_DEEPSEEK.get(split)
        cl47   = CLAUDE_47_SCORES.get(split)
        cl48   = results[split]["accuracy"]
        paper_s = f"{paper:.1f}%" if paper else "N/A"
        cl47_s  = f"{cl47:.1f}%" if cl47 else "N/A"
        cl48_s  = f"{cl48:.1f}%"
        print(f"{split:<22} {paper_s:>20} {cl47_s:>18} {cl48_s:>18}")

    summary = {
        "paper_scores_deepseek_r1": PAPER_SCORES_DEEPSEEK,
        "claude_47_scores": CLAUDE_47_SCORES,
        "claude_48_scores": {s: results[s]["accuracy"] for s in SPLITS},
        "note": f"claude-opus-4.8 is current strongest Claude. {MAX_SAMPLES}/30 samples per split.",
    }
    with open(out_dir / "summary_claude48.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
