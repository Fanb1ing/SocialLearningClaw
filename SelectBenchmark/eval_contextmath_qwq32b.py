#!/usr/bin/env python3
"""
Run DeepSeek-R1 on ContextMATH (paper baseline).
Claude Opus 4.7 results already collected; combine at the end.
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

MODEL_NAME = "gemini-2.5-pro"
MODEL_ID   = "google/gemini-2.5-pro"

PAPER_SCORES = {
    "gemini-2.5-pro": {
        "aime_2024_sg": 73.3,
        "aime_2024_cs": 76.7,
        "aime_2025_sg": 56.7,
        "aime_2025_cs": 50.0,
    }
}

CLAUDE_SCORES = {
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


def call_model(question: str, retries=3) -> str:
    for attempt in range(retries):
        try:
            # Use streaming to avoid HTTP timeout on long reasoning chains
            chunks = []
            with client.chat.completions.create(
                model=MODEL_ID,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": USER_TEMPLATE.format(question=question)},
                ],
                max_tokens=8192,
                temperature=0.0,
                stream=True,
            ) as stream:
                for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        chunks.append(delta)
            return "".join(chunks)
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
        records.append({"id": row["id"], "gold": row["answer"],
                        "predicted": pred, "correct": ok,
                        "output_snippet": output[:300]})
        print(f"    [{i+1:02d}/{len(samples)}] {'✓' if ok else '✗'}  pred={pred!r}  gold={row['answer']!r}")
        time.sleep(1)

    acc = 100.0 * n_correct / len(samples)
    print(f"  => Accuracy: {acc:.1f}%  ({n_correct}/{len(samples)})")
    return {"accuracy": acc, "n_correct": n_correct, "n_total": len(samples), "records": records}


def main():
    out_dir = Path(__file__).parent / "results" / "contextmath"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}\nModel: {MODEL_NAME}  ({MODEL_ID})\n{'='*60}")
    ds_results = {}
    for split in SPLITS:
        ds_results[split] = eval_split(split)

    with open(out_dir / f"{MODEL_NAME}.json", "w", encoding="utf-8") as f:
        json.dump(ds_results, f, indent=2, ensure_ascii=False)

    # ── Summary ────────────────────────────────────────────────
    print("\n\n" + "="*72)
    print("RESULTS SUMMARY vs PAPER")
    print("="*72)
    print(f"{'Split':<22} {'DeepSeek-R1(paper)':>20} {'DeepSeek-R1(ours)':>20} {'Claude-Opus-4.7':>18}")
    print("-"*72)
    for split in SPLITS:
        paper   = PAPER_SCORES["gemini-2.5-pro"].get(split)
        ours_ds = ds_results[split]["accuracy"]
        ours_cl = CLAUDE_SCORES.get(split)
        paper_s = f"{paper:.1f}%" if paper else "N/A"
        ds_s    = f"{ours_ds:.1f}%"
        cl_s    = f"{ours_cl:.1f}%" if ours_cl else "N/A"
        print(f"{split:<22} {paper_s:>20} {ds_s:>20} {cl_s:>18}")

    summary = {
        "paper_scores":  PAPER_SCORES,
        "our_scores": {
            "gemini-2.5-pro": {s: ds_results[s]["accuracy"] for s in SPLITS},
            "claude-opus-4.7":  CLAUDE_SCORES,
        },
        "note": (
            f"Claude-opus-4.7 not in original paper. "
            f"DeepSeek-R1-0528 as paper-era baseline (paper R1 scores: 70.0/66.7/73.3/53.3). "
            f"Each split: {MAX_SAMPLES}/30 samples."
        ),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
