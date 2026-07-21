#!/usr/bin/env python3
"""
ContextMATH × Memory Agent Baselines (Reflexion + ExpeL + A-MEM + TGM).

Reflexion:
  Per-problem retry loop (up to MAX_RETRIES=3 attempts).
  After each wrong answer: LLM generates reflection → injected into next attempt.
  Reflections also accumulate cross-problem (cross-split learning).

ExpeL:
  No per-problem retry (single attempt per problem).
  After each problem: add to experience pool (with trajectory summary + lesson).
  Every 5 problems: LLM extracts generalizable math insights.
  Next problem receives insights + recent successful examples in system prompt.

Model: claude-opus-4.8

Usage:
    cd /data5/fanbingbing/SocialLearningClaw
    .venv/bin/python SelectBenchmark/eval_contextmath_memory.py [--baseline reflexion|expel|both]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent.parent))
from socialclaw.memory_agents import AMemory, ReflexionMemory, ExPeLMemory, TrainableGraphMemory

load_dotenv(Path(__file__).parent.parent / ".env")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

MODEL_NAME = "claude-opus-4.8"
MODEL_ID   = "anthropic/claude-opus-4.8"
MAX_RETRIES = 3   # Reflexion: max attempts per problem
MAX_SAMPLES = 10  # per split

DATA_DIR = Path(__file__).parent / "data" / "ContextMATH"
SPLIT_FILES = {
    "aime_2024_sg": DATA_DIR / "aime_2024_sg-00000-of-00001.parquet",
    "aime_2024_cs": DATA_DIR / "aime_2024_cs-00000-of-00001.parquet",
    "aime_2025_sg": DATA_DIR / "aime_2025_sg-00000-of-00001.parquet",
    "aime_2025_cs": DATA_DIR / "aime_2025_cs-00000-of-00001.parquet",
}
SPLITS = list(SPLIT_FILES.keys())

OUT_DIR = Path(__file__).parent / "results" / "contextmath"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

BASE_SYSTEM = (
    "You are a mathematical reasoning expert. "
    "Solve problems step by step. Always box your final answer using \\boxed{} notation."
)
USER_TEMPLATE = (
    "{question}\n\nSolve this problem carefully. Put your final numerical answer in \\boxed{{}}."
)


# ── answer extraction ─────────────────────────────────────────────────────────

def extract_boxed(text: str) -> str | None:
    matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    if not matches:
        return None
    raw = matches[-1].strip()
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


# ── LLM call ─────────────────────────────────────────────────────────────────

def call_model(system: str, question: str, api_retries: int = 3) -> str:
    for attempt in range(api_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL_ID,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": USER_TEMPLATE.format(question=question)},
                ],
                max_tokens=8192,
                temperature=0.0,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            print(f"    API error (attempt {attempt + 1}): {e}")
            if attempt < api_retries - 1:
                time.sleep(5)
    return ""


# ── Reflexion evaluation ──────────────────────────────────────────────────────

def eval_reflexion(split: str, memory: ReflexionMemory) -> dict:
    """
    Evaluate one AIME split with Reflexion.
    Memory persists across problems within this call (cross-problem learning).
    """
    df = pd.read_parquet(SPLIT_FILES[split])
    samples = df.head(MAX_SAMPLES).to_dict("records")
    records = []
    n_correct = 0

    print(f"\n  [Reflexion | {split}] {len(samples)} problems")
    for i, row in enumerate(samples):
        question = row["question"]
        gold = row["answer"]

        final_pred: str | None = None
        final_correct = False
        attempts_log = []

        for retry in range(MAX_RETRIES):
            mem_block = memory.get_memory_block()
            system = (BASE_SYSTEM + "\n\n" + mem_block) if mem_block else BASE_SYSTEM
            output = call_model(system, question)
            pred = extract_boxed(output)
            correct = answers_match(pred, gold)

            attempts_log.append({"attempt": retry + 1, "pred": pred, "correct": correct})
            print(f"    [{i+1:02d}/{len(samples)}] attempt {retry+1}: {'✓' if correct else '✗'}  pred={pred!r}  gold={gold!r}")

            if correct:
                final_pred = pred
                final_correct = True
                break

            # Generate reflection on failure (don't reveal gold answer)
            task_ctx = (
                f"AIME math problem (split={split}):\n{question}"
            )
            failure_info = f"My answer was {pred!r} but that was wrong."
            reflection = memory.reflect(task_ctx, failure_info)
            if reflection:
                print(f"    [Reflexion] {reflection[:120]}...")
            time.sleep(1)

        if not final_correct:
            final_pred = attempts_log[-1]["pred"]

        n_correct += final_correct
        records.append({
            "id": row["id"],
            "gold": gold,
            "predicted": final_pred,
            "correct": final_correct,
            "n_attempts": len(attempts_log),
            "attempts": attempts_log,
        })
        time.sleep(1)

    accuracy = 100.0 * n_correct / len(samples)
    print(f"  => [Reflexion | {split}] Accuracy: {accuracy:.1f}%  ({n_correct}/{len(samples)})")
    return {"accuracy": accuracy, "n_correct": n_correct, "n_total": len(samples), "records": records}


# ── ExpeL evaluation ──────────────────────────────────────────────────────────

def eval_expel(split: str, memory: ExPeLMemory) -> dict:
    """
    Evaluate one AIME split with ExpeL.
    Single attempt per problem; experience accumulates cross-problem.
    """
    df = pd.read_parquet(SPLIT_FILES[split])
    samples = df.head(MAX_SAMPLES).to_dict("records")
    records = []
    n_correct = 0

    print(f"\n  [ExpeL | {split}] {len(samples)} problems")
    for i, row in enumerate(samples):
        question = row["question"]
        gold = row["answer"]

        mem_block = memory.get_memory_block()
        system = (BASE_SYSTEM + "\n\n" + mem_block) if mem_block else BASE_SYSTEM
        output = call_model(system, question)
        pred = extract_boxed(output)
        correct = answers_match(pred, gold)
        n_correct += correct

        print(f"    [{i+1:02d}/{len(samples)}] {'✓' if correct else '✗'}  pred={pred!r}  gold={gold!r}")

        # Build lesson for experience pool
        task_summary = question[:200]
        if correct:
            lesson = f"Solved correctly. Final answer: {pred}."
        else:
            lesson = f"Failed — predicted {pred!r} (wrong). Review the approach."

        memory.add_experience(
            task=task_summary,
            outcome=correct,
            trajectory=output[:300],
            lesson=lesson,
        )

        records.append({
            "id": row["id"],
            "gold": gold,
            "predicted": pred,
            "correct": correct,
            "n_experiences_before": len(memory.experiences) - 1,
            "n_insights": len(memory.insights),
        })
        time.sleep(1)

    accuracy = 100.0 * n_correct / len(samples)
    print(f"  => [ExpeL | {split}] Accuracy: {accuracy:.1f}%  ({n_correct}/{len(samples)})")
    return {"accuracy": accuracy, "n_correct": n_correct, "n_total": len(samples), "records": records}


# ── A-MEM evaluation ──────────────────────────────────────────────────────────

def eval_amem(split: str, memory: AMemory) -> dict:
    """
    Evaluate one AIME split with A-MEM.
    Single attempt per problem; structured notes evolve cross-problem.
    """
    df = pd.read_parquet(SPLIT_FILES[split])
    samples = df.head(MAX_SAMPLES).to_dict("records")
    records = []
    n_correct = 0

    print(f"\n  [A-MEM | {split}] {len(samples)} problems")
    for i, row in enumerate(samples):
        question = row["question"]
        gold = row["answer"]

        mem_block = memory.get_memory_block(question, top_k=5)
        system = (BASE_SYSTEM + "\n\n" + mem_block) if mem_block else BASE_SYSTEM
        output = call_model(system, question)
        pred = extract_boxed(output)
        correct = answers_match(pred, gold)
        n_correct += correct

        print(
            f"    [{i+1:02d}/{len(samples)}] {'✓' if correct else '✗'}  "
            f"pred={pred!r}  gold={gold!r}  n_notes={len(memory.notes)}"
        )

        lesson = (
            f"Solved correctly with final answer {pred}."
            if correct
            else f"Failed: predicted {pred!r}, which was wrong. Improve contextual extraction and arithmetic checks."
        )
        memory.add_experience(
            task=f"ContextMATH {split} problem:\n{question[:1200]}",
            outcome=correct,
            trajectory=output[:1200],
            lesson=lesson,
            context=f"split={split}; gold={gold}",
        )

        records.append({
            "id": row["id"],
            "gold": gold,
            "predicted": pred,
            "correct": correct,
            "n_notes": len(memory.notes),
        })
        time.sleep(1)

    accuracy = 100.0 * n_correct / len(samples)
    print(f"  => [A-MEM | {split}] Accuracy: {accuracy:.1f}%  ({n_correct}/{len(samples)})")
    return {"accuracy": accuracy, "n_correct": n_correct, "n_total": len(samples), "records": records}


# ── TGM evaluation ────────────────────────────────────────────────────────────

def eval_tgm(split: str, memory: TrainableGraphMemory) -> dict:
    """
    Evaluate one AIME split with trainable graph memory.
    Single attempt per problem; query/path/meta-cognition graph evolves cross-problem.
    """
    df = pd.read_parquet(SPLIT_FILES[split])
    samples = df.head(MAX_SAMPLES).to_dict("records")
    records = []
    n_correct = 0

    print(f"\n  [TGM | {split}] {len(samples)} problems")
    for i, row in enumerate(samples):
        question = row["question"]
        gold = row["answer"]

        mem_block = memory.get_memory_block(
            f"ContextMATH {split} problem:\n{question}",
            top_k=3,
        )
        system = (BASE_SYSTEM + "\n\n" + mem_block) if mem_block else BASE_SYSTEM
        output = call_model(system, question)
        pred = extract_boxed(output)
        correct = answers_match(pred, gold)
        n_correct += correct

        print(
            f"    [{i+1:02d}/{len(samples)}] {'✓' if correct else '✗'}  "
            f"pred={pred!r}  gold={gold!r}  n_meta={len(memory.meta)}"
        )

        lesson = (
            f"Solved correctly with final answer {pred}. Preserve the useful reasoning path."
            if correct
            else (
                f"Failed: predicted {pred!r}, which was wrong. Build a future strategy that improves "
                "problem parsing, equation setup, arithmetic verification, or answer checking."
            )
        )
        meta = memory.add_experience(
            task=f"ContextMATH {split} problem:\n{question[:1600]}",
            outcome=correct,
            trajectory=output[:1600],
            lesson=lesson,
            context=f"split={split}; gold={gold}; predicted={pred}",
            domain="contextmath",
        )

        records.append({
            "id": row["id"],
            "gold": gold,
            "predicted": pred,
            "correct": correct,
            "n_query_nodes": len(memory.queries),
            "n_path_nodes": len(memory.paths),
            "n_meta_nodes": len(memory.meta),
            "last_meta_id": meta.id,
        })
        time.sleep(1)

    accuracy = 100.0 * n_correct / len(samples)
    print(f"  => [TGM | {split}] Accuracy: {accuracy:.1f}%  ({n_correct}/{len(samples)})")
    return {"accuracy": accuracy, "n_correct": n_correct, "n_total": len(samples), "records": records}


# ── main ──────────────────────────────────────────────────────────────────────

def run_reflexion() -> None:
    memory = ReflexionMemory(client=client, model_id=MODEL_ID, max_reflections=15)
    all_results: dict = {}

    print("\n" + "=" * 60)
    print("Baseline: Reflexion  |  Model:", MODEL_NAME)
    print("=" * 60)

    for split in SPLITS:
        all_results[split] = eval_reflexion(split, memory)

    out_path = OUT_DIR / f"reflexion_{MODEL_NAME}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "baseline": "reflexion",
            "model": MODEL_NAME,
            "max_retries": MAX_RETRIES,
            "splits": all_results,
            "summary": {s: all_results[s]["accuracy"] for s in SPLITS},
            "final_reflections": memory.reflections,
        }, f, indent=2, ensure_ascii=False)

    memory.save(OUT_DIR / f"reflexion_{MODEL_NAME}_memory.json")
    _print_summary("Reflexion", all_results)
    print(f"\nResults saved to {out_path}")


def run_expel() -> None:
    memory = ExPeLMemory(client=client, model_id=MODEL_ID, insight_interval=5, max_insights=10)
    all_results: dict = {}

    print("\n" + "=" * 60)
    print("Baseline: ExpeL  |  Model:", MODEL_NAME)
    print("=" * 60)

    for split in SPLITS:
        all_results[split] = eval_expel(split, memory)

    out_path = OUT_DIR / f"expel_{MODEL_NAME}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "baseline": "expel",
            "model": MODEL_NAME,
            "insight_interval": 5,
            "splits": all_results,
            "summary": {s: all_results[s]["accuracy"] for s in SPLITS},
            "final_insights": memory.insights,
            "n_experiences": len(memory.experiences),
        }, f, indent=2, ensure_ascii=False)

    memory.save(OUT_DIR / f"expel_{MODEL_NAME}_memory.json")
    _print_summary("ExpeL", all_results)
    print(f"\nResults saved to {out_path}")


def run_amem() -> None:
    print(f"Loading A-MEM embedding model: {EMBED_MODEL} ...")
    embedder = SentenceTransformer(EMBED_MODEL)
    memory = AMemory(client=client, model_id=MODEL_ID, embedder=embedder, max_notes=120, retrieve_k=5)
    all_results: dict = {}

    print("\n" + "=" * 60)
    print("Baseline: A-MEM  |  Model:", MODEL_NAME)
    print("=" * 60)

    for split in SPLITS:
        all_results[split] = eval_amem(split, memory)

    out_path = OUT_DIR / f"amem_{MODEL_NAME}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "baseline": "amem",
            "model": MODEL_NAME,
            "embed_model": EMBED_MODEL,
            "splits": all_results,
            "summary": {s: all_results[s]["accuracy"] for s in SPLITS},
            "n_notes": len(memory.notes),
            "final_notes": [n.content for n in memory.notes],
        }, f, indent=2, ensure_ascii=False)

    memory.save(OUT_DIR / f"amem_{MODEL_NAME}_memory.json")
    _print_summary("A-MEM", all_results)
    print(f"\nResults saved to {out_path}")


def run_tgm() -> None:
    print(f"Loading TGM embedding model: {EMBED_MODEL} ...")
    embedder = SentenceTransformer(EMBED_MODEL)
    memory = TrainableGraphMemory(
        client=client,
        model_id=MODEL_ID,
        embedder=embedder,
        max_meta=30,
        retrieve_k=3,
    )
    all_results: dict = {}

    print("\n" + "=" * 60)
    print("Baseline: TGM  |  Model:", MODEL_NAME)
    print("=" * 60)

    for split in SPLITS:
        all_results[split] = eval_tgm(split, memory)

    out_path = OUT_DIR / f"tgm_{MODEL_NAME}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "baseline": "tgm",
            "model": MODEL_NAME,
            "embed_model": EMBED_MODEL,
            "top_k_meta": 3,
            "splits": all_results,
            "summary": {s: all_results[s]["accuracy"] for s in SPLITS},
            "n_query_nodes": len(memory.queries),
            "n_path_nodes": len(memory.paths),
            "n_meta_nodes": len(memory.meta),
            "final_meta_cognitions": [
                {
                    "id": m.id,
                    "summary": m.summary,
                    "principles": m.principles,
                    "confidence": m.confidence,
                    "evidence_count": m.evidence_count,
                    "reward_sum": m.reward_sum,
                }
                for m in memory.meta.values()
            ],
        }, f, indent=2, ensure_ascii=False)

    memory.save(OUT_DIR / f"tgm_{MODEL_NAME}_memory.json")
    _print_summary("TGM", all_results)
    print(f"\nResults saved to {out_path}")


def _print_summary(name: str, results: dict) -> None:
    # Vanilla claude-opus-4.8 reference scores (from existing results)
    VANILLA = {"aime_2024_sg": None, "aime_2024_cs": None, "aime_2025_sg": None, "aime_2025_cs": None}
    vanilla_path = OUT_DIR / "claude-opus-4.8.json"
    if vanilla_path.exists():
        with open(vanilla_path) as f:
            vd = json.load(f)
        for s in SPLITS:
            VANILLA[s] = vd.get(s, {}).get("accuracy")

    print(f"\n{'='*65}")
    print(f"SUMMARY — {name} vs Vanilla {MODEL_NAME}")
    print(f"{'='*65}")
    print(f"{'Split':<22}  {'Vanilla':>10}  {name:>12}")
    print("-" * 50)
    for split in SPLITS:
        v = VANILLA.get(split)
        m = results[split]["accuracy"]
        v_str = f"{v:.1f}%" if v is not None else "N/A"
        print(f"{split:<22}  {v_str:>10}  {m:>11.1f}%")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", choices=["reflexion", "expel", "amem", "tgm", "both", "all"], default="both")
    args = parser.parse_args()

    if args.baseline in ("reflexion", "both", "all"):
        run_reflexion()
    if args.baseline in ("expel", "both", "all"):
        run_expel()
    if args.baseline in ("amem", "all"):
        run_amem()
    if args.baseline in ("tgm", "all"):
        run_tgm()


if __name__ == "__main__":
    main()
