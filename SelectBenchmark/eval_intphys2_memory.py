#!/usr/bin/env python3
"""
IntPhys2 × Memory Agent Baselines (Reflexion + ExpeL + A-MEM + TGM).

Both baselines use CROSS-VIDEO memory (no per-video retry):
  After each video classification, check ground truth.
  Update memory based on whether the classification was correct.
  Memory is injected into the system prompt for subsequent videos.

Reflexion:
  After each wrong classification: LLM reflects on the physics cue it missed.
  Reflections accumulate and are injected into future video prompts.

ExpeL:
  After each video: add to experience pool (video description, category, outcome).
  Every 5 videos: LLM extracts generalizable physics classification rules.
  Future videos receive extracted rules + recent successful examples.

Model: claude-opus-4.8

Usage:
    cd /data5/fanbingbing/SocialLearningClaw
    .venv/bin/python SelectBenchmark/eval_intphys2_memory.py [--baseline reflexion|expel|both]
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
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

DATA_DIR     = Path(__file__).parent / "data" / "IntPhys2"
DEBUG_META   = DATA_DIR / "debug_metadata.csv"
DEBUG_VIDEOS = DATA_DIR / "Debug" / "Videos"
OUT_DIR      = Path(__file__).parent / "results" / "intphys2"
EMBED_MODEL  = "BAAI/bge-small-en-v1.5"

SECONDS_PER_FRAME = 1.5
MAX_FRAMES = 24

BASE_SYSTEM = (
    "You are evaluating whether a video shows physically plausible behavior. "
    "Watch the sequence of frames carefully. "
    "Answer ONLY with '1' if the object behavior is consistent with real-world physics, "
    "or '0' if it violates physics (e.g., objects appear/disappear, teleport, "
    "pass through walls, or change shape impossibly). "
    "Output only a single character: 1 or 0."
)

PAPER_SCORES = {
    "gpt-4o":           {"permanence": 59.2, "immutability": 59.1, "continuity": 56.1, "solidity": 56.0, "avg": 57.6},
    "gemini-2.5-flash": {"permanence": 61.6, "immutability": 61.8, "continuity": 55.0, "solidity": 56.1, "avg": 58.6},
    "human":            {"avg": 96.4},
}


# ── data loading ──────────────────────────────────────────────────────────────

def load_metadata() -> list[dict]:
    import pandas as pd
    df = pd.read_csv(DEBUG_META)
    items = []
    for _, row in df.iterrows():
        fname = Path(row["file_name"]).name
        video_path = DEBUG_VIDEOS / fname
        if not video_path.exists():
            continue
        label = 1 if "Possible" in str(row["type"]) else 0
        items.append({
            "video_path": video_path,
            "label": label,
            "filename": fname,
            "category": str(row.get("condition", "?")).lower(),
            "camera": str(row.get("Camera", "Fixed")),
            "type": str(row["type"]),
        })
    return items


def extract_frames(video_path: Path) -> list[str]:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    skip = max(1, int(fps * SECONDS_PER_FRAME))
    frames, idx = [], 0
    while len(frames) < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % skip == 0:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frames.append(base64.b64encode(buf).decode())
        idx += 1
    cap.release()
    return frames


def parse_answer(text: str) -> int | None:
    text = text.strip()
    if text in ("0", "1"):
        return int(text)
    if re.search(r"\byes\b", text, re.IGNORECASE):
        return 1
    if re.search(r"\bno\b", text, re.IGNORECASE):
        return 0
    m = re.search(r"\b([01])\b", text)
    return int(m.group(1)) if m else None


# ── LLM call ─────────────────────────────────────────────────────────────────

def call_model(frames: list[str], system: str, api_retries: int = 3) -> str:
    content = [{"type": "text", "text": "Classify this video as physically plausible (1) or impossible (0)."}]
    for b64 in frames:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    for attempt in range(api_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL_ID,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": content},
                ],
                max_tokens=16,
                temperature=0.0,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            print(f"    API error (attempt {attempt + 1}): {e}")
            if attempt < api_retries - 1:
                time.sleep(5)
    return ""


# ── Reflexion evaluation ──────────────────────────────────────────────────────

def eval_reflexion(items: list[dict], memory: ReflexionMemory) -> dict:
    records = []
    n_correct = 0

    print(f"\n  [Reflexion] Evaluating {len(items)} videos ...")
    for i, item in enumerate(items):
        frames = extract_frames(item["video_path"])
        if not frames:
            print(f"    [{i+1:03d}] SKIP (no frames): {item['filename']}")
            continue

        mem_block = memory.get_memory_block()
        system = (BASE_SYSTEM + "\n\n" + mem_block) if mem_block else BASE_SYSTEM

        output = call_model(frames, system)
        pred = parse_answer(output)
        correct = (pred == item["label"]) if pred is not None else False
        n_correct += correct

        print(
            f"    [{i+1:03d}/{len(items)}] {'✓' if correct else '✗'}  "
            f"pred={pred}  gold={item['label']}  cat={item['category']}  "
            f"n_reflections={len(memory.reflections)}"
        )

        if not correct and pred is not None:
            label_str = "plausible (1)" if item["label"] == 1 else "impossible (0)"
            task_ctx = (
                f"Physics video classification — category: {item['category']}, camera: {item['camera']}.\n"
                f"The video shows an object physics scenario."
            )
            failure_info = (
                f"I predicted {pred} but the correct answer was {item['label']} ({label_str}). "
                f"I missed the physics violation / plausibility cue."
            )
            reflection = memory.reflect(task_ctx, failure_info)
            if reflection:
                print(f"    [Reflexion] {reflection[:120]}...")

        records.append({
            **{k: str(v) if isinstance(v, Path) else v for k, v in item.items()},
            "predicted": pred,
            "correct": correct,
            "n_reflections_used": len(memory.reflections),
        })
        time.sleep(0.5)

    total = len([r for r in records if r["predicted"] is not None])
    acc = 100.0 * n_correct / total if total else 0.0
    print(f"  => [Reflexion] Accuracy: {acc:.1f}%  ({n_correct}/{total})")
    return {"accuracy": acc, "n_correct": n_correct, "n_total": total, "records": records}


# ── ExpeL evaluation ──────────────────────────────────────────────────────────

def eval_expel(items: list[dict], memory: ExPeLMemory) -> dict:
    records = []
    n_correct = 0

    print(f"\n  [ExpeL] Evaluating {len(items)} videos ...")
    for i, item in enumerate(items):
        frames = extract_frames(item["video_path"])
        if not frames:
            print(f"    [{i+1:03d}] SKIP (no frames): {item['filename']}")
            continue

        mem_block = memory.get_memory_block()
        system = (BASE_SYSTEM + "\n\n" + mem_block) if mem_block else BASE_SYSTEM

        output = call_model(frames, system)
        pred = parse_answer(output)
        correct = (pred == item["label"]) if pred is not None else False
        n_correct += correct

        print(
            f"    [{i+1:03d}/{len(items)}] {'✓' if correct else '✗'}  "
            f"pred={pred}  gold={item['label']}  cat={item['category']}  "
            f"n_exp={len(memory.experiences)}  n_insights={len(memory.insights)}"
        )

        label_str = "plausible" if item["label"] == 1 else "impossible"
        task_summary = f"Physics video — category: {item['category']}, camera: {item['camera']}"
        if correct:
            lesson = f"Correctly classified as {label_str}. The physics cue was detectable."
        else:
            lesson = (
                f"Misclassified as {'plausible' if pred == 1 else 'impossible'}; "
                f"correct was {label_str}. Need to look more carefully at {item['category']} violations."
            )

        memory.add_experience(task=task_summary, outcome=correct, lesson=lesson)

        records.append({
            **{k: str(v) if isinstance(v, Path) else v for k, v in item.items()},
            "predicted": pred,
            "correct": correct,
            "n_experiences": len(memory.experiences),
            "n_insights": len(memory.insights),
        })
        time.sleep(0.5)

    total = len([r for r in records if r["predicted"] is not None])
    acc = 100.0 * n_correct / total if total else 0.0
    print(f"  => [ExpeL] Accuracy: {acc:.1f}%  ({n_correct}/{total})")
    return {"accuracy": acc, "n_correct": n_correct, "n_total": total, "records": records}


# ── A-MEM evaluation ──────────────────────────────────────────────────────────

def eval_amem(items: list[dict], memory: AMemory) -> dict:
    records = []
    n_correct = 0

    print(f"\n  [A-MEM] Evaluating {len(items)} videos ...")
    for i, item in enumerate(items):
        frames = extract_frames(item["video_path"])
        if not frames:
            print(f"    [{i+1:03d}] SKIP (no frames): {item['filename']}")
            continue

        query = (
            f"IntPhys2 {item['category']} physics classification, "
            f"camera={item['camera']}, type={item['type']}"
        )
        mem_block = memory.get_memory_block(query, top_k=5)
        system = (BASE_SYSTEM + "\n\n" + mem_block) if mem_block else BASE_SYSTEM

        output = call_model(frames, system)
        pred = parse_answer(output)
        correct = (pred == item["label"]) if pred is not None else False
        n_correct += correct

        print(
            f"    [{i+1:03d}/{len(items)}] {'✓' if correct else '✗'}  "
            f"pred={pred}  gold={item['label']}  cat={item['category']}  "
            f"n_notes={len(memory.notes)}"
        )

        label_str = "plausible" if item["label"] == 1 else "impossible"
        lesson = (
            f"Correctly classified this {item['category']} video as {label_str}."
            if correct
            else (
                f"Misclassified as {'plausible' if pred == 1 else 'impossible'}; "
                f"correct label was {label_str}. Re-check normal behavior versus true {item['category']} violations."
            )
        )
        memory.add_experience(
            task=query,
            outcome=correct,
            trajectory=f"model_output={output!r}",
            lesson=lesson,
            context=f"filename={item['filename']}; gold={item['label']}",
        )

        records.append({
            **{k: str(v) if isinstance(v, Path) else v for k, v in item.items()},
            "predicted": pred,
            "correct": correct,
            "n_notes": len(memory.notes),
        })
        time.sleep(0.5)

    total = len([r for r in records if r["predicted"] is not None])
    acc = 100.0 * n_correct / total if total else 0.0
    print(f"  => [A-MEM] Accuracy: {acc:.1f}%  ({n_correct}/{total})")
    return {"accuracy": acc, "n_correct": n_correct, "n_total": total, "records": records}


# ── TGM evaluation ────────────────────────────────────────────────────────────

def eval_tgm(items: list[dict], memory: TrainableGraphMemory) -> dict:
    records = []
    n_correct = 0

    print(f"\n  [TGM] Evaluating {len(items)} videos ...")
    for i, item in enumerate(items):
        frames = extract_frames(item["video_path"])
        if not frames:
            print(f"    [{i+1:03d}] SKIP (no frames): {item['filename']}")
            continue

        query = (
            f"IntPhys2 physics video classification. category={item['category']}; "
            f"camera={item['camera']}; type={item['type']}; filename={item['filename']}"
        )
        mem_block = memory.get_memory_block(query, top_k=3)
        system = (BASE_SYSTEM + "\n\n" + mem_block) if mem_block else BASE_SYSTEM

        output = call_model(frames, system)
        pred = parse_answer(output)
        correct = (pred == item["label"]) if pred is not None else False
        n_correct += correct

        print(
            f"    [{i+1:03d}/{len(items)}] {'✓' if correct else '✗'}  "
            f"pred={pred}  gold={item['label']}  cat={item['category']}  "
            f"n_meta={len(memory.meta)}"
        )

        label_str = "plausible" if item["label"] == 1 else "impossible"
        pred_str = "unparsed" if pred is None else ("plausible" if pred == 1 else "impossible")
        lesson = (
            f"Correctly classified this {item['category']} video as {label_str}; preserve the useful physical cue checks."
            if correct
            else (
                f"Misclassified as {pred_str}; correct label was {label_str}. Future strategy should reduce "
                f"possible/impossible bias and focus on the specific {item['category']} physical constraint."
            )
        )
        meta = memory.add_experience(
            task=query,
            outcome=correct,
            trajectory=f"model_output={output!r}; pred={pred}; gold={item['label']}",
            lesson=lesson,
            context=(
                f"filename={item['filename']}; category={item['category']}; camera={item['camera']}; "
                f"type={item['type']}; gold={item['label']}; predicted={pred}"
            ),
            domain="intphys2",
        )

        records.append({
            **{k: str(v) if isinstance(v, Path) else v for k, v in item.items()},
            "predicted": pred,
            "correct": correct,
            "n_query_nodes": len(memory.queries),
            "n_path_nodes": len(memory.paths),
            "n_meta_nodes": len(memory.meta),
            "last_meta_id": meta.id,
        })
        time.sleep(0.5)

    total = len([r for r in records if r["predicted"] is not None])
    acc = 100.0 * n_correct / total if total else 0.0
    print(f"  => [TGM] Accuracy: {acc:.1f}%  ({n_correct}/{total})")
    return {"accuracy": acc, "n_correct": n_correct, "n_total": total, "records": records}


# ── summary helpers ───────────────────────────────────────────────────────────

def _by_category(records: list[dict]) -> dict:
    by_cat: dict = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in records:
        if r["predicted"] is not None:
            by_cat[r["category"]]["total"] += 1
            by_cat[r["category"]]["correct"] += int(r["correct"])
    return {cat: 100.0 * d["correct"] / d["total"] for cat, d in by_cat.items() if d["total"]}


def _load_vanilla_accuracy() -> float | None:
    p = OUT_DIR / "claude-opus-4.8.json"
    if p.exists():
        with open(p) as f:
            d = json.load(f)
        return d.get("overall_accuracy")
    return None


def _print_summary(name: str, results: dict) -> None:
    vanilla = _load_vanilla_accuracy()
    print(f"\n{'='*55}")
    print(f"SUMMARY — {name}  |  Model: {MODEL_NAME}")
    print(f"{'='*55}")
    print(f"  {name} accuracy:  {results['accuracy']:.1f}%")
    if vanilla is not None:
        print(f"  Vanilla {MODEL_NAME}: {vanilla:.1f}%")
    print(f"  Chance baseline:  50.0%")
    print(f"  Paper best (Gemini-2.5-Flash): 58.6%")
    print(f"  Human: 96.4%")
    by_cat = _by_category(results["records"])
    if by_cat:
        print("\n  By category:")
        for cat, acc in sorted(by_cat.items()):
            print(f"    {cat:<30} {acc:.1f}%")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", choices=["reflexion", "expel", "amem", "tgm", "both", "all"], default="both")
    args = parser.parse_args()

    items = load_metadata()
    print(f"Loaded {len(items)} videos from {DATA_DIR}")
    if not items:
        print("No videos found. Check that IntPhys2 data is downloaded.")
        return

    if args.baseline in ("reflexion", "both", "all"):
        memory = ReflexionMemory(client=client, model_id=MODEL_ID, max_reflections=20)
        print("\n" + "=" * 55)
        print(f"Baseline: Reflexion  |  Model: {MODEL_NAME}")
        print("=" * 55)
        results = eval_reflexion(items, memory)
        out_path = OUT_DIR / f"reflexion_{MODEL_NAME}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "baseline": "reflexion",
                "model": MODEL_NAME,
                **results,
                "by_category": _by_category(results["records"]),
                "final_reflections": memory.reflections,
                "paper_scores": PAPER_SCORES,
            }, f, indent=2, ensure_ascii=False)
        memory.save(OUT_DIR / f"reflexion_{MODEL_NAME}_memory.json")
        _print_summary("Reflexion", results)
        print(f"\nResults saved to {out_path}")

    if args.baseline in ("expel", "both", "all"):
        memory_e = ExPeLMemory(client=client, model_id=MODEL_ID, insight_interval=5, max_insights=10)
        print("\n" + "=" * 55)
        print(f"Baseline: ExpeL  |  Model: {MODEL_NAME}")
        print("=" * 55)
        results_e = eval_expel(items, memory_e)
        out_path_e = OUT_DIR / f"expel_{MODEL_NAME}.json"
        with open(out_path_e, "w", encoding="utf-8") as f:
            json.dump({
                "baseline": "expel",
                "model": MODEL_NAME,
                **results_e,
                "by_category": _by_category(results_e["records"]),
                "final_insights": memory_e.insights,
                "n_experiences": len(memory_e.experiences),
                "paper_scores": PAPER_SCORES,
            }, f, indent=2, ensure_ascii=False)
        memory_e.save(OUT_DIR / f"expel_{MODEL_NAME}_memory.json")
        _print_summary("ExpeL", results_e)
        print(f"\nResults saved to {out_path_e}")

    if args.baseline in ("amem", "all"):
        print(f"Loading A-MEM embedding model: {EMBED_MODEL} ...")
        embedder = SentenceTransformer(EMBED_MODEL)
        memory_a = AMemory(client=client, model_id=MODEL_ID, embedder=embedder, max_notes=120, retrieve_k=5)
        print("\n" + "=" * 55)
        print(f"Baseline: A-MEM  |  Model: {MODEL_NAME}")
        print("=" * 55)
        results_a = eval_amem(items, memory_a)
        out_path_a = OUT_DIR / f"amem_{MODEL_NAME}.json"
        with open(out_path_a, "w", encoding="utf-8") as f:
            json.dump({
                "baseline": "amem",
                "model": MODEL_NAME,
                "embed_model": EMBED_MODEL,
                **results_a,
                "by_category": _by_category(results_a["records"]),
                "n_notes": len(memory_a.notes),
                "final_notes": [n.content for n in memory_a.notes],
                "paper_scores": PAPER_SCORES,
            }, f, indent=2, ensure_ascii=False)
        memory_a.save(OUT_DIR / f"amem_{MODEL_NAME}_memory.json")
        _print_summary("A-MEM", results_a)
        print(f"\nResults saved to {out_path_a}")

    if args.baseline in ("tgm", "all"):
        print(f"Loading TGM embedding model: {EMBED_MODEL} ...")
        embedder_t = SentenceTransformer(EMBED_MODEL)
        memory_t = TrainableGraphMemory(
            client=client,
            model_id=MODEL_ID,
            embedder=embedder_t,
            max_meta=30,
            retrieve_k=3,
        )
        print("\n" + "=" * 55)
        print(f"Baseline: TGM  |  Model: {MODEL_NAME}")
        print("=" * 55)
        results_t = eval_tgm(items, memory_t)
        out_path_t = OUT_DIR / f"tgm_{MODEL_NAME}.json"
        with open(out_path_t, "w", encoding="utf-8") as f:
            json.dump({
                "baseline": "tgm",
                "model": MODEL_NAME,
                "embed_model": EMBED_MODEL,
                "top_k_meta": 3,
                **results_t,
                "by_category": _by_category(results_t["records"]),
                "n_query_nodes": len(memory_t.queries),
                "n_path_nodes": len(memory_t.paths),
                "n_meta_nodes": len(memory_t.meta),
                "final_meta_cognitions": [
                    {
                        "id": m.id,
                        "summary": m.summary,
                        "principles": m.principles,
                        "confidence": m.confidence,
                        "evidence_count": m.evidence_count,
                        "reward_sum": m.reward_sum,
                    }
                    for m in memory_t.meta.values()
                ],
                "paper_scores": PAPER_SCORES,
            }, f, indent=2, ensure_ascii=False)
        memory_t.save(OUT_DIR / f"tgm_{MODEL_NAME}_memory.json")
        _print_summary("TGM", results_t)
        print(f"\nResults saved to {out_path_t}")


if __name__ == "__main__":
    main()
