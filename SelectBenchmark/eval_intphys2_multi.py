#!/usr/bin/env python3
"""
IntPhys2 evaluation with multiple models:
  - claude-opus-4.8: current strongest Claude (NOT in original paper)
  - gpt-4o: paper baseline (paper Table 2 reports exact scores)

Metric: Binary Classification Accuracy (%) — proportion of videos correctly
classified as plausible (1) or impossible (0). Paper reports this per
Condition × Camera type; we report overall and per-condition averages.

Paper Table 2 (Debug split, Fixed + Moving camera avg):
  GPT-4o:          Permanence=59.2%, Immutability=59.1%, Continuity=56.1%, Solidity=56.0%  avg=57.6%
  Gemini-2.5-Flash: Permanence=61.6%, Immutability=61.8%, Continuity=55.0%, Solidity=56.1% avg=58.6%
  Human:            avg=96.4%
  Chance:           50.0%

Usage:
    cd /data5/fanbingbing/SocialLearningClaw
    .venv/bin/python SelectBenchmark/eval_intphys2_multi.py
"""

import os, re, json, time, base64
from pathlib import Path
from collections import defaultdict
import cv2
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

MODELS = {
    "claude-opus-4.8": "anthropic/claude-opus-4.8",
    "gpt-4o":          "openai/gpt-4o",
}

# Paper Table 2: accuracy per condition (avg of Fixed+Moving camera), Debug split
PAPER_SCORES = {
    "gpt-4o": {
        "permanence":   59.2,
        "immutability": 59.1,
        "continuity":   56.1,
        "solidity":     56.0,
        "avg":          57.6,
    },
    "gemini-2.5-flash": {
        "permanence":   61.6,
        "immutability": 61.8,
        "continuity":   55.0,
        "solidity":     56.1,
        "avg":          58.6,
    },
    "human": {"avg": 96.4},
}
CHANCE = 50.0

DEBUG_META   = Path(__file__).parent / "data" / "IntPhys2" / "debug_metadata.csv"
DEBUG_VIDEOS = Path(__file__).parent / "data" / "IntPhys2" / "Debug" / "Videos"
OUT_DIR      = Path(__file__).parent / "results" / "intphys2"

SECONDS_PER_FRAME = 1.5
MAX_FRAMES = 12  # keep cost low; 10.6s video → ~7 frames

PROMPT = (
    "You are evaluating whether a short video clip shows physically plausible behavior. "
    "I will show you a sequence of frames from a 3D physics simulation. "
    "Carefully examine whether the objects behave consistently with real-world physics laws "
    "(gravity, solidity, object permanence, continuity of motion). "
    "Answer ONLY with a single digit: '1' if the behavior is physically plausible, "
    "or '0' if it violates physics (e.g., objects pass through walls, disappear, "
    "teleport, or change shape impossibly). Output only: 1 or 0."
)

PREV_CLAUDE47_RESULT = {
    "model": "claude-opus-4.7",
    "overall_accuracy": 65.0,
    "note": "All solidity condition, Fixed camera only, 20 videos",
}


def load_items() -> list[dict]:
    df = pd.read_csv(DEBUG_META)
    items = []
    for _, row in df.iterrows():
        fname = Path(row["file_name"]).name
        vpath = DEBUG_VIDEOS / fname
        if not vpath.exists():
            continue
        label = 1 if "Possible" in str(row["type"]) else 0
        items.append({
            "video_path":  vpath,
            "label":       label,
            "condition":   str(row.get("condition", "?")).lower(),
            "camera":      str(row.get("Camera", "Fixed")),
            "game":        str(row.get("game_name", "?")),
            "type":        str(row["type"]),
            "filename":    fname,
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


def call_model(model_id: str, frames: list[str], retries: int = 3) -> str:
    content = [{"type": "text", "text": PROMPT}]
    for b64 in frames:
        content.append({"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": content}],
                max_tokens=8,
                temperature=0.0,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            print(f"    API error (attempt {attempt+1}): {e}")
            if attempt < retries - 1:
                time.sleep(5)
    return ""


def parse_answer(text: str) -> int | None:
    text = text.strip()
    if text in ("0", "1"):
        return int(text)
    m = re.search(r"\b([01])\b", text)
    return int(m.group(1)) if m else None


def evaluate(model_name: str, model_id: str, items: list[dict]) -> dict:
    records = []
    n_correct = 0
    print(f"\n  [{model_name}] {len(items)} videos")
    for i, item in enumerate(items):
        frames = extract_frames(item["video_path"])
        if not frames:
            continue
        output = call_model(model_id, frames)
        pred   = parse_answer(output)
        ok     = (pred == item["label"]) if pred is not None else False
        n_correct += ok
        records.append({**item,
                        "video_path": str(item["video_path"]),
                        "predicted": pred, "correct": ok, "output": output.strip()})
        print(f"    [{i+1:03d}/{len(items)}] {'✓' if ok else '✗'}  "
              f"pred={pred}  gold={item['label']}  "
              f"cond={item['condition']}  cam={item['camera']}")
        time.sleep(0.5)

    valid = [r for r in records if r["predicted"] is not None]
    acc = 100.0 * n_correct / len(valid) if valid else 0.0
    return {"accuracy": acc, "n_correct": n_correct, "n_total": len(valid), "records": records}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    items = load_items()
    print(f"Loaded {len(items)} videos from Debug split (locally available)")

    all_results = {}
    for model_name, model_id in MODELS.items():
        print(f"\n{'='*60}\nModel: {model_name}  ({model_id})\n{'='*60}")
        result = evaluate(model_name, model_id, items)
        all_results[model_name] = result

        # Per-condition breakdown
        by_cond = defaultdict(lambda: {"correct": 0, "total": 0})
        for r in result["records"]:
            if r["predicted"] is not None:
                by_cond[r["condition"]]["total"] += 1
                by_cond[r["condition"]]["correct"] += r["correct"]

        print(f"\n  Overall: {result['accuracy']:.1f}%  ({result['n_correct']}/{result['n_total']})")
        for cond, d in sorted(by_cond.items()):
            acc = 100.0 * d["correct"] / d["total"] if d["total"] else 0
            print(f"  {cond:<20}: {acc:.1f}% (n={d['total']})")

        result["by_condition"] = {
            c: {"accuracy": 100.0*d["correct"]/d["total"] if d["total"] else 0,
                "n": d["total"]}
            for c, d in by_cond.items()
        }
        with open(OUT_DIR / f"{model_name}.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    # Summary
    print("\n\n" + "="*65)
    print("RESULTS vs PAPER  (metric: Binary Classification Accuracy %)")
    print("  Paper Table 2, Debug split, avg Fixed+Moving camera")
    print("="*65)
    print(f"{'Model':<25} {'Overall Acc':>12} {'vs Paper Baseline':>18}")
    print("-"*55)
    for m, r in all_results.items():
        note = "NOT in paper" if m != "gpt-4o" else f"paper={PAPER_SCORES['gpt-4o']['avg']:.1f}%"
        print(f"{m:<25} {r['accuracy']:>11.1f}% {note:>18}")
    print(f"{'GPT-4o (paper)':<25} {'57.6%':>12} {'paper baseline':>18}")
    print(f"{'Gemini-2.5-Flash (paper)':<25} {'58.6%':>12} {'paper baseline':>18}")
    print(f"{'Human (paper)':<25} {'96.4%':>12}")
    print(f"{'Chance':<25} {'50.0%':>12}")

    summary = {
        "metric": "Binary Classification Accuracy (%) — paper Table 2",
        "our_results": {m: r["accuracy"] for m, r in all_results.items()},
        "claude_47_result": PREV_CLAUDE47_RESULT,
        "paper_scores": PAPER_SCORES,
        "chance": CHANCE,
        "dataset_note": "Debug split, Fixed camera only, 20 locally downloaded videos (solidity condition)",
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
