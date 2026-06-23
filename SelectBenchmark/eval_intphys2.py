#!/usr/bin/env python3
"""
Evaluate IntPhys 2 benchmark with Claude Opus 4.7.

IntPhys 2 is a video benchmark testing intuitive physics (Permanence, Immutability,
Continuity, Solidity). Models classify videos as physically plausible (1) or
impossible (0). Chance = 50%, Human = 96.4%, best SOTA (Gemini 2.5 Flash) ~56%.

Paper: https://arxiv.org/abs/2506.09849
Dataset: https://huggingface.co/datasets/facebook/IntPhys2
         https://dl.fbaipublicfiles.com/IntPhys2/IntPhys2.zip

Paper baseline models tested: GPT-4o, Gemini-2.5-Flash, V-JEPA-2, Qwen-VL 2.5
Claude NOT tested in original paper.

Usage:
    cd /data5/fanbingbing/SocialLearningClaw
    .venv/bin/python SelectBenchmark/eval_intphys2.py [--max-scenes 10]
"""

import os, re, json, time, base64, argparse
from pathlib import Path
import cv2
import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

MODEL_NAME = "claude-opus-4.7"
MODEL_ID   = "anthropic/claude-opus-4.7"

# Paper-reported overall accuracy on Debug split (fixed + moving camera avg)
PAPER_SCORES = {
    "GPT-4o":           {"Permanence": 59.2, "Immutability": 59.1, "Continuity": 56.1, "Solidity": 56.0},
    "Gemini-2.5-Flash": {"Permanence": 61.6, "Immutability": 61.8, "Continuity": 55.0, "Solidity": 56.1},
    "Human":            {"Permanence": 99.6, "Immutability": 93.8, "Continuity": 96.7, "Solidity": 95.7},
}
CHANCE = 50.0

DATA_DIR     = Path(__file__).parent / "data" / "IntPhys2"
DEBUG_META   = DATA_DIR / "debug_metadata.csv"
DEBUG_VIDEOS = DATA_DIR / "Debug" / "Videos"
OUT_DIR      = Path(__file__).parent / "results" / "intphys2"
SECONDS_PER_FRAME = 1.5
MAX_FRAMES = 24  # keep cost manageable

PROMPTS = [
    # Prompt 1: binary numeric (most reliable for answer parsing)
    (
        "You are evaluating whether a video shows physically plausible behavior. "
        "Watch the sequence of frames carefully. "
        "Answer ONLY with '1' if the object behavior is consistent with real-world physics, "
        "or '0' if it violates physics (e.g., objects appear/disappear, teleport, "
        "pass through walls, or change shape impossibly). "
        "Output only a single character: 1 or 0."
    ),
]


def download_dataset(max_scenes: int = 10):
    """Download IntPhys2 Debug split videos via HuggingFace API."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Get dataset file list from HuggingFace
    api_url = "https://huggingface.co/api/datasets/facebook/IntPhys2"
    resp = requests.get(api_url + "?full=true", timeout=15)
    if resp.status_code != 200:
        raise RuntimeError(f"HF API failed: {resp.status_code}")

    # List files for Debug split
    siblings = resp.json().get("siblings", [])
    debug_files = [s["rfilename"] for s in siblings if "Debug" in s["rfilename"]]
    print(f"Found {len(debug_files)} Debug split files on HuggingFace")

    # Download metadata JSON first to understand structure
    meta_files = [f for f in debug_files if f.endswith(".json")][:1]
    video_files = [f for f in debug_files if f.endswith(".mp4")]

    base_url = "https://huggingface.co/datasets/facebook/IntPhys2/resolve/main"

    downloaded = 0
    for rfilename in meta_files + video_files:
        local_path = DATA_DIR / rfilename
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if local_path.exists():
            continue
        url = f"{base_url}/{rfilename}"
        r = requests.get(url, timeout=60)
        if r.status_code == 200:
            local_path.write_bytes(r.content)
            downloaded += 1
            if rfilename.endswith(".mp4"):
                print(f"  Downloaded {rfilename} ({len(r.content)//1024}KB)")
        if downloaded >= max_scenes * 4:  # 4 videos per scene
            break

    return DATA_DIR


def extract_frames(video_path: Path) -> list[str]:
    """Extract frames as base64 JPEG strings."""
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


def call_model(frames: list[str], prompt: str, retries: int = 3) -> str:
    content = [{"type": "text", "text": prompt}]
    for b64 in frames:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL_ID,
                messages=[{"role": "user", "content": content}],
                max_tokens=16,
                temperature=0.0,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            print(f"    API error (attempt {attempt+1}): {e}")
            if attempt < retries - 1:
                time.sleep(5)
    return ""


def parse_answer(text: str) -> int | None:
    """Return 1 (plausible) or 0 (impossible) from model output."""
    text = text.strip()
    if text in ("0", "1"):
        return int(text)
    if re.search(r"\byes\b", text, re.IGNORECASE):
        return 1
    if re.search(r"\bno\b", text, re.IGNORECASE):
        return 0
    m = re.search(r"\b([01])\b", text)
    return int(m.group(1)) if m else None


def load_metadata(data_dir: Path = None) -> list[dict]:
    """
    Load metadata from Debug CSV. Label: '_Possible' → 1, '_Impossible' → 0.
    Only returns rows whose video file exists locally.
    """
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
            "scene_id": str(row.get("SceneIndex", "?")),
            "category": str(row.get("condition", "?")),
            "camera": str(row.get("Camera", "Fixed")),
            "game": str(row.get("game_name", "?")),
            "type": str(row["type"]),
            "filename": fname,
        })
    return items


def evaluate(items: list[dict], prompt: str) -> dict:
    records = []
    n_correct = 0
    print(f"\n  Evaluating {len(items)} videos...")
    for i, item in enumerate(items):
        frames = extract_frames(item["video_path"])
        if not frames:
            print(f"    [{i+1:03d}] SKIP (no frames): {item['filename']}")
            continue
        output = call_model(frames, prompt)
        pred = parse_answer(output)
        correct = (pred == item["label"]) if pred is not None else False
        n_correct += correct
        records.append({**item, "video_path": str(item["video_path"]),
                        "predicted": pred, "correct": correct, "output": output.strip()})
        print(f"    [{i+1:03d}/{len(items)}] {'✓' if correct else '✗'}  "
              f"pred={pred}  gold={item['label']}  cat={item['category']}  "
              f"cam={item['camera']}  frames={len(frames)}")
        time.sleep(0.5)

    total = len([r for r in records if r["predicted"] is not None])
    acc = 100.0 * n_correct / total if total else 0.0
    return {"accuracy": acc, "n_correct": n_correct, "n_total": total, "records": records}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-scenes", type=int, default=10,
                        help="Max number of scenes to evaluate (each has 4 videos)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download if data already present")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Download data if needed
    if not args.skip_download or not any(DATA_DIR.rglob("*.mp4")):
        print("Downloading IntPhys 2 Debug split...")
        download_dataset(max_scenes=args.max_scenes)
    else:
        print(f"Using existing data in {DATA_DIR}")

    # Load metadata
    items = load_metadata()
    print(f"Found {len(items)} videos locally")
    if not items:
        print("No videos found. Check download.")
        return
    print(f"Evaluating all {len(items)} available videos")

    prompt = PROMPTS[0]
    results = evaluate(items, prompt)

    with open(OUT_DIR / f"{MODEL_NAME}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Category breakdown
    from collections import defaultdict
    by_cat: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results["records"]:
        if r["predicted"] is not None:
            by_cat[r["category"]]["total"] += 1
            by_cat[r["category"]]["correct"] += r["correct"]

    print(f"\n{'='*60}")
    print(f"Model: {MODEL_NAME}  |  Overall accuracy: {results['accuracy']:.1f}%  "
          f"(chance=50%, best SOTA~56%, human~96%)")
    print(f"{'='*60}")
    print(f"{'Category':<30} {'Accuracy':>10} {'N':>6}")
    print("-"*50)
    for cat, d in sorted(by_cat.items()):
        acc = 100.0 * d["correct"] / d["total"] if d["total"] else 0
        print(f"{cat:<30} {acc:>9.1f}% {d['total']:>6}")

    print("\nPaper comparison (Debug split, avg fixed+moving camera):")
    for m, scores in PAPER_SCORES.items():
        avg = sum(scores.values()) / len(scores)
        print(f"  {m:<25}: {avg:.1f}% avg")
    print(f"  {'Chance baseline':<25}: {CHANCE:.1f}%")

    summary = {
        "model": MODEL_NAME,
        "overall_accuracy": results["accuracy"],
        "by_category": {cat: 100.0*d["correct"]/d["total"] for cat, d in by_cat.items() if d["total"]},
        "paper_scores": PAPER_SCORES,
        "chance_baseline": CHANCE,
        "n_videos": results["n_total"],
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
