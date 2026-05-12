#!/usr/bin/env python3
"""
下载并预处理 CL-bench 数据集。
来源：HuggingFace tencent/CL-bench
"""

import json
import os
import sys
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("请先安装 datasets: pip install datasets")
    sys.exit(1)

DATA_DIR = Path(__file__).parent.parent / "data" / "clbench"
RAW_DIR = DATA_DIR / "raw"
PREPARED_DIR = DATA_DIR / "prepared"


def download():
    """从 HuggingFace 下载原始数据。"""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading tencent/CL-bench ...")
    ds = load_dataset("tencent/CL-bench", trust_remote_code=True)
    ds.save_to_disk(str(RAW_DIR / "hf_dataset"))
    print(f"Saved to {RAW_DIR / 'hf_dataset'}")

    print("Downloading tencent/CL-bench-Life ...")
    ds_life = load_dataset("tencent/CL-bench-Life", trust_remote_code=True)
    ds_life.save_to_disk(str(RAW_DIR / "hf_dataset_life"))
    print(f"Saved to {RAW_DIR / 'hf_dataset_life'}")


def prepare():
    """预处理为统一 JSONL 格式。"""
    from datasets import load_from_disk
    PREPARED_DIR.mkdir(parents=True, exist_ok=True)

    for name, subdir in [("clbench", "hf_dataset"), ("clbench_life", "hf_dataset_life")]:
        src = RAW_DIR / subdir
        if not src.exists():
            print(f"Skip {name}: {src} not found. Run download first.")
            continue

        ds = load_from_disk(str(src))
        split = list(ds.keys())[0]
        records = []
        for item in ds[split]:
            # CL-bench 使用 messages 对话格式
            messages = item.get("messages", [])
            context_parts = []
            question = ""
            answer = ""
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "system":
                    context_parts.append(content)
                elif role == "user":
                    question = content
                elif role == "assistant":
                    answer = content

            meta = item.get("metadata", {})
            rubrics = item.get("rubrics", [])
            records.append({
                "id": meta.get("task_id") or meta.get("context_id") or f"{name}_{len(records)}",
                "context": "\n".join(context_parts),
                "question": question,
                "answer": answer,
                "rubrics": rubrics,
                "meta": meta,
            })

        out_path = PREPARED_DIR / f"{name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(f"{name}: {len(records)} records -> {out_path}")


def inspect():
    """快速查看数据统计。"""
    for path in sorted(PREPARED_DIR.glob("*.jsonl")):
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        sample = json.loads(lines[0]) if lines else {}
        print(f"\n{path.name}: {len(lines)} records")
        print(f"  keys: {list(sample.keys())}")
        print(f"  context length (chars): {len(sample.get('context', ''))}")
        print(f"  question length (chars): {len(sample.get('question', ''))}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CL-bench 数据下载与预处理")
    parser.add_argument("action", choices=["download", "prepare", "inspect", "all"], default="all", nargs="?")
    args = parser.parse_args()

    if args.action in ("download", "all"):
        download()
    if args.action in ("prepare", "all"):
        prepare()
    if args.action in ("inspect", "all"):
        inspect()
