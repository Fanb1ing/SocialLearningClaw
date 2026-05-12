#!/usr/bin/env python3
"""
下载并预处理 ARC-AGI 数据集。

ARC-AGI-1/2：静态网格 JSON，可直接下载。
ARC-AGI-3：交互式环境，需要 API key，本脚本仅做环境准备。
"""

import json
import os
import sys
import zipfile
from pathlib import Path

try:
    import requests
except ImportError:
    print("请先安装 requests: pip install requests")
    sys.exit(1)

DATA_DIR = Path(__file__).parent.parent / "data" / "arc"
RAW_DIR = DATA_DIR / "raw"
PREPARED_DIR = DATA_DIR / "prepared"

# 使用 ghfast.top 镜像加速 GitHub 下载
GHFAST = "https://ghfast.top"

# ARC-AGI-1 官方仓库（静态 JSON）
ARC1_REPO = f"{GHFAST}/https://github.com/fchollet/ARC-AGI"
ARC1_JSON_URL = f"{GHFAST}/https://github.com/fchollet/ARC-AGI/raw/master/data/"

# ARC-AGI-2 官方仓库（静态 JSON）
ARC2_REPO = f"{GHFAST}/https://github.com/arcprize/ARC-AGI-2"

# ARC-AGI-3 官方 Agents 仓库（交互式，需要 API key）
ARC3_AGENTS_REPO = f"{GHFAST}/https://github.com/arcprize/ARC-AGI-3-Agents"


def download_arc1():
    """下载 ARC-AGI-1 训练和评估数据。"""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    arc1_dir = RAW_DIR / "arc1"

    if arc1_dir.exists():
        print(f"ARC-AGI-1 已存在: {arc1_dir}")
        return

    print(f"Cloning ARC-AGI-1 from {ARC1_REPO} ...")
    import subprocess
    subprocess.run(["git", "clone", "--depth", "1", f"{ARC1_REPO}.git", str(arc1_dir)], check=True)
    print(f"Saved to {arc1_dir}/data/{{training,evaluation}}/")


def download_arc2():
    """下载 ARC-AGI-2 数据。"""
    arc2_dir = RAW_DIR / "arc2"

    if arc2_dir.exists():
        print(f"ARC-AGI-2 已存在: {arc2_dir}")
        return

    print(f"Cloning ARC-AGI-2 from {ARC2_REPO} ...")
    import subprocess
    subprocess.run(["git", "clone", "--depth", "1", f"{ARC2_REPO}.git", str(arc2_dir)], check=True)
    print(f"Saved to {arc2_dir}/data/")


def prepare_arc12():
    """将 ARC-AGI-1/2 的静态 JSON 转为统一格式。"""
    PREPARED_DIR.mkdir(parents=True, exist_ok=True)

    for version, subdir in [("arc1", "arc1/data"), ("arc2", "arc2/data")]:
        src = RAW_DIR / subdir
        if not src.exists():
            print(f"Skip {version}: {src} not found.")
            continue

        records = []
        for split in ["training", "evaluation"]:
            split_dir = src / split
            if not split_dir.exists():
                continue
            for json_file in sorted(split_dir.glob("*.json")):
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                records.append({
                    "id": json_file.stem,
                    "split": split,
                    "train": data.get("train", []),
                    "test": data.get("test", []),
                    "version": version,
                })

        out_path = PREPARED_DIR / f"{version}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        print(f"{version}: {len(records)} tasks -> {out_path}")


def setup_arc3():
    """准备 ARC-AGI-3 交互式环境。"""
    arc3_dir = RAW_DIR / "arc3"
    arc3_dir.mkdir(exist_ok=True)
    agents_dir = arc3_dir / "agents"

    if agents_dir.exists():
        print(f"ARC-AGI-3 agents 已存在: {agents_dir}")
    else:
        print(f"\nCloning ARC-AGI-3 agents from {ARC3_AGENTS_REPO} ...")
        import subprocess
        subprocess.run(["git", "clone", "--depth", "1", f"{ARC3_AGENTS_REPO}.git", str(agents_dir)], check=True)

    print("\n=== ARC-AGI-3 环境准备 ===")
    print("ARC-AGI-3 是交互式环境，需要 API key。")
    print(f"官方 Agents 仓库: {ARC3_AGENTS_REPO}")
    print("\n下一步：")
    print("1. 访问 https://arcprize.org/ 申请 API key")
    print("2. cp agents/.env.example agents/.env 并填入 API key")
    print("3. cd agents && uv run main.py --agent=random --game=ls20")
    print("\n注意：ARC-AGI-3 需要通过 API 在线运行，数据集不在本地。")
    print("      本项目的 ARC-AGI-3 适配器将在 Stage 2 实现。")


def inspect():
    """快速查看数据统计。"""
    for path in sorted(PREPARED_DIR.glob("*.jsonl")):
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        sample = json.loads(lines[0]) if lines else {}
        print(f"\n{path.name}: {len(lines)} tasks")
        print(f"  keys: {list(sample.keys())}")
        print(f"  train pairs: {len(sample.get('train', []))}")
        print(f"  test pairs: {len(sample.get('test', []))}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ARC-AGI 数据下载与预处理")
    parser.add_argument("action", choices=["setup", "prepare", "inspect", "all"], default="all", nargs="?")
    args = parser.parse_args()

    if args.action in ("setup", "all"):
        download_arc1()
        download_arc2()
        setup_arc3()
    if args.action in ("prepare", "all"):
        prepare_arc12()
    if args.action in ("inspect", "all"):
        inspect()
