#!/usr/bin/env python3
"""Summarize canonical ContextMATH or IntPhys2 result files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from socialclaw.experiment import BASELINES


def comparison_signature(manifest: dict) -> str:
    """Fields that must match before methods belong in one comparison table."""
    config = manifest.get("config") or {}
    extra = config.get("extra") or {}
    payload = {
        "model": config.get("model"),
        "split": config.get("split"),
        "temperature": config.get("temperature"),
        "base_url": config.get("base_url"),
        "feedback": config.get("feedback"),
        "budget": config.get("budget"),
        "sample_ids": manifest.get("sample_ids"),
        "dataset_fingerprint": manifest.get("dataset_fingerprint"),
        "reserved_demo_ids": extra.get("reserved_demo_ids"),
        "seconds_per_frame": extra.get("seconds_per_frame"),
        "max_frames": extra.get("max_frames"),
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--benchmark", required=True, choices=["contextmath", "intphys2"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="")
    parser.add_argument(
        "--allow-incomparable",
        action="store_true",
        help="Print a table even when protocol/sample manifests differ",
    )
    args = parser.parse_args()

    root = Path(args.output_root) / args.benchmark
    rows = []
    for method in BASELINES:
        candidates = []
        for path in sorted(root.glob(f"{method}/*/*/*/results.json")):
            manifest_path = path.parent / "manifest.json"
            if not manifest_path.exists():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            config = manifest.get("config") or {}
            if config.get("model") != args.model:
                continue
            if args.split and config.get("split") != args.split:
                continue
            candidates.append(path)
        if not candidates:
            continue
        path = candidates[-1]
        payload = json.loads(path.read_text(encoding="utf-8"))
        manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
        metrics = payload["metrics"]
        rows.append(
            {
                "method": method,
                "model": manifest["config"]["model"],
                "split": manifest["config"]["split"],
                "first": metrics["first_attempt_accuracy"],
                "final": metrics["accuracy"],
                "total": metrics["total"],
                "signature": comparison_signature(manifest),
                "run_dir": str(path.parent),
            }
        )

    if not rows:
        print("No matching results found.")
        return
    signatures = {row["signature"] for row in rows}
    if len(signatures) > 1 and not args.allow_incomparable:
        print(
            "Refusing to combine runs with different protocol or evaluated sample IDs.",
            file=sys.stderr,
        )
        for row in rows:
            print(f"- {row['method']}: {row['run_dir']}", file=sys.stderr)
        print("Use --allow-incomparable only for diagnostic viewing.", file=sys.stderr)
        raise SystemExit(2)
    print("| Method | Model | Split | First attempt | Final | N |")
    print("|---|---|---|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['method']} | {row['model']} | {row['split']} | "
            f"{row['first']:.3f} | {row['final']:.3f} | {row['total']} |"
        )


if __name__ == "__main__":
    main()
