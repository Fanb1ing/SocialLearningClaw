#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from socialclaw.schema.trajectory_pipeline import run_prototype_pipeline


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--min-support", type=int, default=2)
    args = parser.parse_args()
    report = run_prototype_pipeline(
        args.corpus, args.output, window_size=args.window_size, min_support=args.min_support
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
