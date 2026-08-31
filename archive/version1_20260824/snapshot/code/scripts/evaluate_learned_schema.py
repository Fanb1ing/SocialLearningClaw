from __future__ import annotations

import argparse
import json

from socialclaw.schema.evaluation import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a finished learned Schema snapshot against Gold.")
    parser.add_argument("--learned", required=True, help="Learned schema.json")
    parser.add_argument("--memory", required=True, help="Matching learned memory.json")
    parser.add_argument("--gold", default="gold/arc_agi3/v1", help="ARC Gold root")
    parser.add_argument("--output", required=True, help="Evaluation output directory")
    arguments = parser.parse_args()
    metrics = run_evaluation(arguments.learned, arguments.memory, arguments.gold, arguments.output)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
