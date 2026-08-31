from __future__ import annotations

import argparse
import json

from socialclaw.schema.window_induction import run_window_induction


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Induce grounded SchemaNodes from frozen trajectory-memory windows."
    )
    parser.add_argument("memory", help="Phase C memory.json path")
    parser.add_argument("output", help="Review output directory")
    arguments = parser.parse_args()
    report = run_window_induction(arguments.memory, arguments.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
