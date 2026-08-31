from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

from socialclaw.v2.agents.prompts import (
    EXPLORATION_INSTRUCTIONS,
    MAIN_INSTRUCTIONS,
    UPDATE_INSTRUCTIONS,
)
from socialclaw.v2.efps import COGNITION_CONTRACT_VERSION


INSTRUCTIONS = {
    "exploration_agent": EXPLORATION_INSTRUCTIONS,
    "main_agent": MAIN_INSTRUCTIONS,
    "update_agent": UPDATE_INSTRUCTIONS,
    "recovery_update_agent": UPDATE_INSTRUCTIONS,
}
METHODS = {
    "exploration_agent": "text",
    "main_agent": "structured",
    "update_agent": "structured",
    "recovery_update_agent": "structured",
}


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _episode_created_at(run_dir: Path) -> str:
    episodes = sorted((run_dir / "trajectory" / "episodes").glob("*.json"))
    if len(episodes) != 1:
        raise ValueError(f"Expected one trajectory episode in {run_dir}")
    payload = json.loads(episodes[0].read_text(encoding="utf-8"))
    return str(payload["episode"]["created_at"])


def _call_record(name: str, audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "agent": name,
        "method": METHODS[name],
        "instructions_sha256": _sha256_text(INSTRUCTIONS[name]),
        "payload_sha256": _sha256_text(str(audit.get("input_text") or "")),
        "image_sha256": [
            str(item["sha256"]) for item in audit.get("image_inputs") or []
        ],
        "output": audit.get("output"),
        "model": audit.get("model"),
        "usage": audit.get("usage") or {},
        "tool_trace": audit.get("tool_trace") or [],
        "usage_rounds": audit.get("usage_rounds") or [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze audited V2 model logical calls into a deterministic fixture."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--reset-on-game-over",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="record the all-game recovery setting (enabled by default)",
    )
    args = parser.parse_args()

    timeline_path = args.run_dir / "timeline.json"
    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    summary = timeline["summary"]
    calls = []
    for event in timeline["events"]:
        audits = event.get("agent_calls") or {}
        order = (
            ("update_agent",)
            if int(event.get("step") or 0) == 0
            else (
                "exploration_agent",
                "main_agent",
                "update_agent",
                "recovery_update_agent",
            )
        )
        for name in order:
            if name in audits:
                calls.append(_call_record(name, audits[name]))

    payload = {
        "format_version": 1,
        "cognition_contract_version": COGNITION_CONTRACT_VERSION,
        "description": (
            "Frozen logical model calls for deterministic environment/cognition replay; "
            "not a fresh provider evaluation."
        ),
        "model": summary["model"],
        "episode_created_at": _episode_created_at(args.run_dir),
        "experiment_config": {
            "game_id": summary["game_id"],
            "model": summary["model"],
            "output_dir_name": args.run_dir.name,
            "max_steps_per_level": summary["max_steps_per_level"],
            "stop_after_levels": summary["stop_after_levels"],
            "reset_on_game_over": bool(args.reset_on_game_over),
            "compact_process": timeline.get("process_prompt_detail") == "compact",
        },
        "expected_summary": summary,
        "source_timeline_sha256": hashlib.sha256(timeline_path.read_bytes()).hexdigest(),
        "calls": calls,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Preserve the audited model object's key order.  It is part of timeline.json's
    # byte-level reproduction even though graph validation itself is order agnostic.
    serialized = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
    with args.output.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            compressed.write(serialized)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "calls": len(calls),
                "compressed_bytes": args.output.stat().st_size,
                "uncompressed_bytes": len(serialized),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
