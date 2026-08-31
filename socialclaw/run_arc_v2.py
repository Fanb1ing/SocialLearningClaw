from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .utils import load_dotenv
from .v2.efps import COGNITION_CONTRACT_VERSION
from .v2.model import OpenAICompatibleVisionModel, RecordedVisionModel
from .v2.runtime import run_arc_online


def _stop_after_levels(value: str) -> int | None:
    normalized = value.strip().lower()
    if normalized in {"all", "unlimited", "none"}:
        return None
    try:
        parsed = int(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "must be a positive integer or 'all'"
        ) from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(str(project_root / ".env"))
    parser = argparse.ArgumentParser(
        description="Run the game-agnostic V2 visual EFPS Agent on a public ARC game"
    )
    parser.add_argument("--game-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--api-key", default="")
    parser.add_argument(
        "--recorded-transcript",
        type=Path,
        help=(
            "replay a committed model-call transcript while re-executing the real "
            "environment and cognition runtime; no API key is used"
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--max-step",
        "--max-steps",
        dest="max_steps",
        type=int,
        default=30,
        help="maximum Agent actions for each level (the budget resets only after a level pass)",
    )
    parser.add_argument(
        "--stop-after-levels",
        type=_stop_after_levels,
        default=1,
        metavar="N|all",
        help="number of levels to test, or 'all' to continue until public WIN/failure",
    )
    parser.add_argument(
        "--reset-on-game-over",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "recover the current level after public GAME_OVER without resetting its "
            "action budget (enabled for every game by default)"
        ),
    )
    parser.add_argument(
        "--compact-process",
        action="store_true",
        help="omit full prompts from process.md while retaining per-step public inputs and EFPS summaries",
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()
    if args.recorded_transcript:
        model = RecordedVisionModel(args.recorded_transcript)
        if model.cognition_contract_version != COGNITION_CONTRACT_VERSION:
            raise SystemExit(
                "Recorded transcript cognition contract does not match current source: "
                f"recorded={model.cognition_contract_version}, "
                f"current={COGNITION_CONTRACT_VERSION}"
            )
        expected = model.experiment_config
        actual = {
            "game_id": args.game_id,
            "model": args.model,
            "output_dir_name": Path(args.output_dir).name,
            "max_steps_per_level": args.max_steps,
            "stop_after_levels": args.stop_after_levels,
            "reset_on_game_over": args.reset_on_game_over,
            "compact_process": args.compact_process,
        }
        mismatched = {
            key: {"expected": expected.get(key), "actual": value}
            for key, value in actual.items()
            if expected.get(key) != value
        }
        if mismatched:
            raise SystemExit(
                "Recorded experiment arguments do not match: "
                + json.dumps(mismatched, ensure_ascii=False, sort_keys=True)
            )
    else:
        api_key = (
            args.api_key
            or os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        if not api_key:
            raise SystemExit(
                "Missing model API key: pass --api-key or configure "
                "OPENROUTER_API_KEY/OPENAI_API_KEY"
            )
        model = OpenAICompatibleVisionModel(
            base_url=args.base_url,
            api_key=api_key,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    summary = run_arc_online(
        args.output_dir,
        game_id=args.game_id,
        model=model,
        max_steps=args.max_steps,
        stop_after_levels=args.stop_after_levels,
        reset_on_game_over=args.reset_on_game_over,
        compact_process=args.compact_process,
    )
    if isinstance(model, RecordedVisionModel):
        model.assert_exhausted()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
