from __future__ import annotations

import argparse
import os
from pathlib import Path

from sentence_transformers import SentenceTransformer

from .agent.openai_compatible import OpenAICompatibleAgent
from .arc_runner import run_arc_agi3
from .experiment import METHODS, ExperimentBudget, ExperimentConfig, write_manifest
from .utils import load_dotenv


MEMORY_METHODS = {"reflexion", "expel", "amem", "tgm"}
PROMPT_METHODS = {"naive", "icl", "rag", "withrule"}


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(str(project_root / ".env"))
    if os.environ.get("ARC_AGI_API_KEY") and not os.environ.get("ARC_API_KEY"):
        os.environ["ARC_API_KEY"] = os.environ["ARC_AGI_API_KEY"]
    arc_hosts = "three.arcprize.org,arcprize.org"
    existing_no_proxy = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
    no_proxy = f"{arc_hosts},{existing_no_proxy}" if existing_no_proxy else arc_hosts
    os.environ["NO_PROXY"] = no_proxy
    os.environ["no_proxy"] = no_proxy

    parser = argparse.ArgumentParser(description="Unified ARC-AGI-3 experiment runner")
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--game-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-name", default="", help="Optional display-safe model name")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=1,
        help="Attempts per level; all paper comparisons should use the same value",
    )
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--memory-update-interval", type=int, default=10)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--use-llm-concepts", action="store_true")
    args = parser.parse_args()

    if args.max_attempts != 1 and args.method != "schema":
        raise SystemExit(
            "ARC prompt/memory baselines currently use one uninterrupted environment attempt. "
            "Use --max-attempts 1 for a comparable run."
        )

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing API key: use --api-key or OPENROUTER_API_KEY")

    config = ExperimentConfig(
        benchmark="arc_agi3",
        method=args.method,
        model=args.model,
        split=args.game_id,
        temperature=args.temperature,
        base_url=args.base_url,
        output_root=args.output_root,
        budget=ExperimentBudget(
            max_attempts=args.max_attempts,
            max_steps=args.max_steps,
            max_tokens_per_call=args.max_tokens,
        ),
        extra={
            "embed_model": args.embed_model if args.method in {"rag", "amem", "tgm", "schema"} else "",
            "memory_update_interval": args.memory_update_interval if args.method in MEMORY_METHODS else None,
        },
    )
    config.validate()

    agent = OpenAICompatibleAgent(
        base_url=args.base_url,
        api_key=api_key,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    model_name = args.model_name or args.model

    if args.method in PROMPT_METHODS:
        from .arc_methods import run_baseline

        embedder = SentenceTransformer(args.embed_model) if args.method == "rag" else None
        run_dir = run_baseline(
            mode=args.method,
            game_id=args.game_id,
            agent=agent,
            embedder=embedder,
            max_steps_per_level=args.max_steps,
            runs_dir=args.output_root,
            render_mode="terminal" if args.render else None,
            model=model_name,
        )
    elif args.method in MEMORY_METHODS:
        from .arc_methods import run_game

        embedder = (
            SentenceTransformer(args.embed_model)
            if args.method in {"amem", "tgm"}
            else None
        )
        run_dir = run_game(
            game_id=args.game_id,
            baseline=args.method,
            agent=agent,
            model_name=model_name,
            runs_dir=args.output_root,
            embedder=embedder,
            max_steps=args.max_steps,
            memory_update_interval=args.memory_update_interval,
        )
    else:
        embedder = SentenceTransformer(args.embed_model)
        run_dir = run_arc_agi3(
            game_id=args.game_id,
            agent=agent,
            embedder=embedder,
            max_steps_per_level=args.max_steps,
            max_retries_per_level=args.max_attempts,
            schema_dir="schema_arc_agi3",
            runs_dir=args.output_root,
            reset_schema=False,
            render_mode="terminal" if args.render else None,
            auto_yes=True,
            always_ask_correction=False,
            correction_conf_threshold=1.1,
            use_llm_concepts=args.use_llm_concepts,
            no_retrieval=False,
            model=model_name,
        )

    write_manifest(
        Path(run_dir),
        config,
        sample_ids=[args.game_id],
        dataset_fingerprint="ARC-AGI-3 remote environment",
    )
    print(f"Results: {run_dir}")


if __name__ == "__main__":
    main()
