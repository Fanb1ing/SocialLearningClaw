from __future__ import annotations

import argparse
import os

from .cc_agent.adapters.openai_compatible import OpenAICompatibleAgent
from .pipeline import RunConfig, run_stage1


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--prepared",
        required=True,
        default="data/pbench/prepared/all.jsonl",
        help="Path to prepared cosmos_reason1 jsonl",
    )
    p.add_argument(
        "--base-url",
        required=True,
        default="https://openrouter.ai/api/v1",
        help="OpenAI-compatible base URL (e.g. https://openrouter.ai/api/v1)",
    )
    # Make api-key optional; prefer env for VS Code debug envFile.
    p.add_argument("--api-key", required=False, default="", help="API key (or set OPENROUTER_API_KEY in env)")
    p.add_argument("--model", required=True, default="moonshotai/kimi-k2.6", help="model name")
    p.add_argument("--max-problems", type=int, default=5)
    p.add_argument("--max-iters", type=int, default=2)
    args = p.parse_args()

    api_key = (args.api_key or "").strip()
    if not api_key:
        # Common env var names
        for k in ["OPENROUTER_API_KEY", "OPENAI_API_KEY", "API_KEY"]:
            v = (os.environ.get(k) or "").strip()
            if v:
                api_key = v
                break

    if not api_key:
        raise SystemExit(
            "Missing API key. Provide --api-key or set OPENROUTER_API_KEY in your environment/.env (VS Code envFile)."
        )

    agent = OpenAICompatibleAgent(base_url=args.base_url, api_key=api_key, model=args.model)

    cfg = RunConfig(prepared_dataset_path=args.prepared, max_problems=args.max_problems)
    cfg.stop.max_iters = args.max_iters

    run_dir = run_stage1(agent=agent, cfg=cfg)
    print(run_dir)


if __name__ == "__main__":
    main()
