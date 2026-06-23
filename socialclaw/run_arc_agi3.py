from __future__ import annotations

import argparse
import os

from sentence_transformers import SentenceTransformer

from .agent.openai_compatible import OpenAICompatibleAgent
from .arc_runner import run_arc_agi3
from .utils import load_dotenv


def main() -> None:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    load_dotenv(os.path.join(project_root, ".env"))

    # .env uses ARC_AGI_API_KEY but the arc_agi SDK reads ARC_API_KEY
    if os.environ.get("ARC_AGI_API_KEY") and not os.environ.get("ARC_API_KEY"):
        os.environ["ARC_API_KEY"] = os.environ["ARC_AGI_API_KEY"]

    # arc_agi SDK (three.arcprize.org) fails with SSL errors through the system proxy,
    # but OpenRouter requires the proxy for regional access.
    # Solution: only bypass proxy for arcprize.org; keep proxy active for everything else.
    _arc_no_proxy = "three.arcprize.org,arcprize.org"
    existing = os.environ.get("no_proxy") or os.environ.get("NO_PROXY") or ""
    combined = f"{_arc_no_proxy},{existing}" if existing else _arc_no_proxy
    os.environ["no_proxy"] = combined
    os.environ["NO_PROXY"] = combined

    p = argparse.ArgumentParser(description="Run ARC-AGI-3 with schema-based reasoning")
    p.add_argument("--game-id", required=True, help="ARC-AGI-3 game ID (e.g. sk48-d8078629)")
    p.add_argument("--base-url", default="https://openrouter.ai/api/v1", help="OpenAI-compatible base URL")
    p.add_argument("--api-key", default="", help="API key (or set OPENROUTER_API_KEY in .env)")
    p.add_argument("--model", default="qwen/qwen2.5-vl-72b-instruct", help="LLM model name")
    p.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5", help="Embedding model")
    p.add_argument("--max-steps", type=int, default=200, help="Max steps per level")
    p.add_argument("--max-retries", type=int, default=3, help="Max retries per level on GAME_OVER")
    p.add_argument(
        "--schema-dir",
        default="schema_arc_agi3",
        help="Schema directory. Default: saved inside run_dir/schema (per-run isolation). "
             "Set explicitly to reuse a previous schema across runs.",
    )
    p.add_argument("--runs-dir", default="runs", help="Root directory for run output (default: runs)")
    p.add_argument("--reset-schema", action="store_true", help="Clear schema before run (useful when reusing a schema-dir)")
    p.add_argument("--render", action="store_true", help="Enable terminal rendering")
    p.add_argument("--auto-yes", action="store_true", help="Auto-skip human questions; LLM generates schema instead")
    p.add_argument("--always-ask-correction", action="store_true", help="Always ask human for correction on wrong answers (debug mode)")
    p.add_argument("--correction-conf-threshold", type=float, default=-1.0,
                   help="Confidence threshold for triggering human correction (default: -1.0, always trigger)")
    p.add_argument("--max-tokens", type=int, default=8192, help="Max tokens per LLM call")
    p.add_argument("--use-llm-concepts", action="store_true", help="Use vision LLM to extract concepts (better quality, higher cost)")
    p.add_argument("--no-retrieval", action="store_true", help="Debug: bypass embedding retrieval, inject ALL schema concepts into every prompt")
    args = p.parse_args()

    api_key = (args.api_key or "").strip()
    if not api_key:
        for k in ["OPENROUTER_API_KEY", "OPENAI_API_KEY", "API_KEY"]:
            v = (os.environ.get(k) or "").strip()
            if v:
                api_key = v
                break
    if not api_key:
        raise SystemExit("Missing API key. Provide --api-key or set OPENROUTER_API_KEY in .env.")

    print(f"Loading embedding model: {args.embed_model} ...")
    embedder = SentenceTransformer(args.embed_model)

    agent = OpenAICompatibleAgent(
        base_url=args.base_url,
        api_key=api_key,
        model=args.model,
        max_tokens=args.max_tokens,
    )

    run_dir = run_arc_agi3(
        game_id=args.game_id,
        agent=agent,
        embedder=embedder,
        max_steps_per_level=args.max_steps,
        max_retries_per_level=args.max_retries,
        schema_dir=args.schema_dir,
        runs_dir=args.runs_dir,
        reset_schema=args.reset_schema,
        render_mode="terminal" if args.render else None,
        auto_yes=args.auto_yes,
        always_ask_correction=args.always_ask_correction,
        correction_conf_threshold=args.correction_conf_threshold,
        use_llm_concepts=args.use_llm_concepts,
        no_retrieval=args.no_retrieval,
        model=args.model,
    )
    print(f"Run completed: {run_dir}")


if __name__ == "__main__":
    main()
