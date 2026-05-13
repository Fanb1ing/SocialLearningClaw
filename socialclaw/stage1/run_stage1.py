from __future__ import annotations

import argparse
import os
from typing import Iterable

from sentence_transformers import SentenceTransformer

from .agent.openai_compatible import OpenAICompatibleAgent
from .dataset.arc import load_prepared_jsonl as load_arc
from .dataset.clbench import load_prepared_jsonl as load_clbench
from .dataset.pbench import load_prepared_jsonl as load_pbench
from .pipeline import PipelineConfig, run_stage1


def _load_dataset(path: str) -> Iterable:
    name = os.path.basename(path).lower()
    if "pbench" in name:
        return load_pbench(path)
    if "clbench" in name or "cl_bench" in name:
        return load_clbench(path)
    if "arc" in name:
        return load_arc(path)
    if "cosmos" in name or "reason" in name:
        return load_pbench(path)
    # Default to pbench format
    return load_pbench(path)


def _load_dotenv(dotenv_path: str) -> None:
    """Load KEY=VALUE pairs from a .env file into os.environ."""
    if not os.path.exists(dotenv_path):
        return
    with open(dotenv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


def main() -> None:
    # Load .env from project root (if present)
    # run_stage1.py is at socialclaw/stage1/run_stage1.py, so go up 2 levels to project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    dotenv_path = os.path.join(project_root, ".env")
    _load_dotenv(dotenv_path)

    p = argparse.ArgumentParser(description="Run Stage 1: Schema-centric reasoning pipeline")
    p.add_argument(
        "--prepared",
        required=True,
        help="Path to prepared dataset jsonl",
    )
    p.add_argument(
        "--base-url",
        required=True,
        default="https://openrouter.ai/api/v1",
        help="OpenAI-compatible base URL",
    )
    p.add_argument(
        "--api-key",
        required=False,
        default="",
        help="API key (or set OPENROUTER_API_KEY / OPENAI_API_KEY in env)",
    )
    p.add_argument("--model", required=True, default="moonshotai/kimi-k2.6", help="LLM model name")
    p.add_argument(
        "--embed-model",
        default="BAAI/bge-small-en-v1.5",
        help="SentenceTransformer embedding model",
    )
    p.add_argument("--max-problems", type=int, default=5)
    p.add_argument("--max-iters", type=int, default=2)
    p.add_argument("--top-k", type=int, default=5, help="Top-K concepts to retrieve")
    p.add_argument("--threshold", type=float, default=0.75, help="Similarity threshold for concept retrieval")
    p.add_argument("--auto-yes", action="store_true", help="Skip human interaction prompts")
    p.add_argument(
        "--always-ask-correction",
        action="store_true",
        help="Always ask human for correction on wrong answers (debug mode, overrides confidence threshold)",
    )
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--schema-dir", default="schema", help="Schema directory. By default schema is saved inside run_dir/schema for per-run isolation. Set explicitly to reuse a previous schema.")
    # Debug flags
    p.add_argument("--reset-schema", action="store_true", help="Clear existing schema before run (useful when reusing a schema-dir)")
    p.add_argument(
        "--problem-id",
        dest="problem_ids",
        action="append",
        default=None,
        help="Run only specified problem id(s). Can be used multiple times.",
    )
    p.add_argument("--dry-run", action="store_true", help="Build prompts only; do not call LLM")
    p.add_argument("--show-prompt", action="store_true", help="Print prompt before sending to LLM")
    p.add_argument("--context-id", default=None, help="Run all tasks under a specific CL-bench context_id (schema iterates across tasks)")
    args = p.parse_args()

    api_key = (args.api_key or "").strip()
    if not api_key:
        for k in ["OPENROUTER_API_KEY", "OPENAI_API_KEY", "API_KEY"]:
            v = (os.environ.get(k) or "").strip()
            if v:
                api_key = v
                break

    if not api_key:
        raise SystemExit(
            "Missing API key. Provide --api-key or set OPENROUTER_API_KEY in your environment/.env."
        )

    print(f"Loading embedding model: {args.embed_model} ...")
    embedder = SentenceTransformer(args.embed_model)

    agent = OpenAICompatibleAgent(
        base_url=args.base_url,
        api_key=api_key,
        model=args.model,
    )

    problems = list(_load_dataset(args.prepared))

    # Filter by context-id for CL-bench iterative schema usage
    if args.context_id:
        problems = [p for p in problems if p.meta.get("context_id") == args.context_id]
        if not problems:
            raise SystemExit(f"No tasks found for context_id={args.context_id}")
        problems.sort(key=lambda p: p.id)
        # When running a single context, always start from empty schema and iterate
        args.reset_schema = True
        print(f"[context-mode] Found {len(problems)} tasks for context={args.context_id}. Schema will iterate across tasks.")

    from .stop_policy import StopConfig

    cfg = PipelineConfig(
        max_problems=args.max_problems,
        top_k_concepts=args.top_k,
        similarity_threshold=args.threshold,
        always_ask_correction=args.always_ask_correction,
        stop=StopConfig(max_iters=args.max_iters),
        runs_dir=args.runs_dir,
        schema_dir=args.schema_dir,
        auto_yes=args.auto_yes,
        reset_schema=args.reset_schema,
        problem_ids=args.problem_ids,
        dry_run=args.dry_run,
        show_prompt=args.show_prompt,
    )

    run_dir = run_stage1(agent=agent, embedder=embedder, problems=problems, cfg=cfg)
    print(f"Run completed: {run_dir}")


if __name__ == "__main__":
    main()
