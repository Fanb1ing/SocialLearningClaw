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
from .stop_policy import StopConfig
from .utils import load_dotenv


def _detect_benchmark(path: str) -> str:
    name = os.path.basename(path).lower()
    if "clbench" in name or "cl_bench" in name:
        return "clbench"
    if "pbench" in name:
        return "pbench"
    if "arc" in name:
        return "arc"
    if "cosmos" in name or "reason" in name:
        return "pbench"
    return "benchmark"


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
    return load_pbench(path)


def main() -> None:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    load_dotenv(os.path.join(project_root, ".env"))

    p = argparse.ArgumentParser(description="Run Schema-centric reasoning pipeline (CL-bench / PBench / ARC)")
    p.add_argument("--prepared", required=True, help="Path to prepared dataset .jsonl")
    p.add_argument("--base-url", default="https://openrouter.ai/api/v1", help="OpenAI-compatible base URL")
    p.add_argument("--api-key", default="", help="API key (or set OPENROUTER_API_KEY in .env)")
    p.add_argument("--model", required=True, help="LLM model name")
    p.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5", help="SentenceTransformer embedding model")
    p.add_argument("--max-problems", type=int, default=5, help="Maximum number of problems to run")
    p.add_argument("--max-iters", type=int, default=2, help="Maximum attempts per problem")
    p.add_argument("--top-k", type=int, default=5, help="Top-K concepts to retrieve from schema")
    p.add_argument("--threshold", type=float, default=0.75, help="Embedding similarity threshold for retrieval")
    p.add_argument("--auto-yes", action="store_true", help="Skip human interaction; use LLM to generate schema")
    p.add_argument("--always-ask-correction", action="store_true", help="Always ask human for correction on wrong answers (debug mode)")
    p.add_argument("--runs-dir", default="runs", help="Root directory for run output (default: runs)")
    p.add_argument(
        "--schema-dir",
        default="schema",
        help="Schema directory. Default: saved inside run_dir/schema (per-run isolation). "
             "Set explicitly to reuse a previous schema across runs.",
    )
    p.add_argument("--reset-schema", action="store_true", help="Clear existing schema before run")
    p.add_argument("--problem-id", dest="problem_ids", action="append", default=None,
                   help="Run only specified problem ID(s). Repeatable.")
    p.add_argument("--dry-run", action="store_true", help="Build prompts only; do not call LLM")
    p.add_argument("--show-prompt", action="store_true", help="Print prompt before sending to LLM")
    p.add_argument("--no-retrieval", action="store_true", help="Debug: bypass embedding retrieval, inject ALL concepts into every prompt")
    p.add_argument("--context-id", default=None, help="Run all tasks under a specific CL-bench context_id")
    p.add_argument("--group-by-context", action="store_true", default=None,
                   help="Group problems by context_id; schema iterates within each context (auto-enabled for CL-bench)")
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
    )

    problems = list(_load_dataset(args.prepared))
    benchmark_name = _detect_benchmark(args.prepared)

    if args.context_id:
        problems = [p for p in problems if p.meta.get("context_id") == args.context_id]
        if not problems:
            raise SystemExit(f"No tasks found for context_id={args.context_id}")
        problems.sort(key=lambda p: p.id)
        args.reset_schema = True
        print(f"[context-mode] Found {len(problems)} tasks for context={args.context_id}.")

    group_by_context = args.group_by_context
    if group_by_context is None:
        group_by_context = benchmark_name == "clbench"

    cfg = PipelineConfig(
        max_problems=args.max_problems,
        top_k_concepts=args.top_k,
        similarity_threshold=args.threshold,
        always_ask_correction=args.always_ask_correction,
        stop=StopConfig(max_iters=args.max_iters),
        runs_dir=args.runs_dir,
        benchmark_name=benchmark_name,
        schema_dir=args.schema_dir,
        auto_yes=args.auto_yes,
        reset_schema=args.reset_schema,
        problem_ids=args.problem_ids,
        dry_run=args.dry_run,
        show_prompt=args.show_prompt,
        group_by_context=group_by_context,
        no_retrieval=args.no_retrieval,
        model=args.model,
    )

    run_dir = run_stage1(agent=agent, embedder=embedder, problems=problems, cfg=cfg)
    print(f"Run completed: {run_dir}")


if __name__ == "__main__":
    main()
