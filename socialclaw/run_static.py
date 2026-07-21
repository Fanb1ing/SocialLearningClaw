from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .benchmarks import BenchmarkAdapter, BenchmarkSample, ContextMathBenchmark, IntPhys2Benchmark
from .experiment import (
    METHODS,
    AttemptResult,
    ExperimentBudget,
    ExperimentConfig,
    SampleResult,
    make_output_dir,
    write_manifest,
    write_results,
)
from .llm import OpenAIChatClient
from .methods import MethodController
from .schema import build_schema_system
from .utils import load_dotenv


def _text_content(text: str, images: Sequence[str]) -> Any:
    if not images:
        return text
    content: List[Dict[str, Any]] = [{"type": "text", "text": text}]
    content.extend(
        {"type": "image_url", "image_url": {"url": image}}
        for image in images
    )
    return content


def _sample_images(benchmark: BenchmarkAdapter, sample: BenchmarkSample) -> List[str]:
    if isinstance(benchmark, IntPhys2Benchmark):
        return benchmark.extract_frame_data_urls(sample)
    return []


def _messages(
    *,
    benchmark: BenchmarkAdapter,
    sample: BenchmarkSample,
    method: str,
    method_context: str,
    demonstrations: Sequence[BenchmarkSample],
    retry_feedback: str,
) -> List[Dict[str, Any]]:
    system = benchmark.system_prompt()
    if method == "withrule":
        system += "\n\n=== Task rules ===\n" + benchmark.rule_context()
    if method_context:
        system += "\n\n" + method_context
    if retry_feedback:
        system += "\n\n" + retry_feedback

    messages: List[Dict[str, Any]] = [{"role": "system", "content": system}]
    if method == "icl":
        for demo in demonstrations:
            messages.append(
                {
                    "role": "user",
                    "content": _text_content(
                        benchmark.user_prompt(demo), _sample_images(benchmark, demo)
                    ),
                }
            )
            if benchmark.name == "contextmath":
                answer = f"\\boxed{{{demo.gold}}}"
            else:
                answer = str(demo.gold)
            messages.append({"role": "assistant", "content": answer})

    messages.append(
        {
            "role": "user",
            "content": _text_content(
                benchmark.user_prompt(sample), _sample_images(benchmark, sample)
            ),
        }
    )
    return messages


def run_static_experiment(
    *,
    config: ExperimentConfig,
    benchmark: BenchmarkAdapter,
    samples: List[BenchmarkSample],
    demonstrations: Sequence[BenchmarkSample],
    api_key: str,
    embed_model: str,
) -> Path:
    output_dir = make_output_dir(config)
    write_manifest(
        output_dir,
        config,
        sample_ids=(sample.id for sample in samples),
        dataset_fingerprint=benchmark.fingerprint(config.split),
        demonstration_ids=(sample.id for sample in demonstrations),
    )

    llm = OpenAIChatClient(
        base_url=config.base_url, api_key=api_key, model=config.model
    )
    embedder = None
    if config.method in {"rag", "amem", "tgm", "schema"}:
        from sentence_transformers import SentenceTransformer

        embedder = SentenceTransformer(embed_model)
    schema_manager = None
    if config.method == "schema":
        schema_manager = build_schema_system(
            output_dir / "schema",
            llm=llm,
            embedder=embedder,
        )
    controller = MethodController(
        method=config.method,
        openai_client=llm.client,
        model=config.model,
        embedder=embedder,
        schema_manager=schema_manager,
    )

    results: List[SampleResult] = []
    for index, sample in enumerate(samples, 1):
        attempts: List[AttemptResult] = []
        retry_feedback = ""
        method_query = (
            benchmark.schema_query(sample)
            if config.method == "schema"
            else sample.prompt
        )
        for attempt_index in range(config.budget.max_attempts):
            context = controller.context(method_query)
            messages = _messages(
                benchmark=benchmark,
                sample=sample,
                method=config.method,
                method_context=context,
                demonstrations=demonstrations,
                retry_feedback=retry_feedback,
            )
            response = llm.complete(
                messages,
                temperature=config.temperature,
                max_tokens=config.budget.max_tokens_per_call,
            )
            evaluation = benchmark.evaluate(response.text, sample)
            attempts.append(
                AttemptResult(
                    attempt=attempt_index + 1,
                    prediction=evaluation.prediction,
                    correct=evaluation.correct,
                    response=response.text,
                    usage=response.usage,
                )
            )
            if evaluation.correct:
                break
            if config.feedback == "binary":
                controller.after_failed_attempt(task=sample.prompt, response=response.text)
                retry_feedback = (
                    "The previous response was evaluated as incorrect. No target answer is "
                    "available; try a materially different, independently checked approach."
                )

        final = attempts[-1]
        if config.feedback == "binary":
            controller.after_sample(
                task=method_query,
                response=final.response,
                correct=final.correct,
                domain=benchmark.name,
            )
        results.append(
            SampleResult(
                sample_id=sample.id,
                correct=final.correct,
                prediction=final.prediction,
                attempts=attempts,
                metadata=sample.metadata,
            )
        )
        print(
            f"[{index:03d}/{len(samples):03d}] {sample.id}: "
            f"{'correct' if final.correct else 'wrong'} prediction={final.prediction!r}"
        )

    write_results(output_dir, results)
    return output_dir


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(str(project_root / ".env"))

    parser = argparse.ArgumentParser(
        description="Run a unified static benchmark experiment (ContextMATH or IntPhys2)"
    )
    parser.add_argument("--benchmark", required=True, choices=["contextmath", "intphys2"])
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="")
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--feedback", choices=["none", "binary"], default="binary")
    parser.add_argument("--num-demos", type=int, default=3)
    parser.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--seconds-per-frame", type=float, default=1.5)
    parser.add_argument("--max-frames", type=int, default=12)
    args = parser.parse_args()

    if args.num_demos < 0:
        raise SystemExit("--num-demos must be >= 0")

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing API key: use --api-key or OPENROUTER_API_KEY")

    if args.benchmark == "contextmath":
        data_dir = args.data_dir or "data/contextmath"
        benchmark: BenchmarkAdapter = ContextMathBenchmark(data_dir)
        split = args.split or benchmark.default_split
        samples = benchmark.load(split, args.max_samples)
        demonstrations = (
            benchmark.load_icl_demonstrations(args.num_demos)
            if args.method == "icl"
            else []
        )
        demonstration_fingerprint = (
            benchmark.fingerprint("math_500_sg") if demonstrations else ""
        )
        reserved_demo_ids: List[str] = []
    else:
        data_dir = args.data_dir or "data/intphys2"
        benchmark = IntPhys2Benchmark(
            data_dir,
            seconds_per_frame=args.seconds_per_frame,
            max_frames=args.max_frames,
        )
        split = args.split or benchmark.default_split
        # IntPhys2 has no separate local demonstration split. Reserve the same
        # prefix for every method so all compared runs evaluate identical IDs.
        loaded = benchmark.load(split, 0)
        if args.num_demos >= len(loaded) and loaded:
            raise SystemExit("--num-demos must leave at least one IntPhys2 evaluation sample")
        reserved = loaded[: args.num_demos]
        evaluation_pool = loaded[args.num_demos :]
        samples = evaluation_pool[: args.max_samples] if args.max_samples else evaluation_pool
        demonstrations = reserved if args.method == "icl" else []
        demonstration_fingerprint = benchmark.fingerprint(split) if demonstrations else ""
        reserved_demo_ids = [sample.id for sample in reserved]

    config = ExperimentConfig(
        benchmark=args.benchmark,
        method=args.method,
        model=args.model,
        split=split,
        temperature=args.temperature,
        base_url=args.base_url,
        output_root=args.output_root,
        feedback=args.feedback,
        budget=ExperimentBudget(
            max_samples=args.max_samples,
            max_attempts=args.max_attempts,
            max_tokens_per_call=args.max_tokens,
        ),
        extra={
            "num_demos": args.num_demos if args.method == "icl" else 0,
            "demonstration_fingerprint": demonstration_fingerprint,
            "reserved_demo_ids": reserved_demo_ids,
            "embed_model": args.embed_model if args.method in {"rag", "amem", "tgm", "schema"} else "",
            "seconds_per_frame": args.seconds_per_frame if args.benchmark == "intphys2" else None,
            "max_frames": args.max_frames if args.benchmark == "intphys2" else None,
        },
    )
    output = run_static_experiment(
        config=config,
        benchmark=benchmark,
        samples=samples,
        demonstrations=demonstrations,
        api_key=api_key,
        embed_model=args.embed_model,
    )
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
