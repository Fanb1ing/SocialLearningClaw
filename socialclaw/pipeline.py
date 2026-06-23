from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from typing import Iterable, List, Optional

from .agent.base import Agent, AgentAttempt, ReasoningTrace, Usage
from .dataset.base import Problem
from .evaluator import evaluate
from .human_io import HumanIO
from .logging import write_episode
from .prompt_builder import build_prompt
from .schema.graph import Concept, Relation, SchemaGraph
from .schema.initializer import SchemaInitializer
from .schema.retriever import RetrieveResult, SchemaRetriever
from .schema.storage import SchemaStorage
from .stop_policy import StopConfig, should_stop
from .types import AttemptRecord, Episode
from .utils import add_concepts_with_embeddings, add_relations_resolved, make_run_dir, resolve_relation_names, save_cmd


@dataclass
class PipelineConfig:
    max_problems: int = 20
    top_k_concepts: int = 5
    similarity_threshold: float = 0.75
    correction_conf_threshold: float = 0.6
    always_ask_correction: bool = False
    stop: StopConfig = field(default_factory=StopConfig)
    runs_dir: str = "runs"
    benchmark_name: str = "benchmark"
    schema_dir: str = "schema"
    auto_yes: bool = False
    reset_schema: bool = False
    problem_ids: Optional[List[str]] = None
    dry_run: bool = False
    show_prompt: bool = False
    group_by_context: bool = False
    no_retrieval: bool = False
    model: Optional[str] = None


def _ensure_schema_files(schema_dir: str) -> tuple[str, str, str, str]:
    os.makedirs(schema_dir, exist_ok=True)
    concepts_path = os.path.join(schema_dir, "concepts.jsonl")
    relations_path = os.path.join(schema_dir, "relations.jsonl")
    embeddings_path = os.path.join(schema_dir, "concept_embeddings.npy")
    concept_ids_path = os.path.join(schema_dir, "concept_ids.json")
    return concepts_path, relations_path, embeddings_path, concept_ids_path


def _reset_schema_dir(schema_dir: str) -> None:
    if os.path.exists(schema_dir):
        shutil.rmtree(schema_dir)
    os.makedirs(schema_dir, exist_ok=True)


def _load_or_init_schema(
    schema_dir: str, reset: bool = False
) -> tuple[SchemaGraph, SchemaStorage, dict]:
    if reset:
        _reset_schema_dir(schema_dir)
    concepts_path, relations_path, embeddings_path, concept_ids_path = _ensure_schema_files(schema_dir)
    storage = SchemaStorage(concepts_path, relations_path, embeddings_path, concept_ids_path)
    graph, embeddings = storage.load()
    return graph, storage, embeddings


def _update_schema_from_feedback(graph: SchemaGraph, trace, correct: bool) -> None:
    """Stage 2: reinforce or correct schema based on evaluation result."""
    delta = 0.05 if correct else -0.05

    for cid in trace.concepts:
        c = graph.get_concept(cid)
        if not c:
            c = graph.get_concept_by_name(cid)
        if c:
            new_conf = max(0.1, min(0.95, c.confidence + delta))
            graph.update_concept(c.id, confidence=new_conf)

    for src, tgt, rel_type in trace.relations:
        src_c = graph.get_concept(src) or graph.get_concept_by_name(src)
        tgt_c = graph.get_concept(tgt) or graph.get_concept_by_name(tgt)
        if src_c and tgt_c:
            r = graph.get_relation(src_c.id, tgt_c.id, rel_type)
            if not r:
                r = graph.find_relation(src_c.name, tgt_c.name, rel_type)
            if r:
                new_weight = max(0.1, min(0.95, r.weight + delta))
                graph.update_relation(r.source, r.target, r.relation_type, weight=new_weight)


def run_stage1(
    *,
    agent: Agent,
    embedder,
    problems: Iterable[Problem],
    cfg: PipelineConfig,
) -> str:
    model = cfg.model or "unknown_model"
    run_dir = make_run_dir(cfg.runs_dir, cfg.benchmark_name, model)
    save_cmd(run_dir)

    initializer = SchemaInitializer(agent)
    human_io = HumanIO(auto_yes=cfg.auto_yes)

    if cfg.group_by_context:
        graph = SchemaGraph()
        storage = None
        embeddings: dict = {}
        retriever = SchemaRetriever(graph, embeddings, embedder, agent=agent)
    else:
        schema_dir = cfg.schema_dir
        if schema_dir == "schema":
            schema_dir = os.path.join(run_dir, "schema")
        graph, storage, embeddings = _load_or_init_schema(schema_dir, reset=cfg.reset_schema)
        retriever = SchemaRetriever(graph, embeddings, embedder, agent=agent)

    problem_list = list(problems)
    if cfg.group_by_context and problem_list:
        problem_list.sort(key=lambda p: (
            p.meta.get("context_id", ""),
            p.meta.get("msg_count", 0),
            p.id,
        ))

    count = 0
    current_context_id: Optional[str] = None
    current_context_dir: Optional[str] = None
    for problem in problem_list:
        if cfg.problem_ids and problem.id not in cfg.problem_ids:
            continue

        if cfg.group_by_context:
            ctx_id = problem.meta.get("context_id", "")
            if ctx_id and ctx_id != current_context_id:
                if current_context_id is not None and storage is not None:
                    storage.save(graph, embeddings)
                    print(f"[context] Switching from {current_context_id[:8]} to {ctx_id[:8]}, saving schema...")
                current_context_id = ctx_id
                current_context_dir = os.path.join(run_dir, ctx_id[:8])
                context_schema_dir = os.path.join(current_context_dir, "schema")
                graph, storage, embeddings = _load_or_init_schema(context_schema_dir, reset=cfg.reset_schema)
                retriever = SchemaRetriever(graph, embeddings, embedder, agent=agent)
                print(f"[context] Schema dir: {context_schema_dir}")

        ep = Episode(problem=problem)
        if cfg.model:
            ep.model = cfg.model

        # 1. Schema retrieve + sufficiency check
        if cfg.no_retrieval:
            concepts = graph.list_concepts()
            result = RetrieveResult(matched=concepts, missing=[])
            sufficient = True
        else:
            result = retriever.retrieve(
                problem, top_k=cfg.top_k_concepts, threshold=cfg.similarity_threshold
            )
            concepts = result.matched
            sufficient = retriever.is_sufficient(result)

        if not sufficient:
            if cfg.auto_yes:
                print(f"[auto-yes] Schema insufficient for {problem.id}, missing: {result.missing}, auto-generating...")
                auto_concepts, auto_relations = initializer.generate_schema(problem)
                add_concepts_with_embeddings(graph, embeddings, embedder, auto_concepts)
                add_relations_resolved(graph, auto_relations)
                ep.flags.append("agent_auto_init")
                result = retriever.retrieve(
                    problem,
                    top_k=cfg.top_k_concepts,
                    threshold=cfg.similarity_threshold,
                )
                concepts = result.matched
            else:
                missing_desc = initializer.describe_missing(problem, concepts, missing=result.missing)
                human_answer = human_io.ask(
                    question=missing_desc["question"],
                    context=missing_desc["context"],
                    hint=missing_desc["hint"],
                )
                if human_answer.strip():
                    new_concepts, new_relations = initializer.parse_human_answer(human_answer, problem)
                    add_concepts_with_embeddings(graph, embeddings, embedder, new_concepts)
                    add_relations_resolved(graph, new_relations)
                    ep.flags.append("human_init_concepts")
                    # Merge human-provided concepts directly; skip re-retrieval that may
                    # miss them due to poor embedding recall for newly added concepts.
                    existing_ids = {c.id for c in concepts}
                    concepts = concepts + [c for c in new_concepts if c.id not in existing_ids]

        for attempt_index in range(cfg.stop.max_iters):
            # 2. Agent answers with schema assistance
            concept_ids = [c.id for c in concepts]
            subgraph = graph.subgraph(concept_ids, depth=1)
            prompt = build_prompt(
                problem=problem,
                subgraph=subgraph,
                attempt_index=attempt_index,
            )

            if cfg.show_prompt:
                print(f"\n=== Prompt for {problem.id} (attempt {attempt_index}) ===")
                print(prompt)
                print("=" * 50)

            if cfg.dry_run:
                attempt = AgentAttempt(
                    answer_text="[DRY_RUN]",
                    reasoning_trace=ReasoningTrace(concepts=[], relations=[], explanation="dry_run"),
                    usage=Usage(input_tokens=0, output_tokens=0, total_tokens=0),
                    raw={"dry_run": True},
                )
            else:
                attempt = agent.answer(
                    prompt=prompt,
                    meta={
                        "problem_id": problem.id,
                        "attempt": attempt_index,
                        "problem_meta": problem.meta or {},
                    },
                )

            ep.attempts.append(
                AttemptRecord(
                    input_prompt=prompt,
                    answer_text=attempt.answer_text,
                    reasoning_trace=attempt.reasoning_trace,
                    usage={
                        "input_tokens": attempt.usage.input_tokens,
                        "output_tokens": attempt.usage.output_tokens,
                        "total_tokens": attempt.usage.total_tokens,
                    },
                )
            )

            # 3. Evaluate
            ev = evaluate(problem, attempt, agent=agent)
            ep.evals.append(ev)

            # 4. Compute schema-based reasoning confidence
            reasoning_confidence = graph.compute_confidence(attempt.reasoning_trace)
            ep.reasoning_trace = attempt.reasoning_trace
            ep.reasoning_confidence = reasoning_confidence

            # 5. Stage 2: Schema reinforcement / correction
            if not cfg.dry_run and attempt.reasoning_trace.concepts:
                _update_schema_from_feedback(graph, attempt.reasoning_trace, ev.correct)
                ep.flags.append("schema_reinforce" if ev.correct else "schema_correct")

            stop, reason = should_stop(ep, cfg.stop)
            if stop:
                ep.stop_reason = reason
                break

        if ep.stop_reason is None:
            stop, reason = should_stop(ep, cfg.stop)
            if stop:
                ep.stop_reason = reason

        # 6. High-confidence wrong -> ask human for correction
        if (
            not cfg.dry_run
            and ep.evals
            and not ep.evals[-1].correct
            and (
                ep.reasoning_confidence > cfg.correction_conf_threshold
                or cfg.always_ask_correction
            )
        ):
            correction = human_io.ask_correction(
                problem=problem,
                attempt=ep.attempts[-1],
                reasoning_confidence=ep.reasoning_confidence,
                eval=ep.evals[-1],
            )
            if correction.strip():
                corrected = initializer.parse_correction(correction, problem)
                for c in corrected.get("add_concepts", []):
                    add_concepts_with_embeddings(graph, embeddings, embedder, [c])
                for r in corrected.get("add_relations", []):
                    resolved = resolve_relation_names(graph, r)
                    if resolved:
                        graph.add_relation(resolved)
                for upd in corrected.get("update_concepts", []):
                    cid = upd.get("id")
                    if cid:
                        # parse_correction uses the concept name as id; resolve by name
                        existing = graph.get_concept(cid) or graph.get_concept_by_name(cid)
                        if existing:
                            graph.update_concept(existing.id, **{k: v for k, v in upd.items() if k != "id"})
                ep.flags.append("human_correction")

        # Save schema after each problem
        storage.save(graph, embeddings)

        # 7. Log episode
        if cfg.group_by_context and current_context_dir:
            msg_count = problem.meta.get("msg_count", 0)
            ep_subdir = f"{msg_count:02d}_{problem.id[:8]}"
            write_episode(current_context_dir, ep, subdir=ep_subdir)
        else:
            write_episode(run_dir, ep)

        count += 1
        if count >= cfg.max_problems:
            break

    return run_dir
