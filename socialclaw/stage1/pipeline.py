from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, List, Optional

from .agent.base import Agent, AgentAttempt, ReasoningTrace, Usage
from .dataset.base import Problem
from .evaluator import evaluate
from .human_io import HumanIO
from .logging import write_episode
from .prompt_builder import build_prompt
from .schema.graph import Concept, Relation, SchemaGraph
from .schema.initializer import SchemaInitializer
from .schema.retriever import SchemaRetriever
from .schema.storage import SchemaStorage
from .stop_policy import StopConfig, should_stop
from .types import AttemptRecord, Episode


@dataclass
class PipelineConfig:
    max_problems: int = 20
    top_k_concepts: int = 5
    similarity_threshold: float = 0.75
    correction_conf_threshold: float = 0.6
    always_ask_correction: bool = False
    stop: StopConfig = field(default_factory=StopConfig)
    runs_dir: str = "runs"
    schema_dir: str = "schema"
    auto_yes: bool = False
    reset_schema: bool = False
    problem_ids: Optional[List[str]] = None
    dry_run: bool = False
    show_prompt: bool = False


def _ensure_schema_files(cfg: PipelineConfig) -> tuple[str, str, str, str]:
    os.makedirs(cfg.schema_dir, exist_ok=True)
    concepts_path = os.path.join(cfg.schema_dir, "concepts.jsonl")
    relations_path = os.path.join(cfg.schema_dir, "relations.jsonl")
    embeddings_path = os.path.join(cfg.schema_dir, "concept_embeddings.npy")
    concept_ids_path = os.path.join(cfg.schema_dir, "concept_ids.json")
    return concepts_path, relations_path, embeddings_path, concept_ids_path


def _reset_schema_dir(schema_dir: str) -> None:
    if os.path.exists(schema_dir):
        shutil.rmtree(schema_dir)
    os.makedirs(schema_dir, exist_ok=True)


def _load_or_init_schema(
    cfg: PipelineConfig,
) -> tuple[SchemaGraph, SchemaStorage, dict]:
    if cfg.reset_schema:
        _reset_schema_dir(cfg.schema_dir)

    concepts_path, relations_path, embeddings_path, concept_ids_path = _ensure_schema_files(
        cfg
    )
    storage = SchemaStorage(concepts_path, relations_path, embeddings_path, concept_ids_path)
    graph, embeddings = storage.load()
    return graph, storage, embeddings


def _resolve_relation_names(graph: SchemaGraph, relation: Relation) -> Optional[Relation]:
    """Convert relation source/target from concept names to concept ids."""
    src = graph.get_concept_by_name(relation.source)
    tgt = graph.get_concept_by_name(relation.target)
    if src and tgt:
        return Relation(
            source=src.id,
            target=tgt.id,
            relation_type=relation.relation_type,
            weight=relation.weight,
            evidence=relation.evidence,
        )
    return None


def _add_concepts_with_embeddings(graph, embeddings, embedder, concepts):
    """Add concepts to graph and compute their embeddings."""
    for c in concepts:
        graph.add_concept(c)
        try:
            emb = embedder.encode(
                f"{c.name}: {c.description}",
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            if emb.ndim == 2:
                emb = emb[0]
            embeddings[c.id] = emb
        except Exception:
            pass


def _add_relations_resolved(graph, relations):
    """Add relations to graph, resolving concept names to ids."""
    for r in relations:
        resolved = _resolve_relation_names(graph, r)
        if resolved:
            graph.add_relation(resolved)


def _update_schema_from_feedback(graph: SchemaGraph, trace, correct: bool) -> None:
    """Stage 2: reinforce or correct schema based on evaluation result.

    - Positive feedback (correct): boost confidence of used concepts and weight of used relations.
    - Negative feedback (wrong): reduce confidence/weight of used concepts and relations.
    """
    delta = 0.05 if correct else -0.05

    for cid in trace.concepts:
        c = graph.get_concept(cid)
        if not c:
            c = graph.get_concept_by_name(cid)
        if c:
            new_conf = max(0.1, min(0.95, c.confidence + delta))
            graph.update_concept(c.id, confidence=new_conf)

    for src, tgt, rel_type in trace.relations:
        # Resolve free-text src/tgt to concept ids via fuzzy name matching
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
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg.runs_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Schema is scoped to the run directory by default (like episode logs).
    # If user explicitly set schema_dir, respect it; otherwise use run_dir/schema.
    schema_dir = cfg.schema_dir
    if schema_dir == "schema" and not cfg.reset_schema:
        schema_dir = os.path.join(run_dir, "schema")
    # If reset_schema is used with default "schema", keep the old global behavior
    # so that --reset-schema still works as a manual clear of the global schema.

    # Temporarily override cfg.schema_dir for this run
    original_schema_dir = cfg.schema_dir
    cfg.schema_dir = schema_dir

    graph, storage, embeddings = _load_or_init_schema(cfg)
    cfg.schema_dir = original_schema_dir

    retriever = SchemaRetriever(graph, embeddings, embedder, agent=agent)
    initializer = SchemaInitializer(agent)
    human_io = HumanIO(auto_yes=cfg.auto_yes)

    count = 0
    for problem in problems:
        if cfg.problem_ids and problem.id not in cfg.problem_ids:
            continue

        ep = Episode(problem=problem)

        # 1. Schema retrieve + sufficiency check
        result = retriever.retrieve(
            problem, top_k=cfg.top_k_concepts, threshold=cfg.similarity_threshold
        )
        concepts = result.matched
        sufficient = retriever.is_sufficient(result)

        if not sufficient:
            if cfg.auto_yes:
                # Auto-generate schema from agent when in auto mode
                print(f"[auto-yes] Schema insufficient for {problem.id}, missing: {result.missing}, auto-generating...")
                auto_concepts, auto_relations = initializer.generate_schema(problem)
                _add_concepts_with_embeddings(graph, embeddings, embedder, auto_concepts)
                _add_relations_resolved(graph, auto_relations)
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
                    _add_concepts_with_embeddings(graph, embeddings, embedder, new_concepts)
                    _add_relations_resolved(graph, new_relations)
                    ep.flags.append("human_init_concepts")
                    result = retriever.retrieve(
                        problem,
                        top_k=cfg.top_k_concepts,
                        threshold=cfg.similarity_threshold,
                    )
                    concepts = result.matched

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
                    raw=attempt.raw,
                )
            )

            # 3. Evaluate
            ev = evaluate(problem, attempt, agent=agent)
            ep.evals.append(ev)

            # 4. Compute schema-based reasoning confidence
            reasoning_confidence = graph.compute_confidence(attempt.reasoning_trace)
            ep.reasoning_trace = attempt.reasoning_trace
            ep.reasoning_confidence = reasoning_confidence

            # 4.5 Stage 2: Schema reinforcement / correction
            if not cfg.dry_run and attempt.reasoning_trace.concepts:
                _update_schema_from_feedback(graph, attempt.reasoning_trace, ev.correct)
                if ev.correct:
                    ep.flags.append("schema_reinforce")
                else:
                    ep.flags.append("schema_correct")

            # Stop policy check
            stop, reason = should_stop(ep, cfg.stop)
            if stop:
                ep.stop_reason = reason
                break

        if ep.stop_reason is None:
            stop, reason = should_stop(ep, cfg.stop)
            if stop:
                ep.stop_reason = reason

        # 5. High-confidence wrong -> ask human for correction
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
                    _add_concepts_with_embeddings(graph, embeddings, embedder, [c])
                for r in corrected.get("add_relations", []):
                    resolved = _resolve_relation_names(graph, r)
                    if resolved:
                        graph.add_relation(resolved)
                for upd in corrected.get("update_concepts", []):
                    cid = upd.get("id")
                    if cid:
                        graph.update_concept(cid, **{k: v for k, v in upd.items() if k != "id"})
                ep.flags.append("human_correction")

        # Save schema after each problem
        storage.save(graph, embeddings)

        # 6. Log episode
        write_episode(run_dir, ep)

        count += 1
        if count >= cfg.max_problems:
            break

    return run_dir
