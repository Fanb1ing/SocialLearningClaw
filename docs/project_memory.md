# Project Memory

This is the active cross-session handoff. Verify details against source before
acting. Never put credentials or private benchmark content here.

## 2026-07-21 — Memory-grounded layered Schema foundation

Completed the new infrastructure described in `temp.md` while preserving the
legacy single-layer ARC schema for compatibility.

Key decisions:

- Concrete task logs live in the new `socialclaw.memory` layer.
- A `SchemaNode` is a generalized world rule and keeps source, positive, and
  negative memory IDs as auditable evidence.
- Level `0` is most general; larger levels are more specific.
- Parent/child and same-level similarity links are symmetric and validated.
- LLM induction returns a structured `create`, `merge`, or `skip` proposal.
- Embeddings and the LLM are dependency-injected; offline deterministic
  behavior remains available for tests and empty-project bootstrapping.
- Reliability feedback, task masking, forgetting decay, duplicate
  consolidation, and atomic persistence are implemented.
- `build_schema_system()` constructs and restores the complete memory/schema
  stack from a state directory.

Primary guide: `docs/schema_architecture.md`.

Verification at completion: `.venv/bin/python -m unittest discover -s tests -v`
passed all 25 tests. No live model/API call was made.

The runner migration identified here was later explicitly requested and is
recorded as completed in the next entry.

## 2026-07-21 — Schema integration and runner migration

Completed the requested migration after the user explicitly authorized it:

- `schema` is accepted by `run_static` for ContextMATH and IntPhys2.
- Static tasks use `SchemaMethodController`; only task input, model response,
  and binary correctness enter memory/schema updates. IntPhys2 retrieval adds
  condition/camera/scene metadata but excludes label-bearing `type` and gold.
- The ARC schema runner was replaced with the layered implementation. Every
  observation/action/environment-result transition becomes a `MemoryRecord`,
  online Schema IDs are injected into prompts, and terminal outcomes update
  schemas actually claimed/used during the level.
- ARC artifacts now record source memory IDs and injected/learned Schema IDs;
  run-local state remains under `schema/memory.json` and `schema/schema.json`.
- Static summarization now includes the `schema` method.
- README now explains the responsibilities of `methods/`, `archive/`, and
  `tests/`; active architecture/baseline/result docs were updated.

Verification: compileall, CLI `--help` smoke checks, `git diff --check`, and
`.venv/bin/python -m unittest discover -s tests -v` passed all 32 tests. The
ARC runner has an offline end-to-end fake-environment test. No real API or ARC
server call was made.

Potential experiment follow-up: run small live smoke experiments on one sample
per benchmark to validate provider-specific structured output and measure the
unaccounted auxiliary Schema induction token cost before full benchmark runs.

## 2026-07-24 — Repository walkthrough

Reviewed the tracked repository file-by-file and explained the execution
entries, benchmark adapters, baseline controllers, ARC loops, durable Memory ->
layered Schema path, compatibility modules, tests, configuration, documentation,
archived results, and vendored game environments. No implementation behavior
was changed. Verification was source inspection plus a clean `git status`;
tests were not run because this was a read-only explanation task. No follow-up
is required.

## 2026-07-24 — Retired legacy Schema and restored baseline report

Moved the obsolete single-layer `Concept`/`Relation` graph, storage, retriever,
initializer, full ARC parser, helpers, and tests to
`archive/code/legacy_schema/`. The active ARC parser now retains only connected
component extraction, color naming, and grid diff utilities; current package
exports no longer expose the legacy API.

Added `docs/baseline_smoke_results.md` after auditing tracked results, Git
history, and local legacy ARC artifacts. The report lists all eight current
baseline names and all three benchmarks. Historical numeric results exist for
`naive/reflexion/expel/amem/tgm` across all three and for
`icl/rag/withrule` on ARC only; no prior ContextMATH or IntPhys2 artifacts were
found for the latter three, so those six cells are explicitly marked missing
rather than inferred. Historical runs use mixed models and protocols and are
not a fair unified comparison.

Verification: compileall, `git diff --check`, legacy-import search, two legacy
ARC summary reconstructions, and `.venv/bin/python -m unittest discover -s
tests -v` passed all 31 active tests. Unfinished follow-up: rerun the six
missing static baseline cells under one unified protocol if a complete numeric
8-by-3 matrix is required.

## 2026-07-24 — ICL/RAG static benchmark retest

Reran ICL and RAG on ContextMATH and IntPhys2 with the historical
`anthropic/claude-opus-4.8` model, temperature 0, binary feedback, and one
attempt. ContextMATH used 10 samples from each AIME split and produced ICL
accuracies of 90%, 80%, 70%, and 70% (77.5% mean), versus RAG accuracies of
100%, 90%, 80%, and 70% (85% mean).

For IntPhys2, the first 3 of 20 local videos were reserved as ICL
demonstrations and excluded from both methods' evaluation sets. On the same
remaining 17 videos, ICL scored 10/17 (58.8%) and RAG scored 12/17 (70.6%).
This 17-video protocol prevents demonstration leakage but is not strictly
sample-identical to the historical 20-video rows. Raw artifacts are under
`outputs/smoke_retest_20260724/`, and the results/configuration notes were
added to `docs/baseline_smoke_results.md`.

Verification: all five `scripts/summarize_static.py` comparisons passed
without `--allow-incomparable`; a manifest audit verified the model, budgets,
sample equality, and IntPhys2 demo/evaluation separation across 10 runs; all
31 unit tests and `git diff --check` passed. Unfinished follow-up:
ContextMATH/IntPhys2 `withrule` remains intentionally untested.
