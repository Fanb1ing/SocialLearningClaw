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
