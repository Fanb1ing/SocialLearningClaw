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

## 2026-07-24 — Benchmark download completeness audit

Audited local files against the current official dataset repositories and the
ARC API without downloading benchmark content. ContextMATH has all five
Hugging Face splits (382/382 rows, 277,033 bytes), but not the two additional
AIME 2024 SG robustness variants from the upstream GitHub repository (60
rows). IntPhys2 has metadata for Debug and Main but only 20 Solidity/Fixed
Debug videos (20/60 Debug, 20/1,416 total; 9,157,704 video bytes); no Main or
Held-Out videos are present. ARC has source for four of the 25 games currently
available to this API account; three are the repository's complete configured
experiment set and `sc25` is extra.

Verification: counted parquet/CSV rows with `.venv/bin/python`, matched local
video basenames to metadata, summed exact local file bytes, inspected all ARC
metadata/source files, queried the official ARC game listing (25 games), and
checked the official ContextMATH GitHub and Hugging Face IntPhys2 file
metadata. No dataset was downloaded. Unfinished follow-up: decide whether the
formal protocol should add ContextMATH robustness variants, IntPhys2 Main
(recommended for the labeled benchmark), and/or all 25 public ARC games;
before ARC runs, also align the SDK `ENVIRONMENTS_DIR` with the vendored
`third_party/arc_agi3_games` path or let the SDK redownload into its default
directory.

## 2026-08-06 — Current-state and gold-Schema readiness review

Reviewed the active Memory -> layered Schema implementation, runner wiring,
experiment protocol, benchmark status, local outputs, and the uncommitted data
preparation work. The architecture and all three `schema` runner integrations
are present, but there is still no live artifact for the current layered
Schema method and no ground-truth/gold Schema annotation specification,
dataset, builder, or evaluator. Gold Schema currently appears only as a target
in `docs/schema_redesign_notes.md` for generation-quality and exploit/explore
evaluation.

The working tree also contains unfinished benchmark expansion work: the local
IntPhys2 adapter loads all 60 Debug videos plus a pinned 300-video Main sample,
and the vendored ARC inventory contains 25 metadata/source pairs. These changes
remain uncommitted and must be preserved and finished separately.

Verification: source/document inspection, `git diff --check`, all 33 offline
unit tests, IntPhys2 loader counts (60 Debug and 300 Main), and ARC inventory
counts (25 metadata and 25 source files) passed. No live API/model call was
made. Follow-up: agree on the gold-Schema scope, ontology/granularity,
provenance format, annotation process, and evaluation matching rules before
constructing the first gold artifact.

## 2026-08-06 — Benchmark expansion finalized

Finished the pending local benchmark expansion. IntPhys2 preparation now pins
Hugging Face revision `a077a2f94e25889016fc6e5983cf21e2ddc25fb2`, retains all
60 Debug videos, and deterministically selects 300 Main videos as 75 complete
four-type scene groups using seed 20260724 and stratified Hamilton allocation.
The generated Main provenance manifest is required by the adapter, and missing
referenced videos now fail fast instead of silently changing the evaluation
set.

Vendored ARC-AGI-3 coverage now contains all 25 locally available full game
versions. `inventory.json` records every metadata/source SHA-256 pair;
host-specific metadata paths are removed; the wrapper uses SDK offline mode;
and run manifests fingerprint the selected local game semantics instead of
claiming a remote environment. Project metadata and README now state Python
3.12+, matching `arc-agi>=0.9.8` upstream metadata. The existing Python 3.11
`.venv` required reinstalling that SDK with its Python metadata check bypassed
for local verification; a clean supported environment should use Python 3.12.

Verification: all 37 offline unit tests, compileall, static/ARC CLI help,
`git diff --check`, IntPhys2 loads/fingerprints (60 Debug, 300 Main), and all
25 ARC environments instantiated and reset successfully in offline mode. No
model call was made. Unfinished follow-up: construct the first gold-Schema
batch using privileged benchmark evidence kept strictly outside ordinary
method updates.

## 2026-08-06 — Gold Schema plan and ARC-first scope

Documented the proposed Gold Schema process in
`docs/gold_schema_generation.md`. Gold Schemas are now explicitly defined as
independent privileged evaluation annotations: unlike active learned
`SchemaNode` objects, they do not need to originate from `MemoryRecord`
evidence. The four release requirements are correctness, scope-relative
completeness, atomicity, and non-leakage.

The first proposed batch is source-guided ARC-AGI-3 annotation for the three
default pinned games: CD82 (6 levels), SK48 (8), and TU93 (9). The plan uses
AST/control-flow extraction, behavior-based naming for obfuscated source,
local transition probes, backward dependency coverage from every
`next_level()` predicate, and a CD82 review gate. It deliberately does not use
black-box gameplay search or include full winning action sequences.

Verification: checked the three source files and level counts, reviewed the
Markdown boundary against the active Memory -> Schema convention, and ran
`git diff --check`. No Gold Schema generation or model call was performed.
Follow-up: obtain user approval, then implement the format/validators and
generate the CD82 pilot before expanding to SK48 and TU93.

## 2026-08-06 — Gold Schema plan translated to Chinese

Rewrote `docs/gold_schema_generation.md` in Chinese while preserving the
approved technical boundary, ARC source-analysis phases, artifact contract,
review gate, acceptance criteria, and deferred IntPhys2/ContextMATH plans.
English is retained only for code identifiers, JSON fields, paths, benchmark
names, and standard technical terms where useful. README now links to the plan
with a Chinese label. Verification: `git diff --check`; Gold generation remains
unstarted.

## 2026-08-06 — First ARC Gold Schema pilot generated

Condensed `docs/gold_schema_generation.md` from the long design into a
three-step source facts -> atomic schemas -> validation/coverage workflow, then
generated the first reviewable Gold artifact for all six levels of pinned game
`cd82-fb555c5d`.

The pilot contains 26 independent, non-memory-grounded Gold nodes covering
target/canvas semantics, per-level palettes, four navigation actions, all eight
ACTION5 regions, the four Level 3–6 edge tools, invalid clicks, the diagonal-
masked goal predicate, and the 100-action limit. A reproducible generator,
versioned format, runtime cases, coverage report, validation report, and compact
Chinese review document were added under `gold/arc_agi3/v1/`.

Verification: source SHA-256 and line anchors validated; 20/20 local runtime
cases passed; 16/16 coverage requirements and all six goal levels are covered;
all 40 offline unit tests, compile checks, and `git diff --check` passed. No
model call or black-box solution search was used. Follow-up: wait for human
review of CD82 granularity and wording before generating SK48/TU93.

## 2026-08-06 — ARC Gold Schema first batch completed

Applied the CD82 human review and completed the first ARC-AGI-3 Gold batch.
Gold nodes now include only reusable mechanisms needed for task planning;
per-level palettes, initial UI state, progress-bar semantics, concrete step
counts, and other raw game facts are excluded. CD82 was reduced from 26 to 18
nodes, its four directionally equivalent 12-cell tools were merged, and the
diagonal-masked goal rule was retained because it changes the win predicate.

Generated source-pinned artifacts for SK48 (10 nodes across 8 levels) and TU93
(9 nodes across 9 levels), plus compact Chinese review documents. SK48 covers
chain selection, extension/retraction, rail translation, recursive pushes,
undo, sequence matching, and movement budget. TU93 covers path movement, three
enemy behaviors, mover-dependent collision outcomes, exit completion, and step
exhaustion. The batch manifest now contains 3 games, 23 levels, and 37 nodes.

Verification: 34/34 local behavior probes, 49/49 coverage requirements, source
hash/line-anchor checks, all 41 offline unit tests, compileall, and
`git diff --check` passed. No model call, black-box solve, or complete winning
action sequence was used. Follow-up: human-review the SK48 and TU93 drafts,
then decide whether to revise them and construct cross-game level 0/1 Gold
abstractions.

## 2026-08-06 — ARC-AGI-3 Gold Schema expanded to all 25 games

Applied the SK48/TU93 human review without changing their approved
granularity: SK48 retains the abstract reference sequence, combined recursive
push/block rule, and non-refunding undo; TU93 retains three separate enemy
behaviors, two mover-dependent collision rules, and only the generic step
budget mechanism. Both reviews and manifest statuses now record acceptance.

Generated the remaining 22 source-pinned games, bringing ARC Gold v1 to all 25
inventory games, 183 levels, and 198 atomic schemas. Every game has Chinese
`review.md`, `schemas.json`, runtime cases, coverage, and validation; the new
top-level Gold README is the review index. The 22 new games remain explicitly
`pending` human semantic review. Runtime evidence for those games is an
all-level load plus advertised-action smoke check; source hash/line anchors are
the primary semantic evidence and the review text does not overstate the smoke
checks as behavioral proof.

Verification: regenerated the full batch; all 25 validations passed, every
level has goal coverage, manifest totals match the inventory, all 42 offline
unit tests passed, and generators compile. No black-box solution search or
winning action sequence was used. Follow-up: review the 22 pending games via
`gold/arc_agi3/v1/README.md`, then revise and mark each accepted as appropriate.

## 2026-08-06 — Provisional cross-game ARC Gold layer generated

Added a reproducible cross-game derivation on top of the 198 game-level
schemas. The artifact contains 12 Level 1 mechanism families and 3 Level 0
system abstractions. Each node explicitly records its member Schema IDs, game
scope, member review statuses, inherited source evidence, and parent/member
relations; Level 1 requires at least three games and Level 0 requires at least
two Level 1 families. The families cover all 25 games.

The entire cross-game layer is intentionally `provisional`, because 22 source
games still await human semantic review. It is indexed from the Gold README and
documented in `docs/gold_schema_generation.md`; future full-batch regeneration
also regenerates the cross-game artifacts. Verification: cross-game source
hashes and member links validated, all pending dependencies are exposed, all
43 offline unit tests and `git diff --check` passed. Follow-up: review
`cross_game/review.md`, especially atomic rejection, structured interpretation,
and whether finite-resource failure should split hard loss from local reset.

## 2026-08-06 — Remaining Gold Schema scope defined

ARC-AGI-3 Gold generation is frozen at the current 25-game plus provisional
cross-game version. Defined the next two pipelines in
`docs/gold_schema_generation.md`: ContextMATH will merge the 120 AIME variants
into 60 source-problem groups and require an executable exact derivation for
each; IntPhys2 will canonicalize the 90 split-level scene groups to 89 unique
four-video groups and require paired temporal evidence plus human-reviewed
violation semantics. Dataset inspection confirmed SG/CS original-question and
answer equality within each ContextMATH pair, complete four-type IntPhys2 scene
groups, and one Debug/Main scene overlap. No Gold artifacts for either
benchmark have been generated yet. Follow-up: generate and review a small
ContextMATH pilot, expand it to all 60 groups, then repeat with one IntPhys2
scene per physical condition before processing all 89 unique scenes.

## 2026-08-06 — ContextMATH Gold Schema pilot generated

Generated the first reviewable ContextMATH Gold batch for six source-problem
groups: AIME 2024 IDs 60, 67, 75, and 84 plus AIME 2025 IDs 0 and 3. Their 12
SG/CS variants share 23 task-level Level 3 schemas while keeping variant
semantic alignments separate from reusable mathematical reasoning. Dataset
answers appear only in privileged witness/validation artifacts, never as
schema nodes. Exact rational, divisor, parameter-counting, and bounded
enumeration witnesses passed all 30 checks and matched all six normalized
answers. Cross-problem Level 2 schemas are deliberately deferred until all 60
problem groups exist, to avoid a taxonomy inferred from a small pilot.

Added a reproducible generator, per-problem alignment/witness/coverage/
validation/review artifacts, a benchmark-specific schema specification, and
three artifact tests. Verification: all 46 offline unit tests, compileall, and
`git diff --check` passed. Follow-up: human-review the six entries indexed by
`gold/contextmath/v1/README.md`, revise the format/granularity, then expand the
same structure to all 60 source-problem groups.

## 2026-08-06 — IntPhys2 four-condition Gold pilot generated

Paused ContextMATH expansion after user feedback that its pilot quality and
granularity need later reconsideration. Generated a reviewable IntPhys2 pilot
covering permanence (`debug:3`), immutability (`main_300:21`), continuity
(`main_300:19`), and solidity (`debug:1`, also the one Debug/Main overlap).
The pilot contains four complete four-video groups, eight paired assessments,
four provisional Level 1 condition invariants, and four provisional Level 3
scene discriminants. Each scene includes a generated contact sheet with source
frame numbers plus source-pinned video/frame hashes and a Chinese review file.

The observed counterfactuals are disappearance/appearance across occlusion,
blue/red property changes, left/right cup switching without a visible path,
and collision responses conditioned on a visible yellow obstacle. The
solidity interpretation is explicitly marked as the most uncertain review
item. Metadata, alias inventory, clip hashes, decoded frame hashes, pairing,
and schema-evidence links passed; visual semantics remain
`provisional_review_pending`. Verification: all 49 offline unit tests,
compileall, and `git diff --check` passed. Follow-up: obtain human review from
`gold/intphys2/v1/README.md` before expanding toward all 89 unique scenes.

## 2026-08-13 — Gold Schema session takeaways consolidated

Added `docs/ground_truth_schema_session_takeaways.md` as the durable Chinese
handoff for the full Gold Schema work session. It records the final definition
and four release requirements, the deliberate exception from learned
memory-grounding, benchmark-specific outcomes and review status, successful
ARC practices, why the ContextMATH pilot must not be scaled mechanically,
IntPhys2 visual-evidence risks, cross-benchmark anti-patterns, the missing Gold
evaluator design, and a staged v2 upgrade checklist. README now links directly
to this handoff. No Gold artifacts or generator behavior were changed.
Verification: documentation inspection and `git diff --check`. Follow-up: on
the next upgrade, preserve v1, define the scoring/evaluation contract first,
review the IntPhys2 pilot (especially Solidity), and redesign ContextMATH's
schema ontology before generating more annotations.

## 2026-08-13 — Learned Schema project walkthrough

Reviewed the active documentation and current source for the three-benchmark
experiment framework and the learned MemoryRecord -> layered SchemaNode path,
with emphasis on trajectory ingestion, LLM `create/merge/skip` induction,
retrieval, binary/environment feedback, masking, decay, consolidation, graph
invariants, persistence, and static/ARC runner integration. Confirmed that the
learned Schema system is distinct from the privileged Gold Schema artifacts
and that legacy Concept/Relation code remains archival only. No business code
or runtime behavior was changed. Verification was documentation/source
inspection plus a working-tree diff review; tests were not run for this
read-only walkthrough. Follow-up: define a small first development slice for
trajectory-level induction and its learned-to-Gold evaluation contract before
tuning maintenance thresholds or scaling experiments.

## 2026-08-13 — ARC learned Schema pipeline plan

Added `docs/arc_learned_schema_pipeline_plan.md` as the implementation plan for
the ARC-first learned Schema workflow. The design replaces immediate per-step
induction with durable transition capture, content-addressed lossless grid and
rendered-image evidence, periodic window induction, level synthesis, and
audited maintenance. It separates evidence confidence from downstream utility,
defines a Gold-isolated evaluator with structured semantic and limited
many-to-many matching, and lists phased code changes and acceptance gates from
offline fake environments through a CD82 live pilot. README now links to the
plan. No runtime behavior was changed. Verification: current source/Gold
contract inspection and `git diff --check`; tests were not run for this
design-only task. Follow-up: obtain design approval, then implement Phase A
(memory contracts and visual asset persistence) before changing induction.

## 2026-08-13 — Deterministic trajectory-corpus plan

Expanded the ARC learned Schema plan so generator/maintenance development no
longer depends on a reliable online LLM Agent. The pipeline now begins with a
benchmark-neutral `TrajectorySource` contract shared by deterministic scripts,
coverage exploration, frozen replay, and future Agent streams. The proposed
ARC corpus combines successful, mechanism, boundary, perturbed, and
coverage-guided trajectories, with measurable coverage gates and a roughly
96-episode CD82 v1 target. Only public-interface trajectories enter formal
induction metrics; existing Gold runtime probes that mutate internal game
state are diagnostic-only. The design also defines evidence tiers, corpus
splits and hashes, replay validation, Gold isolation, and adapter/profile
interfaces so later one-step benchmarks use the same recorder, Memory,
induction, maintenance, and evaluation orchestration. No runtime behavior or
trajectory data was generated. Verification: current ARC probe/source audit,
documentation inspection, and `git diff --check`; tests were not run for this
design-only update. Follow-up: implement the generic trajectory contract and
recorder first, then the CD82 deterministic corpus before Schema induction.

## 2026-08-13 — Trajectory foundation Phase A implemented

Implemented the first reviewable development slice without changing active
benchmark runners. Added benchmark-neutral `TrajectoryEpisode`/`TrajectoryStep`
models, optional decisions, evidence tiers, lifecycle stream events and source/
domain-adapter protocols under `socialclaw.trajectory`. Added a content-
addressed artifact store for lossless little-endian int16 grid `.npy` files and
Agent-view PNGs, including role-independent deduplication, portable references,
file/logical hashes, safe paths and pickle-free grid loading. The atomic
`TrajectoryRecorder` validates step continuity and available actions, persists
after every step, protects the previous snapshot on write failure, and supports
identity-checked resume.

Added 10 focused tests plus a network-free two-step review generator. The
generated sample at `outputs/review/trajectory_contract_phase_a/` records one
changed transition and one blocked no-effect transition; the repeated state
reuses the same grid artifact. Updated README, architecture, Schema architecture,
result-format documentation, the ARC pipeline plan, and added
`docs/trajectory_contract.md` with review instructions. Verification: all 59
offline unit tests, compileall, the demo generator, visual inspection of both
PNGs, and `git diff --check` passed. No LLM/API/environment call was made for the
demo. Follow-up: after user review, implement Phase B's ARC adapter, deterministic
CD82 policies, perturbations, coverage explorer, corpus manifest and replay
validation; active ARC still uses its prior direct MemoryRecord path until that
offline corpus layer is accepted.

## 2026-08-13 — CD82 reliable trajectory corpus Phase B implemented

Implemented the ARC-specific layer on top of the generic trajectory contract:
public-observation normalization, per-step lossless grid plus Agent-view PNG
capture, CD82 task/UI diff separation, public-action recording, ARC coverage
summaries, and fresh-environment frame-by-frame replay. Added a visible-grid
CD82 policy that detects palette buttons and solves the overwrite/tool puzzle
without an LLM, plus deterministic near-miss, budget-loss, single-mechanism,
sequence-perturbation, and seeded exploration scenarios. Generic corpus
loading, split/manifest validation, asset verification, and metadata writing
remain ARC-independent; ARC coverage/replay and CD82 policy are isolated in
domain modules.

Generated the ignored local corpus at
`data/trajectory_corpora/arc_agi3/cd82_v1/`: 96 episodes, 1022 public steps,
450 unique artifact refs, about 13 MiB. It contains 78
`source_guided_natural` and 18 `natural` episodes, covers levels/actions 1–6,
effect and no-effect for every action, three click roles, WIN/GAME_OVER/TIMEOUT,
14 paired groups, and 32 transition signatures. Structural/asset validation
passed and all 96 episodes/1022 steps replayed from fresh environments with
identical frames and statuses. A first attempt revealed that reusing one SDK
environment only reset the current level; that incomplete output was replaced,
and the final generator creates a new offline environment for every episode.
Manual inspection confirmed real 64×64 CD82 frames and no Gold Schema fields.

Added `docs/arc_trajectory_corpus.md` and updated the README, architecture,
trajectory, Schema, result-format, and pipeline-plan docs with review commands
and Phase B status. Verification: all 65 unit tests, trajectory compileall,
real six-level policy WIN in exactly 70 actions, corpus validation, full replay,
visual keyframe inspection, and Gold-string scan passed. No network or LLM API
was used. Follow-up: Phase C should project this frozen corpus into traceable
transition/window/level `MemoryRecord`s without changing the active ARC runner;
each later `SchemaNode` must cite those durable memory IDs.

## 2026-08-13 — Three-game corpora and Phase C Schema prototype

Expanded the offline ARC example from CD82 to SK48 and TU93 without requiring
complete play for every game. Generated and fully replay-validated compact
corpora at `data/trajectory_corpora/arc_agi3/{sk48_v1,tu93_v1}`: SK48 has 24
episodes/345 steps and verified successful progress through 3/8 levels; TU93
has 24 episodes/701 steps and a 185-action 9/9 WIN path. Both include verified
success prefixes, omitted-final-action near misses, action/repeat probes, and
seeded exploration. Fixed paths were found with source-assisted offline search
but published episodes were rerun from fresh environments through public
actions and marked `source_guided_natural`; natural explorers remain separate.
SK48 levels 4–8 are explicitly deferred to trajectory v2 rather than presented
as solved.

Implemented the first Phase C `Trajectory -> Memory -> Schema` prototype in
`socialclaw/schema/trajectory_pipeline.py`. Deterministic projection creates
one durable transition MemoryRecord per step, eight-step window summaries, and
episode memories with stable cross-game-safe IDs, source-memory links, corpus/
trajectory paths, and pre/post grid plus PNG references. Added atomic batch
memory insertion after the initial full run exposed quadratic snapshot writes.
The conservative `transition_bucket_v1` inducer groups repeated transitions by
game/action/target-role/effect class, requires at least two supporting records,
and emits Level 2/3 SchemaNodes whose evidence directly cites transition memory
IDs. It is a pipeline baseline, not the final visual-semantic generator.

The review output at `outputs/review/three_game_schema_prototype_v1/` contains
2545 MemoryRecords (2068 transition, 333 window, 144 episode) from 144 episodes
and 40 grounded SchemaNodes (CD82 17, SK48 11, TU93 12). Full evidence-closure
checks resolved every Schema to transition memory, source trajectory, and
visual asset. No LLM/network or Gold Schema read occurred. Added design/review
documentation and three focused pipeline tests, including a regression for
same-named episodes across games. Verification: 48/48 new corpus episode
replays, all 68 repository tests, compile checks, provenance closure, and docs/
diff checks. Follow-up: Phase D should replace the coarse bucket trigger with
window/keyframe semantic proposals and deterministic create/support/revise/
contradict/skip validation; trajectory v2 can later extend SK48 beyond level 3.

## 2026-08-13 — Weekly-report project synthesis

Reviewed the complete active documentation, project memory, current runners,
batch/summarization scripts, Gold artifacts, baseline audit, and CD82 trajectory
examples to prepare an evidence-based Chinese weekly-report draft. The report
separates the three-benchmark unified experiment protocol from the ARC-only
one-click batch script, summarizes eight baselines plus the layered `schema`
method, explains benchmark-specific Gold construction and review status, and
documents the 96-episode/1022-step CD82 corpus and its content-addressed visual
evidence. No runtime behavior or research artifact was changed. Verification
was source/document/artifact inspection and `git diff --check`; tests were not
run for this documentation-only synthesis. Follow-up: Phase C trajectory to
Memory projection and the learned-to-Gold evaluator remain unfinished.

## 2026-08-13 — Weekly report refreshed through Phase D

Re-audited the available baseline artifacts and the latest trajectory work for
the weekly report. The baseline tables remain historical smoke results rather
than a unified fair comparison: static `withrule` and the current layered
`schema` method still lack comparable live artifacts. Corrected the earlier
weekly-report status because Phase C is now complete. A fresh temporary run of
`semantic_window_v1` over the three-game Phase C memory selected 138 keyframe
transitions from 333 windows and produced 50 grounded nodes (CD82 18, SK48 16,
TU93 16), with 50 create, 1175 support, 22 revise, and 4 skip proposals; it made
no network calls and read no Gold Schema. All 72 offline tests passed. No
tracked runtime artifact or business behavior was changed. Follow-up: the
deterministic window semantics remain a prototype; Phase E maintenance, the
learned-to-Gold evaluator, SK48 levels 4–8 trajectory v2, and online runner
integration are unfinished.

## 2026-08-13 — Phase D implementation and review package completed

Implemented `semantic_window_v1` in `socialclaw/schema/window_induction.py` on
top of the layered `MemoryRecord -> SchemaNode` architecture. The generic
scheduler processes durable window memories through replaceable transition
profiler and proposal-generator contracts, selects non-redundant visual
keyframes, validates game/action/evidence scope, applies audited
create/support/revise/contradict/skip operations, and preserves source and
negative transition IDs. ARC profiling reads only corpus-owned pre/post grids
and derives normalized region/change-scale features; it imports neither game
source nor Gold. Repeated evidence supports an existing node, paired
effect/no-effect observations condition over-broad rules, and singleton groups
are recorded as skips rather than promoted.

The ignored review snapshot at
`outputs/review/three_game_schema_phase_d_v1/` consumes the Phase C 2545-memory
snapshot and produces 50 nodes (CD82 18, SK48 16, TU93 16), 138 selected
keyframes, and a complete proposal audit. Of 2068 transitions, 2064 are source
evidence, 25 are also negative evidence, and four singleton semantic groups are
explicitly uncited/skipped. All 192 unique keyframe grid/PNG artifacts were
content-hash verified; network and Gold reads were zero. Added the reproduction
CLI, five focused tests, review README, and updated architecture/result/plan
docs. Verification: all 73 unit tests, real three-game induction rerun,
graph/keyframe evidence validation, compileall, and diff checks passed.
Follow-up: Phase E should add durable schedule cursors, periodic
merge/dedup/promotion/alias/snapshots and interruption recovery; Phase F then
implements the isolated learned-vs-Gold evaluator. SK48 levels 4–8 remain
trajectory v2 work, not a Phase D blocker.

## 2026-08-13 — Phase E paused; learned-vs-Gold evaluator prioritized

Per user priority, deferred Phase E maintenance and implemented the first
read-only learned-vs-Gold evaluator instead. Added an evaluator-only accepted
ARC Gold loader, canonical learned/Gold views, a deterministic cross-language
structured proxy judge, equivalent/narrower/broader/partial/contradiction/
unrelated relations, split/overmerge accounting, evidence-closure checks, an
independent CLI, and six focused offline tests. Induction and runner modules do
not import the evaluator or Gold loader; the evaluator hashes its learned and
Memory inputs, writes only a separate output directory, and never feeds
alignment back into generation.

Evaluated the Phase D 50-node snapshot against all 37 accepted Gold nodes for
CD82, SK48, and TU93. The conservative proxy found zero strictly equivalent
nodes, graded learned precision 0.562, graded Gold recall 0.415, F1 0.478, and
partial coverage of 24/37 Gold nodes; all 2090 evidence references resolved.
Action-signature recall was 1.0, but observation semantics covered 0/4 and
hazards 0/7. CD82/SK48/TU93 partial coverage was respectively 15/18, 7/10, and
2/9. This means v1 captures action/change correlations but not Gold-level
preconditions, object roles, exact geometry, budgets, enemies, or causal
mechanisms. The graded values are diagnostic proxy scores, not publication
metrics; strict zero is enforced for generic “grid changed/no-change/level
completed” summaries.

The ignored review output is
`outputs/review/three_game_schema_evaluation_v1/`, with config, metrics,
alignments, unmatched lists, cache, and Chinese report. Added
`docs/schema_evaluation.md` and updated the plan, architecture, result format,
and README. Verification: the evaluator reproduction completed with zero
network calls and zero learned-state writes, all 79 repository tests passed,
and compile/diff checks passed. Follow-up: improve induction before management
by learning object/region semantics, preconditions, exact changed masks,
direction mappings, UI budget deltas, and cross-step causal/terminal rules;
then rerun the same evaluator. Freeze a human alignment fixture and calibrate
an independent semantic judge before treating scores as formal metrics.

## 2026-08-13 — Session work consolidated into two review guides

Added two durable Chinese handoff documents requested by the user. The first,
`docs/session_trajectory_memory_summary.md`, consolidates the generic trajectory
contract, content-addressed grid/PNG design, CD82/SK48/TU93 corpus generation
and provenance, replay status, 2545-record Memory projection, limitations, and
step-by-step corpus/Memory/visual review paths. The second,
`docs/session_schema_evaluation_summary.md`, consolidates the Phase C bucket
baseline, Phase D window/keyframe proposal pipeline, evidence validation and
audit, current 50-node output, independent learned-vs-Gold evaluator, metrics,
interpretation, limitations, and exact review/reproduction paths. README links
to both guides. Verified every referenced primary artifact/replay/report path,
cross-checked all counts against current JSON outputs, ran compileall and
`git diff --check`; no runtime behavior changed for this documentation task.
Unfinished work remains unchanged: generation v2 semantics first, Phase E
maintenance deferred, SK48 trajectory v2, human alignment calibration, and
online Agent integration.
