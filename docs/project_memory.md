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

## 2026-08-24 — Version 2 EFPS redesign and reuse audit

Reviewed `0824version2plan.md`, the complete project memory, and the current
ARC runner, environment wrapper, Agent transport, trajectory/asset stack,
Memory/layered-Schema implementation, corpus/replay tooling, evaluator, and
tests. The proposed Version 2 is ARC-only and redefines Schema around a typed
Entity-Feature-Prototype-Schema relation graph, with separate acting/planning,
graph-update, and novelty-driven exploration responsibilities.

The recommended migration freezes the current repository as Version 1 before
implementation. ARC environment/fingerprinting, normalized trajectories,
atomic recording, content-addressed grid/PNG evidence, replay/coverage, and
experiment provenance are the strongest direct-reuse candidates. The current
`SchemaNode`/layered graph/manager/induction path, static benchmark runners,
and Schema-coupled parts of `arc_runner.py` should be archived rather than
carried into Version 2; proposal validation/auditing and evidence provenance
remain useful design patterns. No business code or user input file was
changed. Verification: source/document inspection, clean baseline execution
of all 79 offline tests, and working-tree review. Follow-up: create the V1
archive/manifest first, then implement and validate the EFPS graph contract
before writing Agent prompts or online update behavior.

## 2026-08-24 — V1 frozen and revised EFPS plan saved

Saved the user-approved Version 2 design in
`docs/version2_efps_development_plan.md`. The main Agent is now explicitly the
orchestrator/planner/actor and may call update and exploration child Agents;
the first vertical slice will co-develop the EFPS graph and main Agent on
CD82 Level 1, while trajectory replay remains a debugging tool rather than an
offline prerequisite.

Created the non-importable V1 rollback snapshot at
`archive/version1_20260824/`, pinned to commit
`7c785777775e0b379235d0719ae177c01c5698ef`. It copies 1,739 files
(45,808,048 bytes) covering source, tests, scripts, configs, docs, Gold,
vendored ARC games, three trajectory corpora, review outputs, smoke results,
ContextMATH data, and IntPhys2 metadata/manifests. `FILE_INVENTORY.md`
describes and hashes every copied file; `MANIFEST.sha256` verifies the archive.
Large IntPhys2 videos, legacy data, and legacy outputs remain in place and are
represented by a 25,737-file external SHA-256 list. Credentials, `.venv`,
caches, and private host settings were excluded.

Verification: the archive manifest passed all 1,744 entries, placeholder-only
credential references were inspected, cache/secret files were absent, and all
79 repository tests passed after archiving. No V1 business file was moved or
deleted. Follow-up: begin the `socialclaw.v2` CD82 Level 1 vertical slice with
the minimal EFPS models/graph and child-Agent protocols.

## 2026-08-24 — V2 CD82 Level 1 online cognitive-growth slice

Implemented the first independent `socialclaw.v2` vertical slice without
modifying the frozen V1 Schema implementation. V2 now has evidence-grounded
Entity/Feature/Prototype/Schema models, typed relations, validated atomic graph
transactions, rollback, revision snapshots, and separate main/update/explore
Agent components. The main Agent is the orchestrator/planner/actor and the only
component that chooses environment actions; child Agents only return proposals.

The real offline CD82 environment now runs from an empty cognitive graph and
stops after Level 1. A controlled, source-guided bootstrap actor uses six safe
exploration-child calls and nine update-child calls over nine public actions.
It forms 5 Entities, 10 Feature definitions, 16 assertions, 3 Prototypes,
5 learned Schemas, and 32 typed relations across 19 graph revisions. Repeated
top painting and two palette instances are assimilated into existing Schemas;
new navigation cases and the final top-to-bottom paint transfer accommodate
and generalize existing cognition. All 10 evidence records resolve to the
durable trajectory and content-addressed visual assets.

Generated the ignored review package at
`outputs/review/v2_cd82_level1_zero_start/`, including a Chinese report,
timeline, final graph, 19 snapshots, evidence catalog, normalized episode,
lossless grids and PNGs. Verification: Level 1 completed in 9 actions; fresh
environment replay matched all 9 steps; 20 unique artifact refs, graph types,
Schema evidence closure and snapshot counts passed; compileall, CLI help,
`git diff --check`, and all 83 repository tests passed. Network calls and Gold
Schema reads were zero. Documentation and README now distinguish the V2 main
line from the frozen V1.

Known boundary: the current perception adapter contains CD82 Level 1 visual
primitives and the deterministic actor follows a controlled safe-probe
curriculum. It validates online EFPS evolution, not general autonomous visual
reasoning. Follow-up: replace the bootstrap actor with a structured multimodal
model main Agent under the same child-Agent/evidence/transaction contracts,
then extend persistent cognition through CD82 Levels 2–6.

## 2026-08-27 — V2 CD82 prototype claim and information-flow audit

Audited the V2 prototype after user review. The successful Level 1 run is a
CD82-specific integration fixture, not evidence of general autonomous
learning: perception fixes CD82 regions, colors, roles, positions, and the
goal mask; the main/exploration policies encode the goal and safe action
curriculum; and the update child deterministically maps CD82 actions to
handwritten Schema families. Step-1 `goal` is designer-provided, while its
`prediction` is a hardcoded exploration hypothesis rather than a Schema-based
prediction. Future timelines should represent an unknown/tentative goal and
separate `schema_prediction` from `exploration_hypothesis` with provenance.

Confirmed that the update child proposes Schema operations and the runtime
commits them. Human-reviewable pre/post PNG and lossless grid artifacts exist
for every transition, but current Agents consume parsed CD82 state rather than
Evidence records or images. The main Agent receives the full EFPS graph object
but only reads Schemas in its decision code. The graph currently has five node
types and four instantiated edge types; Evidence is an audit catalog rather
than a graph endpoint. `EXCLUDES` is declared but not materialized, and a
FeatureAssertion-to-FeatureDefinition connection is stored as a field rather
than an explicit edge. Verification was source, timeline, graph, episode, and
artifact-resolution inspection; no runtime code changed and tests were not
rerun. Follow-up: redesign the information boundary and replace all
game-specific cognition/policy logic before presenting the run as a general
learning result.

## 2026-08-27 — Game-agnostic V2 visual Agent and honest CD82 run

Removed the CD82-specific V2 perception, goal, exploration curriculum, action
branches, coordinates, goal mask, and route. V2 now uses one injected
structured-vision model protocol for a Main orchestrator/actor plus Exploration
and Update child Agents. Cognitive inputs contain only the raw public image,
public environment fields, SDK action-argument contracts, raw full-grid deltas,
recent public transitions, and a read-only EFPS view. The model generates all
goal hypotheses, exploration hypotheses, Entities, Features, Prototypes, and
Schemas. Without a cited existing Schema, Main validation forces
`schema_prediction=null`. Static tests prohibit privileged/game-specific V2
imports; game ID remains only in the environment harness.

Replaced the CD82 entry with generic `run_arc_online(game_id, model, ...)` and
reduced output to `report.md`, an input-audited `timeline.json`, one final
`cognition/graph.json`, and replay-required trajectory/grid/PNG evidence.
Timeline deduplicates public observations and action contracts, stores hashed
cognition-view receipts, and records each Agent's image inputs, model output,
usage, selected action, public result, and graph transaction. No duplicate
summary/evidence/manifest or revision snapshots are written. FeatureAssertion
to FeatureDefinition is now an explicit edge and Prototype exclusions are
materialized.

The first real zero-prior run used `anthropic/claude-opus-4.8` on
`cd82-fb555c5d`, stopped honestly at the 20-step limit, and completed 0 levels.
It made 61 model calls, formed 21 Evidence, 19 Entities, 35 Feature definitions,
50 assertions, 2 Prototypes, 10 Schemas, and 117 relations; all Schema evidence
closed and all 20 transitions replayed. It learned local ACTION3/4 movement and
ACTION5 change correlations but misread the scene/goal, missed useful click
coordinates, repeated low-information actions, and produced some conflicting
state-specific Schemas. The run used about 1.286M tokens, so acting views were
subsequently compacted by omitting redundant relation edges and limiting
attached Schema evidence to one image; the run was not relabeled as a success.

Verification: all 83 tests, compileall, CLI help, `git diff --check`, forbidden
dependency/game-ID scans, timeline reference resolution, first-step epistemic
checks, Schema evidence closure, artifact SHA-256 checks, and 20/20 replay
passed. Review output:
`outputs/review/v2_generic_cd82_level1_opus48/`. Follow-up: solve generic
coordinate grounding, repeated-probe suppression, Schema conflict/consolidation,
and executable multi-step planning before another Level 1 run.

## 2026-08-27 — Human-readable chronological Agent audit

Added `socialclaw.v2.reporting.build_process_markdown` and wired every V2 ARC
run to emit `process.md` beside `timeline.json`. The document starts with a
Step 0–20 overview table and then records, in order, each trigger, public image
and state, action contracts, EFPS receipt and recent public transitions,
Exploration/Main outputs, selected action, before/after PNG, public result,
Update proposal, and validator graph delta. “Recent public transitions” is
defined as at most the previous eight public action/result/Evidence summaries;
it contains no hidden goal, Gold, source, or post-hoc interpretation.

Generated the audit for
`outputs/review/v2_generic_cd82_level1_opus48/process.md`: 21 chronological
sections, 156 image links covering 17 content-addressed PNGs, with no missing
targets. Added runtime/test/docs coverage. Verification: 4 V2 tests and all 83
repository tests passed; compileall and `git diff --check` passed. Follow-up:
none for the reporting task; the cognition/planning limitations from the prior
entry remain.

## 2026-08-27 — Semantic transitions, resolvable Evidence, prose exploration

Replaced pixel-only recent transition history with an evidence-grounded
semantic path. After every action, Update must return `transition_analysis`
that attributes visible differences to existing/new Entity IDs using
appeared/disappeared/moved/state_changed/feature_changed. A changed grid with
neither an Entity change nor explicit `unassigned_visual_changes` is rejected;
an unchanged grid cannot assert Entity changes. The normalized result is stored
on the durable Evidence record and passed with each of the last eight public
transitions to the next Exploration/Main decision.

Added `EFPSGraph.resolve_evidence(evidence_id)` and
`annotate_evidence(...)`. Evidence now resolves to its action, public result,
semantic summary, Entity changes, unassigned changes, before/after observation
fingerprints, and artifact IDs. Timeline deduplicates these records in
`input_catalog.evidence`; refs are storage compression only, while the live
Agent payload contains the expanded Evidence. A two-step integration test
confirms the next Exploration call receives both semantic change and resolved
Evidence content.

Exploration now uses a non-JSON `generate_text` model call and returns one prose
advice paragraph to Main; the requested “You have no game...” sentence was
removed from its prompt. The requested “No goal or action meaning is supplied
to you.” sentence was removed from Main. Main/Update remain structured. Updated
the chronological reporter for prose advice, Entity semantics, and Evidence
resolution. The existing Opus48 run predates semantic transition fields, so
its regenerated `process.md` marks all 20 action steps as legacy-missing rather
than inventing post-hoc inputs.

Verification: 7 V2 tests and all 86 repository tests passed; compileall, CLI
help, prompt regression checks, and `git diff --check` passed. Follow-up: the
next real model run is required to evaluate the new Entity-attribution quality;
the historical Opus48 decision sequence was intentionally not rewritten.

## 2026-08-29 — Annotated audit fixes and interrupted semantic rerun

Answered all 22 user annotations directly beside their original locations in
`outputs/review/v2_generic_cd82_level1_opus48/process.md`. Source/SDK checks
confirmed that ACTION1–4 directions are not public; ACTION6 uses display x as
left-to-right column and y as top-to-bottom row; cognition view hashes were
post-run audit fields rather than Agent inputs; `palette` was a model-generated
hypothesis, not a prompt/source leak; the bottom yellow CD82 row is a hidden
source-defined 100-action budget indicator; Update proposes operations while
the deterministic graph transaction commits them; and the apparent repeated
red Entity was an upsert of an existing ID, not a duplicate creation.

Fixed the main visual-boundary defect found by review. The prior V2 sent a
512x512 image with artificial 8-cell gridlines to the model, which plausibly
caused the false 2x2-block segmentation. `PublicARCSession` now stores a raw
64x64 no-overlay `agent_view` used exclusively for model image inputs and a
separate 512x512 guided `review_view` for human Markdown links. ACTION6's
public contract now states the generic x/y display convention. Schema visual
support now attaches the same Evidence's raw before/after pair, not only its
after image.

Expanded Evidence records with resolvable artifact descriptors and expanded
cognition receipts with the actual Entity, Prototype, Schema, and Evidence
summaries sent to Agents; relations, full assertion history, audit log, and
artifact bytes remain omitted. The reporter distinguishes new Entity
candidates from existing Entity upserts, identifies view hashes as audit-only,
shows raw Agent versus guided review images, explains runtime scheduling and
goal hypotheses, and renders full Evidence/EFPS input summaries. Fixed the
legacy empty `unassigned_visual_changes` display.

Attempted the requested fresh Opus48 CD82 Level 1 run in
`outputs/review/v2_generic_cd82_level1_opus48_semantic_20260829/`. It executed
8 actions, then OpenRouter returned HTTP 402 during the Step 9 Main call; only
the OpenRouter key is configured. The directory has `failure.md` and an
unfinished 8-step trajectory, but no final timeline/graph and is explicitly
not an experiment result. The observed artifact metadata confirms Agent images
were 64x64/no-overlay and review images 512x512/guided. Added per-complete-step
partial timeline/graph/process checkpoints for future costly runs; successful
runs delete them, provider failures retain the last complete cognitive state.

Verification: 8 V2 tests and all 87 repository tests passed; compileall and
`git diff --check` passed. Unfinished follow-up: replenish OpenRouter credit,
then rerun from Step 0 into a new output directory and audit the complete new
`process.md`, semantic transitions, Evidence resolution, image roles, replay,
and token usage.

## 2026-08-29 — Complete post-review CD82 rerun

After OpenRouter credit was restored, performed fresh Step-0 runs with the
generic V2 Agent. Two intermediate runs were safely retained as incomplete
checkpoints: one stopped at Step 12 after a truncated Update JSON response, and
one stopped after Step 1 when Update asserted an Entity change for an unchanged
public grid. Added two game-agnostic reliability guards without weakening
evidence validation: structured JSON receives one compact regeneration attempt
after parse failure, and Update receives one correction attempt after a
semantic evidence-constraint violation by re-reading the public result already
present in its original input.
Both attempts aggregate token usage; a second failure still aborts before graph
commit. Added regression tests for both paths.

The final clean run is
`outputs/review/v2_generic_cd82_level1_opus48_semantic_complete3_20260829/`.
It completed the configured 20-action budget and produced final `process.md`,
`report.md`, `timeline.json`, `cognition/graph.json`, and a replayable trajectory.
It did not solve Level 1: `public_levels_completed=0`, so the result is a
complete failed exploration record rather than a successful gameplay claim.
Final cognition contains 21 Evidence records, 7 Entities, 18 Feature
definitions, 24 Feature assertions, 1 Prototype, 16 Schemas, and 58 Relations.
Replay passed all 20 steps; forbidden-read counters are all zero. Steps 13 and
19 each used one semantic correction retry.

Audit verification found 21 chronological sections, 161 valid local Markdown
links with no missing targets, no leftover partial files, and 93 model image
attachments all resolving to 64x64 no-overlay `agent_view` artifacts; guided
`review_view` assets remained human-only. Model usage was 1,566,590 input and
69,925 output tokens (1,636,515 total). Verification: 10 V2 tests and all 89
repository tests passed; compileall and `git diff --check` passed. Follow-up:
human review of the new `process.md`; the main behavioral limitation is failure
to infer and complete Level 1 within 20 actions, and the rapidly growing prompt
size should be optimized before larger experiments.

## 2026-08-29 — EFPS prompt-cost audit and retrieval design

Answered the new artifact annotation directly in the completed run's
`process.md`. Clarified that an Evidence record's three artifact roles are a
durable storage inventory, not three model inputs: only the 64x64 no-overlay
`agent_view` is sent to cognitive Agents; `review_view` is human-only and
`environment_state` is an NPY used for deterministic diff/integrity/replay.
Updated future reporting to state this boundary beside every expanded Evidence.

Clarified the Update correction path. `public_result` is the same public
transition delta already supplied on the first attempt, derived by the runtime
from the environment response and before/after observation; it is not Gold,
source-derived, or newly injected knowledge. Removed the misleading
"authoritative" wording and duplicated result from the retry instruction. The
retry now explicitly says no new observation or external fact is provided.

Audited the 1,636,515-token run: 1,566,590 input and 69,925 output tokens.
Main used 609,537 input tokens (38.91%), Exploration 589,635 (37.64%), and
Update 367,418 (23.45%). Stored cognition-receipt character volume is 62.10%
Evidence, 25.07% Entity/current Features, 8.69% summarized Schema, and 1.29%
Prototype; this is only a proxy because prior logs lack field-level provider
token attribution and the live Schema payload was fuller than the receipt.
Update output is dominated by Schema updates (39.24%), transition analysis
(26.33%), and Entities (21.94%); Main output is dominated by repeated goal
hypotheses (41.22%) and rationale (33.29%).

Added `docs/v2_cognition_retrieval_design.md`. The design separates full durable
EFPS storage from a small prose working set and proposes a read-only
`query_cognition` tool with top-k retrieval and explicit Evidence resolution.
Default inputs become Agent-specific summaries; full nodes, history, and raw
before/after images are retrieved only when needed. It also identifies fields
that stay durable but leave default prompts, replaces full-dict JSON input with
descriptive Markdown, proposes constrained graph-operation tools for Update,
and adds per-section request instrumentation. No live retrieval protocol was
implemented in this review task because it changes the Agent contract and
requires a separately verified experiment. Verification: 10 V2 tests,
compileall, annotation/explanation counts, and `git diff --check` passed.
Follow-up: implement the retrieval contract, field-level usage instrumentation,
and compact Agent I/O, then rerun the same 20-step configuration for comparison.

## 2026-08-29 — Completed second annotated review and narrowed compaction policy

The follow-up review used 13 bracketed annotations, not only the single
`【标注：...】` form previously matched. Added explanations directly beside all
13 markers in the completed run's `process.md` (12 new plus the prior artifact
answer). Verified the initial public grid rather than trusting model labels.
The Step-0 Update had materially incorrect visual Entities/bboxes: it conflated
the large upper-left composite, merged/mislocated two upper markers, and split
the central open-bottom red/white frame plus lower black rectangle incorrectly.
The provider JSON was valid and parsed losslessly; the primary input defect was
shrinking the no-overlay Agent image to 64x64, compounded by unconstrained
visual descriptions. Future `agent_view` is now a 512x512 nearest-neighbor
render with no overlay; `review_view` remains a separate same-size guided image.

Removed `logical_grid_sha256` from the model-facing observation receipt while
retaining it in durable observation/artifact state. Future reporting labels the
guided image as human-only rather than placing it ambiguously under Agent input.
Clarified in the historical process that artifact IDs are not images, current
Agents cannot dereference them, full Evidence descriptors sometimes did bloat
text prompts, and only explicitly attached `agent_view` data reached the model.
Also documented unsupported Main evidence citations, the difference between
hypothesis confidence and action utility, and that Entity transition semantics
were generated by Update rather than deterministic runtime.

Revised `docs/v2_cognition_retrieval_design.md` after user feedback. Compaction
must not hide whole node classes: every Entity, Prototype, and Schema remains in
the default prompt as a short descriptive line; important current Features and
important Relations also remain. Only full Feature history, raw relation-edge
records, repeated artifact metadata, and full Evidence details move behind
retrieval. Important relations are rendered inline (Entity features/Prototype
membership/Schema role bindings). Proposed a persistent hypothesis registry so
Main emits hypothesis deltas instead of repeating the full list every step and
links the selected action to a hypothesis/utility rationale. Verification: all
13 annotations have explanations, 10 V2 tests and all 89 repository tests
passed, compileall and `git diff --check` passed. Follow-up: implement the
prose renderer, read-only cognition tool, hypothesis registry, pixel-grounded
Entity checks, and per-field usage instrumentation before the next live run.

## 2026-08-30 — EFPSGraph data-structure review

Reviewed the current V2 EFPS models, graph implementation, storage, tests, and
architecture docs to answer its concrete representation. `EFPSGraph` is a
custom in-memory typed directed property graph implemented as per-type ID maps
plus a `Relation` edge map, with a separate durable Evidence registry,
revision/audit metadata, copy-validate-commit transactions, and JSON snapshots;
it is not a general graph-library object or the V1 layered Schema graph.
Noted that the implemented edge enum currently has six EFPS relation types,
while Schema support/counterevidence remain ID fields rather than materialized
relation edges. Verification was read-only source inspection; no tests were run
and no business code was changed. Follow-up: continue answering the user's V2
code questions.

## 2026-08-30 — Compact EFPS prompts, retrieval tool, and fresh CD82 run

Implemented compact Markdown cognition inputs for Main, Exploration, and
Update. Every current Entity, Prototype, and Schema remains visible as concise
prose with important current Features and inline typed Relations; full
Evidence history, artifact metadata, raw relation tables, and graph JSON are
not sent by default. The recent-transition window is three semantic
Entity-level summaries. Added a read-only `query_cognition` tool with type/ID/
action filters and summary/decision/evidence detail modes; it reads only the
learned EFPS and durable public Evidence, returns at most 10 records/8,000
characters, and is capped at two tool rounds per logical call. At the cap the
model is forced to answer from existing results instead of aborting.

Added exact audit data for actual prompt text/sections, tool arguments/results,
per-provider-round usage, images, and logical calls. Final runs now write
`token_usage.json` and `token_usage.md`, including Agent/step/request-phase
breakdowns. Provider tokens cannot be allocated exactly to fields, so section
composition is reported only as characters/UTF-8 bytes. Removed automatic
historical Evidence image attachments; current required public images stay
attached, and Evidence details/artifact IDs are queryable. Prototype creation
is Update LLM judgment with no minimum-member/Feature/repetition rule.

Fixed a live-audit defect where an instance-specific FeatureDefinition
description (for example a green object's `color`) appeared beside other
entities sharing that feature name. Instance descriptions now persist on
FeatureAssertion, survive updates/history/storage, and are used by both prose
catalogs and retrieval. Old graphs load with an empty assertion description.

The final fresh run is
`outputs/review/v2_generic_cd82_level1_opus48_compact_retrieval_assertions_20260830/`.
It completed 20 actions but did not pass Level 1. Final cognition has 21
Evidence, 9 Entities, 24 FeatureDefinitions, 33 FeatureAssertions (all with
assertion descriptions), 1 Prototype, 8 Schemas, and 71 Relations. Replay and
graph validation passed; all Evidence IDs resolve; 123 process links resolve;
no partial files remain; Agent input scans found no game ID, review view,
logical grid hash, goal mask, or precomputed route. Forbidden-read counters
are zero.

Usage was 1,042,741 input + 97,378 output = 1,140,119 tokens across 61 logical
calls, 130 provider requests, and 67 cognition queries. Versus the previous
1,636,515-token run, input fell 33.44% and total fell 30.33%, while output rose
39.26%. First requests consumed 406,224 tokens; additional tool/retry requests
consumed 733,895, so query policy remains the main cost bottleneck. Default
learned-cognition text was 79.53% of default section characters; this is not a
provider token share. Verification: all 91 repository tests passed, compileall,
`git diff --check`, replay, graph validation, Evidence resolution, prompt-boundary
scan, and Markdown-link checks passed. Removed two non-final runs created
during development. Follow-up: reduce unnecessary Main/Update queries, add a
persistent Main hypothesis registry, and improve exploration efficiency; the
generic Agent still fails to complete Level 1 within 20 actions.

## 2026-08-30 — Exact cognition-read protocol after process review

Explained every new `【标注：】`/`【标记：】` inline in
`outputs/review/v2_generic_cd82_level1_opus48_compact_retrieval_assertions_20260830/process.md`,
clearly labeling additions as later audit commentary. The review confirmed that
the old lexical query used no extra LLM but treated exact IDs as ranking boosts
and then filled top-k with unrelated records; Update's Main-decision context was
also genuinely truncated at 220 characters.

Replaced the fuzzy interface with deterministic
`read_cognition(command, id, feature_id?)`. Fixed commands now read one exact
Entity, Prototype, Schema, Evidence, Feature history, relation neighborhood, or
agent-visible artifact without search, ranking, summarization, or inference.
Evidence records now carry phase-labeled `current/before/after`
`observation_refs`; `get_artifact` can attach the exact stored public PNG to the
next model request while rejecting review-only images and environment arrays.
Prompts document EFPS/Evidence and the full tool contract; reports include the
actual system instructions and returned tool images. Default catalogs show only
visible Entities, omit redundant active status and empty Prototype options, and
Update receives the full decision hypothesis.

Enforced the design rule that every Schema role binds a Prototype, never an
Entity: prompts require it, the Update translator discards unresolved/empty
bindings, and graph validation rejects and rolls back invalid Schema
transactions. Rejected unstored Main hypotheses are now directed to discarded
inferences instead of fake counterevidence operations. Updated README,
architecture, and retrieval-design documentation. Verification: compileall,
all 92 repository tests, focused artifact-image transport and Schema rollback
tests, and `git diff --check` passed. No real-provider CD82 rerun was started;
new token usage and gameplay behavior remain to be measured in the next live
review run.

## 2026-08-30 — Goal-directed choice, multi-level boundary, and 10-step smoke

Applied the two newest inline review notes. Kept one-step repetition for a
falsifiable reversibility/boundary test. Revised Main and Exploration prompts so
level completion is primary, exploration is only instrumental, predictable
action effect is separated from goal utility, known goal-directed Schemas are
preferred, and reversible states are not revisited without a new material
hypothesis or plan requirement. Added implementation notes beside both new
markers in the reviewed historical `process.md`.

Made multi-level execution explicit. Positive public `level_delta` now selects
the `public_level_boundary` Update phase: completion is the prior action's
terminal effect and the after image is treated as the next level's new scene,
not a scene-wide ordinary action effect. Prior active Entities not reidentified
in that new scene are evidence-groundedly marked disappeared; new same-label
objects receive new Entity IDs unless Update explicitly cites a persistent old
ID. Prototypes, Schemas, Feature history, and Evidence remain continuous.
Documented that `stop_after_levels` is cumulative while `max_steps` is global.

Verification: compileall, `git diff --check`, and all 93 repository tests passed.
The V2 two-level fixture completed two boundaries in two actions and presented
the Level 2 Entity to the second Main call; the real CD82 public-environment test
continued through all six levels. A real zero-prior Opus 4.8 smoke run at
`outputs/review/v2_generic_cd82_level1_opus48_goal_directed_exact_read_smoke10_20260830/`
ran the full 10-action budget without completing Level 1. It produced 11
Evidence, 9 Entities, 3 Prototypes, 6 Prototype-bound Schemas, 101 Relations,
and 724,136 total tokens across 81 provider requests and 47 exact reads. All
tool reads succeeded, 4 stored images were returned, replay passed 10/10, graph
validation passed, 63 process links resolved, forbidden-read counters were
zero, and no partial files remained. First seven decisions explored unknown
semantics; the final three used learned Schemas for goal-configuration attempts.
The failed sandbox-network initialization directory was moved recoverably to
`/tmp/socialclaw_failed_cd82_smoke10_20260830` before the successful output was
given the final directory name. Follow-up: formal multi-level model behavior is
ready to run, but the Agent's CD82 goal inference still did not solve Level 1
within 10 actions and exact-read continuation rounds remain token-heavy.

## 2026-08-30 — Per-level budgets, recoverable GAME_OVER, and three-game formal run

Changed V2 ARC execution so `--max-step`/`--max-steps` is an independent Agent
action budget for each level. Passing a level resets only that budget;
`--stop-after-levels all` continues until public WIN or the first failed level.
Added optional public GAME_OVER recovery for TU93: runtime reset is audited as
an `ENV_RESET` trajectory event, is not an Agent action, preserves the current
public level, does not refund consumed actions, triggers an Update-only scene
realignment, and is supported by replay. Main receives used/remaining per-level
budget. Compact process reports omit full prompts while retaining public input,
EFPS summaries, exact cognition reads, outputs, images, and token statistics.

Added a final tool-free structured-output transport repair after two invalid
JSON responses; all attempt usage remains charged and semantic validators are
unchanged. Two concurrent provider-format interruptions were retained as
incomplete checkpoints and excluded from results; final experiments were run
serially from empty cognition. CD82, SK48, and TU93 each failed Level 1 at its
30-action limit, for 0/1 pass rate each. Token totals were 2,748,352,
2,871,771, and 2,612,454 respectively (8,232,577 combined). TU93 never reached
public GAME_OVER, so its formal run had zero resets; reset behavior was covered
by a same-level budget fixture and a real public-environment level-preservation
check. All three 30-step trajectories replayed exactly, every learned Schema
Evidence ID resolved, compact reports retained EFPS, and all report image links
resolved. Verification: compileall, `git diff --check`, and all 97 tests passed.
Follow-up: the generic Agent still lacks efficient goal inference and planning;
formal online multi-level continuation remains behaviorally unobserved because
no game passed Level 1 within budget.

## 2026-08-31 — All-game GAME_OVER recovery and committed formal-run replay

Generalized the previous recovery option: every V2 game now defaults to
same-level recovery after public `GAME_OVER`; `--no-reset-on-game-over` is the
explicit opt-out. The reset remains an audited, non-Agent `ENV_RESET`, does not
refund the current level's action budget, and triggers only Update-side scene
realignment. A generic fixture now verifies the default rather than a TU93-only
configuration. Removed the `process.md` introductory section headed “先解释：
什么是…”, while retaining each step's actual recent-transition input.

Added a Git-tracked deterministic reproduction bundle under
`experiments/v2_formal_20260830/`: three compressed audited logical model
transcripts, environment/transcript fingerprints, expected summaries, exact
result-log SHA-256 values, critical dependency versions, and a single replay
command. `RecordedVisionModel` validates instructions, payload, and image hashes
for every frozen call, then the normal environment, EFPS validator, trajectory
replay, and report code run unchanged. A clean three-game replay matched
`process.md`, `report.md`, `timeline.json`, both token reports, and final graph
byte-for-byte: CD82/SK48/TU93 remained 0/1 at 30 actions with 2,748,352 /
2,871,771 / 2,612,454 tokens. This is artifact reproduction, not a new provider
trial; fresh online temperature-zero calls are not promised to be identical.

Organized current documentation through `docs/README.md`, moved the original
V2 design input and early CD82 prototype report into `docs/archive/`, and
documented why two provider-interrupted partial runs led to a clean serial
three-game batch. Verification: compileall, all 99 tests, `git diff --check`,
and the full no-API three-game hash-verifying reproduction passed. Follow-up:
no formal run reached Level 2 and none triggered a real GAME_OVER, so online
cross-level behavior and repeated-reset behavior still need future trials.

## 2026-08-31 — Schema triples and global Insight memory

Corrected the V2 cognitive contract. A Schema's semantic content is now exactly
one evidence-grounded `Prototype -> Action -> Output` triple (`prototype_id`,
public action name/arguments, and non-empty observable output). Removed active
role bindings, preconditions, invariants, expected-change lists, and boundary
conditions from the Schema model. Graph contract/format 3 validates exactly one
`TAKES_PROTOTYPE` edge per Schema and can migrate persisted format-2 role-based
graphs into triples while retaining retired fields only as migration metadata.

Added independent global `Insight` memory for rules, constraints, candidate goal
conditions, mechanics, strategies, and other cross-action knowledge. Insights
have create/support/counterevidence/revise operations, kind/scope/confidence/
status, and mandatory durable support Evidence. All Insights appear as concise
text in every Agent's default cognition, are available through exact
`get_insight`, and remain across levels/resets. Main can cite stored Insight IDs
in a dedicated `insight` decision mode without pretending they are Schemas; a
Schema decision may also cite complementary Insights. Timeline receipts,
process/report output, counts, and final graph persistence now include Insights.

Marked the 2026-08-30 frozen three-game transcript as cognition contract 2.
Current contract-3 code rejects it before environment execution so historical
role-based responses cannot be presented as results of the new implementation;
a new online experiment/frozen bundle is required later. No game experiment was
run for this task. Verification: compileall, all 102 repository tests, focused
triple/Insight/migration/Main-decision tests, and `git diff --check` passed.

## 2026-08-31 — Tycho-based V3 EFPS development plan

Reviewed the current contract-3 V2 implementation, its frozen three-game Level
1 results, the Tycho paper, and official Tycho source at commit
`f68912a764372ead0a610db2e1c011d41ce5197e`. Chose a fork-first, pinned-upstream
V3 architecture: Tycho remains the execution, verification, and planning core;
its executable `world_model.py` is the sole dynamics source of truth; EFPS is
an evidence-grounded executable view over that model for entity-to-prototype
classification and `Prototype + Action -> Output` rules. Default control uses
Tycho's actor-orchestrated builder instead of V2's mandatory three-agent call on
every step. V2 stays frozen and import-independent for comparison.

Added `docs/version3_tycho_efps_development_plan.md` and linked it from both
documentation indexes. The plan defines migration boundaries, target modules,
phased implementation, separate dynamics/outcome/plan gates, EFPS integrity
metrics, and a no-paid-call first development slice. Verification covered the
current source and upstream source/documentation, pinned revision, document
links, and whitespace/diff checks. No unit tests or live model/API experiments
were run because this task changed documentation only. Follow-up: implement
Phase 0 upstream pin/parity, then the provider-compatible V3 runner and EFPS
runtime before changing prompts or starting paid game trials.

## 2026-08-31 — V3 Phase 0–2 executable foundation

Imported the official Tycho package and 51-file parity suite at commit
`f68912a764372ead0a610db2e1c011d41ce5197e`, retaining Apache-2.0 attribution,
the public release manifest, and an explicit four-item narrow patch inventory.
The root package preserves Tycho execution, typed workspace history, verifier,
outcome checks, planners, guarded resume, viewer, and actor-orchestrated builder.
`sc-run-arc-v3` defaults to `tycho_efps`/orchestrator, maps useful V2
OpenAI-compatible settings, uses the existing pinned ARC game directory, hashes
local game semantics into the immutable run spec, and rejects any formal run
outside Python >=3.12 with Tycho's exact critical dependency versions.

Added `socialclaw.v3` stable typed Evidence indexing, an EFPS-aware Tycho
workspace/executor, and a standard-library-only runtime copied into each game
sandbox. Executable Prototype matchers and `Prototype + Action -> Output`
handlers require durable Evidence closure, reject duplicate/conflicting triples,
report actual handler attribution, and export a manifest bound to the current
`world_model.py` hash. Every substantive model edit now receives Tycho's normal
dynamics/outcome/planner feedback followed by EFPS audit feedback. Simulated
states never enter the harness-authored Evidence index. V2 source was unchanged.

Added bounded V3 configs, implementation/run documentation, 13 V3 tests, and
upstream-manifest snapshot verification. Verification: Tycho parity 162 passed / 2
skipped; final full repository suite 277 passed / 2 skipped; compileall, config
parsing, runtime-contract diagnostics, and diff checks passed. No model, ARC
online, or paid API call was made. The existing `.venv` is Python 3.11 with
`arc-agi==0.9.8`, so exact Python 3.12 installation/package smoke remains before
any credentialed run. Follow-up: complete that environment gate, then implement
Phase 3 Actor/Builder EFPS prompting and run a separately approved bounded
transport/CD82 Level 1 smoke.

## 2026-09-02 — V3 runtime unblock and first bounded CD82 smoke

Replaced the default environment with Python 3.12.14 and the pinned V3 runtime
(`arc-agi==0.9.9`, `arcengine==0.9.3`, Jinja/Numpy/Pillow/PyYAML pins), then
installed the full project dependency set with CPU-only PyTorch so V2 remains
importable. Preserved the former Python 3.11 environment at
`.venv-py311-backup-20260901/`. `pip check` and the V3 runtime contract pass.

Added audited `--max-actions-per-level` and `--stop-after-levels` limits. They
propagate to workers, stop with distinct reasons, and participate in immutable
run identity. Because this host's Docker socket is inaccessible, added a
Bubblewrap backend whose live doctor verified no outside filesystem access, no
network, cleared secrets, read-only root, writable game workspace, zero effective
capabilities, and no-new-privileges.

The one-call OpenRouter Opus 4.8 text/image/tool smoke passed. The offline CD82
Level 1 trial at `outputs/v3/cd82_level1_5actions_opus48_20260901/` committed
exactly five actions (`ACTION1, ACTION2, ACTION6(31,28), ACTION3, ACTION3`),
completed no level, stopped as `requested_action_limit`, used 10 model calls,
and recorded an estimated $0.89337 inference cost. Audit found that the final
budget-exhausting nonterminal outcome was absent from EFPS evidence; fixed the
opt-in final-observation callback and added a regression test without rewriting
the completed raw artifact. The actor never invoked the builder in this five-step
run (`builder_invocations=0`), left the seeded world model unchanged, and produced
no EFPS manifest, confirming that Phase 3 prompt integration is still required.
Verification: 282 passed, 2 skipped; full dependency
check, sandbox doctor, runtime contract, and `git diff --check` passed. Follow-up:
implement Phase 3 Actor/Builder EFPS prompting before any matched performance
comparison; this smoke does not establish an EFPS improvement.

## 2026-09-02 — V3 architecture and collaborator onboarding

Added `docs/v3_architecture_and_collaboration.md` as the GitHub onboarding entry
for V3. It documents Tycho as the control/execution/planning owner, EFPS as an
Evidence-grounded executable view inside the same `world_model.py`, the V2-to-V3
concept mapping, per-action lifecycle, code ownership boundaries, upstream patch
policy, current Phase 3 gap, and suggested parallel workstreams. Updated the root
and documentation indexes plus the previously stale development-plan status.

The pre-push audit found and removed a `tycho/` ignore rule that would have omitted
both the vendored runtime and `third_party/tycho` provenance/tests from GitHub.
Secret-pattern review matched only documented placeholder examples; no generated
outputs or environments are eligible for commit. Verification: all 18 V3 tests,
`git diff --check`, and local-link validation across 26 Markdown files passed.
No commit or remote push was performed. Follow-up: implement and test Phase 3
Builder/Actor EFPS prompting before describing EFPS as active game cognition.
