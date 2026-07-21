# Codex Project Memory

> Created: 2026-06-27
> Purpose: quick handoff notes for future Codex sessions. Treat this as a compact map, not as ground truth over source code.

## Project Goal

SocialLearningClaw studies proactive social learning for agents. The central claim is that agents should improve through explicit, reusable schema/skill knowledge and human interaction, not only through passive trace reflection. The desired evidence is better sample efficiency, transfer, and learning ceilings than ordinary self-evolution or memory baselines.

Core loop:
1. Extract or retrieve needed concepts from a structured schema graph.
2. Ask humans when important concepts are missing or when a high-confidence answer fails.
3. Answer or act with schema context.
4. Reinforce/correct concept confidence and relation weights from feedback.

## Main Workstreams

### 1. ARC-AGI-3 Experiments

Primary implementation:
- `socialclaw/run_arc_agi3.py`: CLI entry.
- `socialclaw/arc_runner.py`: schema-based ARC-AGI-3 multi-step loop.
- `socialclaw/dataset/arc_agi3.py`: wrapper around `arc_agi.Arcade`.
- `socialclaw/schema/arc_agi3_parser.py`: grid object extraction, spatial relations, action-effect concepts.
- `scripts/run_arc_baselines.py`: zero-shot, few-shot, RAG, with-rule ARC baselines.
- `scripts/run_arc_memory_baseline.py`: Reflexion and ExPeL ARC memory baselines.
- `scripts/eval_arc_summary.py`: summarizes ARC run folders.

Schema ARC behavior:
- Step 0 extracts initial objects before the first prompt, so the first prompt can already include observed concepts.
- BFS object concepts now use stable IDs like `obj_l{level}_{color}_{idx}`, so unchanged frames should update/reuse concepts instead of accumulating per-step duplicates.
- After each action, pre/post grids are compared. The runner adds `action_effect` concepts and `no_effect` or `affected` relations, saves schema immediately, and injects recent action-effect feedback into later prompts.
- If a grid is unchanged, current code reuses previous concepts instead of re-extracting objects.
- Recent human feedback concepts and recent action-effect concepts are force-injected into ARC prompts, bypassing weak embedding recall.
- Outputs live under `runs/arc_agi3/{model}/{timestamp}/`, with per-level `episode.json`, `trajectory.json`, `step_*.json`, `step_*.png`, and run-scoped `schema/`.

ARC baseline set:
- `zero_shot`: grid/action/history only.
- `few_shot`: injects `docs/{game}_fewshot.md`.
- `rag`: online buffer retrieves similar past grid/action/outcome records.
- `withrule`: injects human-written `docs/{game}_rules.md`.
- `schema`: this project's schema method.
- `reflexion` / `expel`: cross-level memory baselines in `scripts/run_arc_memory_baseline.py`.

Common ARC command:
```bash
.venv/bin/python -m socialclaw.run_arc_agi3 \
  --game-id sk48-d8078629 \
  --model google/gemini-2.5-pro \
  --max-steps 200 \
  --auto-yes
```

Run all standard ARC baselines:
```bash
bash scripts/run_all_baselines.sh
```

### 2. Benchmark Selection

Primary docs and scripts:
- `SelectBenchmark/README.md`: rough benchmark notes and possible baselines.
- `SelectBenchmark/report.md`: benchmark suitability report.
- `SelectBenchmark/eval_contextmath*.py`: ContextMATH evaluation variants.
- `SelectBenchmark/eval_intphys2*.py`: IntPhys2 evaluation variants.
- `SelectBenchmark/eval_a2rbench.py`: A2RBench evaluation.
- `SelectBenchmark/eval_contextmath_memory.py` and `eval_intphys2_memory.py`: Reflexion/ExPeL memory baselines for selected benchmarks.

Current selection conclusion from `SelectBenchmark/report.md`:
- IntPhys2 is recommended: current strong models remain far below humans and show a clear bias toward judging videos impossible.
- ContextMATH is partially suitable, especially CS splits where models make contextual extraction and arithmetic mistakes.
- A2RBench is not suitable for this project now because Claude Opus 4.8 scored very high in the local sample.

## Static QA / CL-bench / PBench Pipeline

Primary implementation:
- `socialclaw/run_clbench.py`: CLI for CL-bench, PBench, and static ARC jsonl.
- `socialclaw/pipeline.py`: retrieval, schema initialization, answer, evaluation, reinforcement/correction, episode logging.
- `socialclaw/schema/graph.py`: `Concept`, `Relation`, `SchemaGraph`, fuzzy matching, confidence computation.
- `socialclaw/schema/retriever.py`: LLM concept extraction plus embedding retrieval.
- `socialclaw/schema/initializer.py`: auto schema generation and parsing human answers/corrections.
- `socialclaw/prompt_builder.py`: schema injection prompts.
- `socialclaw/evaluator.py`: exact/MCQ/LLM-as-judge evaluation.

CL-bench important behavior:
- `run_clbench.py` auto-enables `group_by_context` for CL-bench.
- In context mode, tasks are sorted by `context_id`, then `msg_count`, then `id`.
- Each context gets its own schema directory under the run folder.

Common CL-bench command:
```bash
.venv/bin/python -m socialclaw.run_clbench \
  --prepared data/clbench/prepared/clbench.jsonl \
  --model moonshotai/kimi-k2.6 \
  --max-problems 5 \
  --auto-yes
```

## Memory Agent Utilities

Shared modules:
- `socialclaw/memory_agents/reflexion.py`: stores verbal reflections generated after failures.
- `socialclaw/memory_agents/expel.py`: stores an experience pool and periodically distills general insights.

These are baseline mechanisms, not the project's proposed schema graph. They are useful as comparisons because they optimize text memory rather than structured concept/relation knowledge.

## Known Issues / Watch Items

- Some older docs mention issues that appear partly fixed in code. Always verify against source before acting. Examples: stable object IDs and action-effect category injection look fixed now.
- `docs/handoff.md` says timeout after human correction retries the level; it also notes a later thought that correction should maybe continue the game instead of retrying. Check the desired experimental protocol before changing this.
- Human-in-the-loop input is awkward inside Claude/Codex terminal tools because warnings and stdout/stderr can pollute `input()`. Prefer local terminal for real interactive runs, or use `--auto-yes` for automated tests.
- Embedding retrieval with `BAAI/bge-small-en-v1.5` can miss visual/grid concepts and human feedback. ARC currently force-injects recent human/action-effect concepts as a workaround.
- `write_step()` still writes fixed `step_{step:03d}.json` paths inside each level directory. If retrying the same level in the same directory, earlier attempt step files may be overwritten.
- The worktree may contain active user/generated changes. Do not reset or remove them without explicit instruction.

## Project Conventions

- Use `.venv/bin/python` from the repo.
- API keys are loaded from `.env`; OpenRouter is the default LLM endpoint.
- `ARC_AGI_API_KEY` is copied to `ARC_API_KEY` for the ARC SDK.
- ARC code sets `no_proxy` for `three.arcprize.org,arcprize.org` because ARC server access should bypass the proxy while OpenRouter may still need it.
- Run outputs belong in `runs/`; benchmark-selection results belong in `SelectBenchmark/results/`.
