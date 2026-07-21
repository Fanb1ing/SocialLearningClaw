# Handoff Document

> Last updated: 2026-05-26
> Purpose: Summarize current project state so a new Claude session can pick up quickly.

---

## 1. Project Overview

**SocialLearningClaw** — Proactive Social-evolve Agent.
A Schema-based agent learning framework where the agent:
1. Matches problems against a structured Schema graph (Concept + Relation).
2. Proactively asks humans when concepts are missing or when high-confidence reasoning leads to wrong answers.
3. Updates Schema confidence/weights based on evaluation feedback.

Current stage: Stage 1 + Stage 2 implementation complete. ARC-AGI-3 interactive environment is the primary debugging target.

---

## 2. Recent Changes (This Session)

### 2.1 Grid Display Bug — FIXED
- **File**: `socialclaw/stage1/dataset/arc_agi3.py`, `socialclaw/stage1/run_arc_agi3.py`
- **Root cause**: `grid_to_text(grid, max_size=16)` center-cropped 64x64 grids to a 16x16 region. For sk48, the center region happened to be entirely Yellow(4), hiding all actual objects.
- **Fix**: Increased `max_size` from 16 to 32 in both the function default and the call site. LLM now sees a 32x32 center region instead of 16x16.

### 2.2 Meaningless Transformation Rules — FIXED
- **File**: `socialclaw/stage1/schema/arc_agi3_parser.py`
- **Root cause**: `diff_objects_to_rules` did a Cartesian product of all `prev_concepts x curr_concepts`, generating nonsensical rules like `BlueBlob_0 -> transformed_by_ACTION6 -> obj_l0_s1_0/1/2`.
- **Fix**: Implemented object tracking by color + centroid proximity. Each previous object now maps to at most ONE current object with matching color. Eliminates the Cartesian product explosion.

### 2.3 Episode Size Bloat — FIXED
- **File**: `socialclaw/stage1/types.py`, `socialclaw/stage1/pipeline.py`, `socialclaw/stage1/run_arc_agi3.py`
- **Change**: Removed `raw` field from `AttemptRecord`. Episode JSON no longer stores the full raw LLM response / messages array.
- **Note**: `AgentAttempt.raw` still exists for internal agent debugging, but it is not persisted to Episode.

### 2.4 Per-Step Schema Persistence + Action-Effect Learning — DONE
- **File**: `socialclaw/stage1/run_arc_agi3.py`, `socialclaw/stage1/schema/arc_agi3_parser.py`
- **Change**: After each action, the code now captures `pre_grid`, executes the action, captures `post_grid`, and runs `compute_grid_diff()` to detect pixel-level changes.
- **Action-effect concepts/relations**: `build_action_effect_concepts_and_relations()` creates:
  - An `Action_` concept (e.g., `Action_ACTION6_at_18_58`).
  - If no grid change: a `no_effect` relation (weight=0.9).
  - If grid changed: `affected` relations linking the action to post-action objects.
- **Per-step persistence**: `storage.save(graph, embeddings)` is called after every step, not just at level end.
- **Prompt feedback**: `_build_arc_prompt` injects a `Learned action effects:` block so the LLM sees feedback like `ACTION6 at (18,58) had no effect on grid`, preventing repeated ineffective clicks.

### 2.5 Trajectory JSON — DONE
- **File**: `socialclaw/stage1/logging.py`, `socialclaw/stage1/run_arc_agi3.py`
- **Change**: Added `write_trajectory()` in `logging.py`. Each level now outputs `trajectory.json` containing a clean array of step records: `step`, `action`, `x`, `y`, `state`, `grid_changed`, `schema_concepts_added`.

### 2.6 LLM Vision Coordinate Scale Fix — DONE
- **File**: `socialclaw/stage1/schema/arc_agi3_parser.py`
- **Root cause**: The static prompt said "64x64 pixels", causing the vision LLM to return pixel-scale coordinates (e.g., 200–600). These were stored in schema and blindly used by the action model, causing out-of-bounds clicks like `(320, 960)`.
- **Fix**: `_CONCEPT_EXTRACTION_PROMPT_TEMPLATE` is now a dynamic template that accepts `{h}`, `{w}`, `{h_max}`, `{w_max}`. It explicitly instructs the model to return grid-cell indices, not pixel coordinates. `llm_extract_grid_concepts()` validates that all coordinates are within `[0, w-1]` and `[0, h-1]`, discarding out-of-bounds concepts with a warning.

### 2.7 Episode Records Model Name — DONE
- **File**: `socialclaw/stage1/types.py`, `socialclaw/stage1/run_arc_agi3.py`
- **Change**: `Episode` dataclass now has an optional `model: str` field. Every run (ARC-AGI-3, CL-bench, PBench) persists the LLM model name in `episode.json`.

### 2.8 Correction Threshold Lowered for Debugging — DONE
- **File**: `socialclaw/stage1/run_arc_agi3.py`
- **Change**: `correction_conf_threshold` default changed from `0.6` to `-1.0`. This guarantees that any timeout or failure triggers `human_io.ask_correction()`, making it easier to debug the human-in-the-loop correction flow.

### 2.9 Human Correction Flow Verified (3-Step Timeout) — DONE
- **File**: `socialclaw/stage1/run_arc_agi3.py`, `socialclaw/stage1/human_io.py`
- **Change**: Ran a `--max-steps 3` test. Timeout correctly leads to confidence calculation → `ask_correction()` UI (rich panel). The CLI panel displays problem ID, evaluation (TIMEOUT), confidence, original prompt, LLM answer, ground truth (WIN), concepts used, reasoning path, and explanation.
- **Caveat**: In Claude Code's Bash tool, `stderr` warnings interleave with `stdout`, polluting `input()` reads. For real interactive testing, run directly in a local terminal.

### 2.10 Retry After Human Correction — DONE
- **File**: `socialclaw/stage1/run_arc_agi3.py`
- **Root cause**: When timeout occurred after `ask_correction()`, the code hit `stop_reason == "max_iters"` and immediately `break`, terminating the run. The human-provided correction was saved to schema but never used.
- **Fix**: Moved `level_retries` outside the outer loop so it accumulates across retries. When timeout + `"human_correction" in flags`, the level now retries (up to `max_retries_per_level`) instead of breaking. This also fixed the existing `GAME_OVER` retry bug where `level_retries` was reset to 0 at the top of every loop iteration, making `max_retries` never enforced.

### 2.11 Explicit Human Feedback Concept Injection — DONE
- **File**: `socialclaw/stage1/run_arc_agi3.py`
- **Root cause**: `_retrieve_relevant_concepts` relies on BGE embedding similarity between a grid object summary and concept names. Human feedback concepts (e.g., "Stick") often have low similarity to grid queries and get filtered out, so the LLM never sees them in the prompt.
- **Fix**: Added explicit extraction of `source in ("human_feedback", "human_init_concepts")` concepts and merged them into `merged_concepts` before building the prompt. Up to 5 most recent human concepts are always injected, regardless of embedding similarity.

---

## 3. Key Files and Their Roles

> **模块结构已于 2026-05-26 重构**：`socialclaw/stage1/` 扁平化为 `socialclaw/`，CLI 入口由 `run_stage1.py` 改为 `run_clbench.py`。

| File | Role |
|------|------|
| `socialclaw/run_arc_agi3.py` | ARC-AGI-3 CLI 入口（`main()`），调用 `arc_runner.run_arc_agi3()`。 |
| `socialclaw/arc_runner.py` | ARC-AGI-3 核心运行逻辑：多关卡循环、prompt 构建、action 解析、schema 更新、episode 写入。 |
| `socialclaw/run_clbench.py` | CL-bench / PBench / ARC CLI 入口（`main()`），调用 `pipeline.run_stage1()`。 |
| `socialclaw/pipeline.py` | CL-bench / PBench 核心 pipeline：schema 检索 → LLM 答题 → 评估 → schema 更新 → episode 写入。 |
| `socialclaw/utils.py` | 共享工具函数：`load_dotenv`、`make_run_dir`（CST 时间戳）、`add_concepts_with_embeddings`、`resolve_relation_names`、`add_relations_resolved`。 |
| `socialclaw/dataset/arc_agi3.py` | Env wrapper around `arc_agi.Arcade`. `grid_to_image()` renders grid as PNG. |
| `socialclaw/schema/arc_agi3_parser.py` | Grid -> Object -> Concept/Relation parser. `diff_objects_to_rules` does color+centroid matching. `compute_grid_diff` and `build_action_effect_concepts_and_relations` for action-effect feedback. |
| `socialclaw/schema/graph.py` | `SchemaGraph`, `Concept`, `Relation` dataclasses. Confidence computation via geometric mean. |
| `socialclaw/schema/retriever.py` | Embedding retrieval + sufficiency check (LLM extracts concept names, then embedding match). |
| `socialclaw/schema/initializer.py` | Agent auto-generates concepts; parses human answers / corrections. |
| `socialclaw/agent/openai_compatible.py` | OpenAI-compatible API client with retry/fallback. |
| `socialclaw/prompt_builder.py` | Prompt assembly with schema injection (for CL-bench / PBench). |
| `socialclaw/types.py` | `Episode`, `AttemptRecord` dataclasses. |
| `socialclaw/logging.py` | `write_episode()`, `write_trajectory()`, `write_step()`. |
| `socialclaw/human_io.py` | CLI proactive questioning UI (rich-based). |

---

## 4. Known Issues / TODOs

### 4.1 已识别待修复问题（2026-05-17）

11. **人类纠错/补充的回答未正确持久化到 schema**：`parse_correction()` / `parse_human_answer()` 解析后的 concept 疑似未被正确写入 schema，或写入后被覆盖/丢失。根因待查（可能涉及 id 冲突、save 时机）。
12. **人类回答后不应再回到 schema 充足度询问**：当前 `ask()` / `ask_correction()` 结束后会重新进入 `retriever.is_sufficient()` 检查，导致二次提问。应直接跳过 sufficiency 检查，回到主答题循环。
13. **每步重复抽取 grid concepts 造成冗余**：即使 `grid_changed=False`，代码仍会执行 `extract_grid_objects()` 生成新的 post-action concepts，导致 schema 中堆积大量完全相同的 object（如每步都生成新的 `BlueBlob_0`）。应仅在 grid 变化或首次观察时抽取，否则复用上一帧 concepts。
14. **Embedding 检索召回效果差**：`BAAI/bge-small-en-v1.5` 对 grid 对象摘要与人类反馈 concept（如 "Stick"）的语义匹配效果不佳，关键概念经常无法被召回。Stage 3 需考虑替换为视觉-语言嵌入模型或引入负采样/硬匹配策略。
15. **关卡重试时 step 文件被覆盖**：timeout/失败后触发人类纠错并重试关卡时，`write_step()` 的固定路径 `step_{step:03d}.json` 导致前一次尝试的 step 记录被覆盖，丢失完整失败轨迹。调整为人类纠错后不应该为重试游戏，而应该继续游戏。

### 4.2 长期待办

1. **Grid truncation still not full view**: `max_size=32` shows 32x32 center region of a 64x64 grid. Corner objects may still be hidden. If user reports missing colors again, increase to 48 or 64.
2. **Object matching heuristic is simple**: `diff_objects_to_rules` matches only by same color + closest centroid. It cannot track objects that change color. Future improvement: use area overlap or IoU.
3. **Concept duplication**: Auto-generated concepts may have duplicates (e.g. "Sighting Card" vs "Sighting Cards"). Deduplication not yet implemented.
4. **Baseline comparison**: Stage 3 requires baseline A (no Schema) and baseline B (old text Skill). Not started.
5. **Schema cross-benchmark migration**: Stage 3 goal. Not started.
6. **API stability**: `httpx.RemoteProtocolError` can still interrupt long runs. Current mitigation is try-except fallback, but some calls may still crash the process if the error happens outside the wrapped zone.
7. **Action-effect relation deduplication**: The `Learned action effects:` prompt block deduplicates by string equality, but duplicate `no_effect` relations for the same action at the same coordinates may still accumulate in the schema graph. Future improvement: merge or update existing relations instead of appending.
8. **LLM vision concept count**: LLM vision extraction (`llm_extract_grid_concepts`) is capped at 10 objects and adds ~1–2s latency per step. For fast runs, use BFS extraction (`--no-llm-concepts`).
9. **Embedding strategy needs tuning for action-effect learning**: The agent still clicks the same coordinates repeatedly despite `no_effect` relations being recorded. The current embedding retrieval does not effectively surface action-effect feedback to discourage repeated ineffective actions. This is identified as future work (embedding model adjustment or negative sampling) and is deferred to Stage 3.
10. **Interactive stdin pollution in Claude Code Bash**: `warnings.warn()` output on `stderr` gets merged into `stdout` by the Bash tool, causing `input()` in `human_io.ask()` / `ask_correction()` to read warning text instead of user input. For real interactive testing, run directly in a local terminal.

---

## 5. Running

> 完整参数说明见 `docs/cli.md`。

```bash
# ARC-AGI-3 标准运行
.venv/bin/python -m socialclaw.run_arc_agi3 \
  --game-id sk48-d8078629 \
  --model qwen/qwen2.5-vl-72b-instruct \
  --max-steps 200 --auto-yes

# CL-bench 运行
.venv/bin/python -m socialclaw.run_clbench \
  --prepared data/clbench/prepared/clbench.jsonl \
  --model moonshotai/kimi-k2.6 \
  --max-problems 5 --auto-yes
```

Command line: `runs/arc_agi3/<model>/<timestamp>/cmd.txt`
Episode output: `runs/arc_agi3/<model>/<timestamp>/<game_id>_L<level>/episode.json`
Trajectory output: `runs/arc_agi3/<model>/<timestamp>/<game_id>_L<level>/trajectory.json`
Schema output: `runs/arc_agi3/<model>/<timestamp>/schema/`

---

## 6. Data Structures Quick Ref

### Episode JSON
```json
{
  "created_at": "2026-05-14T...",
  "episode": {
    "problem": { "id": "sk48-d8078629_L1", "prompt": "...", ... },
    "attempts": [
      {
        "input_prompt": "...",
        "answer_text": "...",
        "reasoning_trace": { "concepts": [...], "relations": [...], "explanation": "..." },
        "usage": { "input_tokens": 0, "output_tokens": 0, "total_tokens": 0 }
      }
    ],
    "evals": [ { "correct": false, "pred": "ACTION6", "gold": "WIN", "details": "TIMEOUT" } ],
    "reasoning_trace": { ... },
    "reasoning_confidence": 0.0,
    "flags": ["schema_correct"],
    "stop_reason": "max_iters",
    "model": "qwen/qwen2.5-vl-72b-instruct"
  }
}
```

### Trajectory JSON
```json
{
  "created_at": "2026-05-15T...",
  "trajectory": [
    {
      "step": 1,
      "action": "ACTION6",
      "x": 18,
      "y": 58,
      "state": "GameState.NOT_FINISHED",
      "grid_changed": false,
      "schema_concepts_added": ["Action_ACTION6_at_18_58", "NoEffect"]
    }
  ]
}
```

### Concept
- `id`, `name`, `description`, `category`, `confidence` (0~1), `source`, `neighbors`

### Relation
- `source` (concept id), `target` (concept id), `relation_type`, `weight` (0~1), `evidence`

---

## 7. Next Priorities (Suggested)

1. **Verify action-effect feedback in prompt**: Run ARC-AGI-3 for a few steps, inspect the prompt text (via `step_*.json`), and confirm `Learned action effects:` contains `no_effect` lines after an ineffective click.
2. **Verify trajectory.json**: After a level finishes, check `trajectory.json` has the expected array with `grid_changed` and `schema_concepts_added`.
3. **Verify per-step schema persistence**: After step 1, confirm `schema/concepts.jsonl` contains action-effect concepts (e.g., `Action_ACTION6_at_...`).
4. **Verify coordinate bounds**: With `--use-llm-concepts`, inspect `concepts.jsonl` and ensure coordinates are within `[0, grid_h-1]` and `[0, grid_w-1]`.
5. **Implement baseline runs (Stage 3)**.
6. **Add concept deduplication** to `SchemaGraph` or `SchemaInitializer`.
7. **If grid still looks uniform**, increase `max_size` to 48 or 64.
8. **Consider merging duplicate `no_effect` relations** for the same coordinate instead of appending new ones every step.
