# Episode Reading Guide

Each episode records the complete run of a single problem (or one level of ARC-AGI-3). File path:

```
runs/<run_id>/<problem_id>/episode.json
```

---

## Top-level structure

```json
{
  "created_at": "2026-05-13T07:04:22Z",
  "episode": { ... }
}
```

| Field | Meaning |
|-------|---------|
| `created_at` | Timestamp of this run (UTC) |
| `episode` | Core content, see details below |

---

## Episode core fields

### 1. `problem` — Problem information

```json
{
  "id": "Unique problem ID",
  "prompt": "Full problem text sent to LLM",
  "problem_type": "mcq | long_context | arc_grid",
  "retrieval_query": "Condensed text used for embedding retrieval (may be long)",
  "meta": { ... }
}
```

The `meta` field varies by `problem_type`:

**For `long_context` (CL-bench):**
```json
{
  "answer_key": "",
  "context": "Full context text",
  "question": "Question text",
  "rubrics": "Evaluation rubrics",
  "task_id": "Task ID",
  "context_id": "Context ID",
  "context_category": "Category",
  "sub_category": "Sub-category"
}
```

**For `mcq` (PBench):**
```json
{
  "choices": ["Option A", "Option B", ...],
  "answer_key": "A",
  "dataset": "Dataset name",
  "source_file": "Source file path",
  "pbench_id": "PBench ID",
  "qa_index": 0,
  "category": "Category",
  "subcategory": "Sub-category",
  "image": "Image filename (if any)"
}
```

**For `arc_grid` (ARC-AGI-3):**
```json
{
  "game_id": "Game ID",
  "level": 1,
  "step": 0
}
```

---

### 2. `attempts` — List of answering attempts

Each problem may be attempted multiple times (e.g., first attempt wrong, then retry with a more detailed prompt).

```json
[
  {
    "input_prompt": "Full prompt actually sent to LLM (including injected schema)",
    "answer_text": "Raw LLM response text",
    "reasoning_trace": {
      "concepts": ["Sales Enablement", "Coaching Mode"],
      "relations": [["A", "B", "prerequisite"]],
      "explanation": "Brief reasoning explanation"
    },
    "usage": {
      "input_tokens": 1234,
      "output_tokens": 567,
      "total_tokens": 1801
    },
    "raw": {
      "response": { ... },
      "meta": { ... },
      "messages": [ ... ]
    }
  }
]
```

**`raw` field breakdown:**
- `response`: Full LLM API response dict (from OpenAI-compatible API)
- `meta`: Problem metadata passed to the agent
- `messages`: List of message objects sent to the LLM API

---

### 3. `evals` — Evaluation results (one-to-one with attempts)

```json
[
  {
    "correct": false,
    "pred": "Model prediction (e.g. 'A' or summary text)",
    "gold": "Ground truth answer",
    "details": "Error details, e.g. 'llm_judge=wrong' or 'pred=B, gold=A'"
  }
]
```

- `correct=true`: Answered correctly
- `correct=false` + `details="llm_judge=wrong"`: CL-bench judged wrong by LLM-as-judge
- `correct=false` + `details="pred=B, gold=A"`: MCQ option mismatch
- `gold` may be empty string for `long_context` tasks (answer judged by LLM-as-judge)

---

### 4. `reasoning_trace` — Reasoning path from the last attempt

Same content as `attempts[-1].reasoning_trace`, provided for quick access.

---

### 5. `reasoning_confidence` — Schema-based reasoning confidence

**Not output by LLM; computed by the system based on schema:**

- Takes `confidence` of all concepts and `weight` of all relations in `reasoning_trace`
- Geometric mean: `concept_geom * relation_geom`
- Range: 0.0 ~ 1.0
- **Usage**: High confidence (e.g. >0.8) but wrong answer indicates schema error, triggers human correction

---

### 6. `flags` — Event markers (list of strings)

| Flag | Meaning |
|------|---------|
| `agent_auto_init` | `--auto-yes` mode: LLM auto-generated missing schema concepts |
| `human_init_concepts` | Human supplemented missing concepts |
| `schema_reinforce` | Correct answer: related concept/relation confidence/weight +0.05 |
| `schema_correct` | Wrong answer: related concept/relation confidence/weight -0.05 |
| `human_correction` | High-confidence error: human corrected schema |

---

### 7. `stop_reason` — Stop reason

| Value | Meaning |
|-------|---------|
| `max_iters` | Reached max attempt count (default 2) |
| `max_tokens` | Reached max token limit |
| `null` | Normal finish (no stop condition triggered) |

---

## Quick diagnosis workflow

1. **Check `evals[-1].correct`**: Did it answer correctly?
2. **Check `flags`**: Did schema update or human intervention trigger?
3. **Check `reasoning_confidence`**:
   - High (>0.6) but wrong -> schema may be wrong, check concept/relation definitions
   - Low (<0.3) -> LLM barely used schema concepts
4. **Check `attempts[-1].reasoning_trace.concepts`**:
   - Empty `[]` -> LLM declared no concepts used (confidence shows 0.0)
   - Concept names are long sentences -> prompt constraint insufficient, needs optimization
5. **Check `evals[-1].details`**: Understand specific error type

---

## Schema and Episode relationship

- **Old mode**: Global shared `schema/` directory, all episodes shared one schema
- **New mode (current)**: Each `run/` directory has its own `schema/` subdirectory (`runs/<run_id>/schema/`), saved alongside episodes as logs
- Different contexts / games are naturally isolated; multiple tasks within the same context share the same run's schema
