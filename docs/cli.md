# CLI 使用手册

> 本文档覆盖项目中所有命令行入口的完整参数说明。
> **项目根目录**：`/data5/fanbingbing/SocialLearningClaw`
> **Python 环境**：`.venv/bin/python`

---

## 目录

1. [通用说明](#通用说明)
2. [sc-clbench — CL-bench / PBench / ARC 推理](#sc-clbench)
3. [sc-arc-agi3 — ARC-AGI-3 交互式环境](#sc-arc-agi3)
4. [输出目录结构](#输出目录结构)
5. [典型场景示例](#典型场景示例)

---

## 通用说明

### API Key 配置

在项目根目录的 `.env` 文件中配置（优先级低于命令行 `--api-key`）：

```
OPENROUTER_API_KEY=sk-or-...
ARC_AGI_API_KEY=...          # ARC-AGI-3 专用
```

API Key 查找顺序：`--api-key` > `OPENROUTER_API_KEY` > `OPENAI_API_KEY` > `API_KEY`（依次从环境变量中查找）。

### 运行方式

```bash
# 通过 module 运行（推荐）
.venv/bin/python -m socialclaw.run_clbench  [参数...]
.venv/bin/python -m socialclaw.run_arc_agi3 [参数...]

# 或者通过 entry_point（pip install -e . 后可用）
sc-clbench   [参数...]
sc-arc-agi3  [参数...]
```

### Runs 输出路径

所有实验结果保存在：

```
runs/{benchmark}/{model_sanitized}/{YYYYMMDD_HHMMSS}/
```

- `benchmark`：由数据集自动推断（`clbench` / `pbench` / `arc` / `arc_agi3`）
- `model_sanitized`：模型名中的 `/` 替换为 `--`（如 `qwen--qwen2.5-vl-72b-instruct`）
- 时间戳：**东八区（CST, UTC+8）**
- `--runs-dir` 控制根目录，默认为 `runs`

---

## sc-clbench

用于 CL-bench（长上下文阅读理解）、PBench（MCQ 选择题）、ARC-1/2 数据集。

```
.venv/bin/python -m socialclaw.run_clbench --prepared <路径> --model <模型名> [选项...]
```

### 必填参数

| 参数 | 说明 |
|------|------|
| `--prepared <path>` | 准备好的 `.jsonl` 数据集路径 |
| `--model <name>` | LLM 模型名（如 `moonshotai/kimi-k2.6`） |

### 常用可选参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--base-url <url>` | `https://openrouter.ai/api/v1` | OpenAI 兼容 API 地址 |
| `--api-key <key>` | （从环境变量读取） | API Key |
| `--embed-model <name>` | `BAAI/bge-small-en-v1.5` | Embedding 模型（本地加载） |
| `--max-problems <n>` | `5` | 最多处理多少道题 |
| `--max-iters <n>` | `2` | 每道题最多尝试次数 |
| `--top-k <n>` | `5` | Schema 检索返回 Top-K 概念数 |
| `--threshold <f>` | `0.75` | Embedding 相似度阈值（0~1） |
| `--auto-yes` | `False` | 跳过所有人机交互，由 LLM 自动生成 Schema |
| `--runs-dir <dir>` | `runs` | 结果输出根目录 |

### Schema 控制参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--schema-dir <dir>` | `schema` | Schema 保存目录。**默认**：每次 run 在 `run_dir/schema/` 独立保存（隔离）。**显式指定**：跨多次 run 复用同一 Schema。 |
| `--reset-schema` | `False` | 运行前清空 Schema（配合 `--schema-dir` 显式路径使用） |
| `--group-by-context` | CL-bench 自动开启 | 按 `context_id` 分组，每组独立 Schema，组内问题共享 Schema |
| `--context-id <id>` | 无 | 只运行指定 `context_id` 的题目（Schema 自动重置） |

### 调试参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--no-retrieval` | `False` | 跳过 Embedding 检索，把所有 Schema 概念都注入 Prompt（调试用） |
| `--dry-run` | `False` | 只构建 Prompt，不调用 LLM |
| `--show-prompt` | `False` | 打印每次发送给 LLM 的完整 Prompt |
| `--problem-id <id>` | 无 | 只跑指定 ID 的题，可多次使用（如 `--problem-id id1 --problem-id id2`） |
| `--always-ask-correction` | `False` | 每次答错都向人类询问纠错（调试 Human-in-the-loop 流程） |

---

## sc-arc-agi3

用于 ARC-AGI-3 交互式环境（多关卡、逐步 action、视觉 grid）。

```
.venv/bin/python -m socialclaw.run_arc_agi3 --game-id <游戏ID> [选项...]
```

### 必填参数

| 参数 | 说明 |
|------|------|
| `--game-id <id>` | ARC-AGI-3 游戏 ID（如 `sk48-d8078629`） |

### 常用可选参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--base-url <url>` | `https://openrouter.ai/api/v1` | OpenAI 兼容 API 地址 |
| `--api-key <key>` | （从环境变量读取） | API Key |
| `--model <name>` | `qwen/qwen2.5-vl-72b-instruct` | LLM 模型名（须支持视觉输入） |
| `--embed-model <name>` | `BAAI/bge-small-en-v1.5` | Embedding 模型 |
| `--max-steps <n>` | `200` | 每关最大 step 数 |
| `--max-retries <n>` | `3` | 每关 GAME_OVER 后最大重试次数 |
| `--auto-yes` | `False` | 跳过所有人机交互，由 LLM 自动生成 Schema |
| `--max-tokens <n>` | `8192` | 每次 LLM 调用的最大输出 Token 数 |
| `--runs-dir <dir>` | `runs` | 结果输出根目录 |

### Schema 控制参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--schema-dir <dir>` | `schema_arc_agi3` | Schema 保存目录。**默认**：每次 run 在 `run_dir/schema/` 独立保存。**显式指定**：复用已有 Schema（可跨 run 积累知识）。 |
| `--reset-schema` | `False` | 运行前清空 Schema |
| `--use-llm-concepts` | `False` | 用视觉 LLM 提取 Grid 概念（质量更高，耗时更多）。默认用 BFS 连通区域算法。 |

### 调试参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--no-retrieval` | `False` | 跳过 Embedding 检索，注入所有 Schema 概念（调试用） |
| `--render` | `False` | 在终端渲染 Grid（须环境支持） |
| `--always-ask-correction` | `False` | 每次关卡失败都向人类询问纠错 |
| `--correction-conf-threshold <f>` | `-1.0` | 触发人类纠错的 reasoning confidence 阈值（默认 -1.0 = 始终触发） |

---

## 输出目录结构

### CL-bench / PBench / ARC

```
runs/
└── clbench/                                   # benchmark 名
    └── moonshotai--kimi-k2.6/                 # 模型名（/ 替换为 --）
        └── 20260526_153000/                   # 时间戳（CST）
            ├── cmd.txt                        # 本次运行的完整命令行
            ├── {problem_id}/                  # 非 group-by-context 模式（PBench/ARC）
            │   └── episode.json
            └── {ctx_short}/                   # CL-bench group-by-context 模式（context_id 前 8 位）
                ├── schema/                    # 该 context 的 schema 文件
                │   ├── concepts.jsonl
                │   ├── relations.jsonl
                │   ├── concept_embeddings.npy
                │   └── concept_ids.json
                └── {msg_count:02d}_{pid_short}/  # 按 msg_count 排序的题目（problem_id 前 8 位）
                    └── episode.json
```

例如：
```
20260526_153000/
  cmd.txt
  71a2cd92/
    schema/
    02_2bbe2e03/episode.json      ← msg_count=2 的题
    04_72365b51/episode.json      ← msg_count=4 的题
```

**episode.json 格式**：

```json
{
  "created_at": "2026-05-26T...",
  "episode": {
    "problem": { "id": "...", "prompt": "...", "problem_type": "long_context", "meta": {...} },
    "attempts": [
      {
        "input_prompt": "...",
        "answer_text": "...",
        "reasoning_trace": { "concepts": [...], "relations": [...], "explanation": "..." },
        "usage": { "input_tokens": 0, "output_tokens": 0, "total_tokens": 0 }
      }
    ],
    "evals": [{ "correct": true, "pred": "...", "gold": "...", "details": "" }],
    "reasoning_trace": { ... },
    "reasoning_confidence": 0.72,
    "flags": ["schema_reinforce"],
    "stop_reason": "max_iters",
    "model": "moonshotai/kimi-k2.6"
  }
}
```

### ARC-AGI-3

```
runs/
└── arc_agi3/
    └── qwen--qwen2.5-vl-72b-instruct/
        └── 20260526_153000/
            ├── cmd.txt                        # 本次运行的完整命令行
            ├── schema/
            │   ├── concepts.jsonl
            │   ├── relations.jsonl
            │   ├── concept_embeddings.npy
            │   └── concept_ids.json
            └── {game_id}_L{level}/            # 每关一个目录
                ├── episode.json               # 关卡 Episode
                ├── trajectory.json            # 关卡 step 轨迹摘要
                ├── step_001.json              # 每步详细记录
                ├── step_002.json
                └── step_NNN.png               # 每步 Grid 图片
```

**trajectory.json 格式**：

```json
{
  "created_at": "2026-05-26T...",
  "trajectory": [
    {
      "step": 1,
      "action": "ACTION6",
      "x": 18, "y": 58,
      "state": "GameState.NOT_FINISHED",
      "grid_changed": false,
      "schema_concepts_added": ["Action_ACTION6_at_18_58"]
    }
  ]
}
```

---

## 典型场景示例

### CL-bench 完整运行（5道题，自动生成 Schema）

```bash
.venv/bin/python -m socialclaw.run_clbench \
  --prepared data/clbench/prepared/clbench.jsonl \
  --model moonshotai/kimi-k2.6 \
  --max-problems 5 \
  --auto-yes
```

### CL-bench 单 context 运行（Schema 跨题积累）

```bash
.venv/bin/python -m socialclaw.run_clbench \
  --prepared data/clbench/prepared/clbench.jsonl \
  --model moonshotai/kimi-k2.6 \
  --context-id 02d45927-b242-462d-ade7-5fb5915f337e \
  --auto-yes
```

### CL-bench No-Retrieval 对比实验（复用已有 Schema）

```bash
.venv/bin/python -m socialclaw.run_clbench \
  --prepared data/clbench/prepared/clbench.jsonl \
  --model moonshotai/kimi-k2.6 \
  --schema-dir runs/clbench/moonshotai--kimi-k2.6/20260521_061404/02d45927-b242-462d-ade7-5fb5915f337e/schema \
  --no-retrieval \
  --auto-yes
```

### ARC-AGI-3 标准运行（自动生成 Schema，200步限制）

```bash
.venv/bin/python -m socialclaw.run_arc_agi3 \
  --game-id sk48-d8078629 \
  --model qwen/qwen2.5-vl-72b-instruct \
  --max-steps 200 \
  --auto-yes
```

### ARC-AGI-3 人机交互运行（每关失败时向人类提问）

```bash
.venv/bin/python -m socialclaw.run_arc_agi3 \
  --game-id sk48-d8078629 \
  --model qwen/qwen2.5-vl-72b-instruct \
  --max-steps 200
# 注意：须在本地终端运行，不能通过 Claude Code Bash 工具运行（stdin 会被污染）
```

### ARC-AGI-3 复用已有 Schema（no-retrieval 对比实验）

```bash
.venv/bin/python -m socialclaw.run_arc_agi3 \
  --game-id sk48-d8078629 \
  --model qwen/qwen2.5-vl-72b-instruct \
  --schema-dir runs_old/runs_arc_agi3/20260519_061915/schema \
  --no-retrieval \
  --auto-yes
```

### ARC-AGI-3 使用 LLM Vision 提取概念

```bash
.venv/bin/python -m socialclaw.run_arc_agi3 \
  --game-id sk48-d8078629 \
  --model qwen/qwen2.5-vl-72b-instruct \
  --use-llm-concepts \
  --max-steps 200 \
  --auto-yes
```

---

## Schema 文件格式

Schema 存储在 `schema/` 子目录中，由 4 个文件组成：

| 文件 | 格式 | 说明 |
|------|------|------|
| `concepts.jsonl` | JSONL（每行一个 Concept） | 概念：id、name、description、category、confidence、source、neighbors |
| `relations.jsonl` | JSONL（每行一个 Relation） | 关系：source（concept id）、target（concept id）、relation_type、weight、evidence |
| `concept_embeddings.npy` | NumPy `.npy` | Embedding 矩阵，行数 = concept 数，列数 = embedding 维度 |
| `concept_ids.json` | JSON 数组 | Concept ID 列表，与 `concept_embeddings.npy` 行对齐 |

**Concept 字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | `str` | 唯一标识，格式 `concept_{uuid8}` |
| `name` | `str` | 概念名（LLM 直接引用此名） |
| `description` | `str` | 自然语言描述 |
| `category` | `str` | 类别标签（`general` / `level_N` / `action` 等） |
| `confidence` | `float` | 0~1，系统对该概念的确信度 |
| `source` | `str` | `agent_init` / `human_feedback` / `action_effect` 等 |
| `neighbors` | `List[str]` | 邻居 concept id 列表（动态计算） |

**Relation 字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `source` | `str` | concept id |
| `target` | `str` | concept id |
| `relation_type` | `str` | `prerequisite` / `causes` / `part_of` / `located_at` / `analogous` / `related` / `transformed_by_<ACTION>` / `no_effect` / `affected` |
| `weight` | `float` | 0~1，关系强度 |
| `evidence` | `List[dict]` | 来源题目记录 |
