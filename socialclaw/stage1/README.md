# Stage 1: Schema-Centric Reasoning Pipeline

Stage 1 的核心目标是建立 **Schema 图基础设施**，完成**静态 Schema 辅助答题**的端到端验证。

---

## 1. 代码结构

```text
stage1/
  run_stage1.py            # CLI 入口脚本
  pipeline.py              # 主闭环逻辑
  config_schema.py         # （预留）Schema 配置
  types.py                 # Episode、AttemptRecord 等核心数据结构
  prompt_builder.py        # 将 Schema 子图注入 Prompt
  stop_policy.py           # 停止策略（max_iters / confidence / max_tokens）
  evaluator.py             # 答案评估（MCQ / 长文本 / ARC）
  human_io.py              # CLI 主动提问（Rich 美化）
  logging.py               # Episode JSON 落盘
  schema/
    __init__.py
    graph.py               # SchemaGraph、Concept、Relation
    retriever.py           # Embedding 语义检索 + 充足度判断
    initializer.py         # Agent 自动生成 Concept / 解析人类回答
    storage.py             # JSONL + npy 持久化
  agent/
    __init__.py
    base.py                # Agent Protocol、AgentAttempt、ReasoningTrace
    openai_compatible.py   # OpenAI-compatible LLM Provider
  dataset/
    __init__.py
    base.py                # Problem、EvalResult 基类
    pbench.py              # MCQ 数据集加载器
    clbench.py             # 长上下文阅读理解加载器
    arc.py                 # ARC 网格加载器（Stage 1 仅静态接口）
```

---

## 2. 模块职责

### 2.1 Schema 核心 (`schema/`)

| 文件 | 职责 |
|------|------|
| `graph.py` | 定义 `Concept`、`Relation`、`SchemaGraph`。`SchemaGraph.compute_confidence()` 基于几何平均计算 reasoning confidence。 |
| `storage.py` | `SchemaStorage` 负责将 `SchemaGraph` 持久化到 `concepts.jsonl` + `relations.jsonl`，embedding 矩阵存为 `concept_embeddings.npy`。 |
| `retriever.py` | `SchemaRetriever` 使用 SentenceTransformer（BGE）做 embedding 语义检索（cosine similarity），并判断检索到的 concept 是否"充足"。 |
| `initializer.py` | `SchemaInitializer` 让 LLM Agent 自动生成 concept；将人类自由文本回答解析为结构化 `Concept` / `Relation`；将纠错建议解析为 schema 更新操作。 |

### 2.2 Agent (`agent/`)

| 文件 | 职责 |
|------|------|
| `base.py` | 定义 `Agent` Protocol：`answer(prompt, meta) -> AgentAttempt`。`AgentAttempt` 包含 `answer_text` + `reasoning_trace`（概念列表 + 推理路径）+ `usage`。 |
| `openai_compatible.py` | `OpenAICompatibleAgent` 调用任意 OpenAI-compatible API（OpenRouter / SiliconFlow / 本地 vLLM 等）。自动解析 LLM 返回中的 `[推理过程]` 和 `[最终答案]` 块，提取 `reasoning_trace`。支持图片多模态输入。 |

### 2.3 数据集 (`dataset/`)

| 文件 | 职责 |
|------|------|
| `base.py` | `Problem` 基类：`id`、`prompt`、`problem_type`、`meta`。`EvalResult`：correct / pred / gold / details。 |
| `pbench.py` | 加载 MCQ 格式的 prepared JSONL。 |
| `clbench.py` | 加载长上下文阅读理解的 prepared JSONL。 |
| `arc.py` | 加载 ARC 网格的 prepared JSONL（Stage 1 仅使用静态 prompt，交互环境留给 Stage 2）。 |

### 2.4 主链路 (`pipeline.py`)

```text
for problem in dataset:
    1. Schema 检索 + 充足度判断
       └─ 不足 -> CLI 向人类提问 -> 解析为 Concept -> 写入 Schema -> 重新检索

    2. Agent 在 Schema 辅助下答题
       └─ 构建 prompt（渐进式披露）-> LLM 生成 answer + reasoning_trace

    3. 评估 -> EvalResult

    4. 计算 schema-based reasoning confidence（基于几何平均）

    5. 高 confidence 但错误 -> CLI 向人类提问纠错 -> 更新 Schema

    6. 保存 Schema + 落盘 Episode
```

---

## 3. 使用示例

### 3.1 基本运行

```bash
.venv/bin/python -m socialclaw.stage1.run_stage1 \
  --prepared data/pbench/prepared/all.jsonl \
  --base-url https://openrouter.ai/api/v1 \
  --api-key $OPENROUTER_API_KEY \
  --model moonshotai/kimi-k2.6 \
  --max-problems 10
```

### 3.2 指定 Embedding 模型

```bash
.venv/bin/python -m socialclaw.stage1.run_stage1 \
  --prepared data/pbench/prepared/all.jsonl \
  --base-url https://openrouter.ai/api/v1 \
  --model moonshotai/kimi-k2.6 \
  --embed-model BAAI/bge-small-en-v1.5
```

### 3.3 自动模式（跳过人类提问，用于批量测试）

```bash
.venv/bin/python -m socialclaw.stage1.run_stage1 \
  --prepared data/pbench/prepared/all.jsonl \
  --base-url ... --model ... \
  --auto-yes \
  --max-problems 50
```

---

## 4. 调试指南

### 4.1 清空已有 Schema（从头开始）

```bash
--reset-schema
```

这会删除 `--schema-dir` 下的所有已有概念和关系。

### 4.2 只调试某条数据

```bash
--problem-id av_000_0 --problem-id av_000_1
```

可多次使用，只运行指定 ID 的题目。

### 4.3 只构建 Prompt，不调用 LLM（Dry Run）

```bash
--dry-run
```

用于快速验证 Prompt 构建逻辑，不消耗 API token。会生成 Episode，但 answer 标记为 `[DRY_RUN]`。

### 4.4 打印 Prompt 内容

```bash
--show-prompt
```

每次发送给 LLM 前，在终端打印完整 prompt（可与正常模式或 `--dry-run` 联用）。

### 4.5 组合调试示例

```bash
# 清空 schema，只看第 3 题的 prompt 长什么样，不调用 LLM
.venv/bin/python -m socialclaw.stage1.run_stage1 \
  --prepared data/pbench/prepared/all.jsonl \
  --base-url https://openrouter.ai/api/v1 \
  --model moonshotai/kimi-k2.6 \
  --reset-schema \
  --problem-id av_000_2 \
  --dry-run \
  --show-prompt
```

```bash
# 在已有 schema 上跑 5 题，自动跳过提问，打印 prompt
.venv/bin/python -m socialclaw.stage1.run_stage1 \
  --prepared data/pbench/prepared/all.jsonl \
  --base-url https://openrouter.ai/api/v1 \
  --model moonshotai/kimi-k2.6 \
  --auto-yes \
  --show-prompt \
  --max-problems 5
```

---

## 5. Schema 存储格式

默认存储在 `data/schema/`：

| 文件 | 格式 | 内容 |
|------|------|------|
| `concepts.jsonl` | JSONL | 每行一个 `Concept` |
| `relations.jsonl` | JSONL | 每行一个 `Relation` |
| `concept_embeddings.npy` | npy | 概念 embedding 矩阵 |
| `concept_ids.json` | JSON | embedding 矩阵对应的 concept ID 列表 |

---

## 6. Episode 输出格式

每次运行在 `runs/YYYYmmdd_HHMMSS/<problem_id>/episode.json` 生成一个 Episode：

```json
{
  "created_at": "2026-05-11T07:27:10Z",
  "episode": {
    "problem": { "id": "...", "prompt": "...", "problem_type": "mcq", "meta": {...} },
    "attempts": [
      {
        "input_prompt": "...",
        "answer_text": "A",
        "reasoning_trace": { "concepts": [...], "relations": [...], "explanation": "..." },
        "usage": { "input_tokens": 100, "output_tokens": 50, "total_tokens": 150 },
        "raw": { "response": {...} }
      }
    ],
    "evals": [ { "correct": false, "pred": "A", "gold": "B", "details": "..." } ],
    "reasoning_trace": {...},
    "reasoning_confidence": 0.0,
    "flags": ["human_init_concepts", "human_correction"],
    "stop_reason": "max_iters"
  }
}
```

---

## 7. 设计原则

- **Schema 为中心**：所有 reasoning 围绕 `Concept` / `Relation` 展开，confidence 由系统基于 Schema 权重计算，而非 LLM 自评。
- **渐进式披露**：`attempt_index=0` 只注入 concept 名称 + 简短描述；`attempt_index>=1` 注入完整 description + relation 细节。
- **CLI 主动提问**：Schema 不足时问人类补充概念；高 confidence 但错误时问人类纠错。Stage 1 实现提问，Stage 2 再做自动化更新。
- **可插拔 LLM**：通过 `Agent` Protocol 统一封装，token 用量从各厂商 API 的 usage 字段读取并归一化。
