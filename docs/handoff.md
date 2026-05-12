# Handoff 文档

> 本文档供新 Claude Code session 接手时使用。包含所有设计决策、文件位置、环境状态和下一步工作。
> **最后更新：2026-05-12**

---

## 1. 项目定位（一句话）

让 LLM Agent 通过**结构化 Schema 网络**（Concept + Relation + 概率权重）辅助答题，在缺失知识或高自信错误时**主动 CLI 问人**，逐步构建可泛化的概念知识库。

---

## 2. 已完成的工作（截至 2026-05-12）

### 2.1 文档

| 文件 | 内容 |
|------|------|
| `docs/stage1_design.md` | **唯一的设计文档**。包含：Schema 数据结构、运行流程、所有模块接口定义、PromptBuilder 设计、confidence 计算方式、实现顺序 |
| `docs/datasets.md` | 所有数据集的来源、原始结构、处理流程、最终格式、使用方法 |
| `docs/related_work_notes.md` | 8 项可迁移相关工作，用户认为 4(Think-on-Graph)、5(RAPTOR)、7(Neural-Symbolic) 最相关 |
| `README.md` | 项目定位、评测问题、技术框图、Stage 1/2/3 开发阶段 |
| `socialclaw/stage1/README.md` | Stage 1 模块说明、使用示例、调试指南 |

### 2.2 数据集（全部就绪）

```
data/
  pbench/
    prepared/all.jsonl       # 5,636 条 MCQ（含图片）
  clbench/
    prepared/
      clbench.jsonl          # 1,899 条
      clbench_life.jsonl     # 405 条
  arc/
    prepared/
      arc1.jsonl             # 800 tasks
      arc2.jsonl             # 1,120 tasks
    raw/arc3/agents/         # ARC-AGI-3 官方仓库，已配置 API key
```

### 2.3 Stage 1 代码（全部实现并真实运行验证）

```
socialclaw/stage1/
  run_stage1.py            # CLI 入口（含调试参数）
  pipeline.py              # 主闭环逻辑
  types.py                 # Episode、AttemptRecord
  prompt_builder.py        # Schema 子图注入 Prompt
  stop_policy.py           # 停止策略
  evaluator.py             # 答案评估
  human_io.py              # CLI 主动提问（Rich 美化）
  logging.py               # Episode JSON 落盘
  schema/
    graph.py               # SchemaGraph、Concept、Relation + confidence 计算
    retriever.py           # BGE embedding 语义检索 + 充足度判断
    initializer.py         # Agent 自动生成 Concept+Relation / 解析人类回答
    storage.py             # JSONL + npy 持久化
  agent/
    base.py                # Agent Protocol、AgentAttempt、ReasoningTrace
    openai_compatible.py   # OpenAI-compatible Provider，解析 reasoning_trace
  dataset/
    base.py                # Problem、EvalResult
    pbench.py              # MCQ 加载器
    clbench.py             # 长上下文加载器
    arc.py                 # ARC 网格加载器
```

### 2.4 环境

- Python 3.11，venv 在 `.venv/`
- 已安装：`sentence-transformers`, `rich`, `numpy`, `httpx`, `huggingface_hub`
- Embedding 模型：`BAAI/bge-small-en-v1.5`（默认，可换）
- LLM Provider：OpenRouter（默认模型 `moonshotai/kimi-k2.6`）

---

## 3. 关键设计决策（必须遵守）

### 3.1 confidence 不是 LLM 输出，是 Schema-based 计算

- LLM 只输出：使用了哪些 concept（名称）、推理路径、最终答案
- `SchemaGraph.compute_confidence(trace)` 实现：
  - 按 **concept id** 匹配，失败则按 **concept name** 回退匹配
  - concept_geom = 所有匹配到的 concept.confidence 的几何平均
  - relation_geom = 所有匹配到的 relation.weight 的几何平均（无则 1.0）
  - overall = concept_geom * relation_geom
- 高 confidence（>0.8）+ 错误结果 → 触发向人类提问纠错

### 3.2 Schema 初始化由 Agent 自动生成

- `--auto-yes` 模式下，schema 不足时自动调用 `initializer.generate_schema(problem)`
- LLM 输出 JSON `{concepts: [...], relations: [...]}`
- relation 的 source/target 在存储前由 **name 解析为 concept id**
- source 标注：`agent_init` = LLM 生成，`human_feedback` = 人类输入

### 3.3 CLI 主动提问在 Stage 1 实现

- 缺失 concept 时：CLI 问人 → 解析为 concept + relation → 写入 schema → 重新检索
- 高自信错误时：CLI 展示推理路径 + confidence → 问人哪里错了 → 解析为 schema 更新
- 支持 `--auto-yes` 模式（测试时自动跳过提问，改用 Agent 自动生成）

### 3.4 调试入口

```bash
--reset-schema          # 清空已有 schema
--problem-id ID         # 只跑指定题目（可多次使用）
--dry-run               # 只构建 prompt，不调用 LLM
--show-prompt           # 打印 prompt 内容
--auto-yes              # 跳过人类提问，自动用 Agent 生成 schema
```

---

## 4. 真实运行验证结果（2026-05-12）

已真实调用 OpenRouter API 跑通：

1. **av_000_0**（schema 为空）：触发 auto-generate，LLM 生成 **9 concepts + 8 relations**，answer=A correct
2. **av_000_0**（schema 已存在）：Embedding 召回 9 个 concept 注入 prompt，answer=A correct，confidence=0.5
3. **av_000_1**（同视频新问题）：Embedding 召回 8 个 concept（double_yellow_lines 被过滤），answer=A correct，confidence=0.5

Schema 存储位置：`schema/`（项目根目录）
Episode 存储位置：`runs/YYYYMMDD_HHMMSS/<problem_id>/episode.json`

---

## 5. 下一步工作（三个已知问题，用户会在下一 session 解决）

### 5.1 Relation type 对齐

**现状**：LLM 在 reasoning_trace 中"发明" relation type（如 `continuously_run_along`、`of`），但 schema 中定义的是 `located_at`、`part_of`，导致 relation confidence 无法计算。

**修复方向**：
- 方案 A：Prompt 中明确要求 LLM 只能使用 schema 中已有的 relation type
- 方案 B：Agent 输出 reasoning_trace 时，relation type 做模糊匹配/映射

### 5.2 Concept confidence 初始值偏低

**现状**：所有 concept confidence 默认 0.5，relation weight 默认 0.5。

**修复方向**：
- 根据 concept 的"通用性"或"证据强度"分配不同的初始 confidence
- 或者让人类/Agent 在生成时评估并输出 confidence

### 5.3 is_sufficient 阈值粗糙

**现状**：`is_sufficient` 用 `>=3 个 concept` 或 `>=2 个 category` 作为阈值，对于简单题可能过高。

**修复方向**：
- 升级为 LLM 二分类判断（输入 problem + retrieved concepts，输出 sufficient/not）
- 或根据 problem 类型动态调整阈值

---

## 6. 常见陷阱

- **不要把 confidence 交给 LLM 输出**：用户明确反对，必须用 SchemaGraph 计算
- **不要加 tool_calls**：用户不要任何工具框架
- **ARC-AGI-3 是交互式环境**：Stage 1 只做接口抽象，不要试图本地评测
- **几何平均 vs 算术平均**：confidence 计算用几何平均，体现链式依赖的"短板效应"
- **relation 存储的是 concept id，不是 name**：graph.py 中的 relation source/target 始终是 id，pipeline 负责 name -> id 的解析

---

## 7. 快速启动

```bash
# 安装依赖
.venv/bin/pip install -e .

# 单题调试（自动模式，打印 prompt）
OPENROUTER_API_KEY=xxx .venv/bin/python -m socialclaw.stage1.run_stage1 \
  --prepared data/pbench/prepared/all.jsonl \
  --base-url https://openrouter.ai/api/v1 \
  --model moonshotai/kimi-k2.6 \
  --problem-id av_000_0 \
  --auto-yes --show-prompt --max-iters 1

# 清空 schema 从头跑
.venv/bin/python -m socialclaw.stage1.run_stage1 \
  --prepared data/pbench/prepared/all.jsonl \
  --base-url https://openrouter.ai/api/v1 \
  --model moonshotai/kimi-k2.6 \
  --reset-schema --auto-yes --max-problems 5
```

---

## 8. 参考资源

- `docs/stage1_design.md` —— 详细设计文档（模块接口、数据结构、prompt 格式、confidence 算法）
- `docs/datasets.md` —— 数据集全貌和使用方法
- `socialclaw/stage1/README.md` —— Stage 1 代码结构、使用示例、调试指南
- `README.md` —— 项目定位、Stage 划分、评测问题列表
