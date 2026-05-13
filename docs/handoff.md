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
| `docs/related_work_notes.md` | 8 项可迁移相关工作 |
| `README.md` | 项目定位、评测问题、技术框图、Stage 1/2/3 开发阶段（已更新到最新实现状态） |
| `socialclaw/stage1/README.md` | Stage 1 模块说明、使用示例、调试指南 |
| **本文件** | `docs/handoff.md` |

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

### 2.3 Stage 1 + Stage 2 代码（全部实现并真实运行验证）

```
socialclaw/stage1/
  run_stage1.py            # CLI 入口（含调试参数）
  pipeline.py              # 主闭环逻辑（含 Stage 2 动态更新）
  types.py                 # Episode、AttemptRecord
  prompt_builder.py        # Schema 子图注入 Prompt
  stop_policy.py           # 停止策略
  evaluator.py             # 答案评估（含 CL-bench LLM-as-judge）
  human_io.py              # CLI 主动提问（Rich 美化）
  logging.py               # Episode JSON 落盘
  schema/
    graph.py               # SchemaGraph、Concept、Relation + confidence 计算 + relation type 模糊匹配
    retriever.py           # LLM 提取 concept → 逐个 embedding 匹配 + 充足度判断
    initializer.py         # Agent 自动生成 Concept+Relation / 解析人类回答
    storage.py             # JSONL + npy 持久化
  agent/
    base.py                # Agent Protocol、AgentAttempt、ReasoningTrace
    openai_compatible.py   # OpenAI-compatible Provider，解析 reasoning_trace
  dataset/
    base.py                # Problem（含 retrieval_query）、EvalResult
    pbench.py              # MCQ 加载器
    clbench.py             # 长上下文加载器（context + question 拼接）
    arc.py                 # ARC 网格加载器
```

### 2.4 环境

- Python 3.11，venv 在 `.venv/`
- 已安装：`sentence-transformers`, `rich`, `numpy`, `httpx`, `huggingface_hub`
- Embedding 模型：`BAAI/bge-small-en-v1.5`（默认，可换）
- LLM Provider：OpenRouter（默认模型 `moonshotai/kimi-k2.6`）
- API key 保存在项目根目录 `.env` 文件中

---

## 3. 关键设计决策（必须遵守）

### 3.1 Embedding 检索：先提取 concept，再逐个匹配

**不是**直接把 question 文本编码为 embedding 去匹配 schema。

正确流程：
1. LLM 读取题目，提取所需 concept 名称列表
2. 对每个提取出的 concept，单独编码其名称，与 schema 中所有 concept embedding 匹配（取 top-1）
3. 相似度 ≥ threshold 则放入 `matched`，否则放入 `missing`
4. `is_sufficient` = `missing` 为空 且 `matched` 非空

这样做的好处：
- 检索更精准（避免长文本稀释语义）
- `missing` 列表可直接用于主动提问
- 充足度判断天然是 LLM-based，无硬编码阈值

### 3.2 confidence 不是 LLM 输出，是 Schema-based 计算

- LLM 只输出：使用了哪些 concept（名称）、推理路径、最终答案
- `SchemaGraph.compute_confidence(trace)` 实现：
  - 按 **concept id** 匹配，失败则按 **concept name** 回退匹配
  - concept_geom = 所有匹配到的 concept.confidence 的几何平均
  - relation_geom = 所有匹配到的 relation.weight 的几何平均（无则 1.0）
  - overall = concept_geom * relation_geom
- 高 confidence（>0.8）+ 错误结果 → 触发向人类提问纠错

### 3.3 Relation type 模糊匹配

LLM 在 reasoning_trace 中会"发明" relation type（如 `continuously_run_along`、`of`）。
`SchemaGraph.get_relation` / `find_relation` 支持三层回退：
1. **精确匹配**（大小写不敏感）
2. **预定义别名映射**：如 `continuously_run_along` → `located_at`，`of` → `part_of`
3. **字符串相似度**：`difflib.SequenceMatcher` ratio ≥ 0.75，或子串互相包含

### 3.4 Schema 初始化由 Agent 自动生成

- `--auto-yes` 模式下，schema 不足时自动调用 `initializer.generate_schema(problem)`
- LLM 输出 JSON `{concepts: [...], relations: [...]}`
- relation 的 source/target 在存储前由 **name 解析为 concept id**
- source 标注：`agent_init` = LLM 生成，`human_feedback` = 人类输入

### 3.5 Stage 2 动态更新（已实现）

每道题评估后自动执行：
- 正反馈（答对）：相关 concept confidence +0.05，relation weight +0.05（上限 0.95）
- 负反馈（答错）：相关 concept confidence -0.05，relation weight -0.05（下限 0.1）
- Episode flags：`schema_reinforce` / `schema_correct`

### 3.6 CLI 主动提问在 Stage 1 已实现

- 缺失 concept 时：CLI 问人 → 解析为 concept + relation → 写入 schema → 重新检索
- 高自信错误时：CLI 展示推理路径 + confidence → 问人哪里错了 → 解析为 schema 更新
- 支持 `--auto-yes` 模式（测试时自动跳过提问，改用 Agent 自动生成）

### 3.7 CL-bench LLM-as-judge

- CL-bench 为开放式长文本问答，exact match 不现实
- `evaluator.evaluate()` 在 `problem_type == "long_context"` 且 agent 非空时，调用 LLM 判断回答质量
- prompt：题目 + 标准答案 + 模型回答 → LLM 输出 correct/wrong

### 3.8 调试入口

```bash
--reset-schema          # 清空已有 schema
--problem-id ID         # 只跑指定题目（可多次使用）
--dry-run               # 只构建 prompt，不调用 LLM（注意：retrieve 仍会调 LLM 提取 concept）
--show-prompt           # 打印 prompt 内容
--auto-yes              # 跳过人类提问，自动用 Agent 生成 schema
```

---

## 4. 真实运行验证结果（2026-05-12）

已真实调用 OpenRouter API 跑通：

1. **CL-bench 第 1 题**（schema 为空）：LLM 提取 concept → 无匹配 → auto-generate 生成概念 → 重新匹配成功 → Agent 答题 → schema_correct（answer 为空判错，confidence 降低）
2. **CL-bench 第 2 题**（schema 已存在）：LLM 提取 concept → embedding 匹配已有 schema → sufficient → Agent 答题 → schema_correct（LLM-as-judge 判 wrong）
3. **av_000_0 / av_000_1**（PBench）：历史验证通过

Schema 存储位置：`schema/`（项目根目录）
Episode 存储位置：`runs/YYYYMMDD_HHMMSS/<problem_id>/episode.json`

---

## 5. 遗留问题与下一步工作

### 5.1 Schema 跨 benchmark 污染（测试阶段可控）

**现状**：`schema/` 是全局目录，pbench 的驾驶概念和 CL-bench 的游戏概念混在一起。

**缓解**：测试时用 `--reset-schema` 清空后再跑。

**长期**：用户说"测试阶段可以把 schema 清空来简单操作"，不着急隔离。

### 5.2 概念重复

**现状**：Auto-generate 产生重复概念（如 `Sighting Card` 和 `Sighting Cards`、`Dusk Phase` 出现两次）。

**计划**：用户说后面会有去重/遗忘机制，当前不阻塞。

### 5.3 ARC-AGI-3 交互式环境

**现状**：Stage 1 已完成接口抽象，Stage 2 计划跑通多轮 action/observation 循环。

**计划**：待实现。

### 5.4 CL-bench 部分 answer 为空

**现状**：数据集中部分题目 answer 字段为空，此时 evaluator 返回 correct=False。

**计划**：数据集本身问题，不影响主链路。

---

## 6. 常见陷阱

- **不要把 confidence 交给 LLM 输出**：用户明确反对，必须用 SchemaGraph 计算
- **不要加 tool_calls**：用户不要任何工具框架
- **ARC-AGI-3 是交互式环境**：Stage 1 只做接口抽象，不要试图本地评测
- **几何平均 vs 算术平均**：confidence 计算用几何平均，体现链式依赖的"短板效应"
- **relation 存储的是 concept id，不是 name**：graph.py 中的 relation source/target 始终是 id，pipeline 负责 name -> id 的解析
- **dry-run 不会跳过 retrieve**：因为 retrieve 中的 concept 提取是核心逻辑，即使 `--dry-run` 也会调 LLM 提取 concept（只是答题环节跳过）

---

## 7. 快速启动

```bash
# 安装依赖
.venv/bin/pip install -e .

# 单题调试（自动模式，打印 prompt）
.venv/bin/python -m socialclaw.stage1.run_stage1 \
  --prepared data/pbench/prepared/all.jsonl \
  --base-url https://openrouter.ai/api/v1 \
  --model moonshotai/kimi-k2.6 \
  --problem-id av_000_0 \
  --auto-yes --show-prompt --max-iters 1

# 跑 CL-bench（清空 schema，跑 2 题）
.venv/bin/python -m socialclaw.stage1.run_stage1 \
  --prepared data/clbench/prepared/clbench.jsonl \
  --base-url https://openrouter.ai/api/v1 \
  --model moonshotai/kimi-k2.6 \
  --reset-schema --auto-yes --max-problems 2

# 查看 episode
ls runs/
cat runs/YYYYMMDD_HHMMSS/<problem_id>/episode.json
```

---

## 8. 参考资源

- `docs/stage1_design.md` —— 详细设计文档（模块接口、数据结构、prompt 格式、confidence 算法）
- `docs/datasets.md` —— 数据集全貌和使用方法
- `socialclaw/stage1/README.md` —— Stage 1 代码结构、使用示例、调试指南
- `README.md` —— 项目定位、Stage 划分、评测问题列表
