# Stage 2 设计与实现计划（动态更新 + 主动提问）

> 目标：在 Stage 1 静态 Schema 基础上，实现动态更新机制（巩固/纠错）和 CLI 主动提问，并跑通 ARC-AGI-3 交互式环境。
> **原则**：先跑主实验，baseline 后补；动态更新基于 reasoning_trace 中的 concept/relation 使用记录。

---

## 0. 非目标（Stage 2 不做什么）

- 不做 baseline 对比（Stage 3 补充）。
- 不替换 embedding 模型（仍用 `BAAI/bge-small-en-v1.5`）。
- 不做 Schema 跨 benchmark 自动隔离（当前用 `--reset-schema` 手动控制）。

---

## 1. 总体闭环

Stage 2 在 Stage 1 的答题循环之后，增加了**反馈驱动的 Schema 更新**和**高自信错误时的人类纠错提问**。

### 1.1 运行流程（Stage 2 新增部分）

```text
# Stage 1 流程结束后（答题 + 评估已完成）

# 4. 计算 schema-based reasoning confidence
reasoning_confidence = schema_graph.compute_confidence(attempt.reasoning_trace)

# 5. Stage 2: Schema 巩固 / 纠错
if eval.correct:
    # 正反馈：提升 used concept confidence 和 used relation weight (+0.05)
    _update_schema_from_feedback(graph, trace, correct=True)
    episode.flags.append("schema_reinforce")
else:
    # 负反馈：降低 used concept confidence 和 used relation weight (-0.05)
    _update_schema_from_feedback(graph, trace, correct=False)
    episode.flags.append("schema_correct")

# 6. 高自信错误 → CLI 向人类提问纠错
if not eval.correct and (
    reasoning_confidence > cfg.correction_conf_threshold  # 默认 0.6
    or cfg.always_ask_correction  # 调试模式：只要判错就提问
):
    correction = human_io.ask_correction(
        problem=problem,
        attempt=attempt,
        reasoning_confidence=reasoning_confidence,
        eval=eval
    )
    if correction.strip():
        corrected = schema_initializer.parse_correction(correction, problem)
        # 应用更新：add concepts / add relations / update concepts
        for c in corrected.get("add_concepts", []):
            graph.add_concept(c)
        for r in corrected.get("add_relations", []):
            graph.add_relation(r)
        for upd in corrected.get("update_concepts", []):
            graph.update_concept(upd["id"], **{k: v for k, v in upd.items() if k != "id"})
        episode.flags.append("human_correction")

# 7. 持久化
storage.save(graph, embeddings)
logger.write(episode)
```

---

## 2. Schema 动态更新机制

### 2.1 反馈信号定义

| 信号 | 条件 | 更新内容 |
|------|------|----------|
| 正反馈 | `eval.correct == True` | trace 中使用的 concept confidence +0.05，relation weight +0.05 |
| 负反馈 | `eval.correct == False` | trace 中使用的 concept confidence -0.05，relation weight -0.05 |

### 2.2 更新实现

`socialclaw/stage1/pipeline.py` 中：

```python
def _update_schema_from_feedback(graph: SchemaGraph, trace, correct: bool) -> None:
    delta = 0.05 if correct else -0.05

    for cid in trace.concepts:
        c = graph.get_concept(cid)
        if not c:
            c = graph.get_concept_by_name(cid)  # name 回退匹配（支持模糊匹配）
        if c:
            new_conf = max(0.1, min(0.95, c.confidence + delta))
            graph.update_concept(c.id, confidence=new_conf)

    for src, tgt, rel_type in trace.relations:
        # 先用模糊匹配将自由文本的 src/tgt 解析为 schema 中的 concept
        src_c = graph.get_concept(src) or graph.get_concept_by_name(src)
        tgt_c = graph.get_concept(tgt) or graph.get_concept_by_name(tgt)
        if src_c and tgt_c:
            r = graph.get_relation(src_c.id, tgt_c.id, rel_type)
            if not r:
                r = graph.find_relation(src_c.name, tgt_c.name, rel_type)
            if r:
                new_weight = max(0.1, min(0.95, r.weight + delta))
                graph.update_relation(r.source, r.target, r.relation_type, weight=new_weight)
```

**边界保护**：
- confidence / weight 上限 0.95（保留一定不确定性，避免过拟合）
- confidence / weight 下限 0.1（避免彻底遗忘，保留重新激活的可能）

### 2.3 为什么用固定步长（±0.05）

- 简单、可解释、无需超参搜索。
- 几何平均的乘性特性使得少量更新就能显著影响 overall confidence。
- 后续可扩展为基于样本量的贝叶斯更新（如使用 evidence 计数）。

---

## 3. 主动提问机制

### 3.1 缺失 concept 时的提问（Schema 初始化阶段）

当 `retriever.is_sufficient()` 返回 False 时：

**auto-yes 模式**（`--auto-yes`）：
- 自动调用 `initializer.generate_schema(problem)`，由 LLM 输出 concept + relation
- 解析后写入 schema，flag 标记为 `"agent_auto_init"`

**人工模式**（默认）：
- `initializer.describe_missing()` 生成中文提问描述
- `human_io.ask()` 在 CLI 展示问题 + 上下文 + 提示
- 人类输入自由文本，`initializer.parse_human_answer()` 解析为结构化 Concept/Relation
- flag 标记为 `"human_init_concepts"`

### 3.2 高自信错误时的提问（Schema 纠错阶段）

触发条件：
```python
not eval.correct and (
    reasoning_confidence > cfg.correction_conf_threshold  # 默认 0.6
    or cfg.always_ask_correction  # 调试开关
)
```

`human_io.ask_correction()` 展示内容：
- 题目 ID + 评估结果（如 `llm_judge=wrong`）
- **原题 prompt**（前 800 字）
- **LLM 原始回答**（前 600 字）
- **标准答案**（前 600 字）
- **使用的 concepts**
- **推理路径（relations）**
- **explanation**

人类输入纠正建议后，`initializer.parse_correction()` 解析为：
- `add_concepts`: 新增 concept
- `add_relations`: 新增 relation
- `update_concepts`: 修改现有 concept 的 description

---

## 4. 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `similarity_threshold` | 0.75 | Embedding 检索阈值，低于此值视为 missing |
| `correction_conf_threshold` | 0.6 | 高自信错误的 confidence 阈值 |
| `always_ask_correction` | False | 调试模式：只要判错就向人提问 |
| `auto_yes` | False | 是否自动跳过人类提问（用 LLM 生成 schema） |

---

## 5. ARC-AGI-3 交互式环境

> **状态：接口已实现并验证**

ARC-AGI-3 与 PBench/CL-bench 不同：它不是一次性的问答，而是**多轮 action/observation 循环**。Agent 需要：
1. 观察当前网格状态（observation）
2. 推断转换规则
3. 执行动作（如修改网格中的像素）
4. 接收环境反馈

### 5.1 已实现组件

| 文件 | 职责 |
|------|------|
| `dataset/arc_agi3.py` | 封装 `arc_agi.Arcade`，管理多 level 游戏循环，提供 `grid_to_text` 和 `extract_objects` |
| `schema/arc_agi3_parser.py` | Grid -> 连通区域提取 -> Schema Concept（Object + 颜色/位置/面积）+ Spatial Relation（above/below/left_of/right_of）+ Transformation Rule（action 触发 object 变化） |
| `run_arc_agi3.py` | 多轮交互入口：每关循环 observation → extract objects → build prompt → LLM action → step → 通关后强化 rules / 失败后修正 rules |

### 5.2 Schema 表示设计

**Object Concept**：从 64×64 grid 中提取的连通区域
- `name`: `BlueBlob_0`, `RedBlob_1` ...
- `description`: 颜色、位置(top_left/bottom_right)、面积、质心
- `category`: `level_N`

**Spatial Relation**：基于质心的相对位置
- `above` / `below` / `left_of` / `right_of`

**Transformation Rule**：action 执行前后 object 的变化
- `source`: 上一帧的 object id
- `target`: 当前帧的 object id
- `relation_type`: `transformed_by_ACTION1`
- `weight`: 置信度（通关后 +0.05，失败后 -0.05）

### 5.3 运行命令

```bash
.venv/bin/python -m socialclaw.stage1.run_arc_agi3 \
  --game-id sk48-d8078629 \
  --base-url https://openrouter.ai/api/v1 \
  --model moonshotai/kimi-k2.6 \
  --reset-schema --max-steps 200
```

---

## 6. 已知问题与缓解

### 6.1 Schema 跨 benchmark 污染

**现状**：`schema/` 是全局目录，不同 benchmark 的概念混在一起。
**缓解**：测试时用 `--reset-schema` 清空后再跑；运行结束后检查 schema categories 是否与当前 benchmark 一致。
**验证**：`--reset-schema` 已验证有效，新 schema 只包含当前测试题目的概念，无旧领域残留。
**长期**：可按 `--schema-dir benchmark_name_schema/` 分别存储。

### 6.2 概念重复

**现状**：Auto-generate 可能产生重复概念（如 `Sighting Card` 和 `Sighting Cards`）。
**计划**：后续增加去重/遗忘机制，当前不阻塞主实验。

### 6.3 Neighbors Feature（新增）

`Concept` dataclass 新增 `neighbors: List[str]` 字段，`SchemaGraph` 提供 `get_neighbors(cid)` 方法，从 relation 中动态计算邻居 concept id 列表。`storage.py` 在 save 时自动将 `neighbors` 写入 `concepts.jsonl`，加载时回填到 Concept 对象。这便于人类直接阅读 schema 文件，也支持后续基于邻居的子图扩展。

---

## 7. 调试入口

```bash
# 跑 CL-bench（清空 schema，跑 2 题，auto-yes 模式）
.venv/bin/python -m socialclaw.stage1.run_stage1 \
  --prepared data/clbench/prepared/clbench.jsonl \
  --base-url https://openrouter.ai/api/v1 \
  --model moonshotai/kimi-k2.6 \
  --reset-schema --auto-yes --max-problems 2

# 调试提问模式（不带 auto-yes，只要判错就提问）
.venv/bin/python -m socialclaw.stage1.run_stage1 \
  --prepared data/clbench/prepared/clbench.jsonl \
  --base-url https://openrouter.ai/api/v1 \
  --model moonshotai/kimi-k2.6 \
  --reset-schema --always-ask-correction --max-problems 1

# 按 context 迭代跑 CL-bench（同一 context 的 tasks 共享 schema）
.venv/bin/python -m socialclaw.stage1.run_stage1 \
  --prepared data/clbench/prepared/clbench.jsonl \
  --base-url https://openrouter.ai/api/v1 \
  --model moonshotai/kimi-k2.6 \
  --context-id 71a2cd92-6978-4ea8-a37f-d99728129d89 \
  --auto-yes

# 跑 ARC-AGI-3（多轮交互，schema 记录 object 变换规则）
.venv/bin/python -m socialclaw.stage1.run_arc_agi3 \
  --game-id sk48-d8078629 \
  --base-url https://openrouter.ai/api/v1 \
  --model moonshotai/kimi-k2.6 \
  --reset-schema --max-steps 200
```
