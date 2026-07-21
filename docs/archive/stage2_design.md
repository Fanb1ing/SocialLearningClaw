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
    reasoning_confidence > cfg.correction_conf_threshold  # 默认 -1.0（调试时必定触发）
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

`socialclaw/pipeline.py` 中：

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
    reasoning_confidence > cfg.correction_conf_threshold  # 默认 -1.0（ARC-AGI-3），CL-bench 侧默认 0.6
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
| `correction_conf_threshold` | -1.0 | 高自信错误的 confidence 阈值（默认 -1.0 确保调试时必定触发） |
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
| `dataset/arc_agi3.py` | 封装 `arc_agi.Arcade`；`reset()` 返回初始 observation；`get_available_actions(obs)` 从 observation 读取当前可用 actions（而非全量 action space） |
| `schema/arc_agi3_parser.py` | Grid -> 连通区域 BFS 提取 -> Schema Concept（Object + 颜色/位置/面积）+ Spatial Relation（above/below/left_of/right_of，限 nearest 3 邻居）+ Transformation Rule（action 触发 object 变化）+ **Action-Effect Relation**（`no_effect` / `affected`）。新增 `compute_grid_diff`（逐像素对比 pre/post grid）和 `build_action_effect_concepts_and_relations`（生成 action concept 与 effect relation） |
| `run_arc_agi3.py` | 多轮交互入口：每关循环 observation → build prompt（基于已有 schema + action-effect 反馈）→ LLM action → step → **提取 post-action grid → 对比 pre/post grid → 生成 action-effect concepts/relations → 立即持久化 schema** → 通关后强化 rules / 失败后修正 rules；每关输出**标准 Episode** + **Trajectory JSON**（`runs/<run_id>/<game_id>_L<level>/trajectory.json`） |

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

**Action-Effect Relation**：记录单次 action 对 grid 的实际影响（防循环核心）
- **Action Concept**：`id=action_l{N}_s{step}_ACTION6`, `name=Action_ACTION6_at_x_y`, `category=action`, `source=action_effect`
- **`no_effect`**：`source=action_concept`, `target=no_effect_concept`, `relation_type=no_effect`, `weight=0.9`
  - 当 `compute_grid_diff(pre_grid, post_grid)` 发现无像素变化时生成。
  - 直接注入 prompt 的 `Learned action effects:` 区块，例如：`Action_ACTION6_at_18_58 had no effect on grid`
- **`affected`**：`source=action_concept`, `target=post_object_concept`, `relation_type=affected`, `weight=0.7`
  - 当 grid 发生变化时，将 action concept 与受影响的 post-action object 关联。
- **Per-step 持久化**：action-effect concept/relation 生成后立即 `graph.add_relation()` + `storage.save()`，确保下一步 prompt 能读取到最新反馈。

### 5.3 运行命令

```bash
# 标准运行（清空 schema，每关最多 200 步）
.venv/bin/python -m socialclaw.run_arc_agi3 \
  --game-id sk48-d8078629 \
  --base-url https://openrouter.ai/api/v1 \
  --model qwen/qwen2.5-vl-72b-instruct \
  --reset-schema --max-steps 200

# Auto-yes 模式（自动跳过人类提问，由 LLM 生成 schema）
.venv/bin/python -m socialclaw.run_arc_agi3 \
  --game-id sk48-d8078629 \
  --base-url https://openrouter.ai/api/v1 \
  --model qwen/qwen2.5-vl-72b-instruct \
  --reset-schema --max-steps 200 --auto-yes

# 调试模式（只要判错就向人类提问纠错，无视 confidence 阈值）
.venv/bin/python -m socialclaw.run_arc_agi3 \
  --game-id sk48-d8078629 \
  --base-url https://openrouter.ai/api/v1 \
  --model qwen/qwen2.5-vl-72b-instruct \
  --reset-schema --max-steps 200 --always-ask-correction
```

---

## 6. 已知问题与缓解

### 6.1 Schema 跨 benchmark 污染 ✅ 已解决

**旧问题**：`schema/` 曾是全局目录，不同 benchmark 的概念混在一起。
**解决方案**：Schema 已改为 **run-scoped storage**——每个 `run/<run_id>/schema/` 目录独立保存该次运行的 schema，与 episode 日志天然隔离。不同 benchmark / context / 游戏之间不再共享 schema。
**验证**：`--reset-schema` 已验证有效，新 schema 只包含当前测试题目的概念，无旧领域残留。

### 6.2 ARC-AGI-3 Prompt 瘦身与兼容性修复

**问题 1**：每帧 grid 提取 44 个 objects，全部注入 prompt 导致 schema 部分长达 7411 字符，远超原始 prompt（~1400 字符）。
**解决**：
1. `_build_arc_prompt` 限制只注入 **confidence top 10 concepts** + **前 10 条 transformation rules**（`transformed_by_*` 类型）
2. **调整 loop 顺序**：LLM 先基于已有 schema 做决策，执行 action 后，才把当前 frame 的 objects/rules 写进 schema。这样 step 0 的 prompt 完全是原始输入（无 schema 注入），后续 step 才逐步注入历史学到的知识
**效果**：prompt 从 8822 字符降至 ~3000 字符

**问题 2**：ARC-AGI-3 prompt 要求 LLM 输出严格 JSON，但 `_parse_reasoning_trace` 期望 `[Reasoning Process]` / `[Final Answer]` 格式，导致 trace 永远为空，confidence 始终为 0。
**解决**：修改 prompt，在 JSON 的 `reasoning` 字段中嵌入 `concepts_used` / `reasoning_path` / `explanation`；新增 `_parse_agent_action` 从 JSON 中提取结构化 trace 并覆盖 `attempt.reasoning_trace`。
**效果**：reasoning_trace 正确非空，schema confidence 正常计算，可触发高自信纠错。

**问题 3**：API 偶发中断（`httpx.RemoteProtocolError`）或返回空响应，导致进程 crash。
**解决**：`agent.answer()`、`retriever.retrieve()`、`initializer.generate_schema()` 等关键 LLM 调用处添加 try-except，fallback 到默认 action / 视为 schema 充足。
**效果**：进程不再因网络波动 crash，游戏可持续运行。

**问题 4**：`grid_to_text(max_size=16)` 对 64×64 网格做中心裁剪，sk48 的中心 16×16 区域恰好全为 Yellow(4)，导致 LLM 看到的 grid 全是同一种颜色，无法识别物体。
**解决**：将 `max_size` 从 16 提高到 32（函数默认值 + `run_arc_agi3.py` 调用处）。
**效果**：LLM 现在能看到 32×32 中心区域，包含更多颜色变化。

**问题 5**：`diff_objects_to_rules` 对前后帧物体做笛卡尔积（prev_concepts × curr_concepts），生成无意义规则，如 `BlueBlob_0 -> transformed_by_ACTION6 -> obj_l0_s1_0/1/2`。
**解决**：改为基于颜色匹配 + 质心距离最近邻匹配，每个前帧物体最多映射到一个后帧物体。
**效果**：规则数量从 O(n²) 降至 O(n)，且每条规则都有明确的单一目标。

**问题 6：Schema 仅在关卡结束时持久化，LLM 无法感知中间步骤的 action 效果**
**解决**：
1. 每一步执行 action 后，立即调用 `storage.save(graph, embeddings)` 持久化 schema。
2. 同时生成 `trajectory.json`，便于人类离线审查 agent 行为。
**效果**：LLM 在 step N+1 的 prompt 中即可看到 step N 的 `no_effect` 反馈，避免重复无效点击。

**问题 7：LLM vision 概念提取返回 pixel 坐标，导致 action 越界**
**解决**：
1. 将静态 prompt 改为 template `_CONCEPT_EXTRACTION_PROMPT_TEMPLATE`，动态填入 `{h}`, `{w}`, `{h_max}`, `{w_max}`。
2. 明确要求返回 grid-cell 索引（0~h-1, 0~w-1），而非 image pixel 坐标。
3. `llm_extract_grid_concepts()` 中对解析出的 `top_left` / `bottom_right` 做边界校验，越界则丢弃并 warning。
**效果**：concept 坐标被约束在 grid 范围内，action 模型不再收到 (320, 960) 这类越界坐标。

### 6.3 概念重复

**现状**：Auto-generate 可能产生重复概念（如 `Sighting Card` 和 `Sighting Cards`）。
**计划**：后续增加去重/遗忘机制，当前不阻塞主实验。

### 6.4 Neighbors Feature（新增）

`Concept` dataclass 新增 `neighbors: List[str]` 字段，`SchemaGraph` 提供 `get_neighbors(cid)` 方法，从 relation 中动态计算邻居 concept id 列表。`storage.py` 在 save 时自动将 `neighbors` 写入 `concepts.jsonl`，加载时回填到 Concept 对象。这便于人类直接阅读 schema 文件，也支持后续基于邻居的子图扩展。

### 6.5 已识别待修复问题（2026-05-17）

**问题 8：人类纠错/补充的回答未正确写入 schema**
- 现象：`human_io.ask_correction()` / `ask()` 返回的文本经 `parse_correction()` / `parse_human_answer()` 解析后，concept 未被持久化到 schema，或写入后被后续步骤覆盖/丢失。
- 根因待查：可能涉及 `_add_concepts_with_embeddings` 重复调用、id 冲突、或 `storage.save()` 时机问题。
- 状态：**待修复**，当前作为已知 issue 记录。

**问题 9：人类回答后流程不应再询问 schema 充足度，应直接回到解题**
- 现象：在缺失 concept 的 `ask()` 流程或纠错 `ask_correction()` 流程结束后，代码会重新执行 `retriever.is_sufficient()` 或进入下一轮 sufficiency check，导致再次向人类提问，而不是立刻基于更新后的 schema 继续答题/继续下一 step。
- 期望：人类一旦给出回答，立即将回答应用到 schema，跳过 sufficiency 二次确认，直接回到主答题循环。
- 状态：**待修复**。

**问题 10：每步重复抽取 grid concepts 造成冗余**
- 现象：`run_arc_agi3.py` 每一步都会调用 `extract_grid_objects()` / `llm_extract_grid_concepts()` 生成新的 post-action concepts，即使 grid 没有任何变化（`grid_changed=False`）。这导致 schema 中堆积大量完全相同的 object concepts（如 `BlueBlob_0` 在 step1/step2/step3 各有一条记录）。
- 期望：若 `compute_grid_diff` 发现 grid 未变化，应复用上一帧的 object concepts，不生成新的 concept；仅当 grid 变化或首次观察时才抽取。
- 状态：**已修复**（2026-05-26，`arc_runner.py` 中当 `grid_changed=False` 且 `prev_concepts` 非空时，直接复用 `prev_objects`/`prev_concepts`，不再重新提取）。

**问题 11：Embedding 检索召回效果差**
- 现象：`_retrieve_relevant_concepts` 使用 `BAAI/bge-small-en-v1.5` 对 grid 对象摘要（如 "ARC grid with 44 objects, colours Blue(10)..."）与 schema concept 做 embedding 相似度匹配，实际召回结果与人类直观预期差距大。人类反馈的 concept（如 "Stick"）常常无法被召回，导致 prompt 中缺失关键信息。
- 根因：通用 embedding 模型对网格/视觉对象的语义理解不足；query 构造过于简单。
- 计划：Stage 3 考虑（1）用视觉-语言嵌入模型替代纯文本 BGE；（2）引入负采样让 action-effect 反馈更具区分度；（3）或改用基于属性/规则的硬匹配替代纯 embedding 召回。
- 状态：**已识别为未来工作，Stage 3 评估后决定方案**。

**问题 12：关卡重试时 step 文件被覆盖**
- 现象：timeout/失败后触发人类纠错并重试关卡时，`write_step()` 使用固定路径 `step_{step:03d}.json`，导致同一 level 目录下的 step 文件被新尝试覆盖，丢失前一次失败的完整轨迹。
- 期望：每次 level 尝试应有独立的子目录或前缀（如 `attempt_1/step_001.json`），保留所有历史尝试记录。
- 状态：**待修复**。

---

## 7. 调试入口

**CL-bench 多轮对话修复（2026-05-19）**：
- 原始数据为多轮 messages 格式，预处理脚本 `scripts/download_clbench.py` 原先只保留最后一条 user/assistant 消息，中间轮次全丢。
- 修复后所有非最后轮次的消息以 `[role]: content` 格式拼接为 context，并加入 `msg_count` 用于正确排序。
- Pipeline 中 `group_by_context` 模式按 `msg_count` 排序（替代字母序），确保 schema 按正确对话顺序累积。
- 每个 context 的 schema 保存到 `run_dir/<context_id>/schema/` 独立目录，避免覆盖。
- 修复效果：两个 context 测试从 0/6 → 4/6 correct。

```bash
# 跑 CL-bench（清空 schema，跑 2 题，auto-yes 模式）
.venv/bin/python -m socialclaw.run_clbench \
  --prepared data/clbench/prepared/clbench.jsonl \
  --base-url https://openrouter.ai/api/v1 \
  --model qwen/qwen2.5-vl-72b-instruct \
  --reset-schema --auto-yes --max-problems 2

# 调试提问模式（不带 auto-yes，只要判错就提问）
.venv/bin/python -m socialclaw.run_clbench \
  --prepared data/clbench/prepared/clbench.jsonl \
  --base-url https://openrouter.ai/api/v1 \
  --model qwen/qwen2.5-vl-72b-instruct \
  --reset-schema --always-ask-correction --max-problems 1

# 按 context 迭代跑 CL-bench（同一 context 的 tasks 共享 schema，按 msg_count 正确排序）
.venv/bin/python -m socialclaw.run_clbench \
  --prepared data/clbench/prepared/clbench.jsonl \
  --base-url https://openrouter.ai/api/v1 \
  --model qwen/qwen2.5-vl-72b-instruct \
  --context-id 71a2cd92-6978-4ea8-a37f-d99728129d89 \
  --group-by-context --auto-yes

# 跑 ARC-AGI-3（多轮交互，schema 记录 object 变换规则）
.venv/bin/python -m socialclaw.run_arc_agi3 \
  --game-id sk48-d8078629 \
  --base-url https://openrouter.ai/api/v1 \
  --model qwen/qwen2.5-vl-72b-instruct \
  --reset-schema --max-steps 200
```
