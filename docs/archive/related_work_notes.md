# Schema 相关技术：可迁移工作与建议

> 整理时间：2026-05-10
> 重点标注：用户认为 **4（Think-on-Graph）、5（RAPTOR）、7（Neural-Symbolic 概率图学习）** 最相关。

---

## 1. Voyager (CVPR 2023) — Skill Library + Embedding 检索

- **核心思想**：Minecraft 中 Agent 自动将可执行代码 skill 存入 library，用 embedding 检索，失败时合成新 skill。
- **可迁移模块**：SkillRetriever → SchemaRetriever 的 embedding 索引结构；skill 的「执行验证」机制可迁移为 schema 的「推理验证」机制。
- **Gap**：Voyager 的 skill 是可执行代码（有明确执行结果验证），我们的 Schema 是推理知识（验证需要人类反馈或评测器）。
- **迁移点**：把「代码执行验证」改成「答题评测验证」。

---

## 2. Expel (ICLR 2024) — 从对比经验中提取规则

- **核心思想**：从 success trajectory 和 failure trajectory 的对比中，用 LLM 归纳出自然语言规则（insight）。
- **可迁移模块**：Schema 初始化/更新时，可用 Expel 思想从「做对的题」和「做错的题」对比中，自动提取 Concept 和 Relation。
- **迁移点**：将朴素的 knowledge points 拼接升级为 Expel 式的「对比归纳生成 schema」。

---

## 3. Reflexion (NeurIPS 2023) — 语言化的自我反思

- **核心思想**：Agent 用语言描述自己的错误，将反思文本存入 memory，后续检索复用。
- **可迁移模块**：Schema 纠错环节中的「agent 自行更新」可用 Reflexion 模式——错误后让 Agent 自己写「错误分析 + 概念关系修正建议」，再解析更新 SchemaGraph。
- **迁移点**：`reflexion_generate(problem, trace, feedback) -> correction_draft -> apply_correction(graph, draft)`。

---

## 4. Think-on-Graph (ACL 2024) — LLM 在知识图谱上推理 ⭐重点

- **核心思想**：不把 KG 三元组全部塞进 prompt，而是让 LLM 在 KG 上做束搜索（beam search），每一步只探索当前最相关的邻居节点。
- **可迁移模块**：直接对应 README 框图第 3 步「在 schema 知识网络中 reasoning」。
- **迁移点**：
  - 实现 `reason_on_graph(problem, graph, start_concepts)`：
    1. 从检索到的 start concepts 出发
    2. LLM 每一步选择「探索哪个 relation / neighbor」
    3. 收集路径上的 concept 描述作为最终推理上下文
  - 相比直接塞全部 schema，更省 token，且能处理更大网络。
- **实现成本**：低。不需要额外训练，纯 prompt + graph traversal。
- **优先级**：**最高**，建议 Stage 1 就接入。

---

## 5. RAPTOR (NAACL 2024) — 层次化树状检索 ⭐重点

- **核心思想**：对文档做递归聚类和摘要，构建摘要树；检索时从高层抽象节点向底层细节节点下钻。
- **可迁移模块**：Schema 库的层次化组织。底层是具体概念（如「摩擦力方向判断」），高层是抽象概念（如「力与运动」）。
- **迁移点**：
  - Schema 膨胀到几百个 concept 时，用 RAPTOR 思想做层次化聚类。
  - 检索先命中高层 cluster，再下钻到具体 concept。
- **优先级**：中等。Schema 规模小的时候可先不做，但架构上预留层次化字段（如 `concept.parent_id`）。

---

## 6. LEMA / GEPA — 从错误中学习的优化框架

- **核心思想**：收集错误样本，通过 LLM 生成优化后的 prompt / 策略。
- **可迁移模块**：README 中提到「可以调 GEPA，比如问了问题之后怎么优化 skill」。GEPA 的「专家反馈 → 策略优化」管道可套用在「人类回答 → Schema 更新」上。
- **迁移点**：把 GEPA 的「文本策略优化」替换为「Schema 子图更新」（修改 relation weight / 新增 concept）。

---

## 7. Neural-Symbolic 概率图学习 ⭐重点

- **相关工作**：DeepProbLog, Neural Theorem Provers, Bayesian Concept Learning (Tenenbaum et al.)
- **可迁移模块**：Schema 中「概率」的更新机制可借鉴贝叶斯更新：
  - 正反馈（答对）：`P(relation) ← P(relation) + α · (1 - P(relation))`
  - 负反馈（答错）：`P(relation) ← P(relation) - β · P(relation)`
  - 高自信错误（Agent 推理路径概率高但结果错）：向人类提问，因为可能是 schema 结构本身有误（缺失节点或错误边）。
- **迁移点**：比简单「计数投票」更有理论依据，也符合 README 中「概率比较低则自行更新，概率高则问人」的直觉。
- **优先级**：高。Stage 2 做动态更新时优先引入。

---

## 8. DSPy (Stanford NLP) — 模块化的 LLM Pipeline

- **核心思想**：用编程方式定义和优化 LLM pipeline（检索 → 生成 → 评估），自动优化 prompt。
- **可迁移模块**：整个 Stage 1 pipeline 的结构化可借鉴 DSPy 的「模块 + 签名 + 优化器」思想。
- **迁移点**：把 `build_prompt → agent.answer → evaluate → update_schema` 定义成 DSPy 风格模块链，后续可用 BootstrapFewShot 等优化器自动调优 prompt。

---

## 综合建议（按优先级排序）

| 优先级 | 工作 | 落地阶段 | 说明 |
|--------|------|----------|------|
| P0 | Think-on-Graph | Stage 1 | 直接实现 schema reasoning，成本低收益高 |
| P0 | BGE Embedding | Stage 1 | 先用开源模型，不自己训 |
| P1 | Neural-Symbolic 贝叶斯更新 | Stage 2 | 动态更新时引入，有理有据 |
| P1 | Expel 对比归纳 | Stage 2 | Schema 初始化/更新时自动提取 concept |
| P2 | RAPTOR 层次化 | Stage 2/3 | Schema 规模大时再做，先预留字段 |
| P2 | Reflexion | Stage 2 | 纠错时让 agent 自己写修正建议 |
| P2 | GEPA | Stage 2/3 | 人类反馈后的策略优化 |
| P3 | DSPy | Stage 3 | 全局 pipeline 优化，偏工程化 |

---

## Schema 形式建议

采用 **Hybrid（结构化 + 自然语言）**：
- 不要纯 KG 三元组（LLM 不好理解）
- 也不要纯文本（无法做概率更新）
- 每个 Concept 既有 `description`（给 LLM 看），又有 `embedding` 和 `relation`（给系统做检索和更新）
