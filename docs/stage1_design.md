# Stage 1 设计与实现计划（Schema 为中心）

> 目标：建立 Schema 图基础设施，完成静态 Schema 辅助答题的端到端验证。
> **原则**：先跑主实验，baseline 后补；Schema 初始化由 Agent 自动生成；CLI 主动提问在 Stage 1 实现。

---

## 0. 非目标（Stage 1 不做什么）

- 不做 Schema 动态更新（巩固/纠留给 Stage 2）。
- 不跑 baseline 对比（Stage 3 补充）。
- 不自己训练 embedding 模型（先用开源 BGE）。
- ARC-AGI-3 交互式环境 Stage 1 只做接口抽象，Stage 2 跑通。

---

## 1. 总体闭环（与 README 新框图对齐）

### 1.1 核心数据结构

```text
Concept:
  id: str
  name: str
  description: str          # 给 LLM 看的自然语言描述
  category: str             # 类别标签（物理/数学/逻辑...）
  confidence: float         # 0~1，系统对这条 concept 的确信度
  source: str               # "agent_init" | "human_feedback" | ...
  created_at: str
  neighbors: List[str]      # 邻居 concept id 列表（动态计算，存储时写入 JSONL）

Relation:
  source: str               # concept id
  target: str               # concept id
  relation_type: str        # "prerequisite" | "causes" | "analogous" | ...
  weight: float             # 0~1，概率化权重
  evidence: List[dict]      # [{problem_id, correct: bool}, ...]

RetrieveResult:
  matched: List[Concept]    # 从 schema 中成功匹配到的概念
  missing: List[str]        # LLM 提取出但未在 schema 中找到匹配的概念名称

SchemaGraph:
  concepts: Dict[str, Concept]
  relations: List[Relation]
  # 方法：add/remove/update, subgraph_extract, probability_propagate
```

存储：
- `schema/concepts.jsonl` —— 每行一个 Concept（含 `neighbors` 字段，embedding 单独存 `.npy`）
- `schema/relations.jsonl` —— 每行一个 Relation
- `schema/concept_embeddings.npy` + `concept_ids.json` —— 对齐的 embedding 矩阵

### 1.2 运行流程（伪代码）

```text
for problem in dataset:
  episode = init(problem)

  # 1. Schema 检索 + 充足度判断
  # 1a. LLM 提取问题所需 concept 名称列表
  # 1b. 逐个与 schema concept 做 embedding 相似度匹配
  result = schema_retriever.retrieve(problem, top_k=5, threshold=0.75)
  # sufficient = missing 为空 且 matched 非空（LLM-based，无硬编码阈值）
  sufficient = schema_retriever.is_sufficient(result)

  if not sufficient:
    # CLI 向人类提问，描述缺失的 concept
    missing_desc = schema_initializer.describe_missing(problem, result.matched, missing=result.missing)
    human_answer = human_io.ask(
      question=missing_desc["question"],
      context=problem.prompt,
      hint=missing_desc["hint"]
    )
    # 将人类回答解析为 concept + relation，写入 schema
    new_concepts = schema_initializer.parse_human_answer(human_answer, problem)
    schema_graph.add_concepts(new_concepts)
    episode.flags.append("human_init_concepts")
    # 重新检索（现在 schema 已补充）
    result = schema_retriever.retrieve(problem, top_k=5, threshold=0.75)

  concepts = result.matched

  # 2. Agent 在 Schema 辅助下答题
  subgraph = schema_graph.subgraph(concepts)
  prompt = prompt_builder.build(problem, subgraph)
  attempt = agent.answer(prompt)
  episode.attempts.append(attempt)

  # 3. 评估（CL-bench 使用 LLM-as-judge）
  eval = evaluator.evaluate(problem, attempt, agent=agent)
  episode.evals.append(eval)

  # 4. 计算 schema-based reasoning confidence
  reasoning_confidence = schema_graph.compute_confidence(attempt.reasoning_trace)

  # 5. Stage 2: Schema 巩固 / 纠错
  if eval.correct:
    # 正反馈：提升 used concept confidence 和 used relation weight (+0.05)
    schema_graph.reinforce(attempt.reasoning_trace)
    episode.flags.append("schema_reinforce")
  else:
    # 负反馈：降低 used concept confidence 和 used relation weight (-0.05)
    schema_graph.correct(attempt.reasoning_trace)
    episode.flags.append("schema_correct")

  # 6. 若错误且 reasoning_confidence 很高（或开启调试模式）→ CLI 向人类提问纠错
  if not eval.correct and (reasoning_confidence > 0.6 or cfg.always_ask_correction):
    correction = human_io.ask_correction(
      problem=problem,
      attempt=attempt,
      reasoning_confidence=reasoning_confidence,
      eval=eval
    )
    corrected = schema_initializer.parse_correction(correction, problem)
    schema_graph.update(corrected)
    episode.flags.append("human_correction")

  episode.reasoning_trace = attempt.reasoning_trace
  episode.reasoning_confidence = reasoning_confidence

  logger.write(episode)
```

### 1.3 Episode 数据结构

每道题一次运行形成一个 `Episode`（可序列化 JSON）：

- `problem`: `{id, split, domain, prompt, choices?, answer_key?}`
- `attempts[]`: 每次对 Agent 的一次调用
  - `input_prompt`: 最终给 Agent 的 prompt
  - `answer`: 结构化答案
  - `reasoning_trace`: 使用了哪些 concept / relation
  - `usage`: `{input_tokens, output_tokens, total_tokens}`
  - `raw`: 原始返回
- `evals[]`: 与 attempts 对齐
  - `correct`: bool
  - `pred`: 预测（规范化）
  - `gold`: 标准答案（规范化）
  - `details`: 误差说明
- `reasoning_trace`: Agent 声明使用的 concept/relation 路径
- `reasoning_confidence`: 系统基于 schema 计算的推理置信度（见 3.4 节）
- `flags[]`: 标记，如 `"human_init_concepts"`, `"human_correction"`
- `stop_reason`: `max_iters|max_tokens|...`

所有 episode 都会落盘：`runs/YYYYmmdd_HHMMSS/<problem_id>/episode.json`。

---

## 2. 文件结构

```text
socialclaw/
  stage1/
    run_stage1.py            # CLI 入口（含 --reset-schema / --problem-id / --context-id / --dry-run / --auto-yes 等调试参数）
    run_arc_agi3.py          # ARC-AGI-3 交互式运行入口（多轮 action/observation + schema rule 学习）
    types.py                 # Problem 基类、Episode、EvalResult
    pipeline.py
    prompt_builder.py
    stop_policy.py
    evaluator.py
    human_io.py              # CLI 主动提问
    schema/
      __init__.py
      graph.py               # SchemaGraph、Concept、Relation
      retriever.py           # Embedding 检索 + 充足度判断
      initializer.py         # Agent 自动生成初始化 concept
      storage.py             # JSONL + npy 读写
      arc_agi3_parser.py     # ARC-AGI-3 Grid -> Object -> Concept/Relation 解析器
    agent/
      __init__.py
      base.py                # Agent 基类
      openai_compatible.py   # 可插拔 LLM provider
    dataset/                 # 数据集加载器
      base.py
      pbench.py              # MCQ
      clbench.py             # 长上下文阅读理解
      arc.py                 # ARC 网格（静态接口）
      arc_agi3.py            # ARC-AGI-3 交互式环境包装器
    logging.py
```

---

## 3. 模块接口定义

### 3.1 SchemaGraph

`socialclaw/stage1/schema/graph.py`

```python
@dataclass
class Concept:
    id: str
    name: str
    description: str
    category: str = "general"
    confidence: float = 0.5
    source: str = "agent_init"
    created_at: str = ""
    neighbors: List[str] = field(default_factory=list)

@dataclass
class Relation:
    source: str
    target: str
    relation_type: str = "related"
    weight: float = 0.5
    evidence: List[dict] = field(default_factory=list)

class SchemaGraph:
    def __init__(self, concepts_path, relations_path):
        ...
    def add_concept(self, c: Concept) -> None: ...
    def add_relation(self, r: Relation) -> None: ...
    def get_concept(self, cid: str) -> Optional[Concept]: ...
    def get_concept_by_name(self, name: str) -> Optional[Concept]:
        """支持精确匹配、大小写不敏感、子串包含、difflib 模糊匹配。"""
    def get_neighbors(self, cid: str) -> List[str]:
        """从 relations 中动态计算邻居 concept id 列表。"""
    def subgraph(self, concept_ids: List[str], depth: int = 1) -> "SchemaGraph": ...
    def compute_confidence(self, trace: ReasoningTrace) -> float:
        """
        根据 reasoning_trace 中使用的 concept 和 relation，
        计算 schema-based reasoning confidence。
        见 3.4 节详细说明。
        """
        ...
    def save(self) -> None: ...
```

### 3.2 SchemaRetriever

`socialclaw/stage1/schema/retriever.py`

```python
@dataclass
class RetrieveResult:
    matched: List[Concept]   # 成功匹配到的 schema 概念
    missing: List[str]       # 未匹配到的概念名称（需补充）

class SchemaRetriever:
    def __init__(self, graph: SchemaGraph, embedder, agent: Optional[Agent] = None):
        ...

    def retrieve(self, problem: Problem, top_k=5, threshold=0.75) -> RetrieveResult:
        """
        两步检索：
        1. 调用 LLM 提取问题所需 concept 名称列表。
        2. 对每个提取出的 concept，计算其 name 的 embedding，
           与 schema 中所有 concept embedding 做相似度匹配，取 top-1。
        3. 相似度 >= threshold 则放入 matched，否则放入 missing。
        """
        ...

    def is_sufficient(self, result: RetrieveResult) -> bool:
        """
        充足度判断：missing 为空 且 matched 非空 即 sufficient。
        无需硬编码阈值，因为 missing 列表本身由 LLM 提取产生。
        """
        ...
```

**关键变化**：不再直接把 `problem.prompt` 编码为 query embedding，而是先让 LLM 提取所需 concept，再逐个匹配。这样检索结果更精准，且 missing 列表可直接用于主动提问。

### 3.3 SchemaInitializer

`socialclaw/stage1/schema/initializer.py`

```python
class SchemaInitializer:
    def __init__(self, agent: Agent):
        ...
    def generate_concepts(self, problem: Problem) -> List[Concept]:
        """
        让 Agent 读题，判断回答该问题需要哪些 concept，
        并输出结构化 concept + relation。
        """
        ...
    def describe_missing(self, problem: Problem, concepts: List[Concept], missing: List[str] = None) -> dict:
        """生成向人类提问的描述（问题 + 提示）。若传入 missing，会在问题中明确指出缺少哪些概念。"""
        ...
    def parse_human_answer(self, answer: str, problem: Problem) -> List[Concept]:
        """将人类自由文本回答解析为结构化 Concept/Relation。"""
        ...
    def parse_correction(self, correction: str, problem: Problem) -> dict:
        """将人类纠错建议解析为 schema 更新操作。"""
        ...
```

### 3.4 Agent 接口与 Reasoning Confidence

`socialclaw/stage1/agent/base.py`

```python
@dataclass
class ReasoningTrace:
    concepts: List[str]       # 使用的 concept id 列表
    relations: List[tuple]    # [(source, target, relation_type), ...]
    explanation: str          # Agent 对推理过程的自然语言解释

@dataclass
class AgentAttempt:
    answer_text: str
    reasoning_trace: ReasoningTrace   # Agent 声明使用了哪些 concept / relation
    usage: Usage
    raw: Dict[str, Any]

class Agent:
    def answer(self, *, prompt: str, meta: Dict[str, Any]) -> AgentAttempt:
        ...
```

**关键：reasoning_confidence 不是 LLM 输出，是系统基于 Schema 计算**

`SchemaGraph.compute_confidence(trace: ReasoningTrace) -> float` 的实现：

```python
def compute_confidence(self, trace: ReasoningTrace) -> float:
    """
    Schema-based reasoning confidence 计算逻辑：

    1. 收集 trace 中使用的所有 concept 的 confidence
    2. 收集 trace 中使用的所有 relation 的 weight
    3. 综合计算：
       - concept_conf = 所有 concept.confidence 的几何平均
       - relation_conf = 所有 relation.weight 的几何平均
       - overall_confidence = concept_conf * relation_conf

    几何平均比算术平均更能体现"链式依赖"：
    只要路径上有一个低置信度节点/边，整体 confidence 就会显著下降。

    边界情况：
    - 如果 trace 为空（Agent 没有声明使用任何 concept），返回 0.0
    - 如果 trace 中引用了不存在的 concept/relation，忽略并打 warning
    """
    import math

    concept_scores = []
    for cid in trace.concepts:
        c = self.get_concept(cid)
        if c:
            concept_scores.append(c.confidence)

    relation_scores = []
    for src, tgt, rel_type in trace.relations:
        r = self.get_relation(src, tgt, rel_type)
        if r:
            relation_scores.append(r.weight)

    if not concept_scores:
        return 0.0

    concept_geom = math.prod(concept_scores) ** (1 / len(concept_scores))
    relation_geom = math.prod(relation_scores) ** (1 / len(relation_scores)) if relation_scores else 1.0

    return concept_geom * relation_geom
```

**Relation Type 模糊匹配**：
- LLM 在 reasoning_trace 中可能"发明" relation type（如 `continuously_run_along`、`of`）。
- `SchemaGraph.get_relation` 和 `find_relation` 支持三层回退匹配：
  1. **精确匹配**（大小写不敏感）
  2. **预定义别名映射**：如 `continuously_run_along` → `located_at`，`of` → `part_of`
  3. **字符串相似度**：`difflib.SequenceMatcher` ratio ≥ 0.75，或子串互相包含
- 这确保了即使 LLM 使用了 schema 中不存在的 relation type 变体，confidence 计算仍能尽可能匹配到已有 relation。

**为什么这样设计**：
- confidence 反映的是"当前 schema 对这条推理路径有多确信"
- 高 confidence + 错误结果 = schema 结构本身可能有问题（concept 定义错误或 relation 方向/类型错误）
- 这恰好触发向人类提问纠错的条件

#### 可插拔 LLM Provider

Stage 1 主路径为**自研 Agent + 可插拔 LLM Provider（Multi-Provider）**：
- 统一通过 `Agent` 抽象封装不同厂商 API（OpenAI-compatible / Anthropic / 其它）
- **tokens**：从各厂商 API 的 usage 字段读取（适配层做归一化，保证 `input/output/total`）
- Agent 只负责生成 `answer_text` 和 `reasoning_trace`（概念使用声明），不负责输出 confidence

### 3.5 HumanIO（CLI 主动提问）

`socialclaw/stage1/human_io.py`

```python
class HumanIO:
    def ask(self, question: str, context: str, hint: str) -> str:
        """
        终端交互式提问。
        示例输出：
          ────────────────────────────────
          [主动提问] 当前问题缺少必要的概念：
          问题：{question}
          上下文：{context[:200]}...
          提示：{hint}
          请输入你的回答（多行，Ctrl+D 结束）：
          ────────────────────────────────
        """
        ...

    def ask_correction(self, problem: Problem, attempt: AgentAttempt,
                       reasoning_confidence: float, eval: EvalResult) -> str:
        """
        高自信错误时向人类提问，请求纠正 schema。
        展示 Agent 的推理路径和 schema confidence，询问哪里错了。
        """
        ...
```

**关键设计**：
- 用 `rich` 库美化 CLI 输出（问题高亮、推理路径树状展示）。
- 支持 `--auto-yes` 模式（测试时自动跳过提问，记录为 skipped）。
- 人类回答的文本交给 `schema_initializer.parse_human_answer()` 解析为结构化 Concept/Relation。

### 3.6 PromptBuilder：LLM 如何利用 Schema 信息

`socialclaw/stage1/prompt_builder.py`

#### 输入
- `problem: Problem`
- `subgraph: SchemaGraph`（检索到的 concept + 它们之间的 relation 构成的子图）
- `attempt_index: int`（第几次尝试，用于渐进式披露）

#### 输出
- 一个完整 prompt 字符串，包含：系统指令 + Schema 知识网络 + 题目 + 输出格式要求

#### Schema 文本化格式（给 LLM 看）

**方式 A：扁平列表（简单直接，Stage 1 默认）**

```text
【可用概念网络】

概念 1：摩擦力方向判断
  描述：摩擦力的方向总是与相对运动（或相对运动趋势）的方向相反。
  相关概念：→ 相对运动（prerequisite）、→ 受力分析（causes）

概念 2：受力分析
  描述：分析物体受到的所有外力，包括重力、支持力、摩擦力等。
  相关概念：→ 牛顿第二定律（causes）

...

请基于上述概念网络推理本题，并在回答中说明：
1. 你使用了哪些概念（列出概念名称或编号）
2. 概念之间的推理路径（如：概念A → 关系 → 概念B）
```

**方式 B：Think-on-Graph 逐步探索（进阶，Stage 2 接入）**

不一次性塞入全部子图，而是让 LLM 逐步选择探索方向：

```text
当前已知概念：摩擦力方向判断
可选下一步：
  A. 探索 "相对运动"（prerequisite）
  B. 探索 "受力分析"（causes）
  C. 直接基于现有信息作答

请选择：A

[系统返回 "相对运动" 的描述]
...
```

Stage 1 先用 **方式 A**，实现成本低；Stage 2 再升级为 **方式 B**。

#### 输出格式约束（强制 LLM 结构化返回）

System prompt 根据 schema 是否为空给出不同约束：

**Schema 非空时**：
```text
注意：
1. 必须使用上方【可用概念网络】中的精确概念名称（如 'Sales Enablement'），
   不要添加解释或创造新名称。
2. 推理路径节点必须是概念名称，格式为：概念A -> 关系类型 -> 概念B。
   不要使用描述性句子作为节点。
3. 若题目是选择题，[最终答案]只输出选项字母（如 A / B / C）。
```

**Schema 为空时**：
```text
注意：
1. 当前概念网络为空，请在[推理过程]中直接列出你从题目中识别出的关键术语名称。
   名称应简洁（不超过10个字），不要输出解释性句子或括号备注。
2. 推理路径可省略，或仅使用你列出的术语名称。
3. 若题目是选择题，[最终答案]只输出选项字母（如 A / B / C）。
```

**注意**：LLM 不输出 confidence。confidence 由系统在收到 `reasoning_trace` 后，根据 schema 中 concept/relation 的权重计算得出。

- `prompt_builder` 负责把这个格式约束拼接进 system prompt。
- `agent/openai_compatible.py` 负责用 regex/json 解析 LLM 的返回，提取 `reasoning_trace`（concepts + relations）和 `answer_text`。

#### 渐进式披露（按 attempt_index）

- `attempt=0`：只注入 concept 名称 + 简短描述（省 token）。
- `attempt>=1` 且错误：注入完整 description + relation 细节。
- Token 压力大时：只保留与问题 embedding 相似度最高的 topK concept，截断长 description。

### 3.7 Problem 基类与 Evaluator

`socialclaw/stage1/dataset/base.py`

```python
@dataclass
class Problem:
    id: str
    prompt: str                # 给 LLM 答题的完整 prompt
    problem_type: str          # "mcq" | "long_context" | "arc_grid"
    meta: Dict[str, Any]
    retrieval_query: str = ""  # 用于 embedding 检索的精简文本（如 CL-bench 取 question 前 2000 字）

@dataclass
class EvalResult:
    correct: bool
    pred: Any
    gold: Any
    details: str = ""

class Evaluator:
    def evaluate(self, problem: Problem, attempt: AgentAttempt, agent: Optional[Agent] = None) -> EvalResult:
        """
        评估逻辑：
        - MCQ：提取选项字母，exact match。
        - long_context：若 agent 提供且 gold 非空，使用 LLM-as-judge
          （让 LLM 判断 pred 是否语义等价于 gold）；否则 fallback 到 exact match。
        - arc_grid：exact match。
        """
        ...
```

**CL-bench LLM-as-judge**：
- CL-bench 为开放式长文本问答，gold 答案往往是数百字的文档，exact match 不现实。
- `evaluate()` 在 `problem_type == "long_context"` 且 `agent` 非空时，构造评估 prompt：
  `题目 + 标准答案 + 模型回答 → LLM 输出 correct/wrong`。
- 这避免了因摘要/改写导致的假阴性，同时不引入外部评估框架。

---

## 4. 环境配置

- Python 版本：建议 3.11/3.12
- venv 已创建在项目根目录 `.venv/`
- 使用 `.venv/bin/python` 运行脚本
- 主要依赖：`datasets`, `sentence-transformers`, `rich`, `pydantic`, `numpy`

---

## 5. 实现顺序建议

1. `schema/graph.py` + `schema/storage.py` —— Schema 数据建模与持久化
2. `schema/retriever.py` —— Embedding 检索（接入 BGE）
3. `schema/initializer.py` —— Agent 自动生成 concept
4. `agent/openai_compatible.py` —— 可插拔 LLM Agent
5. `dataset/clbench.py` + `dataset/arc.py` —— 数据集接口
6. `prompt_builder.py` —— Schema 注入式 prompt 组装
7. `pipeline.py` + `run_stage1.py` —— 拼主链路
8. 跑主实验，记录指标

---

## 6. Stage 2 已实现内容（补充）

### 6.1 Schema 巩固 / 纠错

`pipeline.py` 中每道题评估后自动执行：

```python
def _update_schema_from_feedback(graph, trace, correct):
    delta = 0.05 if correct else -0.05
    # 更新 trace 中使用的所有 concept 的 confidence
    for cid in trace.concepts:
        c = graph.get_concept(cid) or graph.get_concept_by_name(cid)  # 支持模糊匹配
        if c:
            new_conf = max(0.1, min(0.95, c.confidence + delta))
            graph.update_concept(c.id, confidence=new_conf)
    # 更新 trace 中使用的所有 relation 的 weight
    # 先用模糊匹配将自由文本的 src/tgt 解析为 schema 中的 concept
    for src, tgt, rel_type in trace.relations:
        src_c = graph.get_concept(src) or graph.get_concept_by_name(src)
        tgt_c = graph.get_concept(tgt) or graph.get_concept_by_name(tgt)
        if src_c and tgt_c:
            r = graph.get_relation(src_c.id, tgt_c.id, rel_type) or graph.find_relation(src_c.name, tgt_c.name, rel_type)
            if r:
                new_weight = max(0.1, min(0.95, r.weight + delta))
                graph.update_relation(r.source, r.target, r.relation_type, weight=new_weight)
```

- 正反馈（答对）：`+0.05`（上限 0.95）
- 负反馈（答错）：`-0.05`（下限 0.1）
- Episode flags：`schema_reinforce` / `schema_correct`

### 6.2 高自信错误 → 人类提问纠错

当 `reasoning_confidence > 0.6`（或开启 `--always-ask-correction` 调试模式）但结果错误时，触发 `human_io.ask_correction()`：
- CLI 展示原题、LLM 回答、标准答案、推理路径 + confidence 值
- 人类输入纠正建议，由 `initializer.parse_correction()` 解析为 schema 更新操作
- 支持添加 concept、添加 relation、修改 concept description
- 新增 `--always-ask-correction`：调试模式下只要判错就提问，无视 confidence 阈值

### 6.3 CLI 主动提问（缺失 concept）

`auto-yes` 模式：自动调用 `initializer.generate_schema()` 生成 concept + relation，跳过人类。
非 `auto-yes` 模式：CLI 向人类提问，传入 `missing` 列表，人类回答后解析入库。

### 6.4 关键设计不变

- **Confidence 仍由 SchemaGraph 计算**，不交给 LLM 输出。
- **Relation source/target 存储的是 concept id**，不是 name（pipeline 负责 name → id 解析）。
- **几何平均**体现链式依赖的"短板效应"。
