# SocialLearningClaw Version 2：EFPS 认知图与 Agent 开发方案

状态：V1 已冻结；受控 CD82 bootstrap actor 已于 2026-08-27 移除并替换为游戏无关的视觉
Main/Exploration/Update Agent。Main/Update 使用结构化输出，Exploration 只向 Main 返回文本建议。
第一次真实 CD82 Level 1 测试在 20 步内失败，但完成了从空图到
evidence-grounded EFPS 的真实在线闭环。早期结果见
`docs/archive/v2_cd82_level1_prototype.md`；正式三游戏结果与复现入口见
`experiments/v2_formal_20260830/README.md`。下一步优先解决
通用视觉坐标、重复探索抑制和 Schema consolidation，不增加 CD82 规则。

## 1. 项目现状与 Version 2 目标

SocialLearningClaw V1 是一个研究 Agent Schema Learning 的实验仓库，同时支持 ARC-AGI-3、
ContextMATH 和 IntPhys2，并比较八种 baseline 与一个 memory-grounded layered `schema` 方法。
当前 learned Schema 主线是：环境 transition 写入 `MemoryRecord`，再归纳为带 level、父子关系、
相似关系、可靠度、反馈、遗忘与维护机制的 `SchemaNode`。

Version 2 不再继续扩展这一多层结构。项目后续只研究 ARC-AGI-3，并让 Schema 的定义与皮亚杰的
认知发展理论对齐：Schema 是一系列被主体视为等价的行动中可重复、可迁移和可泛化的组织结构。
主体把已有 Schema 应用于新的情景和对象；结果符合预期时发生同化，结果不能由当前认知解释时
发生顺应。

Version 2 的核心类型图对象为 Entity、Feature、Prototype 和 Schema，以下统称 EFPS；另设不受
Schema 三元组格式限制的全局 Insight/Rule：

```text
具体 ARC observation
        ↓
Entity detection / tracking
        ↓
Feature assertions
        ↓
Prototype membership
        ↓
Schema = Prototype → Action → Output
        ↕
global Insight / Rule / Goal condition
        ↓
Concrete ARC action and prediction
```

## 2. 已确认的重构原则

1. 完整冻结当前 V1，不直接修改旧 Schema 实现。
2. 复用稳定的 ARC 环境、Trajectory、视觉资产、replay、coverage 和实验记录能力。
3. 在 `socialclaw.v2` 下建立独立实现，不让 V2 运行路径依赖旧 layered Schema。
4. 不再沿用 `SchemaNode.level`、父子层级、相似边等多层 Schema 语义。
5. 所有 learned Schema、Insight 及其关键关系仍必须引用持久 evidence ID，可以回到 trajectory、动作、
   observation、环境结果和无损视觉资产。
6. 主 Agent 本身就是 orchestrator。它负责理解当前任务、维护整体执行状态、规划和选择动作，并
   根据需要调用更新子 Agent与探索子 Agent；不存在第四个独立 orchestrator Agent。
7. 认知图与主 Agent 同步迭代。第一条开发纵向切片直接选择一个 ARC 游戏的 Level 1，不要求先在
   冻结轨迹上完成整套离线认知图。现有 trajectory/replay 仍作为调试、回归和失败复现工具。

## 3. EFPS 数据定义

### 3.1 Entity

Entity 是当前 ARC 关卡中被主体观察、区分和追踪的具体对象，例如某一个红色方块、可点击按钮、
玩家、敌人、出口或 UI 元素。

建议至少记录：

- `entity_id`：当前 episode/level 内稳定的实例 ID；
- observation 中的位置、mask、边界框和时间范围；
- 当前 Feature assertions；
- Prototype memberships 及置信度；
- 首次、最近一次和支持当前判断的 evidence IDs；
- active、occluded、disappeared、merged-candidate 等跟踪状态。

Entity 不能按 frame 重复创建。对象移动、颜色变化或短时遮挡时，应优先更新同一个 Entity；只有
无法与现有实例匹配时才新增 Entity。

### 3.2 Feature

Feature 是关于 Entity 或 Prototype 的原子属性。应区分 Feature 的定义与某次带证据的断言：

```text
FeatureDefinition: movable
FeatureAssertion: entity_12 has_feature movable, confidence=0.74, evidence=[...]
```

Feature 至少分为：

- `intrinsic`：颜色、形状、面积、纹理、条纹等自身属性；
- `state`：当前位置、激活、损坏、选中、朝向等可变状态；
- `affordance`：可移动、可点击、可推动、可拾取等主体交互属性；
- `relational`：阻挡某类对象、伤害邻居、位于某对象内部或旁边等关系属性。

每个 assertion 保存 subject、predicate、object/value、scope、confidence、positive/negative evidence
和有效时间。情境关系不能被误写成永久固有属性。

### 3.3 Prototype

Prototype 是主体根据共享关键 Feature 把多个 Entity 视为等价后形成的原型。Prototype 不是某个
具体物体，也不是传统 class label；它服务于 Schema 的迁移和压缩。

建议记录：

- defining、optional 和 exclusion Features；
- Entity memberships 与置信度；
- 支持、反例和最近修订证据；
- split/merge lineage，仅用于审计，不形成新的 active 多层 Schema；
- 被哪些 Schema 作为输入 Prototype 使用。

一个 Entity 可以在不同认知角色下属于多个 Prototype，例如同时属于“红色方块”和“可推动障碍物”。
更新系统必须避免仅因为出现一个新 Entity 就创建一个同名 Prototype。

### 3.4 Schema

Schema 的语义定义严格是一个三元组：

```text
(input Prototype, public Action pattern, observable Output)
```

- `prototype_id`：动作所作用或适用的一个输入类型；具体 Entity 只能经 Prototype membership 使用它；
- `action`：公开 action 名和参数模式；
- `output`：该 action 在该 Prototype 上产生的可观察结果。

`schema_id`、置信度、状态、修订次数、正反 Evidence ID 和审计 metadata 是三元组的管理字段，
不是额外语义槽。Schema 不再存 role bindings、preconditions、invariants 或 boundary conditions。
一个 transition 涉及多个对象时，应为主要输入选择明确 Prototype，并把跨对象约束保存为 Insight，
而不是把 Schema 扩回任意规则容器。

### 3.5 全局 Insight / Rule

Insight 保存不能自然表达为一个 `Prototype → Action → Output` 的可复用认识，例如：

- constraint：撞到某类墙时不能进入被占据位置；
- goal：某种公开状态变化可能表示关卡完成条件；
- mechanic：多个动作共享的资源、开关或状态机制；
- strategy：基于多条 Schema 组合出的行动原则；
- rule/other：其他由公开 Evidence 支持、可被反证和修订的全局陈述。

每个 Insight 保存 `kind`、`statement`、`scope`、confidence、status、support/counter Evidence IDs。
它不是游戏手册或隐藏事实；初始时为空，只能由 Update 根据 Agent 已获得的公开 Evidence 提出。
Main 可以引用 Insight ID 进入 `insight` 决策模式，或与 Schema 一起使用。

## 4. EFPS 关系图

V2 使用一个 typed relation graph，不使用 V1 的 level/parent/child/similar 层级。核心关系包括：

```text
Entity --HAS_FEATURE--> FeatureAssertion
Entity --INSTANCE_OF--> Prototype
Prototype --DEFINED_BY--> FeatureDefinition
Prototype --EXCLUDES--> FeatureDefinition
Schema --TAKES_PROTOTYPE--> Prototype
Schema --SUPPORTED_BY--> Evidence
Schema --CONTRADICTED_BY--> Evidence
Insight --SUPPORTED_BY--> Evidence
Insight --CONTRADICTED_BY--> Evidence
```

关系必须是类型安全的，并由 validator 阻止 dangling reference、错误端点、无证据 Schema 和非法
状态。Graph snapshot 与 revision log 均使用原子写入；一次更新子 Agent 的多项修改作为一个
transaction 提交，任何一项失败都整体回滚。

## 5. 同化、顺应与最小修改原则

### 5.1 同化

当环境反馈可以由已有认知解释时：

- 把新观察到的 Entity 映射到已有 Prototype；
- 支持已有 Feature assertion、Schema 或 Insight；
- 更新置信度、使用统计和有效范围；
- 不无理由增加 Entity、Prototype 或 Schema。

### 5.2 顺应

当结果与预测不一致时，更新子 Agent 按以下优先级寻找最小修改：

1. 检查 Entity segmentation/tracking 是否错误；
2. 修正 Entity 与 Prototype 的 membership；
3. 新增、撤销或限定 Feature assertion；
4. 修订已有 Schema 三元组，或把条件/范围修订为 Insight；
5. 在有区分 Feature 与多组证据时分化 Prototype；
6. 修改 Schema 的输入 Prototype；
7. 只有现有结构无法解释时才创建新 Prototype、Schema 或 Insight。

更新操作采用固定枚举：

```text
ADD_ENTITY / UPDATE_ENTITY / RETIRE_ENTITY
ADD_FEATURE_ASSERTION / REVISE_FEATURE_ASSERTION / RETRACT_FEATURE_ASSERTION
LINK_ENTITY_PROTOTYPE / UNLINK_ENTITY_PROTOTYPE
CREATE_PROTOTYPE / REVISE_PROTOTYPE / SPLIT_PROTOTYPE / MERGE_PROTOTYPE
CREATE_SCHEMA / REVISE_SCHEMA / ADD_SCHEMA_SUPPORT / ADD_SCHEMA_COUNTEREVIDENCE
CREATE_INSIGHT / REVISE_INSIGHT / ADD_INSIGHT_SUPPORT / ADD_INSIGHT_COUNTEREVIDENCE
SKIP
```

创建 Prototype、Schema 或 Insight 的 proposal 必须同时给出：现有结构为什么不足、预计解决什么预测错误、
引用哪些证据，以及为什么更小的修改不够。

## 6. Agent 组织方式

### 6.1 主 Agent：orchestrator、planner 与 actor

主 Agent 是唯一控制环境动作和总体流程的 Agent。它负责：

- 接收当前 observation、可用动作、历史和 EFPS/Insight 认知；
- 判断当前目标和认知状态；
- 从 Prototype-level Schema 形成 plan；
- 检查当前 Entity 是否属于 Schema 的输入 Prototype；
- 预测动作结果并执行下一动作；
- 比较预测与真实结果；
- 判断是继续执行、调用更新子 Agent，还是调用探索子 Agent；
- 审核子 Agent proposal，再交给确定性 validator 和 graph transaction。

建议主 Agent 输出结构化、可审计的决策，而不要求保存模型私有 chain-of-thought：

```json
{
  "goal_hypotheses": [{"text": "...", "confidence": 0.72, "evidence_ids": []}],
  "decision_mode": "explore|schema|insight",
  "selected_action": {"name": "ACTION3", "arguments": {}},
  "schemas_used": ["schema_x"],
  "schema_prediction": "...",
  "insights_used": ["insight_y"],
  "insight_application": "...",
  "exploration_hypothesis": null,
  "rationale": "..."
}
```

### 6.2 更新子 Agent

更新子 Agent 没有环境行动权限。它接收 pre/post observation、实际 action、主 Agent 的 prediction、
使用过的 Schema/Insight、当前 EFPS/Insight 认知和持久 evidence IDs，返回同化/顺应 proposal。

主 Agent只决定是否调用和是否接受其建议；真正的合法性、证据闭包和原子应用由确定性代码保证。

### 6.3 探索子 Agent

探索子 Agent 没有直接环境行动权限。它在出现新 Entity、新 action、低置信 membership、Schema
预测冲突、无法解释的 transition 或多个竞争假设时生成探索候选。

每个候选至少包含：要区分的假设、建议 action、预期的不同结果、信息增益、不可逆风险、动作成本
和重复探针惩罚。主 Agent从中选择实际执行动作：

```text
priority = expected_information_gain
           - irreversible_risk
           - action_cost
           - repeated_probe_penalty
```

## 7. 现有代码复用边界

### 7.1 直接复用

- `socialclaw/dataset/arc_agi3.py`：只在 public environment gateway 后复用离线环境和 rendering；
- `socialclaw/trajectory/models.py`：Observation、Action、Decision、StepResult、Episode、Outcome；
- `socialclaw/trajectory/recorder.py`：连续性校验、逐步原子持久化和 resume；
- `socialclaw/trajectory/corpus.py`、`arc_corpus.py`：原子 JSON 和逐帧 replay；
- `socialclaw/memory/assets.py`：grid/PNG 内容寻址、去重、hash 和路径安全；
- provenance 和 budget 设计模式；V2 不计算会读取游戏实现的 environment fingerprint。

### 7.2 拆分后复用

- `agent/openai_compatible.py` 保留模型请求、图像输入、重试和 token usage，替换旧 Concept/Relation
  parser 与单一回答协议；
- `arc_runner.py` 只参考环境状态、available-action 校验和终局处理，不复制其 Schema-coupled loop；
- 不复用 `schema/arc_agi3_parser.py` 的对象、颜色或区域语义；V2 gateway 只计算全网格 raw diff；
- `schema/window_induction.py` 只复用 proposal、validator、audit、skip/contradict 的设计模式；
- `schema/evaluation.py` 只复用输入隔离、hash、alignment cache 和 evidence closure 思路。

### 7.3 冻结后禁止进入 V2 运行路径

- V1 `SchemaNode`、`LayeredSchemaGraph`、`SchemaManager`、旧 induction/storage/system；
- `schema/trajectory_pipeline.py` 与旧 `window_induction.py` 的 learned Schema 语义；
- `methods/schema.py` 和旧 `arc_runner.py` 的即时 layered-Schema loop；
- ContextMATH、IntPhys2 runner、adapter、Gold 生成与静态实验逻辑；
- ARC baseline loop 继续作为 V1 对照保存，但 V2 production package 不导入它。

ARC Gold v1 保留为历史机制级对照，不能直接宣称为 V2 EFPS Gold；它缺少系统的 Entity、Feature、
Prototype membership、Schema 三元组和全局 Insight 标注。

## 8. 建议目录

```text
socialclaw/v2/
  efps/
    models.py
    relations.py
    graph.py
    operations.py
    validator.py
    storage.py
  context.py
  model.py
  agents/
    main_agent.py
    update_agent.py
    exploration_agent.py
    protocols.py
    prompts.py
  runtime/
    public_arc.py
    arc_online.py
  evaluation/
    graph_metrics.py
    schema_metrics.py
    exploration_metrics.py
socialclaw/run_arc_v2.py
tests/v2/
```

V2 通过 import 使用稳定基础设施，不复制第二份 Trajectory、ArtifactStore 或 ARC wrapper。

## 9. 开发计划

### Phase 0：冻结 V1

- 建立日期化归档目录；
- 保存代码、测试、脚本、配置、设计文档、Gold、关键 trajectory corpus 和 review 结果；
- 保存 Git commit、工作树状态、逐文件 SHA-256 和逐文件用途说明；
- 不保存 `.env`、API key、`.venv`、cache 或主机私有配置；
- 运行当前完整测试，形成 V1 基线。

### Phase 1：一个游戏 Level 1 的纵向骨架

先选择 `cd82-fb555c5d` Level 1，因为本地环境、成功轨迹、视觉资产和人工机制审核最完整。

同时搭建最小 EFPS models、graph、主 Agent loop 和子 Agent protocol。此阶段允许模型、图表示和
prompt 同步迭代，不要求先完成通用离线认知图。

最小运行链路：

```text
reset Level 1
  → normalize observation and save evidence
  → Update model proposes initial Entities/Features/Prototypes from raw image
  → Exploration model writes prose advice; Main model selects one action
  → execute and record the public pixel transition
  → Update model compares public before/action/after and attributes changes to Entities
  → enrich the resolvable Evidence record with semantic Entity changes
  → validate and commit graph transaction
  → repeat until WIN/GAME_OVER/step limit
```

### Phase 2：EFPS 图与更新门禁完善

- 补齐 typed relations、graph invariants、atomic snapshot 和 revision log；
- 完成最小修改优先级；
- 支持 support、counterevidence、membership 修正和 prototype split；
- 用 Level 1 运行中暴露的问题同步调整 Entity/Feature/Prototype/Schema 定义。

### Phase 3：主 Agent planning/reasoning 完善

- 从相关 EFPS 子图构造 prompt，不注入完整图；
- 显式检查 Schema 输入 Prototype 对当前 Entity 的适用性；
- 支持短计划、结果预测、计划继续/重规划；
- 避免重复 no-effect action；
- 记录所用 Schema/Insight、prediction/application 和子 Agent 调用原因。

### Phase 4：探索子 Agent 完善

- 实现 novelty 与 unexplained-transition 触发；
- 生成可以区分竞争假设的低风险动作；
- 在给 Main 的简洁文本中解释信息增益、风险、重复成本和预期可区分结果；
- 由主 Agent决定是否执行，探索子 Agent不直接控制环境。

### Phase 5：从 Level 1 向完整游戏扩展

扩展顺序建议：CD82 Level 1 → CD82 全六关 → SK48 → TU93 → 其他 ARC 游戏。每次扩展都检查
Prototype、Schema 和 Insight 是否跨关卡复用，避免退化为每关重新建图。

### Phase 6：V2 评测

至少记录：Entity tracking consistency、Feature 稳定性、Prototype 数量与复用率、split/merge 次数、
Schema prediction accuracy、Schema/Insight 跨关卡复用率、Insight 反证率、unexplained transition rate、graph edit rate、
exploration information gain、actions-to-win 和 evidence closure。

现有 trajectory/replay 用于精确复现在线失败和做回归，不作为“必须先离线完成”的开发门槛。

## 10. 第一阶段验收标准

V1 冻结完成后，第一个 V2 milestone 以 CD82 Level 1 为准：

1. 环境 observation/action/result 全部进入现有 durable trajectory；
2. 每个 Entity、Prototype membership 和 Schema 都能回到 evidence ID；
3. 主 Agent 能读取相关 EFPS 子图、产生合法 action，并说明 Schema binding；
4. 主 Agent 能按需调用更新或探索子 Agent；
5. 子 Agent 不能直接行动或绕过 validator 修改图；
6. no-effect 不重复创建 Entity/Prototype/Schema；
7. 所有 graph transaction 可审计、可回放、失败可回滚；
8. 完成一个可重复 fake-model 闭环和一次不使用专用规则的真实在线运行；真实 Level 1 是否成功单独报告。

通过这一纵向切片后，再决定哪些表示需要泛化，而不是提前为全部 25 个游戏设计过重的抽象层。
