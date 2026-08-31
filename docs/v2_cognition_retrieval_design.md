# V2 认知输入精简与按需检索设计

日期：2026-08-29；实现更新：2026-08-30

当前状态：紧凑 Markdown 输入、全部 EFPS 核心节点与全局 Insight 目录、精确只读
`read_cognition`、工具调用审计和逐 provider request token 统计已经实现。2026-08-30 的首次紧凑
实验仍使用下文记录的旧 `query_cognition` 模糊检索；审计发现其会在指定 ID 后继续补足 top-k，现已
删除。下面第 5 节中建议的图操作工具与 hypothesis registry 仍是后续优化，不是本轮实现的一部分。

## 1. 当前运行的实际成本

审计对象：
`outputs/review/v2_generic_cd82_level1_opus48_semantic_complete3_20260829/`。

| Agent | 调用数 | 输入 token | 输入占比 | 输出 token | 总 token |
|---|---:|---:|---:|---:|---:|
| Main | 20 | 609,537 | 38.91% | 17,445 | 626,982 |
| Exploration | 20 | 589,635 | 37.64% | 15,493 | 605,128 |
| Update | 21 | 367,418 | 23.45% | 36,987 | 404,405 |
| 合计 | 61 | 1,566,590 | 100% | 69,925 | 1,636,515 |

输入占总 token 的 95.73%。单步输入从 Step 1 的 13,159 增长到 Step 20
的 119,026；Step 13 和 Step 19 因一次证据约束纠错重试分别达到
115,047 和 145,381。

现有日志不能把 provider input token 精确归因到单个字段，因为运行时只
保存调用总 usage 和压缩后的认知输入回执，没有保存字段级计量。以 41 个
认知输入回执的 JSON 字符数作可审计代理，分布如下：

| 回执字段 | 字符占比 |
|---|---:|
| Evidence 摘要 | 62.10% |
| Entity 与当前 Feature assertions | 25.07% |
| Schema 摘要 | 8.69% |
| Prototype | 1.29% |
| 其他清单/计数/策略 | 2.85% |

Schema 的真实请求占比高于 8.69%，因为回执只保留 Schema 摘要，而旧请求
发送了完整 preconditions、expected changes、invariants、boundary
conditions 和 Evidence ID 列表。Evidence 还同时出现在认知图、最近
transition 和 Update 输入中，形成重复。

输出也有明显冗余。Update 输出字符中，Schema updates 占 39.24%，
transition analysis 占 26.33%，Entity 占 21.94%；Main 输出中，重复的
goal hypotheses 占 41.22%，rationale 占 33.29%。

## 2. 原则：完整持久化不等于完整提示

EFPS 继续作为完整、可回退、Evidence-grounded 的持久认知库。Agent 默认
只收到一个小型 working set；需要历史细节时通过只读工具按精确 ID 读取。审计
必须同时保存默认 working set、每次读取参数、原样返回结果和最终实际输入。

认知分三层：

1. 热上下文：当前画面、公开 action 合同、最近 3 条 Entity 级 transition、
   所有 Entity/Prototype/Schema 的一行文本目录，以及最相关节点的稍详细摘要。
2. 温存储：可由工具按 ID 返回的节点原始记录、Schema 的支持与反例、Entity 的
   当前及历史 Feature。
3. 冷存储：完整 Evidence artifact 描述、Feature 历史、audit log、派生 Relation、
   replay 数据。仅审计或明确解析时读取。

## 3. Agent 可调用的精确只读工具

统一入口是 `read_cognition(command, id, feature_id?)`。它是存储读取器，不是问答或搜索系统。

参数约定：

- `command`：固定枚举；
- `id`：必须是默认目录或先前精确结果中出现的完整持久 ID；
- `feature_id`：仅 `get_feature_history` 可选使用，用于限定一个精确 FeatureDefinition ID；
- 不接受自然语言 query、node type filter、top-k、detail 或 action name。

命令和返回内容：

| command | `id` 类型 | 原样返回内容 |
|---|---|---|
| `get_entity` | Entity ID | Entity、其 FeatureAssertions、相邻 typed Relations |
| `get_prototype` | Prototype ID | Prototype、引用的 FeatureDefinitions、相邻 Relations |
| `get_schema` | Schema ID | 完整 `Prototype → Action → Output` 三元组与正反证据 ID |
| `get_insight` | Insight ID | 全局 rule/constraint/goal/mechanic/strategy、范围、置信度与正反证据 ID |
| `get_evidence` | Evidence ID | action/result、语义变化、未归属变化、带 phase 的公开观察引用和 agent-view artifact IDs |
| `get_feature_history` | Entity ID | 该 Entity 保存的 FeatureAssertions；可用 `feature_id` 精确过滤 |
| `get_relations` | EFPS 节点 ID | 与该节点相连的原始 typed Relation records |
| `get_artifact` | artifact ID | artifact 描述；若为 agent-visible PNG，同时把保存的原图附入下一轮模型输入 |

成功返回 `{"ok": true, "command": ..., "id": ..., "record" 或 "records": ...}`。失败返回
`{"ok": false, ..., "error": ...}`，错误包括 `not_found`、`invalid_command`、
`not_agent_visible` 和 `stored_file_unavailable`。后端只做字典查找和访问边界过滤，不调用另一个
LLM，不扩展召回、不排名、不总结、不推断。Evidence 里的 `review_view` 和 `environment_state` 不会
暴露给 Agent；只有它在原交互中有权看到的 `agent_view` PNG 能由 `get_artifact` 取回。

## 4. 各 Agent 的默认输入

### Main

- 当前原始画面和公开状态；
- 精简 action 合同；
- 最近 3 条 transition 的一行语义摘要；
- 所有 Entity、Prototype、Schema 的一行文本目录；
- 最相关节点的重要 Feature、Relation、正反证据摘要；
- Exploration 的一段建议；
- `read_cognition` 精确读取。

不再默认发送所有节点的完整字段和历史；但所有 Entity、Prototype、Schema
仍以简短文本形式可见，避免检索器过早过滤掉 Agent 可能需要的对象。

### Exploration

- 当前原始画面和公开 action；
- 最近 3 条 transition；
- action 探索覆盖摘要；
- 所有 Entity、Prototype、Schema 的一行文本目录；
- 最重要的冲突、低置信 Schema、关键 Relation 和未解决问题；
- `read_cognition` 精确读取。

### Update

- 当前 action、公开 result、原始 before/after 图片；
- 所有 Entity、Prototype、Schema 的一行文本目录；
- 与 changed bounds 相交或近期变化的 Entity 的重要 Feature；
- action name 匹配的 Schema 的完整工作摘要；
- 必要时用 `read_cognition` 按 ID 读取历史。

Update 不需要完整 Main rationale、全部旧 Evidence 或无关 Schema。
Prototype 是否创建由 Update LLM 根据当前公开 Evidence 自主判断，不设置最少成员数、
最少共同 Feature 数或重复出现次数等人为门槛；确定性 validator 只负责结构、引用和
Evidence grounding。

## 5. 输入输出格式

模型输入改成短 Markdown/描述性文本，而不是对整个 Python dict 做 JSON dump。
本地仍保留结构化对象用于 validator 和审计。工具参数与最终环境 action 可以保留
最小结构化协议，因为 runtime 必须无歧义地执行和校验；认知内容本身以文本摘要
提供。

输出精简建议：

- Exploration：最多一小段建议；现状已基本满足。
- Main：一个 action tool call，加一条正在检验的假设及引用 ID；不再每步重复完整
  goal hypotheses 和长 rationale。
- Update：优先调用受约束的图操作工具，如 `upsert_entity`、`assert_feature`、
  `create_or_revise_schema`、`attach_transition_semantics`，避免一次返回庞大 JSON。
- 没有变化的节点不重复输出；未更新的字段保持原值。

## 6. 持久字段取舍

| 类型 | 完整持久化 | 默认送给 Agent | 按需检索 |
|---|---|---|---|
| Evidence | action/result、before/after IDs、语义变化、Entity 归因 | 最近且相关的一行摘要 | 完整记录和原始图 |
| Entity | ID、当前 bbox/status、证据和当前 Feature | 所有 Entity 一行；重要 Feature 保留 | 完整 Feature 历史 |
| FeatureDefinition | ID、名称、类型、证据 | 通常不单独发送 | 定义详情 |
| FeatureAssertion | 当前值、置信度、证据；历史留在审计 | 合并进相关 Entity 一行 | 完整历史 |
| Prototype | 定义特征、成员、证据 | 所有 Prototype 一行，含成员关系与重要定义 Feature | 完整置信度和证据 |
| Schema | Prototype、action、output、正反 Evidence | 所有 Schema 一行显示完整三元组 | 完整 Schema 与证据链 |
| Insight | 类型、全局陈述、范围、置信度、正反 Evidence | 所有 Insight 一行 | 完整 Insight 与证据链 |
| Relation | 保留为图索引或由节点引用重建 | 重要关系内联为自然语言 | 完整邻域和底层边 |
| Audit log | 完整保存 | 不发送 | 人类审计/回放 |

Artifact metadata 应集中存储并由 ID 引用，不应在每个认知视图里重复宽高、renderer、
路径和 audience。`review_view` 永远不进入认知检索结果。

当前 Relation 是 EFPS 节点之间的类型边：Entity `HAS_FEATURE` 某个
FeatureAssertion，FeatureAssertion `ASSERTS_FEATURE` 某个 FeatureDefinition，
Entity `INSTANCE_OF` Prototype，Prototype `DEFINED_BY`/`EXCLUDES` Feature，
Schema `TAKES_PROTOTYPE` Prototype。Insight 是全局命题，不为了图形整齐而制造虚假端点边。
这些关系不是都应删除；应保留关系语义，但避免把
可由节点字段重建的 relation_id、metadata、重复 Evidence 数组整表发送。默认文本
把重要边内联到节点，例如“entity_x 属于 prototype_p”或“schema_s 以 prototype_p 为输入”。

Main 的 goal hypotheses 冗余发生在跨步骤：旧 prompt 要求每一步重新输出完整列表，
同一假设又随 Main decision 进入 Update 输入、timeline 和报告。新设计应建立持久的
hypothesis registry；Main 默认看到当前短列表，每步只输出 `create/revise/drop` 变化，
并用 `selected_hypothesis_id` 明确本次动作服务于哪条假设。

## 7. 预算和验证目标

- 每个 Agent 的默认文本上下文目标不超过 4,000 token；
- 单次工具检索结果不超过 2,000 token；
- 每个决策最多两轮精确认知读取；
- 日志新增每次请求的字段字符数、估算 token、图片数和 provider 总 usage；
- 对比实验同时报告过关率、Entity 归因完整率、Schema 命中/反例率、总 token 和
  每步 token 增长曲线。

预计可以消除大部分全图重复输入，但具体节省比例必须通过新运行测量，不能用字符
占比直接冒充 provider token 占比。

## 8. 2026-08-30 实现与实测结果

最终同版本运行位于
`outputs/review/v2_generic_cd82_level1_opus48_compact_retrieval_assertions_20260830/`。
它从空图在线运行 20 个动作，未通过 CD82 Level 1；trajectory replay 和最终 graph
validation 均通过。最终图含 21 Evidence、9 Entity、24 FeatureDefinition、33
FeatureAssertion、1 Prototype、8 Schema 和 71 typed Relation。

与旧版 1,636,515 total tokens 相比，新版使用 1,042,741 input、97,378 output，合计
1,140,119，total 减少 30.33%，input 减少 33.44%；output 因工具选择、续轮和更完整
Update 输出增加 39.26%。61 个逻辑 Agent 调用产生 130 个 provider requests 和 67 次
旧版 `query_cognition`：首轮请求占 406,224 tokens（35.63%），69 个追加请求占 733,895
tokens（64.37%）。这说明紧凑默认输入有效，但过度查询仍是主要成本。

Provider 不提供字段级 token。可精确报告的是：默认 prompt 的 `Current learned
cognition` 共 523,141 字符，占默认输入区段字符 79.53%；工具共返回 288,986 字符；
Update 全部输出 54,079 tokens。最后一项混合了场景、transition 语义和图 proposal，
不能冒充“纯认知图输出 token”。完整逐 Agent、逐步、逐 provider request 和工具调用
数据见该运行的 `token_usage.md`/`token_usage.json`。

实测还发现并修复了 Feature 描述归属问题：共享 `FeatureDefinition(name=color)` 不能
保存某个具体 Entity 的描述。描述现持久化在各 Entity 的 `FeatureAssertion` 上，紧凑
目录和查询工具均读取 assertion-level 描述；最终运行 33/33 assertions 均有自己的
描述。Prototype 创建仍由 Update LLM 自主判断，没有最少成员、Feature 或重复次数门槛。

## 9. 2026-08-30 精确读取协议修订

对上述实验的人工审查确认了旧查询的两个问题：自然语言字段诱导 Agent 把工具当问答系统；即使请求
包含精确 `ids`，本地打分器也只提升该记录权重，仍用无关节点补足 top-k。当前代码已用第 3 节的
`read_cognition` 完全替换该检索器，并同步完成以下修订：

- Evidence 新增 `observation_refs`，明确 `current/before/after`、fingerprint 与 artifact IDs；
- `get_artifact` 能将精确保存的 agent-view PNG 动态附回模型，而非只返回无意义的 ID；
- 三个 Agent 的 system prompt 统一说明 EFPS、Evidence、工具命令/参数/返回值；
- 新报告在时间线前记录实际 system instructions，并审计每次精确读取参数、原始返回值和图片；
- 默认目录仅列当前可见 Entity，省略冗余 `status=active` 及空的 optional/excludes；
- Schema 创建和修订必须形成完整的 `Prototype → Action → Output` 三元组，translator 与 graph
  validator 双重拒绝直接绑定 Entity、缺失 Prototype 或缺失 Output；
- 不符合三元组的全局规律进入 Evidence-grounded Insight，并通过 `get_insight` 精确读取；
- Update 的 Main decision 学习上下文不再截断为 220 字符。

该修订已通过本地结构和 runtime 测试，但尚未用昂贵的真实 provider 重跑 CD82，因此新的 token
消耗与游戏表现不能从旧实验外推。
