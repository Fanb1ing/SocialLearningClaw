# ARC learned Schema 全流程开发方案

状态：Phase A–D 已实现离线原型；三个 ARC 游戏已完成 trajectory -> Memory -> window/keyframe
Schema induction。根据当前审查优先级，Phase E 定时管理暂缓，先实现 Phase F evaluator 并用评测
反馈改进 Schema 生成。当前不改变 ARC/ContextMATH/IntPhys2 runner 行为。

## 1. 目标与边界

本功能需要完整跑通：

```text
TrajectorySource（scripted / explorer / replay / Agent）
        ↓
观察并执行动作或读取冻结轨迹
        ↓
保存可重放的 step / level 轨迹记忆
        ↓
按窗口和关卡从记忆归纳 learned Schema
        ↓
按计划执行更新、冲突处理、合并、去重、升层、mask 和遗忘
        ↓
下一步/下一关检索并使用 learned Schema
        ↓
run 结束后离线比较 learned Schema 与 Gold Schema
```

必须保持两条边界：

1. learned Schema 必须引用持久化 `MemoryRecord` ID，且最终能追溯到原始
   observation/action/result transition；
2. Gold Schema 只能由离线 evaluator 读取，runner、记忆生成器和 Schema 管理器
   均不得导入或读取 `gold/`。

第一阶段先用 `cd82-fb555c5d` 做端到端 pilot。它的 18 条 game-level Gold
Schema 已经过人工修订，适合用来校准记忆粒度和 evaluator；SK48/TU93 在接口稳定后加入。

## 2. 当前实现需要修正的关键点

当前代码已经能把每个 transition 写成 `MemoryRecord` 并立即执行一次
`create/merge/skip`，但还不适合作为正式实验方法：

- 每一步都立即归纳，单步噪声和偶然变化容易膨胀成 Schema；
- 每个 transition 只包含文字摘要，Agent 实际看到的视觉输入没有成为 Memory evidence；
- 每一步产生的规则都会先获得正反馈，即使它只是一次观察、最终关卡失败；
- Agent 未声明使用 Schema 时，runner 会把全部注入 Schema 当成 used，credit assignment
  过宽；
- 单个 `reliability_weight` 同时承担“规则是否真实”和“规则是否帮助通关”，两个含义混在一起；
- 生成器一次只处理一条 Memory，不能从一段轨迹联合提炼规律或识别反例；
- 管理只有同层相似度阈值合并、衰减和 mask，没有显式冲突、修订、拆分、升层及操作日志；
- learned Schema 与 ARC Gold v1 字段不一致，目前没有正式匹配和评分器。

因此第一版不应在现有逐 step 归纳上继续叠加阈值，而应先把记忆和归纳单位改正确。

## 3. 轨迹数据集优先的开发模式

Schema 生成和管理的开发不应依赖在线 LLM Agent 恰好产生高质量动作。系统先建设一个
可复现的 trajectory corpus，再让离线 replay 和在线 Agent 共用后续 Memory -> Schema
pipeline。这样可以独立回答“给定同一批轨迹，Schema 模块是否变好”，不会把 Agent 行为波动
误认为 Schema 算法变化。

### 3.1 统一轨迹合同

新增 benchmark-neutral 的 `TrajectoryEpisode` 和 `TrajectoryStep`。ARC 的一关包含很多 step；
ContextMATH/IntPhys2 等一次性任务只是只有一个 step 的普通 episode，不需要另写一套 pipeline。

```text
TrajectoryEpisode
  episode_id, benchmark, task_id, split, actor, provenance
  initial_observation, steps[], terminal_outcome, metadata

TrajectoryStep
  step_index
  observation: text + structured state + artifact refs
  available_actions[]
  action: name + arguments
  result: next observation + state delta + environment status
  decision: optional response/rationale/schema IDs
```

其中：

- `actor` 标记 `scripted_policy`、`coverage_explorer`、`replay`、`human` 或 `llm_agent`；
- `decision` 是可选字段，Schema 生成不能要求存在 LLM chain-of-thought；
- observation artifact 使用通用 `MemoryArtifactRef`，既可引用 ARC grid/PNG，也可在未来引用视频
  帧、题目附件或其他模态；
- task/step/terminal 三层结构对所有 benchmark 相同，ARC 只是在 `steps` 数量上更长；
- corpus 文件和在线 runner 必须经过同一个 `TrajectoryRecorder`，避免出现两套语义不同的日志。

Schema pipeline 只接收标准化 `TrajectoryEpisode` 流，不直接依赖 ARC environment、Agent client
或某个 benchmark 的原始 sample class。

### 3.2 四类轨迹来源

实现统一 `TrajectorySource` 接口，至少支持：

1. `ScriptedPolicySource`：确定性规则/人工脚本从公开 observation 选择 action；
2. `CoverageExplorerSource`：固定 seed 的 action/state coverage 探索；
3. `RecordedReplaySource`：从冻结 corpus 原样重放，不调用环境或 LLM；
4. `AgentTrajectorySource`：未来接真实 LLM/human Agent，输出完全相同的数据合同。

所有来源最终都调用同一个 `TrajectoryRecorder.record_step()`。Memory builder 不知道动作由谁生成，
因此离线调试与在线实验之间只替换 source，不替换 Schema 代码。

### 3.3 ARC 可靠轨迹如何生成

ARC v1 corpus 采用互补的五类采集策略，而不是只保存成功通关轨迹：

#### A. 确定性成功轨迹

- 使用只读取公开 observation 的 game-specific policy；
- CD82 可根据画面中的目标、画布、调色板和工具计算动作，不需要 LLM；
- SK48/TU93 的首版策略可由人工编写并通过本地环境逐步验证；
- 每关至少重复 replay 两次，所有 observation/action/state hash 必须一致；
- 成功轨迹提供目标、长程依赖和 utility evidence，但不独占 corpus。

这些轨迹可由开发者参考源码设计 policy，但实际记录必须从 `env.reset()` 开始，只调用公开
observation 与 action 接口。manifest 将其标为 `source_guided_natural`，用于评估“从可靠轨迹提炼
Schema”的能力，不宣称代表自主 Agent 探索能力。

#### B. 单机制正向轨迹

- 每个 level 对每个 available action 至少采集一个有效变化样本；
- 对 ACTION6 等带参数动作，覆盖对象中心、边缘、背景和不同可点击角色；
- 保存 action effect、非局部连锁变化、选择状态和资源变化；
- 同一机制跨多个 level 重复采集，为 Level 3 -> Level 2 promotion 提供真实支持。

#### C. 反例与边界轨迹

- blocked/no-effect action；
- 相同 action 在 trigger 成立/不成立时的配对；
- invalid/background click、重复点击、最小/最大长度、墙边和资源临界点；
- WIN 前 near-miss、GAME_OVER、TIMEOUT 和预算耗尽；
- 这些轨迹用于学习 constraint/exception，也用于验证错误 Schema 会被 revise/contradict。

#### D. 成功轨迹扰动

对已验证成功序列执行确定性变换：删除一个动作、替换方向、重复动作、交换相邻动作、对 click
做小幅坐标偏移，并从环境重新执行，而不是人工伪造结果。保留导致相同成功、局部失败、无效果
或终局失败的不同结果，形成接近决策边界的正负 pair。

#### E. 覆盖导向探索

使用固定 seed 和有限预算探索尚未覆盖的 bucket：

- `(level, action signature, changed/no-effect)`；
- environment status；
- state-delta signature 和 changed-region 类型；
- click target role/region；
- action pair、重复效果及资源变化；
- 新 logical-state hash。

优先选择能增加 bucket coverage 的分支；达到连续若干 episode 无新增覆盖时停止。探索器目标来自
公开 transition 特征，不读取 Gold 文本或 Schema ID。

### 3.4 “全面”的可度量标准

corpus 生成器必须输出 `coverage.json`，不能只凭轨迹数量声称全面。第一版至少报告：

- level 覆盖率；
- available action 覆盖率；
- action 的 changed/no-effect 双侧覆盖；
- ACTION6 参数角色和空间区域覆盖；
- WIN/GAME_OVER/TIMEOUT terminal 覆盖；
- state-delta signature 数量；
- trigger/blocked、success/near-miss 等 paired-case 数量；
- 跨 level 重复机制的 evidence 数量；
- observation/grid hash 去重率和 replay determinism；
- 每个轨迹来源及每种终局的 episode/step 分布。

Gold coverage 只在 corpus 冻结后由离线 evaluator 计算，不能反向把未命中的 Gold 文本喂给
explorer 补数据。否则测到的是 Gold-guided annotation，而不是从任务经历学习 Schema。

CD82 v1 建议以 coverage gate 为最终条件，同时设一个便于估算成本的初始目标：

- 6 条可重复成功轨迹，每个 level 一条；
- 至少 12 条 near-miss/terminal failure，每个 level 至少两条；
- 至少 36 条 action、click-role、blocked/effect 单机制轨迹；
- 至少 24 条成功序列扰动轨迹；
- 至少 18 条固定 seed 的 coverage exploration 轨迹。

即首批约 96 个 episode。数量只是预算，不是质量结论：未达到 action/effect、终局、pair 和跨
level evidence coverage 时继续采集；较少 episode 已达到全部 gate 时可以提前停止。每一类保留
独立 scenario ID，后续可以只 replay 某个失败模式调试 Schema revise/merge。

### 3.5 数据可信度与泄漏分级

每条 episode 必须带 `evidence_tier`：

- `natural`：从 reset 开始，只通过公开 observation/action 到达；可用于正式 induction 指标；
- `source_guided_natural`：policy 设计参考了源码，但实际轨迹只走公开接口；可用于固定轨迹上的
  Schema 抽取实验，必须披露为 privileged behavior；
- `state_injected_probe`：直接修改内部状态、sprite 或 budget；只能做机制单元测试；
- `synthetic`：fake environment 或手写 transition；只能做接口/算法回归。

现有 Gold `runtime_cases.json` 中有些 case 会直接修改内部位置、画布或预算，因此只能借鉴其
测试思想或标为 `state_injected_probe`，不得直接转换为正式训练轨迹，也不得进入主指标。

建议冻结三套数据：

```text
tests/fixtures/trajectories/        小型 synthetic 回归 fixture，可提交 Git
data/trajectory_corpora/arc_agi3/   可复现生成的本地大 corpus，默认不提交大资产
configs/arc_agi3/trajectories/      scenario/policy/seed/split 配置和 manifest，可提交 Git
```

开发协议分开报告：

1. `fixed-corpus induction`：在同一批 source-guided/natural 轨迹上比较不同 Schema 算法；
2. `online end-to-end`：使用真实 Agent 轨迹，报告 Agent 与 Schema 的联合效果；
3. `held-out generalization`：CD82 用于开发，SK48/TU93 的冻结 corpus 和 Gold 用于接口稳定后的
   held-out 检查，避免反复针对所有 Gold 调 prompt。

### 3.6 corpus 格式和复现

```text
corpus_root/
  manifest.json
  coverage.json
  episodes/<episode-id>.json
  assets/grids/<sha256>.npy
  assets/images/<sha256>.png
  splits/{train,dev,test}.json
  validation.json
```

manifest 固定 game source hash、环境版本、collector 版本、policy 名称和 hash、seed、scenario、
预算、视觉渲染参数及 episode IDs。每次生成后重新 replay，验证 action 合法、pre/post 链连续、
终局一致、asset hash 正确；不一致的 episode 不发布。

corpus 本身保存事实轨迹，不保存 Gold Schema 文本、Gold ID 或 learned-to-Gold alignment。

### 3.7 后续 benchmark 的扩展接口

通用层只定义协议，不写 `if benchmark == "arc_agi3"`：

```text
TrajectoryDomainAdapter
  task_identity(raw_task)
  normalize_observation(raw_observation) -> Observation
  normalize_action(raw_action) -> Action
  describe_transition(pre, action, post) -> TransitionFeatures
  terminal_outcome(raw_result) -> Outcome
  is_informative(features) -> bool

SchemaDomainProfile
  schema_kinds
  default_scope
  induction_instructions
  canonicalize_schema(node)
```

ARC adapter 负责 grid、available actions、click 参数、环境状态和多 step；未来静态 benchmark
adapter 把题目/视频作为 observation、模型回答作为 action、判分作为 result，生成一个 step 的
episode。window scheduler 对单步 episode 会在 terminal 时立即 flush，因此无需特殊分支。

通用 `TrajectoryRecorder`、artifact store、corpus validator、Memory projector、Schema scheduler、
maintenance 和 evaluator orchestration 都不得导入 ARC SDK。benchmark-specific 内容限制在 adapter、
policy 和 Gold canonicalizer 中。这样后续迁移主要是新增 profile/adapter，而不是复制 runner。

## 4. ARC 记忆设计

### 4.1 三种记忆粒度

#### A. `transition`：原始事实层

每个环境 step 先同步持久化一条不可变的 transition memory：

- run/game/level/attempt/step；
- Agent 收到的 prompt、当前 available actions、模型原始响应和 reasoning trace；
- pre-grid 与 Agent 实际看到的 rendered PNG；
- 标准化 action 及参数；
- post-grid、环境状态、changed regions、对象摘要；
- 当步注入、Agent 明确声明使用的 Schema IDs；
- 指向视觉资产的引用和哈希。

transition memory 是 learned Schema 的最终事实证据。即使归纳调用失败，它也必须已经落盘。

#### B. `window_summary`：归纳工作记忆

每积累一批有信息量的 transition，生成一条派生 summary memory：

- `source_memory_ids` 指向窗口内全部 transition；
- 合并连续重复动作和重复 no-effect，但不删除原始 transition；
- 标出首次状态变化、动作效果差异、潜在前置条件、反例和终止事件；
- 保存用于多模态归纳的 keyframe 引用。

它用于控制上下文长度，不能代替原始证据。新 Schema 至少要引用一个原始 transition ID；
可同时引用 summary ID。

#### C. `level_episode`：关卡结果与长期 credit

关卡结束时保存完整 episode summary：

- 全部 transition/window memory IDs；
- WIN、GAME_OVER、TIMEOUT 或其他终止原因；
- Agent 明确使用过的 Schema、使用位置和使用后的局部结果；
- 成功路径中的有效机制、失败路径中的未证实假设和矛盾；
- 本关未覆盖但值得继续探索的问题。

level outcome 主要更新 Schema 的使用价值，不直接把失败解释为规则为假。

### 4.2 视觉记忆如何保存

不把 base64 或整张图片嵌入 `memory.json`。每个唯一 grid 保存两种内容寻址资产：

```text
schema/
  assets/
    grids/<logical-grid-sha256>.npy
    images/<render-sha256>.png
```

- `.npy` 是无损、可重放的整数 grid，是环境状态的主证据；加载时禁止 pickle；
- `.png` 是 Agent 当时实际看到的渲染结果，是视觉模型输入和人工审核证据；
- logical grid hash 由 shape、固定 dtype 和 C-order cell bytes 计算，不依赖文件路径；
- PNG 另存渲染参数、尺寸和 SHA-256；
- Memory 中只保存相对路径、hash、shape、dtype、role（`pre_state`、`agent_view`、
  `post_state`）和 render metadata；
- 相邻 step 的 post-grid 与下一 step 的 pre-grid 内容相同时复用同一资产；
- 原始 transition 全部保存，但归纳模型只接收 keyframes，避免视觉 token 无限制增长。

keyframe 默认选择：关卡初始帧、每种 action 首个有效变化的 pre/post、效果与过去不一致的
pre/post、关键 no-effect 代表、GAME_OVER/WIN 前后帧。对连续相同 no-effect 只送一组图片，
其余通过计数和 Memory ID 保留。

ARC 的文本对象摘要和 grid diff 是检索/筛选特征，不能替代 grid 与 PNG。Schema 归纳器在
需要命名视觉对象、区分区域或解释非局部变化时读取 keyframe；简单且明确的移动/no-effect
规则可以只用结构化 grid diff，以节省多模态调用成本。

### 4.3 建议的数据模型扩展

在通用 memory 层增加：

```text
MemoryArtifactRef
  artifact_id, role, media_type, relative_path, sha256, metadata

MemoryRecord
  scope: transition | window_summary | level_episode | feedback
  source_memory_ids: list[str]
  artifacts: list[MemoryArtifactRef]

MemoryEvent
  input_artifact_ids: list[str]
  output_artifact_ids: list[str]
```

`MemoryStore` 验证引用存在、hash 格式正确、派生 Memory 不形成 provenance cycle。
删除/遗忘 Schema 时不删除这些原始资产。

## 5. 从记忆生成 Schema

### 5.1 触发时机

拆成三个不同频率的操作：

1. **每 step**：只保存 transition，保证环境 loop 不因辅助 LLM 失败而丢轨迹；
2. **每 N 个 informative transitions 或关卡终止**：执行一次 window induction，建议 pilot
   默认 `N=8`；
3. **每关结束**：执行 level synthesis 和一次管理任务；长关卡可每 4 个 induction batch
   额外维护一次。

`informative` 指 grid/state 变化、动作效果与已有预测不一致、新 action/参数组合、终止事件。
重复 no-effect 会被计数，但不单独触发每一次归纳。

### 5.2 生成输入

每次 induction 输入：

- 当前 window summary 和选中的原始 transition；
- keyframe 或结构化 pre/post grid diff；
- 同 game、scope 相容的候选 learned Schema；
- 候选的正向证据、反例摘要和当前 confidence；
- 本批已处理过的 Memory IDs，防止重复消费。

induction 可以读取 Memory 中真实发生过的完整成功/失败轨迹，但不能读取源码、Gold Schema 或
隐藏环境规则；生成结果必须是原子、可复用机制，不得把完整获胜 action sequence 复制成 Schema
或随后注入 Agent 的上下文。

### 5.3 输出操作

生成器从单一 proposal 改为可返回多个原子 proposal：

- `create`：新建一个有充分 transition evidence 的原子规则；
- `support`：新证据支持已有规则，只追加 evidence，不重写语义；
- `revise`：新证据支持对已有规则增加前置条件、constraint 或 exception；
- `contradict`：观察结果与规则预测冲突，记录反例，交给管理器决定降权、拆分或废弃；
- `skip`：证据重复、只是 UI/配置事实、只描述具体动作序列，或证据不足。

每个 proposal 必须显式列出 `evidence_memory_ids`；管理器只接受当前 batch 可见且真实存在的
Memory ID。LLM 不直接修改图，所有操作经过确定性 validator 后由 Manager 应用。

### 5.4 learned Schema 的语义字段

为与 ARC Gold 对齐，learned node 增加一组通用结构化字段：

- `title`；
- `kind`：`observation_semantics`、`action_precondition`、`action_effect`、
  `state_transition`、`constraint`、`hazard`、`goal`；
- `trigger`、结构化 `action_sequence`、`expectation`；
- `constraints`、`exceptions`；
- `benchmark`、`game_scope`、`level_scope`；
- `abstraction_level`。

learned 专属字段继续保存 memory evidence、graph relations、状态和权重。Gold v1 文件不改写；
evaluator 把 learned 与 Gold 分别转成一个只读 canonical view。

ARC 层级暂定：

- Level 3：只被单个关卡或局部视觉配置支持；
- Level 2：同一个游戏的多个关卡共同支持的机制；
- Level 1：至少多个游戏支持的机制族；
- Level 0：由多个 Level 1 家族支持的系统抽象。

只跑单游戏 pilot 时允许生成 Level 3，并在至少两个不同 level 提供一致证据后提升到 Level 2；
不在单游戏数据上自动生成 Level 0/1。

## 6. 反馈与 Schema 管理

### 6.1 分开“真实性”和“有用性”

建议把当前单一权重拆成：

- `evidence_confidence`：transition 是否持续支持该规则；预测命中时上升，真实反例时下降；
- `utility_weight`：Agent 使用该 Schema 后的局部动作效果和关卡表现；
- `reliability_weight`：为兼容检索而保留的派生综合分数，不再直接作为唯一状态。

WIN 可以提高使用过的 Schema 的 utility，但 GAME_OVER 不能自动证明这些 Schema 为假。
只有 trigger 成立、Schema 给出了可检查预测、实际 transition 与预测冲突时，才写 negative
evidence 并降低 evidence confidence。

只给 Agent 明确声明使用、且确实在当步 prompt 中注入的 Schema 分配 credit。未声明时保持
unknown，不再把全部 injected IDs 当成 used。

### 6.2 定时管理顺序

每次 maintenance 按固定顺序执行并写审计日志：

1. provenance 和图结构校验；
2. 消费尚未处理的 support/contradiction evidence；
3. 同 scope、同 kind 的精确/近重复检测；
4. 语义等价节点合并，保留全部 evidence 和 ID redirect；
5. 冲突节点修订、按条件拆分，无法消解则降低 confidence 或 mask；
6. 多 level 一致证据触发 Level 3 -> Level 2 promotion；
7. 根据访问、证据量、反例和 utility 执行 mask/deprecate；
8. 生成不可变 snapshot 和 maintenance report。

不物理删除 Memory，也不在第一版物理删除 Schema。合并后的旧 ID 写入 alias/tombstone，保证
历史 step artifact 和 evaluator 仍能解析。

建议输出：

```text
schema/
  maintenance/events.jsonl
  snapshots/checkpoint_<n>/schema.json
  summaries/window_<n>.json
  summaries/level_<n>.json
```

## 7. learned Schema vs Gold Schema evaluator

### 7.1 运行边界

evaluator 是独立离线 CLI，只读取一个已结束 run 的 Schema snapshot 和指定 Gold 目录。
它不实例化 `SchemaManager`、不回写 learned state，也不向 Agent/生成器返回匹配结果。

默认只评估 manifest 标记为人工接受的 game Gold；pending/provisional 节点单独报告，不进入正式
主分数。第一阶段只评 CD82 game-level Gold，不评 provisional cross-game 层。

### 7.2 匹配流程

1. **结构校验**：learned evidence ID、scope、层级、action 格式和 graph links 有效；
2. **scope 过滤**：game 和 level 不相容的节点不能匹配；
3. **候选召回**：kind、action signature、关键词和 embedding 产生小候选集；
4. **语义判定**：温度 0 的独立 judge 对 canonical 字段判断
   `equivalent / learned_narrower / learned_broader / partial / contradiction / unrelated`；
5. **组合覆盖**：允许最多 3 个原子 learned nodes 联合覆盖一个 Gold，也允许一个 learned
   node 覆盖多个 Gold，但必须显式记录 split/merge 类型，不能靠重复计分；
6. **人工校准**：CD82 建立一小批人工 alignment fixture，用于固定 judge prompt 和阈值。

embedding 只用于候选召回，不能直接决定正确匹配。所有 judge 输入、输出、置信度和理由缓存到
artifact，保证复查和避免重复费用。

### 7.3 指标

Schema quality 主指标：

- `gold_recall`：Gold 机制被 learned Schema 完整覆盖的比例；
- `learned_precision`：active learned Schema 被 Gold 支持的比例；
- `semantic_f1`；
- `contradiction_rate`；
- `scope_accuracy` 和 `level_accuracy`；
- `evidence_traceability`；
- `split_rate`、`overmerge_rate` 和 unmatched 节点列表。

学习过程另报：

- 每百个 informative transitions 的 active Schema 数量（density）；
- 新 level 首次命中旧 Schema 的比例；
- retrieved、claimed、locally useful Schema 的 precision/recall；
- checkpoint 的 Gold recall/precision 曲线；
- 首次发现每条 Gold 机制所需的 steps；
- Agent 通关率、steps 和 Schema 辅助调用 token，避免只优化静态文本相似度。

输出：

```text
evaluation/
  config.json
  metrics.json
  alignments.json
  unmatched_learned.json
  unmatched_gold.json
  judge_cache.jsonl
  report.md
```

## 8. 端到端运行协议

完整开发链路分六段执行：

1. 用 fake environment 验证通用 trajectory contract、recorder、asset 和 replay；
2. 不调用 LLM Agent，以确定性 policy/explorer 生成并冻结 CD82 corpus；
3. 从空 Schema state replay corpus，按 step 投影 Memory，按窗口调用 Schema induction；
4. 每个 episode/level 结束生成 summary、maintenance 和 snapshot；
5. corpus 全部消费后显式调用 evaluator，与 CD82 Gold 生成独立报告；
6. 最后才把 source 换成真实 Agent，验证 online 与 replay 经过同一 recorder/pipeline。

第一轮调试允许 Schema induction/judge 使用 LLM API，但动作轨迹完全离线确定；进一步做单元测试时
使用 fake generator/judge，整个测试套件不需要网络。这样 Agent API、Schema API 和 evaluator API
三个变量可以分别控制。

runner 和 learned state 分开：run artifacts 永远在本次输出目录；新增显式
`--schema-state-dir` 才允许跨 run 复用。默认仍为 run-local，防止不同实验互相污染。

CLI 建议增加：

```text
generate-arc-trajectories --game-id ... --scenario-set ... --seed ...
replay-schema-corpus --corpus ... --state-dir ...
--schema-induction-interval 8
--schema-maintenance-batches 4
--schema-state-dir <optional>
--schema-reset
--schema-keyframe-limit 12
--schema-snapshot-every-level
```

这些字段全部写入 `manifest.json`。正式比较时必须固定模型、step/attempt/token budget、视觉渲染、
induction/maintenance schedule 和初始 Schema snapshot。

## 9. 需要修改的代码

### 9.1 现有文件

| 文件 | 修改内容 |
|---|---|
| `socialclaw/memory/models.py` | 增加 Memory scope、source IDs、artifact refs 和 event 的输入/输出视觉引用 |
| `socialclaw/memory/store.py` | 校验派生 Memory 引用；支持新格式加载与迁移 |
| `socialclaw/memory/bank.py` | 按 game/level/scope/evidence 状态过滤检索 |
| `socialclaw/benchmarks/base.py` | 增加通用 trajectory/domain adapter 协议 |
| `socialclaw/dataset/arc_agi3.py` | 保持环境包装；补充 recorder 需要的公开 observation/action/outcome 元数据 |
| `socialclaw/schema/node.py` | 增加 kind、scope、constraints/exceptions、双权重、alias/tombstone 等字段 |
| `socialclaw/schema/induction.py` | 单条 proposal 改为多 transition、多模态、批量原子 proposals |
| `socialclaw/schema/manager.py` | 拆分 ingest/induce/feedback/maintenance；增加冲突、修订、promotion 和审计 |
| `socialclaw/schema/layered_graph.py` | 安全 merge/redirect、冲突关系、promotion 后的层级校验 |
| `socialclaw/schema/layered_storage.py` | Schema format v2、snapshot 和 v1 迁移 |
| `socialclaw/schema/system.py` | 组装 asset store、induction scheduler、maintenance journal |
| `socialclaw/schema/arc_agi3_parser.py` | 输出可序列化 grid/object/diff 特征，供 keyframe 和候选筛选使用 |
| `socialclaw/arc_runner.py` | 先保存 transition，再按窗口归纳；关卡总结、精确 credit、checkpoint |
| `socialclaw/run_arc.py` | 新增 schedule/state/keyframe CLI 和 manifest 字段 |
| `socialclaw/logging.py` | step artifact 与 memory ID、asset ID、checkpoint ID 对齐 |
| `socialclaw/agent/openai_compatible.py` | 复用持久化 Agent-view PNG；支持归纳器需要的多图消息构造 |
| `socialclaw/experiment.py` | 固定 Schema schedule、初始 snapshot 和视觉资产参数 |

### 9.2 建议新增文件

| 文件 | 职责 |
|---|---|
| `socialclaw/trajectory/models.py` | benchmark-neutral episode/step/observation/action/outcome 合同 |
| `socialclaw/trajectory/source.py` | scripted/explorer/replay/agent 四类 `TrajectorySource` 接口 |
| `socialclaw/trajectory/recorder.py` | 所有 source 共用的原子 step recorder 和连续性校验 |
| `socialclaw/trajectory/corpus.py` | manifest、split、coverage、validation 和 corpus reader/writer |
| `socialclaw/trajectory/replay.py` | 冻结 corpus 的确定性 replay 与 Memory 投影入口 |
| `socialclaw/trajectory/arc_agi3.py` | ARC adapter、coverage feature 和 public-interface collector |
| `socialclaw/trajectory/arc_policies.py` | CD82/SK48/TU93 policy、扰动器和 coverage explorer |
| `configs/arc_agi3/trajectories/*.json` | 可版本化的 scenario、seed、预算、split 和 coverage gate |
| `socialclaw/memory/assets.py` | 内容寻址 grid/PNG 保存、去重、hash 校验和安全加载 |
| `socialclaw/schema/pipeline.py` | source-independent trajectory -> Memory -> induction -> maintenance orchestration |
| `socialclaw/schema/arc_memory.py` | ARC transition/window/level Memory builder 和 keyframe selector |
| `socialclaw/schema/scheduler.py` | informative step 计数、induction/maintenance 触发和恢复游标 |
| `socialclaw/schema/evaluation.py` | canonical view、候选召回、组合匹配和指标计算 |
| `socialclaw/schema/gold_loader.py` | 只供离线 evaluator 使用的 ARC Gold v1 loader |
| `scripts/generate_arc_trajectory_corpus.py` | 生成、重放验证和冻结 ARC v1 corpus |
| `scripts/replay_schema_corpus.py` | 不运行 Agent 地消费 corpus 并生成 learned Schema snapshots |
| `scripts/evaluate_learned_schema.py` | evaluator CLI 与报告输出 |
| `tests/test_trajectory_contract.py` | 单步/多步 episode、adapter 和 source 一致性测试 |
| `tests/test_trajectory_corpus.py` | corpus split、coverage、determinism 和 evidence-tier 隔离测试 |
| `tests/test_memory_assets.py` | grid/PNG round-trip、去重、hash 和坏引用测试 |
| `tests/test_arc_schema_pipeline.py` | fake ARC 的完整 step -> memory -> induction -> maintenance -> reload 测试 |
| `tests/test_schema_maintenance.py` | merge、冲突、promotion、alias、mask 和 evidence 保留测试 |
| `tests/test_schema_evaluator.py` | 人工 alignment fixture、组合匹配、泄漏边界和指标测试 |

现有 `tests/test_layered_schema.py`、`tests/test_arc_schema_runner.py` 和
`tests/test_gold_schema.py` 继续保留并迁移到新数据合同。

## 10. 推荐实现顺序与验收门

### 阶段 A：通用轨迹合同、recorder 与视觉资产

先完成 benchmark-neutral 数据模型、source/adapter、asset store、transition 持久化和 reload。
验收：同一 grid 去重；PNG 与 Agent 输入一致；单步和多步 episode 都可 replay；辅助调用失败不
丢记录；通用模块不导入 ARC SDK。

状态：已完成。实现和审查方式见 [`trajectory_contract.md`](trajectory_contract.md)。

### 阶段 B：ARC 可靠轨迹 corpus

先实现 CD82 observation policy、机制脚本、扰动器和 coverage explorer，生成 natural /
source-guided-natural corpus。验收：全部 episode 从 reset 经公开 action 产生、二次 replay hash
一致、coverage report 达标、Gold ID/text 不在 corpus 中。随后冻结 SK48/TU93 为 held-out corpus。

状态：已完成三个示例游戏。CD82 为 96 episodes / 1022 steps；SK48/TU93 各 24 episodes，SK48
v1 明确只验证到 3/8 level，TU93 验证到 9/9。全部 144 episode replay 一致。实现与人工审查见
[`arc_trajectory_corpus.md`](arc_trajectory_corpus.md)。SK48 4–8 level 留给 trajectory v2，不阻塞
Schema 模块开发。

### 阶段 C：离线 replay -> Memory

把冻结 corpus 投影为 transition/window/level Memory。验收：原始 episode 和 Memory 一一可追溯，
相同 corpus 重跑产生相同 Memory provenance，完全不调用 Agent API。

状态：原型已完成并扩展到三个示例游戏。144 episodes / 2068 steps 投影为 2545 条 transition、
window、episode MemoryRecord，稳定 ID、跨游戏隔离和视觉 evidence 闭包通过验证；一个不读取 Gold
的确定性 bucket baseline 已从这些 Memory 生成 40 个 SchemaNode。当前 SK48 轨迹成功进度为
3/8，明确留到 trajectory v2；详见
[`trajectory_schema_prototype.md`](trajectory_schema_prototype.md)。下一阶段用相同 Memory 合同开发
真正的 window/keyframe 语义归纳，不再改动投影层接口。

### 阶段 D：窗口归纳

完成 scheduler、keyframe、批量 proposal 和 validator。验收：固定轨迹能从多条证据创建原子
Schema；重复 no-effect 不膨胀；每个 node 引用真实 transition ID；fake generator 可离线测试。

状态：已完成确定性 `semantic_window_v1` 原型。333 个窗口从 2068 个 transition 中选择 138 个
关键帧，形成 50 个 SchemaNode；2064 个 transition 被 source 引用，4 个 singleton 明确 skip。
`create/support/revise/contradict/skip` 合同、grounded validator、视觉 artifact 校验和逐提案 audit
均已实现；详见 [`window_schema_induction.md`](window_schema_induction.md)。下一阶段在同一 graph
合同上实现定时维护，不回退到逐 step 建 Schema。

### 阶段 E：定时管理

完成双权重、support/contradiction、merge、promotion、alias 和 snapshot。验收：维护前后 graph
有效，历史 ID 可解析，Memory 和视觉 evidence 不丢失；corpus replay 中断后可从游标恢复。

状态：按当前开发决策暂缓。现有 Phase D snapshot 保持不变，先评估和改进生成质量。

### 阶段 F：evaluator

先用手写 learned fixture 对齐 CD82 Gold，冻结匹配类别、组合规则和指标，再接真实 learned
snapshot。验收：完全匹配、过宽、过窄、拆分、合并、矛盾和无关七类用例均稳定通过。

状态：第一版离线结构化 proxy 已完成并在三个已审核 Gold 游戏上运行。50 条 learned 对 37 条 Gold
的 strict precision/recall 均为 0，graded precision/recall 为 0.562/0.415，partial coverage 为
24/37，证据可追溯率为 1.0。七类关系与 fake fixture 均有离线测试；结果说明当前生成器主要学到
动作相关性，尚未学到 Gold 粒度的前置条件、对象语义、精确效果和 hazard。详见
[`schema_evaluation.md`](schema_evaluation.md)。正式论文指标仍需冻结人工 alignment fixture 并校准
独立语义 judge。

### 阶段 G：在线 source 兼容与真实 Agent pilot

将 `RecordedReplaySource` 替换为 `AgentTrajectorySource`，先跑 CD82 小预算 smoke。验收：除
actor/provenance 外的数据合同一致，同一 pipeline 无分支工作。在线 Agent 结果用于最终端到端实验，
不阻塞 Schema 模块日常开发。

## 11. 第一阶段完成标准

- ARC 每个 step 都有可重放 transition memory 和无损视觉 evidence；
- 无需调用 Agent API 即可生成/重放一批含成功、失败、边界和反例的确定性 ARC 轨迹；
- corpus 有来源分级、固定 split、覆盖报告、环境/source/policy hash 和 replay validation；
- `state_injected_probe` 不进入正式 Schema induction/evaluator 主指标；
- Schema induction 与 maintenance 失败不影响主轨迹持久化；
- Schema 只引用存在的 Memory，且可递归追溯到 transition；
- 生成按窗口执行，不再默认每步创建规则；
- feedback 区分规则真实性与任务有用性；
- maintenance 可恢复、可审计，不破坏历史 ID；
- evaluator 与 runner 进程和 import 路径隔离，不发生 Gold 泄漏；
- CD82 输出 learned precision/Gold recall/F1/contradiction/evidence 指标；
- 同一 Schema pipeline 同时接受多步 ARC replay、在线 Agent stream 和未来单步 benchmark episode；
- fake environment 全流程测试、现有离线测试、compileall 和 `git diff --check` 全部通过；
- 真实模型/API pilot 的任务调用与辅助 Schema 调用 token 分开记录。
