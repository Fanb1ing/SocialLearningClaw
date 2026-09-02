# Version 3 开发方案：Tycho 世界模型基座 + EFPS 认知

日期：2026-09-02
状态：Phase 0–2、Python 3.12 运行时和 bounded smoke 已完成；Phase 3–5 尚未开始

实现进度和验证结果见 [V3 实现状态](v3_implementation_status.md)，代码边界和协作入口见
[V3 架构与协作指南](v3_architecture_and_collaboration.md)。默认 `.venv` 已升级到 Python 3.12；
首个 credentialed transport/CD82 bounded smoke 已完成，但 Builder 未触发，因此它只验证管线，
不构成 EFPS 性能证据。

## 1. 结论

V3 不应继续修补 V2 的 `EFPSGraph + Main/Exploration/Update` 在线闭环，而应以 Tycho 的
可执行世界模型和 actor-controlled builder 为主干，再把 EFPS 收敛为世界模型中的两类可执行认知：

1. 哪些当前 `Entity` 被同一个可复用 `Prototype` 分类器视为同类；
2. 哪个 `Prototype` 在某个公开 `Action` 下产生哪个可观察 `Output`，并由可执行
   transition rule 实现。

V3 的唯一动力学真相应是 agent 编写的 `world_model.py`。EFPS 不是另一张独立更新的图，而是该程序
可导出的、带 Evidence 引用的认知视图。这样可以保留 EFPS 的研究问题，同时直接获得 Tycho 的 hidden
state、历史回放验证、搜索规划、计划 guard、跨关迁移和选择性建模能力。

建议的默认策略是 Tycho 的 `orchestrator`：actor 是唯一环境行动者，并在认为值得时调用专门的
world-model builder。V2 的固定每步 Exploration/Main/Update 三连调用不进入 V3 主循环。

## 2. 阅读依据与版本边界

本方案核对了：

- Tycho 论文：[Tycho: Active Abstraction with Programmatic World Models for ARC-AGI-3](https://arxiv.org/abs/2607.28287)；
- 官方源码：[NIMI-research/Tycho](https://github.com/NIMI-research/Tycho)，审阅 commit
  `f68912a764372ead0a610db2e1c011d41ce5197e`；
- 当前仓库 V2 contract 3 的 EFPS models/graph、三个 Agent、prompt、runtime、正式实验与冻结转录。

Tycho 源码使用 Apache-2.0；后续 vendoring 必须保留其 LICENSE、作者信息、原始 commit、修改清单和
可复现配置。论文的结果不能直接当作本项目的结果：论文每个 policy 只有一次 public-25 随机运行，
其中 5 个游戏用于 harness 开发，而且强表现依赖很大的推理预算。论文自己也报告 Opus 5 的公开集运行
使用约 15.1k 次模型调用。V3 必须在相同模型、预算和游戏协议下重新做本地 A/B。

## 3. 为什么 V2 会陷入死胡同

### 3.1 表示不能被 planner 直接执行

当前 Schema 是自然语言 `Prototype -> Action -> Output` 三元组；Main 可以引用它作下一步预测，却不能
从一个明确的 hidden state 连续模拟多步动作。全局约束、目标、机制和策略又进入独立 `Insight`，所以
真正规划需要 LLM 每一步重新拼接 Schema、Insight、画面和历史。

### 3.2 更新、行动与探索被切成高成本的固定流水线

V2 每个动作通常固定执行 Exploration、Main、Update，并允许精确读取工具续轮。历史正式三游戏实验在
90 个动作上使用约 823 万 token，仍全部停在 Level 1。更多调用没有形成能被普通搜索算法使用的状态转移
系统。

### 3.3 graph 可以结构合法，但语义仍然错误

现有 validator 能检查 ID、edge type、Evidence closure 和 Schema 三元组完整性，但无法证明对象切分、
Prototype membership、动作输出或目标假设正确。错误认知会继续成为后续 prompt 的上下文，并诱发重复
探索和冲突 Schema。

### 3.4 画面被逐帧重新解释，hidden state 不稳定

V2 虽然追踪 Entity，却依赖 Update 对 before/after 图片重新描述。工具选择、选中色、计数器、camera
offset、敌人 phase 等不可见状态没有一个被 transition 持续推进的强制载体。

### 3.5 目标、动力学和计划没有独立验收

Tycho 的 SK48 失败案例说明：transition 可以准确，但错误 `outcome` 会让搜索系统高效地走向错误目标。
V2 目前也缺少三个独立判据：动力学 replay、terminal outcome、从当前 threaded state 出发的 validated
plan。

## 4. V3 总体架构

```text
ARC public engine
    |
    v
Tycho typed evidence workspace
decision / transient / completion / fatal / reset / next-level frames
    |
    +------------------------------+
    |                              |
    v                              v
Actor (唯一行动者)          World-model builder（actor 按需调用）
直接推理 / 探索 / 采用计划       编辑同一个 world_model.py
    |                              |
    +--------------+---------------+
                   v
         Executable World Model
         init_state(grid0, level)
         transition(state, action)
         render(state)
         outcome(state)
         optional actions/subgoals/heuristic/planner_key
                   |
          +--------+---------+
          |                  |
          v                  v
   replay verifier       BFS/A*/custom planner
   dynamics/outcome      validated plan + frame hashes
          |
          v
   EFPS audit view（由同一程序导出，不反向形成第二套动力学）
   Entity -> Prototype membership
   Prototype -> Action -> Output executable rule
```

四条硬边界：

1. 只有 actor 可以提交环境动作；builder 只编辑模型并返回建议。
2. observed Evidence 与 simulated state 严格分开；模拟 rollout 永远不能写回真实交互记录。
3. `world_model.py` 是动力学唯一真相；EFPS manifest 只能从其源码/运行时 introspection 生成。
4. verifier 通过只表示“已观察轨迹上一致”，不表示未知分支正确；每次真实行动后仍重新观察和检查。

## 5. EFPS 如何进入 Tycho 世界模型

### 5.1 精简认知定义

V3 不再保留 V2 的通用 typed property graph。研究对象收敛为：

```text
EntityInstance
  entity_id
  prototype_ids
  state fields / geometry
  supporting Evidence IDs

PrototypeDef
  prototype_id
  reusable matcher or parser role
  descriptive metadata
  supporting Evidence IDs

SchemaRule
  schema_id
  prototype_id
  public action pattern
  observable output description
  executable handler / transition branch
  support and counter Evidence IDs
```

Feature 不再是不断增长的独立节点表；真正影响分类和转移的 Feature 进入 Prototype matcher、Entity state
field 或程序谓词。原来的 Insight 不再作为第二个全局图：

- 动力学约束进入 `transition`；
- 通关/死亡条件进入 `outcome`；
- 规划分解进入 `subgoals`、`heuristic` 或 `planner.py`；
- 尚未证实的竞争假设进入 `notes/world_model.md`，不能冒充已执行 Schema。

### 5.2 一个来源、两个视图

新增一个很薄的 EFPS runtime/annotation library，供 agent-authored world model 选择使用：

- `@prototype(...)` 注册分类器、说明和 Evidence；
- `@schema_rule(...)` 注册 `Prototype + Action + Output` 与实际 handler；
- `classify(state)` 导出当前 Entity memberships；
- `applied_schema_ids(state)` 暴露最后一次 transition 真正触发的规则；
- `export_efps()` 生成只读 audit manifest。

禁止维护一个可与程序独立修改的 `graph.json`。如果 rule 只出现在 manifest、却没有对应 handler 或
transition branch，validator 必须拒绝它。反过来，允许 world model 使用普通 Python 分支处理暂时无法
自然 EFPS 化的机制，但 audit 要把这些 transition 标为 `unattributed`，不能假装已有 Schema。

### 5.3 保留 Tycho 的自由表示能力

Tycho 的重要设计是“不预先强制 object-centric decomposition”。有些游戏更像 UI、cellular automaton、
counter 或 finite-state controller。V3 因此把 EFPS 作为显式认知 profile，而不是强制所有内部 state
只能是对象列表：

- 对象型游戏：Entity/Prototype/Schema rule 可以直接驱动 transition；
- UI/规则系统：panel、cursor、tool、mode 等可以作为认知 Entity/Prototype；
- 若强行 EFPS 会降低可执行模型质量，builder 必须记录 `efps_applicability=partial|not_applicable` 和原因，
  主模型仍可使用适合该游戏的 state representation。

这不是放弃 EFPS，而是把“何时这种抽象值得使用”纳入 Tycho 所说的 active abstraction，并支持后续
比较纯 Tycho 与 Tycho+EFPS 的收益和失败面。

### 5.4 Evidence grounding

Tycho 当前以 `level_L/turn_NNN`、terminal、death、animation、attempt archive 保存 typed Evidence。
V3 在 harness 侧为每条真实 Evidence 生成稳定 ID，至少绑定：run、game、level、attempt、turn、frame
hash、公开 action 和 Evidence role。新增 `wmlib.evidence_refs()` 供 builder 精确引用。

每个 Prototype 和 Schema rule 必须引用至少一个已存在的真实 Evidence ID；counterevidence 同样引用
真实记录。初始模板和未证实假设不能携带伪造 Evidence。生成的 EFPS manifest 要通过 closure 检查并
绑定 `world_model.py` SHA-256 与 workspace version。

## 6. 直接采用 Tycho 的部分

以下部分原则上按上游结构采用，而不是从 V2 重新实现：

- typed interaction history：decision、transient animation、completion terminal、fatal terminal、reset、
  next-level initialization 分离；
- 每游戏持久 workspace、跨关 notes/program、reset attempt archive；
- `init_state / transition / render / outcome` 世界模型合同；
- `UNKNOWN=-1` 的局部 abstention、prediction coverage 和 bounded observation variants；
- 从 level 初始状态按真实动作推进的 threaded hidden state；
- dynamics verifier、outcome verifier、first-divergence 定位；
- BFS、A*、subgoal、custom planner 与 `planner_key`；
- canonical replay 后才写出的 validated plan；
- world-model source hash、起点 frame hash、每步 predicted frame hash guard；
- actor-controlled builder、level-boundary consolidation、builder 独立短上下文；
- sandbox、resume journal、workspace causal versioning、viewer 和 scoring/replay 工具。

采用时先做 upstream parity，不在第一阶段同时重构命名、日志和行为。

## 7. V2 资产的保留、迁移与退役

| V2 资产 | V3 决策 | 原因 |
|---|---|---|
| `socialclaw/v2` 全目录 | 冻结保留，不作为 V3 import 依赖 | 保证旧结果可审计，避免混合合同 |
| Main Agent 概念 | 迁移为 Tycho actor | 仍是唯一 orchestrator/planner/actor |
| Exploration Agent | 不进入默认主循环 | 探索由 actor 选择；以后仅作可选 ablation/helper |
| Update Agent | 由 world-model builder 替代 | 更新结果必须成为可执行、可 replay 的程序 |
| EFPSGraph/Relation/Feature history | 不迁移到在线主状态 | 它是双重真相和 prompt 膨胀的主要来源 |
| Prototype 与三元组 Schema 语义 | 迁移到 executable EFPS annotations/rules | 保留项目核心研究问题 |
| Insight | 分流到 transition/outcome/subgoals/notes | 避免无边界全局文本库 |
| `read_cognition` | 不迁移 | actor/builder 使用 workspace 文件、wmlib 和 Python 精确查询 |
| ARC trajectory/grid/PNG assets | 保留作回归和交叉检查 | 不在 Tycho 主循环中重复写第二份 canonical log |
| 当前 ARC wrapper | 保留作 fixture/oracle | V3 正式运行使用 Tycho harness 的 terminal/reset/animation 语义 |
| model provider/frozen transcript 经验 | 复用设计，另写 Tycho transport adapter | 不能直接复用 V2 structured-call contract |
| process/token reporting | 保留需要的审计指标 | viewer/workspace 是主审查入口，避免重复 artifact |
| Gold Schema | 保留为隔离的离线分析 | 严禁进入 actor、builder、world model 或 verifier |

旧 V2 graph 可提供一个只读转换器，帮助人类查看“哪些 Prototype/Schema 概念值得迁移”，但正式 V3
零先验实验不得自动把旧游戏知识注入 workspace。

## 8. 代码布局建议

```text
tycho/                    # canonical pinned package; preserves upstream absolute imports

third_party/tycho/
  LICENSE
  UPSTREAM.md              # repo、commit、导入日期、patch 清单
  PUBLIC_RELEASE_MANIFEST.json
  tests/                   # byte-identical upstream parity suite
  tycho -> ../../tycho     # 仅为上游测试保留原 repository-relative layout

socialclaw/v3/
  efps_runtime.py          # executable Prototype/Schema registration
  efps_evidence.py         # stable Evidence IDs and closure
  efps_export.py           # model -> audit manifest
  hooks.py                 # 对 Tycho 的窄扩展点
  provider_adapter.py      # 如需 OpenRouter，适配 Tycho tool protocol
  reporting.py             # 本项目新增指标，不复制 viewer

configs/v3/
  upstream_parity/
  tycho_efps/

tests/v3/
  test_upstream_contract.py
  test_efps_runtime.py
  test_evidence_closure.py
  test_model_verifier.py
  test_guarded_plan.py
  test_no_privileged_reads.py

scripts/run_v3_arc.py
```

导入策略采用 pinned source snapshot，而不是运行时从 GitHub 拉取；正式代码和实验不能依赖网络获取
上游。对上游文件的修改集中在明确 hook/patch，避免不可追踪的大规模 transplant。

## 9. 分阶段实施计划

### Phase 0：冻结边界与 upstream parity

目标：证明仓库内的 Tycho snapshot 与官方 contract 一致。

1. vendor 上述 pinned commit，保留许可证和 provenance；
2. 建立独立 Python 3.12 环境并对齐 `arc-agi==0.9.9` 等关键版本；
3. 原样运行 Tycho credential-free tests 和最小 fake-game harness；
4. 固定 upstream orchestrator 配置、prompt hash、tool schema hash 和 smoke manifest；
5. 不接入 EFPS，不发真实模型请求。

验收：上游测试全过；一个 fake game 能记录 typed Evidence、编辑/验证模型、规划、guard 和 resume；
snapshot 与 patch manifest 可复核。

### Phase 1：接入仓库而不改变 Tycho policy

目标：增加 `sc-run-arc-v3` 和项目级配置/结果入口，同时保持 upstream 行为。

1. 接入现有 25-game inventory 和 offline engine 路径；
2. 解决当前 `arc-agi>=0.9.8` 与 Tycho `0.9.9` 的版本固定；
3. 增加本项目 model provider adapter，但不改变 actor/builder prompt；
4. 把 run manifest、source fingerprint、token/cost 和 resume 状态纳入统一审计；
5. 用 mock/recorded model 验证确定性结果，不复用 contract-2/3 的旧 V2 响应。

验收：同一 fake/recorded 输入下，vendor Tycho CLI 与 `sc-run-arc-v3` 的动作、workspace、verifier 和
summary 一致。

### Phase 2：EFPS executable layer

目标：加入 EFPS，但不破坏 world-model 自由表示和 planner。

1. 实现 Prototype、Entity membership、Schema rule registry；
2. 实现 stable Evidence ID 和 `wmlib.evidence_refs()`；
3. 实现 rule execution attribution、counterevidence 和 EFPS manifest；
4. 给 verifier 增加 Evidence closure、unknown reference、重复/冲突 triple、规则未执行等检查；
5. 建立三类 deterministic fixtures：对象移动、UI toggle、hidden counter。

验收：每条导出的 Schema 都由真实 Evidence 支持，并与实际 transition handler 相连；模拟 rollout 不会
产生 Evidence；普通非 EFPS world model 仍可运行并明确报告 attribution coverage 为 0/partial。

### Phase 3：Tycho+EFPS actor/builder policy

目标：让 EFPS 成为有用的 inductive bias，而不是额外表格。

1. actor beliefs 只保留已证实 Prototype/Schema、竞争目标和待区分假设；
2. builder 优先检查已有 Prototype/rule 是否能同化新 Evidence，再做最小 accommodation；
3. builder 修改模型后同时得到 dynamics、outcome、plan、EFPS 四类反馈；
4. 只允许 actor 按需调用 builder；删除默认每步 Explore/Update；
5. level boundary consolidation 持久化共享 mechanics，清理关卡局部 Entity，不清理 Prototype/rules；
6. 对 `efps_applicability` 和未归因 transition 明确审计。

验收：模型修订能在 first divergence 上提高 replay；跨关能复用 mechanics；错误 outcome 不会因为
dynamics 通过而被隐藏；actor 可选择 bypass model。

### Phase 4：规划与死胡同防护

目标：把“少走弯路”变成可测试机制。

1. planner 从 current threaded state 搜索，不从最新 grid 重新初始化；
2. `game_over` 状态剪枝，RESET 属于 outer protocol，不学习成 Schema；
3. validated plan 必须绑定 model hash、起点 frame hash 和逐步 predicted hashes；
4. 每次只提交一个动作；真实 frame 不匹配立即失效计划并定位 divergence；
5. 增加 repeated state-action、oscillation、无新信息 probe、连续 no-plan 的诊断；
6. no-plan 必须区分：目标未知、actions 不完整、搜索预算不足、reachable graph 穷尽、模型报错。

验收：测试覆盖 stale plan、world-model edit 后失效、click action focus、hidden state threading、GAME_OVER
剪枝、RESET attempt archive 和跨关边界。

### Phase 5：实验阶梯

所有真实付费调用另行获得确认后再执行：

1. credential-free 全测试；
2. 一个模型 transport 的单调用 smoke；
3. CD82 Level 1 bounded smoke；
4. CD82/SK48/TU93 Level 1 matched trial；
5. 仅当前四级门通过后运行三游戏全关；
6. 最后才考虑 public-25。

必须至少比较：

- upstream Tycho orchestrator；
- Tycho+EFPS orchestrator；
- 可选 no-world-model，用于估计 durable Evidence 本身的贡献；
- V2 旧结果只作历史背景，除非重新用相同模型、预算、SDK 和计分协议运行，否则不进入同一统计表。

每个配置先做多 seed/多 trial 的小范围重复，再扩游戏数；不能用单次成功替代可靠性。

## 10. 指标与验收门

### 游戏表现

- levels completed、games won、RHAE、每关 scored actions；
- 第一次过关前探索动作、reset 数、GAME_OVER 数；
- 重复 `(state hash, action)`、reversible oscillation、no-op 重复率。

### 世界模型

- initial render accuracy；
- accepted/strict transition match、known-cell accuracy、prediction coverage；
- level-complete/game-over recall 与 false-positive rate；
- first divergence 类型和修复后 recovery；
- plan found/validated/followed/diverged，搜索节点和 plan 长度。

### EFPS

- Prototype/Schema Evidence closure；
- Entity membership churn 与跨关 Prototype reuse；
- transition 的 Schema attribution coverage；
- 相同 Prototype+Action 的冲突 Output 数；
- create/support/revise/counter 的数量和 Evidence；
- EFPS profile 被采用、部分采用或 bypass 的游戏比例。

### 推理成本

- actor/builder 调用数、input/output/cache token、费用、latency；
- builder 每次修订带来的 verifier 增量和后续动作收益；
- 单关 peak context 与 emergency compaction；
- 环境动作效率与 inference cost 分开报告。

三个必须独立通过的门：

1. dynamics gate：模型在相关当前关 prefix 上可执行且 replay 达标；
2. outcome gate：目标/死亡分类有 terminal Evidence，或明确标记为未证实；
3. plan gate：候选路线经 canonical replay 到 `level_complete`，并由 hashes guard。

不能用 dynamics accuracy 替代 outcome 或 plan gate。

## 11. 第一轮开发切片

第一轮只做到 Phase 0–2，不进行真实付费游戏实验。建议 PR/提交边界：

1. `vendor-tycho`: pinned upstream、license、provenance、原样 tests；
2. `v3-parity-runner`: CLI/config/provider/mock replay，行为 parity；
3. `v3-evidence-ids`: typed Evidence 的稳定 ID 与 closure；
4. `v3-efps-runtime`: executable Prototype/Schema rules 与 manifest；
5. `v3-verifier-tests`: dynamics/outcome/plan/EFPS 四通道 fixture。

到此进行人工 review：确认 EFPS 不是第二套 world model、上游 patch 足够窄、所有 Schema 都能回到真实
Evidence，再开始 Phase 3 prompt/policy 和真实 CD82 smoke。

## 12. 暂不做的事情

- 不修补 V2 graph 以充当 V3 planner；
- 不把 V1 layered Schema、Gold Schema 或游戏源码输入 V3 Agent；
- 不从旧三游戏运行自动 warm-start 新 world model；
- 不在第一阶段新增第四个 exploration/update agent；
- 不把模型 replay 一致性写成“已发现真实规则”；
- 不为追求 Tycho 论文分数直接运行昂贵 public-25；
- 不在未完成 upstream parity 前同时重写 harness、viewer、planner 和 provider。

## 13. 主要风险

1. **EFPS 过强约束表示。** 用可选 profile、明确 bypass 和 A/B 衡量，而不是强制所有游戏对象化。
2. **目标错误被高效 planner 放大。** outcome gate 与 dynamics gate 分开，terminal 未观察时保留竞争假设。
3. **程序对已见轨迹过拟合。** 跨关共享 transition、禁止无理由按 level 写 dynamics 分支，并报告未见分支风险。
4. **UNKNOWN 逃避验证。** 单独报告 coverage，设置非零/最低 coverage gate，不把 vacuous match 当成功。
5. **上游漂移。** 固定 commit、依赖、prompt/tool hashes 和 patch manifest；升级另做一次 parity。
6. **成本继续过高。** actor 决定何时 builder，默认不每步构建；动作成本和推理成本分别设硬上限。
7. **双日志和双 replay 冲突。** Tycho workspace/harness 是 V3 canonical record，V2 trajectory 只做回归导出。

最终成功标准不是“V3 中出现 EFPS 文件”，而是：有限真实 Evidence 被压缩成一个可执行、可反证、可搜索
且能跨关复用的模型；EFPS 能解释其中的对象同类性和动作效果规则，并在 matched trial 中提高通过率、
动作效率、可靠性或可审计性，而不显著恶化推理成本。
