# Architecture

## Version 2 EFPS vertical slice

当前开发主线位于 `socialclaw.v2`，只面向 ARC-AGI-3。它不修改或导入 V1 的
`SchemaManager`/`LayeredSchemaGraph`，而是在真实环境循环中维护一个 typed
Entity–Feature–Prototype–Schema graph 和独立的全局 Insight/Rule 记忆：

```text
raw public image + public action contracts + read-only EFPS/Insights
                    |                         |
                    v                         v
          exploration model child      main model Agent
                    \                    /  (only actor)
                     -> selected action
                              |
                     public environment
                              |
                 raw before/action/after evidence
                              |
                    update model child
                              |
              validated atomic EFPS transaction
```

主 Agent 是唯一动作决策者；探索子 Agent只返回一段给 Main 的探索建议文本，更新子 Agent只提出
同化/顺应 graph operations。确定性 validator 检查类型、引用、evidence closure 和完整 transaction，
任何 operation 失败都不修改当前图。所有 evidence 指向 durable trajectory observation/action/
result 和内容寻址 grid/PNG，并可通过 `EFPSGraph.resolve_evidence` 由 ID 取回完整公开记录。每次
action 后，Update 还必须将像素差分归属为 Entity 级变化；无法归属时显式记录为 unassigned，而不把
cell 数量当作语义。Schema 的语义核心严格为一个 `Prototype → Action → Output` 三元组，validator
要求一个已存 Prototype、一个公开 action pattern 和一个非空可观察 output；适用范围、障碍规律、
通关条件、跨动作机制或策略不能再混入 Schema 字段，而是进入独立的 Evidence-grounded Insight。
Insight 可被创建、支持、反证和修订，并可由 Main 以 `insight` 决策模式直接使用。V2 没有确定性语义
perception：第一次 Entity、Feature、Prototype 和 Insight 也由 Update 模型从公开画面/transition
提出；没有 transition 时 validator 仍禁止创建动作 Schema。

同一 runtime 接受任意完整 ARC game ID，但 game ID 只用于评测 harness 创建环境，不进入 Agent
认知 payload。V2 不导入 Gold loader、游戏源码、专用 policy、坐标、goal mask、environment
fingerprint 或路线。环境实现当然必须在 gateway 背后执行游戏，Agent只能访问公开 observation/action。
timeline 逐步记录共享输入收据、实际附加图片、三个 Agent输出、动作结果和图事务；runtime 同时从
它生成按 Step 展开的 `process.md`，供人直接核对触发、输入、输出和前后 PNG。只保存最终图，
不保存每次 revision snapshot。第一个真实测试在 CD82 Level 1 的 20 步预算内未过关，这一失败用于
暴露通用认知循环的真实瓶颈，而不是由脚本补成成功。V1 已冻结在 `archive/version1_20260824/`；下文描述
的三 benchmark/layered Schema 均为 V1 架构。

Agent 视觉和人类审查视觉是两个 artifact：`agent_view` 是公开 64×64 frame 最近邻放大到
512×512 的无辅助线 RGB 映射，`review_view` 则是同尺寸、每 8 cell 加线的定位图。模型调用只能
选 `agent_view`。`logical_grid_sha256` 仅用于持久完整性/去重，不进入认知 prompt。默认模型输入是
Markdown 描述：当前公开状态、动作合同、最近 3 条 Entity 级 transition、全部 Entity/Prototype/
Schema/Insight 的紧凑目录；不发送整图 JSON、完整 Evidence 历史或重复 artifact 元数据。Main、Exploration、
Update 均可调用只读 `read_cognition(command, id, feature_id?)`，以固定命令和精确 ID 读取 Feature
历史、typed Relation、Schema/Insight 证据链、公开 Evidence 和 agent-visible PNG。该工具不做自然语言
检索、排名、总结或二次 LLM 推理。历史 Evidence 图像不再自动附加到每次请求。在线 runner 每个完整 cognitive step
原子覆盖 partial timeline/graph/process checkpoint，成功完成后清理；外部 provider 中断时保留最后
完整步骤，避免昂贵模型输出只留在内存中。最终 token 报告区分逻辑 Agent 调用、工具续轮和纠错重试，
逐 provider request 保存精确 usage，并将 section 字符占比明确标为输入组成代理而非 token 归因。
FeatureDefinition 只定义共享名称/类型；实体实例的可读描述持久化在 FeatureAssertion，避免共享
`color`、`orientation` 等定义时发生跨 Entity 描述污染。

V2 多关循环以公开 `levels_completed/level_delta` 判断边界。当 `level_delta > 0` 时，更新 phase 为
`public_level_boundary`：上一关完成是动作的终止效果，after 画面作为下一关的新当前场景读取；旧关卡
中未被明确识别为持续存在的 active Entity 会以同一 Evidence 标记为 disappeared。可复用 Prototype、
Schema、Insight、Feature 历史和 Evidence 不清空，因此 Main 下一步在新画面上继续使用同一认知库。
`stop_after_levels` 是累计关数，`max_steps` 是每关独立动作上限；只有公开关卡完成才为下一关
建立新的完整预算。任何游戏进入公开 `GAME_OVER` 后，runtime 默认重开当前关，写入非 Agent action
的 `ENV_RESET` 和 Update-only 场景重对齐，但不会退回本关已消耗的 action。可显式关闭恢复；
Level 通过率与本关超时仍按原预算统计。

模型逻辑调用可以由 `RecordedVisionModel` 冻结回放。它在每次返回响应前校验当前
instructions、文本 payload 和图片序列哈希，然后重新执行环境、EFPS transaction、validator、
trajectory replay 和报告生成。该模式用于字节级 artifact 复现，不计作新的在线模型 trial；
2026-08-30 的冻结包使用重构前角色型 Schema，当前三元组 Schema/Insight 合同会被哈希校验拒绝，
不能将旧响应重放后冒充新实现实验。新合同完成在线实验后应建立新的冻结包。

## Separation of concerns

项目把 benchmark、method 和 experiment protocol 分开：

1. `BenchmarkAdapter` 只负责加载样本、构造任务输入和判分。
2. `MethodController` 只负责静态实验的方法状态、检索和 binary feedback 更新。
3. `ExperimentConfig` 固定模型、数据 split、预算、feedback 权限和输出位置。
4. Runner 负责循环、LLM 调用和标准化 artifact。

这样模型配置、答案解析、采样和 retry 不再散落在每个 baseline 脚本中。

## Trajectory foundation

`socialclaw.trajectory` 定义了独立于 benchmark 和 actor 的单步/多步轨迹合同。确定性脚本、
coverage explorer、冻结 replay 和未来真实 Agent 都通过
`EpisodeStarted -> StepObserved -> EpisodeFinished` 事件流表达经历；ARC 一关是多 step，静态任务
是相同合同下的一步 episode。

`TrajectoryRecorder` 每步原子保存 JSON，并用内容寻址资产引用无损 grid 和 Agent-view PNG。
通用层、三个 ARC 示例 corpus、trajectory-to-Memory 投影和离线窗口归纳当前已经实现和测试，但
尚未接入正式 runner；现有 ARC `schema` loop 仍使用原来的直接 `MemoryRecord` 写入路径。Phase D
通过 benchmark profiler 选择视觉关键帧，再以统一 proposal/validator 合同更新 SchemaGraph；重复
step 只增加 evidence，不膨胀节点。合同与审查方式见
[通用任务轨迹合同](trajectory_contract.md)和
[ARC 可靠轨迹语料](arc_trajectory_corpus.md)及
[Window/keyframe Schema 归纳](window_schema_induction.md)。

## Static benchmarks

`socialclaw.run_static` 驱动 ContextMATH 与 IntPhys2：

```text
dataset -> BenchmarkSample -> method context -> LLM -> adapter.evaluate
                                      ^                    |
                                      |--- binary feedback-|
```

Methods 永远拿不到 `gold`；其更新接口只有任务、模型响应和 `correct: bool`。

## ARC-AGI-3

ARC 是交互式环境，因此保留专用 loop，但由 `socialclaw.run_arc` 统一分发：

- `naive/icl/rag/withrule` 使用同一个 prompt baseline loop。
- `reflexion/expel/amem/tgm` 使用同一个 online-memory loop。
- `schema` 使用当前 schema-aware loop。

统一入口负责给三类 loop 传入相同的模型、step budget、token budget 和输出根目录，并在运行目录写入相同的 `manifest.json`。

## Schema lifecycle across benchmarks

三个 benchmark 现在共享同一套 layered Schema 生命周期：

- ContextMATH：题目与模型回答形成 episode，binary correctness 更新 Schema；
- IntPhys2：视频任务及不含标签的 condition/camera/scene metadata 参与检索，
  binary correctness 更新 Schema；
- ARC-AGI-3：每个 observation/action/environment-result transition 立即写入
  Memory 并归纳 action-effect Schema，关卡 WIN/GAME_OVER/TIMEOUT 再更新本关使用过的 Schema。

统一架构包括 `MemoryRecord`、`SchemaNode`、自动生成/融合、反馈、mask、
遗忘、去重、持久化和可替换 LLM/embedding 接口。完整设计见
[Memory-grounded layered Schema architecture](schema_architecture.md)。

Learned-vs-Gold 比较通过结束后独立 CLI 执行；Gold loader 不进入上述 runner/induction import 路径，
alignment 也不写回 learned state。当前三游戏结构化 proxy 结果与边界见
[Learned Schema vs Gold Schema 评测](schema_evaluation.md)。

旧 `Concept`/`Relation` 模块已经移到
`archive/code/legacy_schema/`，不再随 `socialclaw` 包安装。当前三个正式
runner 的 `schema` method 只使用分层架构。
