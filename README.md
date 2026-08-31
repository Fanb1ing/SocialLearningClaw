# SocialLearningClaw

SocialLearningClaw 是一个研究 Agent Schema Learning 的实验仓库。当前开发主线是 ARC-only
Version 2：以 Entity–Feature–Prototype–Schema（EFPS）typed graph 加全局 Insight/Rule 表达主体认知，主 Agent
同时负责 orchestrator、planning 和行动，并按需调用更新与探索子 Agent。

旧的三 benchmark layered-Schema 系统已冻结为 V1，完整回退快照位于
`archive/version1_20260824/`。V1 统一评测三个 benchmark：

- **ARC-AGI-3**：多步、交互式抽象推理。
- **ContextMATH**：带有上下文扰动的数学推理。
- **IntPhys2**：视频中的物理合理性判断。

项目比较八个 baseline 和一个研究方法：

`naive`、`icl`、`rag`、`withrule`、`reflexion`、`expel`、`amem`、`tgm`、`schema`。

`schema` 已接入全部三个 benchmark，并统一使用 memory-grounded 分层
Schema：具体任务轨迹先写入 Memory，再自动归纳、融合和更新 SchemaNode。
ARC runner 也已经迁移，不再使用早期的单层 Concept/Relation 图。

## 项目结构

```text
socialclaw/
  agent/                 OpenAI-compatible ARC agent
  arc_methods/           ARC 交互环境专用的 prompt/memory baseline loops
  benchmarks/            ContextMATH / IntPhys2 adapters
  dataset/               ARC-AGI-3 environment wrapper and shared types
  memory_agents/         Reflexion / ExPeL / A-MEM / TGM
  memory/                新架构的 episode/knowledge/skill 记忆与持久化
  trajectory/            通用单步/多步任务轨迹合同、source 和原子 recorder
  methods/               静态 benchmark 的方法生命周期控制器
  schema/                分层 Schema、自动归纳、检索、反馈和维护
  v2/                    通用视觉模型 Agent、EFPS graph、更新/探索子 Agent与 runtime
  experiment.py          Protocol, budget, manifest, result types
  run_static.py          ContextMATH / IntPhys2 entry point
  run_arc.py             ARC-AGI-3 entry point
  run_arc_v2.py          任意 ARC game ID 的 V2 通用在线认知入口
configs/arc_agi3/        Fixed ICL examples and human-written rule baselines
scripts/                 Batch execution and ARC summarization
tests/                   Offline unit tests
docs/                    Active documentation
archive/                 Frozen V1 snapshot and retired source code
third_party/             Downloaded ARC-AGI-3 game environments
data/                    Local datasets; ignored by Git
outputs/                 Generated experiments; ignored by Git
```

旧的 CL-bench、PBench、Cosmos-Reason1、ARC-1/2 数据和历史运行均保存在 `data/legacy/` 与 `outputs/legacy/`，不会进入新实验汇总。

### `methods/`、`archive/`、`tests/` 分别做什么

- `socialclaw/methods/` 是静态实验的“方法生命周期层”。`run_static` 把当前任务
  交给这里的 controller，统一完成上下文检索和 binary feedback 更新。它不负责
  加载数据、判分或直接操作 ARC 环境。ARC 因为是多步交互环境，baseline loop
  放在 `arc_methods/`，layered Schema loop 放在 `arc_runner.py`，但遵守相同的
  feedback 与 artifact 协议。
  Reflexion、ExPeL、A-MEM、TGM 的具体数据结构在 `memory_agents/`，新的 Schema
  数据结构在 `memory/` 与 `schema/`，`methods/` 只协调它们的生命周期。
- `archive/` 保存历史 benchmark-selection 结果、来源说明和退役的单层 Schema
  源码；`docs/archive/` 保存旧设计、旧 CLI 和 handoff 文档。它们用于追溯，
  不会被当前 runner 导入，也不能和当前统一协议的结果自动合并。
- `tests/` 是不调用真实 API 的离线回归测试，覆盖 benchmark 解析、实验协议、
  Memory/Schema 数据结构、持久化、反馈、遗忘、合并和 ARC runner 的关键转换。
  修改代码后用它确认没有破坏现有行为。

## 安装

要求 Python 3.12+（当前 `arc-agi>=0.9.8` 的上游要求）。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

在 `.env` 中设置模型服务和 ARC-AGI-3 key：

```dotenv
OPENROUTER_API_KEY=...
ARC_AGI_API_KEY=...
```

## 运行 ContextMATH / IntPhys2

```bash
python -m socialclaw.run_static \
  --benchmark contextmath \
  --method naive \
  --split aime_2024_sg \
  --model anthropic/claude-opus-4.8 \
  --max-samples 10
```

```bash
python -m socialclaw.run_static \
  --benchmark intphys2 \
  --method rag \
  --split main_300 \
  --model anthropic/claude-opus-4.8 \
  --max-samples 17
```

把 `--method` 替换为任一 baseline 或 `schema` 即可。Schema 状态会写入当前
run 的 `schema/memory.json` 与 `schema/schema.json`。默认每题一次 attempt；如果研究 retry，所有对比方法必须显式使用相同的 `--max-attempts`。
IntPhys2 支持完整 Debug 60 视频与固定的 Main 300 视频样本。默认预留当前
split 的开头 3 个视频给 ICL；所有方法都会排除同一保留集。

首次准备 IntPhys2 数据：

```bash
.venv/bin/python scripts/prepare_intphys2_data.py \
  --main-samples 300 --seed 20260724 --workers 8
```

脚本默认固定 Hugging Face revision，并生成 `sample_300_manifest.json`；runner
会拒绝缺少 manifest 或视频不完整的 `main_300`。

## 运行 ARC-AGI-3

### Version 2：无游戏规则的通用视觉 Agent

V2 从空 EFPS/Insight 认知库启动。Main 和 Update 使用结构化视觉输出，Exploration 只返回一段给 Main 的文本
建议；认知 Agent只接收
原始公开画面、公开状态、SDK 动作参数合同、公开转移差异和只读 EFPS，不接收 game ID 对应的
规则、目标、对象标签、专用坐标、goal mask、Gold、源码或预制路线：

```bash
.venv/bin/python -m socialclaw.run_arc_v2 \
  --game-id cd82-fb555c5d \
  --model anthropic/claude-opus-4.8 \
  --output-dir outputs/review/my_cd82_run \
  --max-step 30 --stop-after-levels all --compact-process
```

每次运行生成按时刻展开的 `process.md`，
直接列出触发、Agent输入/输出、动作前后 PNG、公开 transition 和 EFPS 图增量；`timeline.json` 作为
机器审计源，另保留最终 graph 和 replay 所需 trajectory/grid/PNG。不再保存重复
summary/evidence/manifest 或每 revision snapshot。旧 Schema 合同下的历史真实结果见
[2026-08-30 三游戏实验](experiments/v2_formal_20260830/README.md)；它不能代表当前三元组 Schema/Insight 实现。早期 20 步原型审查已移至
[历史文档](docs/archive/v2_cd82_level1_prototype.md)。
最新实现与 token 对比见
[V2 认知输入精简与按需检索](docs/v2_cognition_retrieval_design.md)。

每个 action 后，Update 必须比较 before/after 图片，把公开像素差分解释为具体 Entity 的
`appeared/disappeared/moved/state_changed/feature_changed`；无法归属的变化必须显式保留为
`unassigned_visual_changes`。Schema 的语义严格是一个带证据的
`Prototype → Action → Output` 三元组；墙阻挡、通关条件、跨动作机制或策略等不适合该三元组的知识
保存为独立的全局 Insight/Rule。默认 prompt 只放最近 3 条 transition，以及所有 Entity、Prototype、
Schema、Insight 的紧凑自然语言目录；不再把整张图、完整 Evidence 历史或 artifact 元数据逐步重复发送。
三个 Agent 都可按需调用只读 `read_cognition(command, id, feature_id?)`。它只接受固定命令和精确
持久 ID，直接返回保存的节点（含 `get_insight`）、Feature 历史、Relation、Evidence 或 agent-visible artifact；不做自然
语言检索、相似度排序、摘要或二次 LLM 推理。`get_evidence` 返回动作、公开 result、Entity 变化和带
`before/after/current` phase 的 observation 引用；`get_artifact` 会把精确保存的公开 PNG 重新附给
Agent，且不会暴露仅供人类审查的辅助线图片或内部环境数组。
具体对象的 Feature 描述保存在 `FeatureAssertion`，避免共享 `color` 等 FeatureDefinition 时把一个
Entity 的实例描述误套到另一个 Entity。

视觉 artifact 分为两个严格角色：认知 Agent 只接收由公开 64×64 frame 最近邻放大到 512×512、无任何辅助线
的 `agent_view`；`process.md` 链接的是另存的 512×512 `review_view`，后者每 8 cell 画定位线，仅供
人类审查。历史 Evidence 图片不再自动塞进每次请求；Agent 需要历史细节时先使用精确认知读取，当前
调用仍只附当前决策或更新必需的公开图。长时间真实运行每完成一步会覆盖一个 `*.partial.*`
checkpoint；成功结束后自动删除，模型/API
中断时保留最后一个完整认知步骤并明确标记为非最终结果。

多关运行使用累计公开 `level_delta`：例如 `--stop-after-levels 6` 会在从本次 reset 起累计完成
6 关后停止；`--max-step` 是每一关独立的动作预算，通过一关后才为下一关重新计数。任何游戏发生公开
`GAME_OVER` 时默认重开当前关，但已经消耗的本关动作不会退回；可用 `--no-reset-on-game-over` 关闭。
发生公开 level boundary 时，Update 把上一关
完成本身作为终止动作效果，并把 after 图像作为下一关的新场景重新观察，不会把整幅换关画面误学成
该动作的普通视觉效果。未在新场景中重新识别的旧 Entity 会退出默认可见目录；Prototype、Schema、
Insight 和 Evidence 继续跨关保留。

运行结束还会写入 `token_usage.json` 与 `token_usage.md`：前者逐逻辑调用、逐 provider request 记录
精确 input/output token、图片和工具调用，后者给人类阅读 Agent/Step 分布。prompt 各 section 的字符数
也会记录，但不会冒充 provider token 的字段级精确归因。

### Version 1 runners

```bash
python -m socialclaw.run_arc \
  --method schema \
  --game-id sk48-d8078629 \
  --model google/gemini-2.5-pro \
  --max-steps 200 \
  --max-attempts 1
```

运行全部九种方法和三个默认游戏：

```bash
bash scripts/run_all_baselines.sh
```

仓库固定保存了当前 25 个 ARC 游戏版本，runner 使用 SDK offline mode；每个
run 的 dataset fingerprint 来自对应本地 metadata 与环境源码。刷新官方库存需
有效 ARC API key：`.venv/bin/python scripts/download_arc_games.py`。

汇总：

```bash
python scripts/eval_arc_summary.py \
  --runs-dir outputs \
  --model google/gemini-2.5-pro
```

ContextMATH/IntPhys2 汇总：

```bash
python scripts/summarize_static.py \
  --benchmark contextmath \
  --split aime_2024_sg \
  --model anthropic/claude-opus-4.8
```

## 验证

```bash
python -m unittest discover -s tests -v
```

`experiments/v2_formal_20260830/` 保留重构前角色型 Schema 的冻结历史输入和结果哈希。当前三元组
Schema/Insight 合同会使其输入哈希校验主动失败，避免把旧模型响应伪装成新实现结果；新合同需另开实验。

详细说明从 [文档索引](docs/README.md) 开始：

- [Version 2 EFPS 完整开发方案](docs/version2_efps_development_plan.md)
- [旧 Schema 合同下的 V2 三游戏实验](experiments/v2_formal_20260830/README.md)
- [V2 通用 Agent 的早期 CD82 Level 1 测试（历史）](docs/archive/v2_cd82_level1_prototype.md)
- [V2 认知输入精简与按需检索](docs/v2_cognition_retrieval_design.md)
- [V1 冻结归档与回退说明](archive/version1_20260824/README.md)
- [Architecture](docs/architecture.md)
- [Benchmarks](docs/benchmarks.md)
- [Baselines](docs/baselines.md)
- [Experiment protocol](docs/experiment_protocol.md)
- [Result format](docs/results_format.md)
- [Memory-grounded layered Schema](docs/schema_architecture.md)
- [通用任务轨迹合同](docs/trajectory_contract.md)
- [ARC 可靠轨迹语料（Phase B）](docs/arc_trajectory_corpus.md)
- [三游戏 Trajectory -> Memory -> Schema 原型](docs/trajectory_schema_prototype.md)
- [Window/keyframe Schema 归纳（Phase D）](docs/window_schema_induction.md)
- [Learned Schema vs Gold Schema 评测](docs/schema_evaluation.md)
- [本次开发总览：ARC 轨迹、视觉资产与 Memory](docs/session_trajectory_memory_summary.md)
- [本次开发总览：Schema 生成与 learned-vs-Gold 评测](docs/session_schema_evaluation_summary.md)
- [ARC learned Schema 全流程开发方案](docs/arc_learned_schema_pipeline_plan.md)
- [Gold Schema 构建方案](docs/gold_schema_generation.md)
- [Ground Truth Schema 本轮总结与下次升级指南](docs/ground_truth_schema_session_takeaways.md)
- [ARC-AGI-3 Gold Schema v1](gold/arc_agi3/v1/README.md)
- [ContextMATH Gold Schema 第一批审核稿](gold/contextmath/v1/README.md)
- [IntPhys2 Gold Schema 四类物理规则 pilot](gold/intphys2/v1/README.md)
- [Historical baseline smoke results](docs/baseline_smoke_results.md)
