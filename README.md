# SocialLearningClaw

SocialLearningClaw 是一个研究 Agent Schema Learning 的实验仓库。当前统一评测三个 benchmark：

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
  methods/               静态 benchmark 的方法生命周期控制器
  schema/                分层 Schema、自动归纳、检索、反馈和维护
  experiment.py          Protocol, budget, manifest, result types
  run_static.py          ContextMATH / IntPhys2 entry point
  run_arc.py             ARC-AGI-3 entry point
configs/arc_agi3/        Fixed ICL examples and human-written rule baselines
scripts/                 Batch execution and ARC summarization
tests/                   Offline unit tests
docs/                    Active documentation
  archive/                 Historical results and retired source code
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

详细说明：

- [Architecture](docs/architecture.md)
- [Benchmarks](docs/benchmarks.md)
- [Baselines](docs/baselines.md)
- [Experiment protocol](docs/experiment_protocol.md)
- [Result format](docs/results_format.md)
- [Memory-grounded layered Schema](docs/schema_architecture.md)
- [Gold Schema 构建方案](docs/gold_schema_generation.md)
- [ARC-AGI-3 Gold Schema v1](gold/arc_agi3/v1/README.md)
- [ContextMATH Gold Schema 第一批审核稿](gold/contextmath/v1/README.md)
- [IntPhys2 Gold Schema 四类物理规则 pilot](gold/intphys2/v1/README.md)
- [Historical baseline smoke results](docs/baseline_smoke_results.md)
