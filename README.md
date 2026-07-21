# SocialLearningClaw

SocialLearningClaw 是一个研究 Agent Schema Learning 的实验仓库。当前统一评测三个 benchmark：

- **ARC-AGI-3**：多步、交互式抽象推理。
- **ContextMATH**：带有上下文扰动的数学推理。
- **IntPhys2**：视频中的物理合理性判断。

项目比较八个 baseline 和一个研究方法：

`naive`、`icl`、`rag`、`withrule`、`reflexion`、`expel`、`amem`、`tgm`、`schema`。

`schema` 当前只接入 ARC-AGI-3。它仍是早期的单层 Concept/Relation 实现；新的分层 Schema 与 Agent 架构见 [Schema redesign notes](docs/schema_redesign_notes.md)，将在后续单独重写。

## 项目结构

```text
socialclaw/
  agent/                 OpenAI-compatible ARC agent
  benchmarks/            ContextMATH / IntPhys2 adapters
  dataset/               ARC-AGI-3 environment wrapper and shared types
  memory_agents/         Reflexion / ExPeL / A-MEM / TGM
  methods/               Unified baseline lifecycle
  schema/                Current schema graph implementation
  experiment.py          Protocol, budget, manifest, result types
  run_static.py          ContextMATH / IntPhys2 entry point
  run_arc.py             ARC-AGI-3 entry point
configs/arc_agi3/        Fixed ICL examples and human-written rule baselines
scripts/                 Batch execution and ARC summarization
tests/                   Offline unit tests
docs/                    Active documentation
archive/                 Historical benchmark-selection results
third_party/             Downloaded ARC-AGI-3 game environments
data/                    Local datasets; ignored by Git
outputs/                 Generated experiments; ignored by Git
```

旧的 CL-bench、PBench、Cosmos-Reason1、ARC-1/2 数据和历史运行均保存在 `data/legacy/` 与 `outputs/legacy/`，不会进入新实验汇总。

## 安装

要求 Python 3.11+。

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
  --split debug \
  --model anthropic/claude-opus-4.8 \
  --max-samples 17
```

把 `--method` 替换为任一 baseline 即可。默认每题一次 attempt；如果研究 retry，所有对比方法必须显式使用相同的 `--max-attempts`。
IntPhys2 默认预留开头 3 个视频给 ICL，因此当前 20 个本地视频最多评测其余 17 个；所有方法都会排除同一保留集。

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
