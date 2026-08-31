# Baseline 简单测试结果

更新时间：2026-07-24。

这份报告重新盘点了仓库、Git 历史和本地 `outputs/legacy/`，汇总此前实际
运行过的 ContextMATH、IntPhys2 和 ARC-AGI-3 简单测试。它覆盖当前注册表中的
全部八个 baseline 名称，并补充了 2026-07-24 对 `icl`、`rag` 的
ContextMATH 和 IntPhys2 重测。表中只填写有原始 artifact 支持的结果；没有
找到的运行明确写为“未运行/未找到”，不推测、不补造。

## 完整性结论

| Baseline | ContextMATH | IntPhys2 | ARC-AGI-3 |
|---|---|---|---|
| `naive` | 有 | 有 | 有 |
| `icl` | **有（2026-07-24 补测）** | **有（2026-07-24 补测）** | 有 |
| `rag` | **有（2026-07-24 补测）** | **有（2026-07-24 补测）** | 有 |
| `withrule` | **未找到历史运行** | **未找到历史运行** | 有 |
| `reflexion` | 有 | 有 | 有 |
| `expel` | 有 | 有 | 有 |
| `amem` | 有 | 有 | 有 |
| `tgm` | 有 | 有 | 有 |

因此，原来的跨 benchmark 报告是
[`docs/archive/memory_baseline_summary.md`](archive/memory_baseline_summary.md)；
它完整记录的是当时定义的 `naive + reflexion + expel + amem + tgm` 实验，
不是后来扩展后的“八个 baseline × 三个 benchmark”完整矩阵。Git 历史中也只
找到这五种方法的静态 benchmark 结果文件。本次已补齐 `icl/rag`；
`withrule` 的 ContextMATH 和 IntPhys2 仍按要求暂不运行。

## 2026-07-24 ICL/RAG 补测协议

- 模型保持为历史实验使用的 `anthropic/claude-opus-4.8`，经 OpenRouter
  调用；温度为 0，binary feedback，单次尝试。
- ContextMATH 仍使用四个 AIME split、每个 split 前 10 题，单次调用最多
  8192 tokens。ICL 的 3 个演示来自独立的 `math_500_sg` split；RAG 使用
  `BAAI/bge-small-en-v1.5` 检索模型。
- IntPhys2 保持历史视频抽帧配置：每 1.5 秒一帧、最多 24 帧、单次调用最多
  16 tokens。由于本地没有独立演示集，前 3 个视频只保留作 ICL 演示；为使
  ICL 与 RAG 严格使用同一评测样本，两者均在剩余 17 个视频上测试。
- manifest 审计确认：四个 ContextMATH split 内 ICL/RAG 的题目 ID 分别
  一致；IntPhys2 的 17 个评测视频 ID 和 3 个保留演示视频 ID 在两种方法间
  一致，且演示与评测不重叠。

本次原始 manifest、逐样本记录和汇总保存在
[`outputs/smoke_retest_20260724/`](../outputs/smoke_retest_20260724/)。

## ContextMATH

模型为 Claude Opus 4.8。每个 split 取 10 题，共 40 题。表中单位为准确率
百分比。

| Baseline | AIME 2024 SG | AIME 2024 CS | AIME 2025 SG | AIME 2025 CS | Mean |
|---|---:|---:|---:|---:|---:|
| `naive` | 80.0 | 60.0 | 70.0 | 70.0 | 70.0 |
| `icl`| 90.0 | 80.0 | 70.0 | 70.0 | 77.5 |
| `rag` | 100.0 | 90.0 | 80.0 | 70.0 | 85.0 |
| `withrule` | — | — | — | — | - |
| `reflexion` first attempt | 90.0 | 100.0 | 80.0 | 80.0 | 87.5 |
| `reflexion` final, 最多 3 attempts | 100.0 | 100.0 | 90.0 | 90.0 | 95.0 |
| `expel` | 90.0 | 90.0 | 70.0 | 70.0 | 80.0 |
| `amem` | 90.0 | 100.0 | 90.0 | 80.0 | 90.0 |
| `tgm` | 90.0 | 90.0 | 70.0 | 70.0 | 80.0 |

原始结果位于
[`archive/results/benchmark_selection/contextmath/`](../archive/results/benchmark_selection/contextmath/)；
本次 ICL/RAG 结果位于
[`outputs/smoke_retest_20260724/contextmath/`](../outputs/smoke_retest_20260724/contextmath/)。
需要注意，Reflexion 得到了额外 retry，A-MEM 的旧实现还记录过含 gold 的
post-evaluation context，因此这些数字不能直接作为当前统一协议下的公平比较。

## IntPhys2

模型为 Claude Opus 4.8；Debug/Solidity/Fixed Camera，共 20 个视频。

| Baseline | Accuracy | Correct / Total |
|---|---:|---:|
| `naive` | 70.0 | 14 / 20 |
| `icl` | 58.8 | 10 / 17 |
| `rag`| 70.6 | 12 / 17 |
| `withrule` | — | - |
| `reflexion` | 65.0 | 13 / 20 |
| `expel` | 60.0 | 12 / 20 |
| `amem` | 55.0 | 11 / 20 |
| `tgm` | 65.0 | 13 / 20 |

原始结果位于
[`archive/results/benchmark_selection/intphys2/`](../archive/results/benchmark_selection/intphys2/)；
本次 ICL/RAG 结果位于
[`outputs/smoke_retest_20260724/intphys2/`](../outputs/smoke_retest_20260724/intphys2/)。
历史方法测试了全部 20 个视频，而 ICL/RAG 为了隔离 3 个演示视频，只评测
相同的剩余 17 个视频。因此补测的两行彼此可比，但不能与 20-video 历史行
视作完全相同的样本协议。

## ARC-AGI-3

ARC 的 prompt baseline 使用 Gemini 2.5 Pro；memory baseline 使用 Claude
Opus 4.8。下面汇总本地 artifact 中每种方法最后一组可识别运行。`x/y` 表示
完成关卡数/该次运行记录的关卡数，而不是完整游戏总关卡成功率。

| Baseline | CD82 | SK48 | TU93 |
|---|---|---|---|
| `naive` | 0/1, GAME_OVER, 100 steps | 0/1, TIMEOUT, 200 | 0/1, GAME_OVER, 50 |
| `icl` | 0/1, GAME_OVER, 100 | 1/2, 308 total | 0/1, GAME_OVER, 50 |
| `rag` | 0/1, GAME_OVER, 100 | 0/1, TIMEOUT, 200 | 0/1, GAME_OVER, 50 |
| `withrule` | 1/2, 105 total | 1/2, 222 total | 0/1, GAME_OVER, 50 |
| `reflexion` | 0/1, TIMEOUT, 50 | 0/1, TIMEOUT, 50 | 0/1, GAME_OVER, 50 |
| `expel` | 0/1, TIMEOUT, 50 | 0/1, TIMEOUT, 50 | 0/1, GAME_OVER, 50 |
| `amem` | 0/1, TIMEOUT, 50 | 0/1, TIMEOUT, 50 | 0/1, GAME_OVER, 50 |
| `tgm` | 0/1, TIMEOUT, 50 | 0/1, TIMEOUT, 50 | 0/1, GAME_OVER, 50 |

对应运行保存在
[`outputs/legacy/runs/`](../outputs/legacy/runs/)。其中 prompt baseline 使用
`arc_zero_shot`、`arc_few_shot`、`arc_rag`、`arc_withrule` 旧目录名；
memory baseline 使用 `arc_memory_*`。

## Schema 对照结果

`schema` 是研究方法，不属于八个 baseline。旧单层 Schema 的 Gemini 2.5 Pro
ARC 运行在 CD82、SK48、TU93 上均未完成第一关，分别记录 100、200、50 steps。
这不是当前 `MemoryRecord -> SchemaNode` 分层实现的结果；当前分层 Schema 尚未
发现可用于三 benchmark 总表的历史 live smoke artifact。

## 可复核命令

本次静态补测的 manifest 可比性和汇总可逐 split 复核：

```bash
.venv/bin/python scripts/summarize_static.py \
  --output-root outputs/smoke_retest_20260724 \
  --benchmark contextmath \
  --model anthropic/claude-opus-4.8 \
  --split aime_2024_sg

.venv/bin/python scripts/summarize_static.py \
  --output-root outputs/smoke_retest_20260724 \
  --benchmark intphys2 \
  --model anthropic/claude-opus-4.8 \
  --split debug
```

ContextMATH 的另外三个 split 可将第一条命令的 `--split` 替换为
`aime_2024_cs`、`aime_2025_sg`、`aime_2025_cs`。不加
`--allow-incomparable` 即可成功汇总，表示同一 split 内 ICL/RAG 的协议签名
和样本 ID 已通过工具检查。

ARC 汇总可由本地 artifact 重建：

```bash
.venv/bin/python scripts/eval_arc_summary.py \
  --runs-dir outputs \
  --model google/gemini-2.5-pro \
  --include-legacy \
  --allow-incomparable

.venv/bin/python scripts/eval_arc_summary.py \
  --runs-dir outputs \
  --model claude-opus-4.8 \
  --include-legacy \
  --allow-incomparable
```

`--allow-incomparable` 只用于查看历史结果，因为两组模型、step budget、retry
和反馈协议并不一致，不能据此宣称严格的横向优劣。
