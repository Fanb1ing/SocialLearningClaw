# Architecture

## Separation of concerns

项目把 benchmark、method 和 experiment protocol 分开：

1. `BenchmarkAdapter` 只负责加载样本、构造任务输入和判分。
2. `MethodController` 只负责 baseline 状态、检索和 binary feedback 更新。
3. `ExperimentConfig` 固定模型、数据 split、预算、feedback 权限和输出位置。
4. Runner 负责循环、LLM 调用和标准化 artifact。

这样模型配置、答案解析、采样和 retry 不再散落在每个 baseline 脚本中。

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

## Current schema boundary

当前 Schema 是 `Concept + Relation + embedding` 单层图，包含：

- ARC object observations；
- spatial relations；
- action-effect nodes and relations；
- confidence/weight feedback；
- persistence and retrieval。

本次整理没有把它改写为计划中的分层 SchemaNode，只修复了稳定 object ID 覆盖学习置信度、重复 relation 和 grid shape change 等明确工程问题。

