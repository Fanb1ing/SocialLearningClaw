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

## Schema boundaries

现有 ARC runner 仍使用 `Concept + Relation + embedding` 单层图，包含：

- ARC object observations；
- spatial relations；
- action-effect nodes and relations；
- confidence/weight feedback；
- persistence and retrieval。

新的 memory-grounded 分层架构已经作为独立基础层落地，包括
`MemoryRecord`、`SchemaNode`、自动生成/融合、反馈、mask、遗忘、去重、
持久化和可替换 LLM/embedding 接口。完整设计见
[Memory-grounded layered Schema architecture](schema_architecture.md)。

两套类型暂时并存：旧类型保证现有 ARC 实验可复现，新类型供下一阶段
runner 迁移使用，避免在没有实验协议确认时静默改变行为与 artifact 格式。
