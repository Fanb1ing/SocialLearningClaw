# Architecture

## Separation of concerns

项目把 benchmark、method 和 experiment protocol 分开：

1. `BenchmarkAdapter` 只负责加载样本、构造任务输入和判分。
2. `MethodController` 只负责静态实验的方法状态、检索和 binary feedback 更新。
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

旧 `Concept`/`Relation` 模块暂时保留，只用于历史代码/数据兼容；当前三个
正式 runner 的 `schema` method 都不再依赖它。
