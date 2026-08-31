# Documentation Index

文档按“当前 V2 主线、冻结 V1 参考、历史材料”分组。运行命令以本页链接到的当前文档为准；
旧结果文档只用于追溯，不代表当前 CLI 或预算语义。

## 当前 V2 主线

- [V2 三游戏正式实验与确定性复现](../experiments/v2_formal_20260830/README.md)：
  当前可提交、可逐文件哈希校验的 CD82/SK48/TU93 实验入口；
- [Architecture](architecture.md)：Agent 权限、EFPS、证据、每关预算、GAME_OVER 恢复；
- [Version 2 EFPS 开发方案](version2_efps_development_plan.md)：类型图和纵向开发路线；
- [认知输入精简与精确读取](v2_cognition_retrieval_design.md)：`read_cognition` 工具与 token 审计设计；
- [Project memory](project_memory.md)：跨会话简要开发记录，不是运行规范。

## 冻结 V1 与共享基础设施

- [V1 完整冻结快照](../archive/version1_20260824/README.md)；
- [Benchmark 说明](benchmarks.md)、[baseline 说明](baselines.md)、
  [实验协议](experiment_protocol.md)、[结果格式](results_format.md)；
- [Layered Schema 架构](schema_architecture.md)、[轨迹合同](trajectory_contract.md)、
  [ARC 轨迹语料](arc_trajectory_corpus.md)、[window/keyframe 归纳](window_schema_induction.md)；
- [Learned-vs-Gold 评测](schema_evaluation.md)与
  [Gold Schema 构建](gold_schema_generation.md)。Gold 只属于 V1 独立评测路径，不能进入 V2 Agent。

## 历史材料

`docs/archive/` 保存退役设计、旧 CLI、handoff、早期 V2 输入和原型实验。
`archive/version1_20260824/` 是可回退的 V1 文件快照。两者都不应作为当前 V2 runner 的实现来源。

仓库根目录、`docs/` 和 `experiments/` 中的 Markdown 会进入 Git；运行生成的 `outputs/`、
本地数据 `data/`、密钥 `.env` 和虚拟环境 `.venv/` 均被忽略。正式可复现模型响应放在
`experiments/v2_formal_20260830/transcripts/`，不依赖被忽略的本地输出目录。
