# Version 3 实现状态

日期：2026-09-01

V3 已完成 Phase 0–2、正式 Python 3.12 运行时和第一轮 bounded integration smoke；
尚未开始 Phase 3 prompt/policy，因此当前实验不能用于判断 EFPS 是否提升游戏性能。

## 已实现

- 固定 Tycho 官方 commit `f68912a764372ead0a610db2e1c011d41ce5197e`；顶层 `tycho/`
  保留上游包结构，`third_party/tycho/` 保存 Apache-2.0 LICENSE、citation、release manifest、
  原始测试与 patch inventory。
- 上游差异全部列在 `third_party/tycho/UPSTREAM.md`：approach/workspace 注册、本地环境
  fingerprint、bounded experiment limits，以及 Docker 不可用时的 Bubblewrap 隔离后端。
- `sc-run-arc-v3` 正式入口：默认 `tycho_efps + orchestrator`，复用 Tycho parallel runner、
  typed terminal/reset history、resume、verifier、planner、guard、viewer 和 run spec。
- runner 复用仓库已有 `third_party/arc_agi3_games/`，并将每个实际 game metadata/source
  fingerprint 写入 immutable policy source identity；游戏内容变化会让 resume fail closed。
- 默认 `.venv` 已更新为 Python 3.12.14 与 Tycho 精确关键依赖；旧 Python 3.11 环境保存在
  `.venv-py311-backup-20260901/`，便于恢复和历史排查。
- `--max-actions-per-level` 与 `--stop-after-levels` 是 coordinator/worker 共用的硬限制，
  同时进入 immutable run spec 与 manifest。
- Bubblewrap fallback 通过实时 doctor 才可启动：只暴露只读 Python runtime 与当前 workspace，
  清空继承环境、禁网、只读根目录并 drop capabilities。
- V2 OpenRouter/OpenAI-compatible 参数和环境变量到 Tycho transport 的兼容映射；不记录密钥。
- `EvidenceRef/EvidenceIndex`：ID 绑定 run、game、role、相对位置、level、attempt、turn、公开
  action 与 frame content hash，不绑定绝对主机路径；只扫描 Tycho 真实 workspace，不执行模型。
- Tycho workspace 在每次 decision、terminal、death、animation 和 reset 后刷新 Evidence index；
  `wmlib.evidence_refs()` 提供精确查询，模拟 rollout 不会写入 Evidence。
- 沙箱内标准库-only `efps_runtime.py`：纯分类器 Prototype、Entity membership、可执行
  `Prototype + Action -> Output` handler、实际 rule attribution、重复/冲突 triple 拒绝和 Evidence closure。
- `efps_audit.py` 将 manifest 绑定到 `world_model.py` SHA-256；每次世界模型实质编辑后，在 Tycho
  dynamics/outcome/planner 自动反馈之后继续运行 EFPS audit。
- 两个审计配置：`configs/v3/upstream_parity.yaml` 与
  `configs/v3/tycho_efps_orchestrator.yaml`；另有本次小试验配置
  `configs/v3/cd82_level1_5actions.yaml`。它们不包含 provider 凭据。
- action budget 截断后的最后一个非终态结果现在通过 opt-in callback 写入 EFPS Evidence，
  避免缺失最后一次 `Prototype + Action -> Output` 的观察。

## 验证

- `pip check`：无 broken requirements；V3 runtime contract 通过（Python 3.12.14、ARC 0.9.9）。
- Bubblewrap live doctor 通过 filesystem/network/env/capability 策略检查。
- 全仓最终回归：282 passed，2 skipped；`git diff --check` 通过。
- 单次 OpenRouter transport smoke 通过：Opus 4.8 正确处理文本、PNG 和工具调用。
- CD82 offline Level 1 bounded smoke：5 个动作、0 关、RHAE 0、无错误，停止原因为
  `requested_action_limit`；10 次模型调用，运行记录估算费用约 0.89337 美元。该短跑中
  builder 未触发、world model 保持 seed 状态且没有 EFPS manifest，直接证明 Phase 3 尚未接通。

## 尚未完成

1. Phase 3：调整 Actor/Builder prompt，使 builder 主动使用 EFPS contract，而不只是在 workspace 中可用。
2. 为 EFPS manifest 增加 run-level attribution、membership churn、跨关 prototype reuse 报告。
3. 增加 stale plan、oscillation、repeated state-action、no-information probe 等 Phase 4 诊断。
4. 修完 Phase 3 后再做 upstream Tycho vs Tycho+EFPS 的 matched CD82/SK48/TU93 trial。

当前不能宣称 V3 改善了游戏成绩；现阶段完成的是可运行架构和 credential-free correctness baseline。
