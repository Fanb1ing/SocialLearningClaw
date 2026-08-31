# Window/keyframe Schema induction（Phase D）

状态：三游戏离线原型已跑通。该阶段复用 Phase C 的 2545 条 `MemoryRecord`，不调用 Agent、LLM 或
Gold Schema。

## 数据流和扩展边界

```text
window_summary -> transition Memory IDs
                         |
                         v
              benchmark TransitionProfiler
                         |
             semantic profile + keyframes
                         |
                         v
             WindowProposalGenerator
          create/support/revise/contradict/skip
                         |
                         v
             grounded ProposalValidator
                         |
                         v
                  SchemaNode graph
```

Scheduler、proposal 合同、validator、applier 和 audit 是 benchmark-neutral 的。ARC 只实现
`ARCVisualTransitionProfiler`：它从 Memory 引用的 lossless pre/post grid 计算 task-area diff，将变化
归一为区域（如 CD82 `canvas/tool_area`、SK48 `chain_playfield`、TU93 `maze_board`）和
`none/local/medium/global` scale。迁移新 benchmark 时实现 `TransitionProfiler.profile()`；多步 ARC 和
单步任务仍使用同一 window scheduler。未来多模态 LLM 生成器实现 `WindowProposalGenerator`，通过
`generator_factory` 替换当前确定性生成器，validator/applier 无需更换。

## 关键帧与视觉记忆

原始 transition 全部保留，但只有以下帧进入关键帧 manifest：

- 首次出现的 `(game, action, target role, effect, region, scale)`；
- level completion、WIN、GAME_OVER 边界。

重复 no-effect 只作为 support，不重复占用视觉上下文。`keyframes.json` 保存 transition Memory ID 和
pre/post grid、Agent-view PNG 引用；Schema metadata 中保存正向/负向 `keyframe_memory_ids`。因此未来
LLM 可只看关键图，Schema 的完整统计证据仍可回到全部 transition。

## 提案语义与门禁

- `create`：全局至少两条同语义 transition 时创建原子 Level-3 Schema；
- `support`：同语义后续证据只追加 source Memory、level scope 和可靠度，不新建节点；
- `revise`：同一 action/role 同时出现 effect 与 no-effect 时，把相反 outcome 记为 negative evidence，
  并将原规则限定为 context-dependent；
- `contradict`：合同和 applier 已支持明确矛盾，当前确定性生成器不会无依据伪造矛盾；
- `skip`：全 corpus 支持数不足时记录原因，但不改 graph。

Validator 在应用前检查 evidence ID 存在、必须是 transition、game/action scope 一致、target 存在，
并阻止重复 semantic key。每个提案无论是否改图都写入 `audit.json`。Graph 保存后再次用全量 Memory
ID 校验所有 source/positive/negative 引用。

## 三游戏结果与审查

输出目录：`outputs/review/three_game_schema_phase_d_v1/`。

| 指标 | 结果 |
|---|---:|
| transition / window Memory | 2068 / 333 |
| selected keyframes | 138 |
| learned Schema | 50（CD82 18、SK48 16、TU93 16） |
| source-cited transitions | 2064 |
| negative-evidence transitions | 25 |
| skipped singleton transitions | 4 |
| proposals | create 50 / support 1175 / revise 22 / skip 4 |
| verified unique keyframe artifacts | 192 |

建议先读 review 包的 `README.md`，再选一个 Schema：从 `memory_index.source` 回查 Phase C
`memory.json`，打开对应 pre/post PNG，最后在 `audit.json` 搜 Schema ID 查看它如何创建、支持或修订。

复现命令：

```bash
.venv/bin/python scripts/induce_schema_from_memory_windows.py \
  outputs/review/three_game_schema_prototype_v1/memory.json \
  outputs/review/three_game_schema_phase_d_v1
```

专项测试为 `tests/test_window_schema_induction.py`，覆盖真实 content-addressed grid 特征、关键帧去重、
离线 fake generator 注入、五类 operation 合同、低支持 skip、错误 evidence 拒绝和 negative
evidence。Phase E 将在此 graph 上
加入定时 batch、合并/去冗余、promotion、alias/snapshot 和恢复游标；Phase F 再开发独立 Gold
evaluator。
