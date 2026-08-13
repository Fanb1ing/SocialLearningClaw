# Learned Schema vs Gold Schema evaluator

状态：已完成第一版离线 evaluator，并评测 Phase D 的三个 ARC 游戏 learned snapshot。Phase E 管理
按当前开发优先级暂停。

## 评测隔离

入口为 `scripts/evaluate_learned_schema.py`。它只读取已经结束的：

- learned `schema.json`；
- 与之对应的 `memory.json`；
- 指定版本的 Gold 目录。

Gold loader 只加载 manifest 中 `accepted` 或 `accepted_and_revised` 的游戏。归纳器、scheduler 和
runner 不导入 evaluator/Gold loader；评测过程不实例化 SchemaManager、不修改 learned snapshot，
也不把 alignment 返回生成流程。`config.json` 保存 learned/Memory hash、Gold 版本和零写回声明。

## 第一版判定方式

learned 和 Gold 都先转为只读 canonical view：game、level、kind、action signature、target role、
trigger、expectation 和双语概念标签。确定性 `structured_arc_proxy_v1` 比较这些结构，输出：

- `equivalent`；
- `learned_narrower`；
- `learned_broader`；
- `partial`；
- `contradiction`；
- `unrelated`。

它有意保守：只写“grid changes/no change/level completes”的 learned 规则即使 action 一致，也最多
算 partial；target role 不一致、只有 action 而没有效果概念重叠时不算覆盖。一个 learned 对多个
Gold 标为 broader；多个 learned 对一个 Gold 标为 narrower。所有 Schema evidence ID 都必须在冻结
Memory 中解析。

这仍然是结构化 proxy，而不是最终论文 judge。它适合回答当前算法到底缺了什么，以及比较同一
corpus 上后续生成器版本；最终 semantic precision/recall 需要冻结人工 alignment fixture，再校准
独立 LLM judge 或扩大人工复核。

## 当前三游戏结果

评测对象是 Phase D 的 50 条 learned Schema，对照 37 条已人工接受的 Gold：CD82 18、SK48 10、
TU93 9。

| 指标 | 当前结果 |
|---|---:|
| strict learned precision | 0.000 |
| strict Gold recall | 0.000 |
| graded learned precision | 0.562 |
| graded Gold recall | 0.415 |
| graded semantic F1 | 0.478 |
| partial Gold coverage | 24/37 |
| action-signature recall | 1.000 |
| evidence traceability | 1.000 |

分游戏 partial Gold coverage：CD82 15/18，SK48 7/10，TU93 2/9。按 Gold 类型：

| Gold kind | coverage | 解释 |
|---|---:|---|
| action_effect | 21/23 | 多数只匹配到动作和变化区域，缺精确机制 |
| goal | 3/3 | 只知道哪些动作后过关，不知道完整目标条件 |
| hazard | 0/7 | transition diff 尚未识别预算和敌人机制 |
| observation_semantics | 0/4 | 尚未从视觉中学习目标、画布、参考链或出口角色 |

因此“像不像”的准确结论是：动作层面像，机制层面不像。当前 learned Schema 已找到正确的动作族和
部分作用区域，但没有一条达到 Gold 的完整 trigger + effect 语义。大量 `learned_broader` 也说明
`ACTION5 changes canvas` 把 CD82 八种不同几何填充规则合并得过宽；TU93 的敌人、碰撞、预算几乎
完全缺失。

## 审查与复现

输出：`outputs/review/three_game_schema_evaluation_v1/`。

1. `report.md`：中文结论、分游戏/类型结果、未覆盖清单；
2. `metrics.json`：机器可读指标；
3. `alignments.json`：每一对匹配的 relation、分数、分项和理由；
4. `unmatched_gold.json` / `unmatched_learned.json`：两侧缺口；
5. `judge_cache.jsonl`：可重复审阅的判定缓存；
6. `config.json`：输入路径/hash 和隔离声明。

```bash
.venv/bin/python scripts/evaluate_learned_schema.py \
  --learned outputs/review/three_game_schema_phase_d_v1/schema.json \
  --memory outputs/review/three_game_schema_prototype_v1/memory.json \
  --gold gold/arc_agi3/v1 \
  --output outputs/review/three_game_schema_evaluation_v1
```

下一步应先改进 Schema 生成，而不是做 maintenance：从关键帧/连续 window 提取 precondition、对象
角色、方向映射、精确 changed mask、预算/UI delta 和跨 step 因果，使 learned 节点能表达 Gold
级别的机制。之后用相同 evaluator 比较 v1/v2，确认改善来自生成器而非 Gold 泄漏。
