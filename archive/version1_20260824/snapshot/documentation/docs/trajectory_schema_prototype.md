# 三游戏 Trajectory -> Memory -> Schema 原型

状态：Phase C 原型已完成，并已作为 Phase D window/keyframe 归纳器的冻结 Memory 输入。它以
CD82、SK48、TU93 三份 corpus 为输入，不调用 LLM、不读取 Gold Schema。

## 轨迹 v1 的边界

| 游戏 | episodes | steps | 已验证成功进度 | 说明 |
|---|---:|---:|---:|---|
| CD82 | 96 | 1022 | 6/6 | 完整 Phase B corpus |
| SK48 | 24 | 345 | 3/8 | v1 到第三关；第 4–8 关留给 trajectory v2 |
| TU93 | 24 | 701 | 9/9 | 185 步固定路径可完整 `WIN` |

SK48/TU93 corpus 均包含成功前缀、删除末步的 near-miss、单动作/重复动作探针和固定 seed 探索。
固定成功路径由源码辅助的离线状态搜索找到，但发布轨迹只从全新环境走公开 action API，因此标为
`source_guided_natural`；无 Gold 引导的随机轨迹标为 `natural`。两份语料 48/48 episode 全部逐帧
replay 一致。数量和成功关卡不是完整覆盖声明，manifest/coverage 明确记录 v2 follow-up。

## Phase C Memory 投影

`TrajectoryMemoryProjector` 产生三种已有 `MemoryRecord`，不引入第二套 Memory 类型：

1. 每个 step 一条 `transition` memory：动作、effect/no-effect、changed cells、level delta、status，
   以及 pre/post grid 和 PNG 引用；
2. 每 8 step 一条 `window_summary` memory，`source_memory_ids` 指向窗口内 transition；
3. 每个 episode 一条 `level_episode` memory，指向全部 transition/window 并保存 terminal outcome。

ID 由 benchmark、game、episode、step 的稳定 hash 生成，重复 replay 不产生新 ID；不同游戏中的
同名 episode 不会碰撞。视觉 ref 保留 corpus 根目录和相对资产路径，可以从 Schema evidence 一直
解析到 `.npy`/PNG 文件。`JsonMemoryStore.put_many()` 让一次投影只做一次原子快照写入，避免逐条
保存的 O(n²) I/O。

## 原型 Schema 算法

`transition_bucket_v1` 是一个有意保守的确定性 baseline：

```text
transition memories
  -> group by (game, action, target_role, effect_class)
  -> require support_count >= 2
  -> aggregate level scope and changed-cell range
  -> emit one atomic SchemaNode
```

`effect_class` 为 `effect`、`no_effect`、`level_completion`、`win` 或 `game_over`。跨多个 level 的
重复证据生成 Level 2，同一 level 生成 Level 3。每个 Schema 的 `memory_index.source` 直接列出
所有支持它的 transition memory IDs；window/episode memory 只提供上下文，不替代原始证据。

该算法适合验证管线、证据闭包和形成一个可比较的零成本 baseline，但不是最终语义归纳器：它只按
观测 bucket 总结相关性，尚不能自动发现视觉 trigger、对象身份、条件式动作效果或长动作序列。
Phase D 已在相同 Memory 输入上实现 keyframe/window-based proposal generator，输出 create/support/
revise/contradict/skip，再由确定性 validator 应用；详见
[`window_schema_induction.md`](window_schema_induction.md)。

## 当前真实输出

三游戏运行结果：144 episodes 被投影为 2068 条 transition、333 条 window 和 144 条 episode
memory，共 2545 条 MemoryRecord；原型生成 40 个 SchemaNode：CD82 17、SK48 11、TU93 12。
每个节点至少有两条 transition evidence；全量检查确认 Schema -> memory -> trajectory JSON ->
grid/PNG 的引用闭包有效。网络调用和 Gold 读取均为 0。

```bash
.venv/bin/python scripts/prototype_schema_from_corpus.py \
  --corpus data/trajectory_corpora/arc_agi3/cd82_v1 \
  --corpus data/trajectory_corpora/arc_agi3/sk48_v1 \
  --corpus data/trajectory_corpora/arc_agi3/tu93_v1 \
  --output outputs/review/three_game_schema_prototype_v1
```

审查顺序：

1. `report.json` 看 Memory/Schema 数量和零 API/Gold 声明；
2. `schema.json` 看可读规则、support count 和 `memory_index.source`；
3. 用任一 source ID 在 `memory.json` 找到 episode/step、pre/post artifact refs；
4. 根据该 memory 的 `corpus_root` 和 artifact `relative_path` 打开原始 `.npy`/PNG。

专项测试为 `tests/test_trajectory_schema_pipeline.py`，覆盖稳定投影、跨游戏同名 episode、视觉
provenance、最小重复证据和 Schema evidence 完整性。

Phase C 的 40 个节点是粗粒度 baseline；当前继续开发和后续评测默认使用 Phase D 的 50 节点
`semantic_window_v1` snapshot，不把二者混为同一个算法版本。
