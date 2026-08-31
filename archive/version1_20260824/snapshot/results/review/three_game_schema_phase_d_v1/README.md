# Phase D 三游戏窗口归纳审查包

这个目录是确定性、零 LLM 的 `semantic_window_v1` 输出。输入是 Phase C 已冻结的
`../three_game_schema_prototype_v1/memory.json`，包含 CD82、SK48、TU93 三个 ARC 游戏。

## 先看什么

1. `report.json`：整体规模、提案类型、未被引用的低支持 transition 数；
2. `schema.json`：50 条 Schema 的 trigger/action/expectation、正向证据、负向证据和关键帧 Memory ID；
3. `keyframes.json`：138 个被挑中的视觉 transition，含 pre/post grid 与 Agent-view PNG 引用；
4. `audit.json`：每个窗口的每次 `create/support/revise/skip`、依据、validator 判定和受影响 Schema ID。

当前结果把 2068 个 transition 压缩为 50 个 Schema；2064 个 transition 成为 Schema source evidence。
剩余 4 个只出现一次的语义 bucket 被明确 `skip`，可在 `audit.json` 搜索
`below_global_min_support`。25 个 transition 同时作为相反 outcome 的 negative evidence，促使过宽规则
被加上“其他 context 可能产生不同结果”的条件限定。

例如 `schema_efbba1756ef42ddf` 表示 CD82 中 `ACTION5` 会在 canvas 产生 medium-scale 变化；
`memory_index.source` 可回查 Phase C `memory.json`，再用该 Memory 的 `corpus_root` 加 artifact 的
`relative_path` 打开 `.npy` 或 PNG。不要只审查英文句子：应同时检查它的 action/effect/region、
support、negative evidence 和真实 pre/post 画面。

## 复现

```bash
.venv/bin/python scripts/induce_schema_from_memory_windows.py \
  outputs/review/three_game_schema_prototype_v1/memory.json \
  outputs/review/three_game_schema_phase_d_v1
```

运行时会逐个校验视觉 grid；输出前还会校验全部关键帧 grid/PNG 的内容 hash。当前报告中
`network_calls=0`、`gold_schema_reads=0`。这是归纳原型，不是 learned-vs-Gold 评分结果；Gold evaluator
将在 Phase F 以独立进程读取 learned snapshot。
