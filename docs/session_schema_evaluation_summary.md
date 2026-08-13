# 本次开发总览（二）：Schema 生成与 learned-vs-Gold 评测

更新日期：2026-08-13  
状态：第一版 Schema 生成和离线评测已跑通；Phase E 定时管理按用户要求暂缓。

## 1. 这部分要解决什么

输入是已经冻结的 `MemoryRecord`，输出必须是当前项目统一的 layered `SchemaNode`：

```text
transition / window / episode Memory
                    ↓
     window + visual keyframe selection
                    ↓
        Schema proposal generation
                    ↓
 grounded validator + proposal audit
                    ↓
              SchemaNode graph
                    ↓
      独立、只读 learned-vs-Gold evaluator
```

硬约束：

- 不重新引入旧 `Concept` / `Relation`；
- 每个 learned Schema 必须引用存在的 durable transition Memory IDs；
- Gold 不进入轨迹、Memory、归纳或 runner；
- evaluator 不能修改 learned state，也不能把 alignment 反馈给生成器；
- ARC-specific 特征放在 profiler，scheduler/proposal/validator 对其他 benchmark 可复用。

## 2. Phase C：最小可运行 Schema baseline

第一版 `transition_bucket_v1` 先验证完整数据闭包：

```text
transition Memory
  -> group by (game, action, target_role, effect_class)
  -> require support >= 2
  -> aggregate level scope / changed-cell range
  -> emit SchemaNode with all source Memory IDs
```

`effect_class` 包括 effect、no_effect、level_completion、win 和 game_over。该 baseline 从 2545 条
Memory 生成 40 个节点：CD82 17、SK48 11、TU93 12。

它证明了 `Trajectory -> Memory -> Schema` 和 evidence closure 可以工作，但 trigger 只有“动作在某
level 可用”，没有视觉对象、精确条件或长期因果，因此不是最终生成算法。

代码：`socialclaw/schema/trajectory_pipeline.py`。  
输出：`outputs/review/three_game_schema_prototype_v1/schema.json`。

## 3. Phase D：window/keyframe Schema 生成

第二版 `semantic_window_v1` 使用 333 条 window Memory 调度，不再每个 step 创建节点。

### 3.1 ARC 视觉 transition profiler

`ARCVisualTransitionProfiler` 从 transition Memory 引用的 pre/post lossless grid 计算：

- action 和 target role；
- effect/no-effect/level completion；
- changed cell count；
- `none/local/medium/global` change scale；
- 归一化区域，例如 CD82 `canvas/tool_area`、SK48 `chain_playfield`、TU93 `maze_board`；
- pre/post grid 与 Agent-view PNG references。

Profiler 只读 corpus artifact，不导入游戏源码，不读取 Gold。

### 3.2 关键帧

原始 2068 条 transition 全部保留，但只有以下 transition 进入 keyframe manifest：

- 首次出现的 `(game, action, target role, effect, region, scale)`；
- level completion / WIN / GAME_OVER 边界。

最终选择 138 个关键 transition、192 个唯一 grid/PNG artifacts。重复 no-effect 只追加 support，不
重复占用未来多模态 LLM 上下文。

### 3.3 五类 Schema proposal

- `create`：全 corpus 至少两条同语义证据才创建节点；
- `support`：追加同语义 evidence、level scope 和 reliability，不新建节点；
- `revise`：同 action/role 出现 effect 与 no-effect 时，写入 negative evidence 并限定规则是
  context-dependent；
- `contradict`：合同、validator 和 applier 已支持明确反例；确定性生成器不会无依据制造 contradiction；
- `skip`：支持不足的 singleton 留在 audit，不写入 graph。

### 3.4 Grounded validator

每个 proposal 应用前检查：

- evidence Memory 是否存在；
- 是否为 transition scope；
- game/action scope 是否一致；
- support/revise/contradict 的 target Schema 是否存在；
- create 是否缺 trigger/action/expectation；
- 是否重复 semantic key。

每个操作无论应用与否都进入 `audit.json`。保存前 graph 再检查全部 source/positive/negative Memory
引用以及 keyframe IDs。

### 3.5 Phase D 实际结果

| 项目 | 结果 |
|---|---:|
| transition Memory | 2068 |
| window Memory | 333 |
| selected keyframes | 138 |
| learned Schema | 50 |
| CD82 / SK48 / TU93 | 18 / 16 / 16 |
| source-cited transitions | 2064 |
| negative-evidence transitions | 25 |
| uncited singleton transitions | 4 |
| create / support / revise / skip | 50 / 1175 / 22 / 4 |
| rejected proposals | 0 |
| network / Gold reads | 0 / 0 |

主要实现：`socialclaw/schema/window_induction.py`。  
复现入口：`scripts/induce_schema_from_memory_windows.py`。

```bash
.venv/bin/python scripts/induce_schema_from_memory_windows.py \
  outputs/review/three_game_schema_prototype_v1/memory.json \
  outputs/review/three_game_schema_phase_d_v1
```

## 4. learned-vs-Gold evaluator

在用户决定暂缓 Phase E 后，优先开发了独立 evaluator。

### 4.1 隔离边界

`scripts/evaluate_learned_schema.py` 只读取已经结束的 learned snapshot、对应 Memory 和 Gold 目录。
Gold loader 只加载 manifest 标记为 `accepted` / `accepted_and_revised` 的游戏：

- CD82：18 条 Gold；
- SK48：10 条 Gold；
- TU93：9 条 Gold。

总计 37 条。Evaluator 保存输入绝对路径和 SHA-256，不实例化 SchemaManager，不回写 learned state。
Induction/runner 模块不导入 `gold_loader` 或 `evaluation`。

### 4.2 Canonical view 与匹配关系

learned 和 Gold 分别转换为只读 canonical fields：

- game / level scope；
- kind；
- action family / target role；
- trigger / expectation；
- region、effect class 和双语概念标签；
- learned evidence IDs。

第一版确定性 `structured_arc_proxy_v1` 输出：

- `equivalent`；
- `learned_narrower`；
- `learned_broader`；
- `partial`；
- `contradiction`；
- `unrelated`。

它有意限制宽松匹配：只有 action 一样不够；只写 `grid changes/no change/level completes` 的 learned
Schema 最多是 partial。一个 learned 对多个 Gold 记为 broader，多个 learned 对一个 Gold 记为
narrower。

### 4.3 评测结果

| 指标 | 结果 |
|---|---:|
| learned / Gold | 50 / 37 |
| strict learned precision | 0.000 |
| strict Gold recall | 0.000 |
| graded learned precision | 0.562 |
| graded Gold recall | 0.415 |
| graded semantic F1 | 0.478 |
| partially covered Gold | 24/37 |
| supported learned | 44/50 |
| action-signature recall | 1.000 |
| evidence traceability | 1.000（2090/2090） |

分游戏：

| 游戏 | learned | Gold | graded precision | graded recall | partial coverage |
|---|---:|---:|---:|---:|---:|
| CD82 | 18 | 18 | 0.531 | 0.533 | 15/18 |
| SK48 | 16 | 10 | 0.520 | 0.448 | 7/10 |
| TU93 | 16 | 9 | 0.640 | 0.142 | 2/9 |

按 Gold kind：

| Gold kind | partial coverage | 当前含义 |
|---|---:|---|
| action_effect | 21/23 | 多数只捕捉 action + 粗粒度变化 |
| goal | 3/3 | 知道哪些 transition 后过关，不知道完整目标条件 |
| hazard | 0/7 | 没学到预算、敌人、失败机制 |
| observation_semantics | 0/4 | 没学到目标、画布、参考链、出口等对象角色 |

### 4.4 “像不像 Gold”的结论

准确结论是：**动作层面像，机制层面不像。**

例如 learned：

```text
ACTION5 changes the canvas at medium scale.
```

Gold 的实际粒度是：

```text
活动工具位于顶部且颜色为 C 时，ACTION5 将画布第 0–4 行的 50 个格子设为 C，区域外保持不变。
```

当前 learned 知道 `ACTION5 -> canvas changes`，但没有工具位置、当前颜色、精确 mask 和 unchanged
constraint。因此 CD82 八种几何工具容易被一个过宽节点覆盖。TU93 更明显：普通方向动作能对上，
但敌人追击、延迟 follower、碰撞阶段和步数预算全部缺失。

当前 `contradiction_count=0` 不代表规则都正确。宽泛规则通常无法被严格判为矛盾；主要错误形态是
过宽、过窄和遗漏，而不是明确说反。

`graded F1=0.478` 是诊断 proxy，不是正式论文主分数。正式 semantic score 还需要冻结人工
alignment fixture，并校准独立 LLM judge 或扩大人工审核。

## 5. 具体怎么审查

### 5.1 审查最终 learned Schema

目录：`outputs/review/three_game_schema_phase_d_v1/`

建议顺序：

1. `README.md`：审查说明；
2. `report.json`：总体数量和零网络/Gold 声明；
3. `schema.json`：50 个 node 的 trigger/action/expectation、support、negative 和 keyframes；
4. `keyframes.json`：138 条视觉 transition 的 pre/post artifact refs；
5. `audit.json`：每个 window 的 create/support/revise/skip、判定理由和受影响 Schema。

### 5.2 从 Schema 一直追到原始视觉

1. 在 `schema.json` 选择一个 node；
2. 复制 `memory_index.source` 中的 Memory ID；
3. 在 `outputs/review/three_game_schema_prototype_v1/memory.json` 搜索；
4. 查看 game、episode、step、action 和 pre/post artifacts；
5. 打开 `corpus_root/episodes/<episode>.json` 核对原 transition；
6. 打开 `corpus_root/assets/<artifact.relative_path>` 查看 `.npy` 或 PNG；
7. 在 `audit.json` 搜 Schema ID，查看它何时 create、support 或 revise。

### 5.3 审查 learned-vs-Gold 评测

目录：`outputs/review/three_game_schema_evaluation_v1/`

1. `report.md`：中文结论和未覆盖 Gold 清单；
2. `metrics.json`：完整机器可读指标；
3. `alignments.json`：每个 learned-Gold pair 的 relation、score、分项和原因；
4. `unmatched_gold.json`：当前生成器真正缺失的机制；
5. `unmatched_learned.json`：没有获得 Gold 支持的 learned 节点；
6. `judge_cache.jsonl`：可重复审阅的判定；
7. `config.json`：learned/Memory hash、Gold review status 和零写回声明。

复现：

```bash
.venv/bin/python scripts/evaluate_learned_schema.py \
  --learned outputs/review/three_game_schema_phase_d_v1/schema.json \
  --memory outputs/review/three_game_schema_prototype_v1/memory.json \
  --gold gold/arc_agi3/v1 \
  --output outputs/review/three_game_schema_evaluation_v1
```

## 6. 当前还没做好什么

### 6.1 Schema 生成

1. trigger 仍以动作和事后 effect bucket 为主，尚未从 pre-state 学出真正可预测的条件；
2. 没有稳定的对象身份/跟踪，因此不能描述工具位置、链头方向、敌人角色或出口；
3. changed grid 只归纳 region/scale，没有学习精确 changed mask、颜色替换和不变区域；
4. window scheduler 尚未归纳长动作序列、延迟效果和跨 step 因果；
5. UI 行目前主要被排除出 task diff，导致预算、剩余步数等 hazard 信号没有结构化进入 Schema；
6. deterministic generator 适合复现，但没有真正的视觉语义推理；未来 LLM generator 已预留协议，
   尚未实现和校准；
7. 当前所有节点是原子 Level 3 风格结果，跨 level promotion 和更高层抽象尚未做。

### 6.2 评测

1. `structured_arc_proxy_v1` 是规则型 proxy，无法完全理解中英文复杂语义；
2. 尚未冻结人工 reviewed alignment fixture；
3. 尚未实现独立 LLM semantic judge 与 judge cache 重用；
4. equivalent/broader/narrower 的组合匹配仍是启发式，不应作为最终论文指标；
5. 当前只评三个已审核 ARC 游戏，未评 cross-game provisional Schema；
6. 没有 checkpoint learning curve，因为当前只有一个最终 learned snapshot。

### 6.3 管理与在线流程

Phase E 根据用户决定暂缓，所以下列内容尚未实现：

- 定时 merge 和去冗余；
- conflict split；
- Level 3 -> Level 2 promotion；
- alias/tombstone；
- maintenance snapshot、cursor 和中断恢复；
- 在线 runner 中的完整 Agent -> recorder -> Memory -> induction 流程。

当前优先级应该是先做 Schema generation v2，让规则从“动作导致画面变化”升级为“什么对象在什么
条件下发生什么精确变化”，再用同一 evaluator 比较 v1/v2；管理应在生成质量明显改善后继续。

## 7. 代码、测试与文档入口

核心代码：

- `socialclaw/schema/trajectory_pipeline.py`
- `socialclaw/schema/window_induction.py`
- `socialclaw/schema/evaluation.py`
- `socialclaw/schema/gold_loader.py`
- `scripts/prototype_schema_from_corpus.py`
- `scripts/induce_schema_from_memory_windows.py`
- `scripts/evaluate_learned_schema.py`

专项测试：

- `tests/test_trajectory_schema_pipeline.py`
- `tests/test_window_schema_induction.py`
- `tests/test_schema_evaluator.py`

阶段文档：

- `docs/trajectory_schema_prototype.md`
- `docs/window_schema_induction.md`
- `docs/schema_evaluation.md`
- `docs/arc_learned_schema_pipeline_plan.md`

验证结果：全仓 79 个测试通过，compileall 和 `git diff --check` 通过；归纳和 evaluator 均为零网络
调用，evaluator 对 learned state 零写入。

