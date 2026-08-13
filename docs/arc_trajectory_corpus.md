# ARC 可靠轨迹语料（Phase B）

状态：CD82 v1 已实现并通过验收；另有 SK48/TU93 紧凑示例 corpus。它们是后续离线
`Trajectory -> Memory -> Schema` 开发的固定输入，不依赖在线 LLM Agent 的动作质量。

三游戏当前规模与成功边界见
[三游戏 Trajectory -> Memory -> Schema 原型](trajectory_schema_prototype.md)。其中 SK48 v1 只承诺
可靠完成 3/8，剩余关卡留给 trajectory v2；TU93 v1 已完成 9/9。

## 1. 这批数据是什么

CD82 是一个 64×64 可交互画面中的多关卡作画游戏：左上角显示 10×10 目标图，下方中央是
10×10 画布，周围的工具决定涂色区域，顶部按钮选择颜色，底部显示剩余动作预算。它不是
2×2 ARC 静态网格。

生成器使用真实本地 `cd82-fb555c5d` 环境。每条 episode 都创建全新的 offline 环境，从整局
`reset()` 开始，只读取公开 observation，并通过公开 `ACTION1`…`ACTION6` 执行动作。生成和回放
都不调用网络或 LLM。

## 2. 96 条 episode 的构成

| 类型 | 数量 | 作用 |
|---|---:|---|
| 逐级确定性成功 | 6 | 分别完成第 1…6 关；最长一条以 70 个动作获得 `WIN` |
| near-miss | 11 | 删除/替换各关成功序列末步，保存未完成边界 |
| 动作预算失败 | 1 | 重复合法但阻塞的动作，真实获得 `GAME_OVER` |
| 单机制 | 36 | 方向有效/阻塞、8 个工具位置作画、palette/background/detail click |
| 成功序列扰动 | 24 | omit、duplicate、replace、swap、插入背景 click，结果由环境判定 |
| 固定种子探索 | 18 | 每条 8 步，补充动作和 click 组合；唯一 `natural` 来源 |

前 78 条标为 `source_guided_natural`：策略设计参考了游戏实现，但记录本身没有改写隐藏状态。
18 条固定种子探索标为 `natural`。`state_injected_probe` 和 `synthetic` 不进入这批正式 induction
输入。

## 3. 数据位置和格式

默认生成到本地 ignored 目录：

```text
data/trajectory_corpora/arc_agi3/cd82_v1/
  manifest.json                 环境、生成脚本和 policy hash
  coverage.json                 可量化覆盖及 gate
  validation.json               episode/split/asset 校验
  replay_validation.json        每条轨迹全新环境逐帧回放结果
  episodes/<episode-id>.json    原子持久化的事实轨迹
  assets/grids/<hash>.npy       无损状态；Schema evidence 的主视觉事实
  assets/images/<hash>.png      Agent-view 和人工审查图
  splits/train.json
  review/README.md              中文图示审查入口
```

单个 step 保存 pre-observation、公开 available actions、实际 action、post-observation、环境状态和
结构化 grid diff。CD82 底行是动作预算 UI；`grid_changed` 如实保留它的变化，
`task_state_changed` 则排除底行，避免把阻塞动作误当作工具/画布机制。

视觉不嵌入 JSON。每个状态同时保存内容寻址的 `.npy` 和 512×512 PNG，相邻或跨 episode 的相同
状态自动去重。JSON 中的 artifact ref 保存相对路径、文件 hash、logical grid hash、shape、dtype、
renderer 和 role。

## 4. 当前验收结果

2026-08-13 冻结审查运行：

- 96 episodes，1022 steps，450 个去重后的 artifact refs，占用约 13 MiB；
- 六个 level、六种 action 全覆盖；每种 action 都有 effect 与 no-effect；
- click 覆盖 `palette_button`、`background`、`edge_detail_tool`；
- terminal 覆盖 `WIN`、`GAME_OVER`、`TIMEOUT`；
- 14 个 paired-case group、32 种 transition signature；
- 通用结构、split、内容 hash 和无损 grid 校验通过；
- 96/96 episode、1022/1022 step 从新环境重放后 grid 与 status 完全一致；
- corpus JSON/审查文档未发现 Gold Schema 文本、ID 或 alignment 字段。

这些 gate 证明语料满足当前 Phase B 的可复现与行为覆盖合同，不代表其已经覆盖 CD82 的全部
语义。后续 learned-vs-Gold 结果可能暴露缺失证据，但不得用 Gold 文本反向修改这批开发输入。

## 5. 生成和审查

重新生成完整语料：

```bash
MPLCONFIGDIR=/tmp/socialclaw-mpl \
  .venv/bin/python -u scripts/generate_arc_trajectory_corpus.py --force
```

先看真实画面和动作解释：

```text
data/trajectory_corpora/arc_agi3/cd82_v1/review/README.md
```

再依次审查：

1. `coverage.json` 的七个 gate 是否均为 `true`；
2. `episodes/mechanism_paint_pos_0.json` 的单步 ACTION5 前后及 `task_changed_cells: 50`；
3. `episodes/success_through_level_6.json` 的 70 步完整路径和六次 `level_delta: 1`；
4. `episodes/near_miss_action_budget_game_over.json` 的 100 次合法动作及真实 `GAME_OVER`；
5. `replay_validation.json` 是否为 96 条全部 `passed`。

不想花时间重新执行环境回放时可以传 `--skip-replay` 做开发中间检查，但这种产物不能冻结为
正式 induction 输入。

## 6. 后续 benchmark 如何复用

`models.py`、`source.py`、`recorder.py`、`corpus.py` 和 `memory/assets.py` 不导入 ARC SDK；通用
corpus writer 通过参数接受 domain coverage。ARC 环境规范化和重放在 `arc_agi3.py` / 
`arc_corpus.py`，CD82 动作语义只在 `arc_policies.py`。

迁移到另一个 benchmark 时保留 episode、artifact、split、manifest、validation 和 evidence-tier
合同，只新增 `TrajectoryDomainAdapter`、可选 deterministic policy、domain coverage/replay。
ContextMATH 等一次性任务仍是同一结构下的单 step episode，不需要复制 Memory/Schema pipeline。
