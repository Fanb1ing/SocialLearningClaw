# 本次开发总览（一）：ARC 轨迹生成、视觉资产与 Memory

更新日期：2026-08-13  
状态：离线三游戏原型已跑通；尚未接入真实在线 Agent。

## 1. 这部分要解决什么

Schema 模块的开发不能长期依赖不稳定的 LLM Agent 临场探索。本次先建立一条可复现的数据链：

```text
确定性策略 / 固定 seed 探索 / 冻结 replay / 未来真实 Agent
                         ↓
              通用 TrajectoryEpisode
                         ↓
       step 级 observation-action-result 轨迹
                         ↓
          lossless grid + Agent-view PNG
                         ↓
        transition / window / episode Memory
```

这样可以固定轨迹输入，只比较 Memory 和 Schema 算法本身。ARC 是多 step 游戏；未来
ContextMATH、IntPhys2 或其他一次性任务只是同一合同下只有一个 step 的 episode，不需要另建数据
管线。

本次没有优化 Agent 的决策能力，也没有调用 LLM Agent API 生成这些轨迹。

## 2. 通用轨迹合同

新增 `socialclaw.trajectory`，核心数据结构包括：

- `TrajectoryEpisode`：benchmark、task/game、split、actor、evidence tier、初始观察、steps 和终局；
- `TrajectoryStep`：step index、当前观察、可用动作、实际动作、结果和可选 decision；
- `Observation`：文字、结构化状态以及视觉 artifact references；
- `Action`：动作名称和参数，例如 `ACTION6 + target_role`；
- `StepResult`：下一观察、environment status 和 task state delta；
- `TrajectoryRecorder`：每一步原子落盘、连续性校验、resume 身份校验；
- `TrajectorySource` / `TrajectoryDomainAdapter`：隔离 scripted、explorer、replay、Agent 与 benchmark。

Recorder 会在每步完成后保存完整 episode snapshot。连续性错误、不可用 action 或原子替换失败时，
不会破坏上一个有效版本。

主要代码：

- `socialclaw/trajectory/models.py`
- `socialclaw/trajectory/source.py`
- `socialclaw/trajectory/recorder.py`
- `socialclaw/trajectory/corpus.py`
- `socialclaw/trajectory/arc_agi3.py`
- `socialclaw/trajectory/arc_policies.py`

## 3. ARC 视觉输入如何保存

没有把 base64 图片塞进 Memory JSON。每个唯一画面用内容寻址方式保存两种资产：

```text
assets/
  grids/<logical-grid-sha256>.npy
  images/<png-content-sha256>.png
```

### 3.1 `.npy`：环境事实主证据

- 保存无损二维整数 grid；
- 统一为 little-endian `int16`；
- logical hash 包含 dtype、shape 和 C-order cells；
- 加载时 `allow_pickle=False`；
- 内容相同的 pre/post state 自动复用同一个文件；
- 加载时同时检查文件 SHA-256、大小和 logical grid hash。

### 3.2 `.png`：Agent 实际视觉证据

- 保存与 Agent-view 一致的 ARC 渲染图；
- 保存 cell size、宽高、renderer 等参数；
- PNG 内容相同则复用；
- 便于人工查看“模型当时实际看到了什么”。

JSON 中只保存 `MemoryArtifactRef`：artifact ID、role、media type、相对路径、SHA-256、大小和
metadata。路径必须位于 corpus asset root 内，拒绝绝对路径和 `..` 逃逸。

主要代码：`socialclaw/memory/assets.py`。

## 4. 三个 ARC 游戏的离线轨迹

本次按用户要求使用三个游戏作为示例，并明确区分“完整解完”和“轨迹够用”。

| 游戏 | episodes | steps | 已验证成功进度 | 说明 |
|---|---:|---:|---:|---|
| CD82 | 96 | 1022 | 6/6 | 完整 v1 corpus |
| SK48 | 24 | 345 | 3/8 | v1 到第 3 关；4–8 关留给 trajectory v2 |
| TU93 | 24 | 701 | 9/9 | 185-action 固定路径可完整 WIN |
| 合计 | 144 | 2068 | — | 三份 corpus 全部 replay 验证 |

### 4.1 CD82

实现了只读取公开可见 grid 的策略：识别目标、画布、调色板、活动工具和边缘工具，完成六关。
corpus 含成功、near-miss、预算失败、单机制探针、动作扰动和固定 seed 探索。96/96 episodes、
1022/1022 steps 从 fresh environment 逐帧 replay 一致。

### 4.2 SK48

离线搜索得到前三关可复现动作路径：14、30、33 steps。随后从 fresh environment 仅通过公开 action
API 重新记录。v1 另外包含删末步 near-miss、动作重复探针和随机自然轨迹。没有声称 8/8 完成。

### 4.3 TU93

离线状态搜索得到九关固定路径，共 185 actions；发布轨迹从 fresh environment 公开执行到 WIN。
同样加入 near-miss、动作探针和随机自然轨迹。

### 4.4 轨迹来源与可信度

- `natural`：固定 seed 探索，只使用公开 observation/action；
- `source_guided_natural`：设计策略时参考过源码，但发布 episode 必须从 reset 公开执行；
- `state_injected_probe`：只允许机制测试，不进入正式归纳；
- `synthetic`：只允许接口回归。

当前三份正式 corpus 没有 Gold Schema ID、Gold 文本或 learned-to-Gold alignment。轨迹生成、Schema
生成和 evaluator 是三条隔离路径。

## 5. Corpus 的目录与可复现信息

每份 corpus 使用相同格式：

```text
corpus_root/
  manifest.json
  coverage.json
  validation.json
  replay_validation.json
  episodes/<episode-id>.json
  assets/grids/<sha256>.npy
  assets/images/<sha256>.png
  splits/<split>.json
  review/
```

本地 corpus：

- `data/trajectory_corpora/arc_agi3/cd82_v1/`
- `data/trajectory_corpora/arc_agi3/sk48_v1/`
- `data/trajectory_corpora/arc_agi3/tu93_v1/`

这些大体积生成物默认被 Git 忽略；生成代码、测试和文档在仓库中。

生成入口：

```bash
# CD82 完整 corpus
MPLCONFIGDIR=/tmp/socialclaw-mpl .venv/bin/python \
  scripts/generate_arc_trajectory_corpus.py

# SK48 与 TU93 示例 corpus
MPLCONFIGDIR=/tmp/socialclaw-mpl .venv/bin/python \
  scripts/generate_arc_example_corpora.py
```

## 6. 从轨迹投影到 Memory

`TrajectoryMemoryProjector` 不引入第二套 Memory 模型，而是产生已有的 `MemoryRecord`：

### 6.1 transition Memory

每个 step 一条，共 2068 条。保存：

- benchmark、game、episode、step 和 level；
- action name、arguments、target role；
- task state changed、changed cell count、level delta、status；
- corpus root 和原 episode JSON 相对路径；
- pre/post lossless grid 和 PNG references；
- evidence tier。

它是 Schema 的最终事实证据。Schema node 必须引用这些 durable transition Memory IDs。

### 6.2 window summary Memory

默认每 8 steps 一条，共 333 条。保存窗口范围和 `source_memory_ids`，指向窗口中的原始 transition。
它是归纳调度单元，不替代原始事实。

### 6.3 episode Memory

每个 episode 一条，共 144 条。保存完整 transition/window ID 列表、step count、evidence tier 和终局。

总计：

| Memory scope | 数量 |
|---|---:|
| transition | 2068 |
| window summary | 333 |
| episode | 144 |
| 合计 | 2545 |

Memory ID 使用 benchmark + game + episode + step/window 的稳定 hash；重复投影不会新增 ID，同名 episode
出现在不同游戏也不会碰撞。`JsonMemoryStore.put_many()` 一批只写一次原子 snapshot，避免逐条写入
造成 O(n²) I/O。

实现：`socialclaw/schema/trajectory_pipeline.py`。  
冻结 Memory：`outputs/review/three_game_schema_prototype_v1/memory.json`。

复现：

```bash
.venv/bin/python scripts/prototype_schema_from_corpus.py \
  --corpus data/trajectory_corpora/arc_agi3/cd82_v1 \
  --corpus data/trajectory_corpora/arc_agi3/sk48_v1 \
  --corpus data/trajectory_corpora/arc_agi3/tu93_v1 \
  --output outputs/review/three_game_schema_prototype_v1
```

## 7. 具体怎么审查

### 7.1 先审查 corpus 是否是真实游戏轨迹

1. 打开任意 `episodes/*.json`；
2. 查看 `initial_observation` 和连续 `steps`；
3. 确认每步 action 属于 available actions；
4. 检查前一步 post grid 是否等于后一步 pre grid；
5. 根据 artifact `relative_path` 打开对应 `.npy` 或 PNG；
6. 查看 `replay_validation.json` 是否逐帧一致。

推荐入口：

- `docs/arc_trajectory_corpus.md`
- `data/trajectory_corpora/arc_agi3/cd82_v1/replay_validation.json`
- `data/trajectory_corpora/arc_agi3/sk48_v1/replay_validation.json`
- `data/trajectory_corpora/arc_agi3/tu93_v1/replay_validation.json`

### 7.2 审查一条 Memory 是否能回到视觉事实

1. 打开 `outputs/review/three_game_schema_prototype_v1/memory.json`；
2. 搜索 `"memory_scope": "transition"`；
3. 记录 `episode_id`、`step_index`、`corpus_root` 和 `trajectory_path`；
4. 打开原 episode，核对相同步骤的 action/result；
5. 用 `corpus_root/assets/<relative_path>` 打开 pre/post grid 或 PNG；
6. 检查 Memory 中的 action、changed cells、level delta 和 status 是否与轨迹一致。

### 7.3 审查视觉资产

代码级门禁见 `tests/test_memory_assets.py`，包含：

- grid lossless round-trip 和去重；
- PNG hash 和复用；
- 文件损坏检测；
- 路径逃逸和 object array 拒绝。

## 8. 当前还没做好什么

1. SK48 只有 3/8 关成功轨迹，4–8 关留给 trajectory v2；
2. 尚未把真实在线 `AgentTrajectorySource` 接入 ARC runner；
3. 当前可靠轨迹包含 source-guided 路径，适合调试归纳器，但不能冒充自主 Agent 探索表现；
4. 当前 Memory 主要保存 transition facts，尚未把对象追踪、UI budget delta、角色变化和跨 step 因果
   直接结构化为字段；视觉事实仍可从 grid 重算；
5. 尚未做跨 session 的实时 recorder cursor 与在线中断恢复；冻结 corpus/replay 和原子 episode resume
   已实现；
6. trajectory v2 还需要增加更有针对性的 hazard、collision、预算临界点、对象选择和前置条件配对轨迹。

## 9. 验证情况

- 三份 corpus 共 144 episodes / 2068 steps；
- CD82 96/96、SK48/TU93 48/48 episode replay 一致；
- 2545 条 Memory 稳定投影；
- Schema evidence 可递归解析到 transition、episode 和视觉资产；
- 当前仓库全量 79 个测试通过；
- compileall 和 `git diff --check` 通过；
- 轨迹与 Memory 阶段网络/LLM 调用为 0，Gold Schema 读取为 0。

更细的阶段文档：

- `docs/trajectory_contract.md`
- `docs/arc_trajectory_corpus.md`
- `docs/trajectory_schema_prototype.md`

