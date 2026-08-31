# 通用任务轨迹合同

状态：Phase A 通用合同与 Phase B CD82 corpus 均已实现。当前仍未接入现有 ARC runner；下一步
由冻结 corpus 离线生成 Memory。

## 1. 为什么单独建立轨迹层

Schema 生成需要稳定、可重放的输入，不能依赖在线 Agent 每次都产生相同质量的动作。轨迹层把
“谁产生动作”和“怎样从经历生成 Schema”分开：

```text
scripted policy / coverage explorer / frozen replay / Agent
                              ↓
                 Trajectory lifecycle events
                              ↓
                    TrajectoryRecorder
                              ↓
                后续 Memory -> Schema pipeline
```

ARC 一关是多 step episode；未来静态 benchmark 是同一合同下的单 step episode。

## 2. 当前数据结构

实现位于 `socialclaw/trajectory/`：

- `TrajectoryEpisode`：task、actor、来源等级、初始 observation、steps 和 terminal outcome；
- `TrajectoryStep`：pre-observation、available actions、执行 action、post-result 和可选 decision；
- `Observation`：文本、结构化状态和任意模态 artifact refs；
- `EpisodeStarted / StepObserved / EpisodeFinished`：离线 replay 与在线流共用的生命周期事件；
- `TrajectoryDomainAdapter`：未来 benchmark-specific 规范化边界；
- `TrajectoryRecorder`：每接受一个 step 就原子替换 episode JSON。

`decision` 可以记录 Agent response、简短 rationale 和 Schema IDs，但不是必填。Schema 生成不得
依赖模型私有 chain-of-thought；确定性脚本和 coverage explorer 没有 LLM reasoning 也能产生完整
轨迹。

## 3. 证据来源等级

- `natural`：只走公开 observation/action 接口；
- `source_guided_natural`：policy 设计参考源码，但记录仍只走公开接口；
- `state_injected_probe`：修改内部状态，只允许机制诊断；
- `synthetic`：fake environment/手写 fixture，只允许接口回归。

后续正式 corpus 和 evaluator 必须按等级过滤，不能让后两类进入主指标。

## 4. 视觉和结构化资产

`ContentAddressedArtifactStore` 位于 `socialclaw/memory/assets.py`。当前支持：

- 无损二维 grid：统一保存为 little-endian `int16` `.npy`，读取时固定
  `allow_pickle=False`；
- Agent-view 图片：确定性编码为 PNG；
- JSON 只保存 `MemoryArtifactRef`，包括 role、相对路径、SHA-256、字节数和 metadata；
- 相同内容复用同一路径和 artifact ID，即使它在不同 step 中扮演不同 role；
- 同时校验文件 SHA-256 和 grid logical hash；路径禁止绝对路径和 `..`。

资产目录约定：

```text
<trajectory-root>/
  episodes/<episode-id>.json
  assets/grids/<logical-sha256>.npy
  assets/images/<content-sha256>.png
```

## 5. recorder 保证的约束

1. episode ID 不能包含路径分隔符；
2. step index 必须从 0 连续递增；
3. 当前 step observation 必须等于上一 step 的 post-observation；
4. 若环境提供 available actions，执行 action 必须在其中；
5. finalized episode 不能继续追加 step；
6. 新 snapshot 完整写入并 `fsync` 后才以 `os.replace` 替换旧文件；
7. 验证失败或原子替换失败时，旧 JSON 和 recorder 内存状态均保持不变；
8. resume 时会核对 episode/task/actor/evidence tier 和初始 observation。

## 6. Phase A 最小合同示例

不调用网络或 Agent，生成两步 synthetic ARC-style 轨迹：

```bash
.venv/bin/python scripts/demo_trajectory_contract.py \
  --output-dir outputs/review/trajectory_contract_phase_a
```

期望摘要：

```json
{
  "episode_id": "review.arc.trajectory-contract-v1",
  "evidence_tier": "synthetic",
  "steps": 2,
  "terminal_status": "TIMEOUT",
  "grid_asset_files": 2,
  "image_asset_files": 2
}
```

审查时重点检查：

1. `episodes/review.arc.trajectory-contract-v1.json` 中 step 0 为有效变化、step 1 为 blocked；
2. step 0 的 post-grid artifact ID 等于 step 1 的 pre-grid artifact ID；
3. step 1 的 pre/post-grid artifact ID 相等，因此 no-effect 没有复制资产；
4. `assets/images/` 两张 PNG 分别对应初始和移动后的状态；
5. provenance 明确写有 `network_calls: 0`，证据等级为 `synthetic`。

`outputs/` 是 ignored generated artifact；需要重新审查时使用一个新的输出目录，避免覆盖旧结果。

## 7. 验证

专项测试：

```bash
.venv/bin/python -m unittest \
  tests.test_memory_assets tests.test_trajectory_contract -v
```

覆盖 grid/PNG round-trip、内容去重、坏 hash、危险路径、单步/多步 episode、事件 replay、状态
连续性、不可用动作、resume 身份检查，以及原子替换失败不破坏旧 snapshot。

## 8. Phase B 的真实 ARC 语料

Phase B 已增加 ARC adapter、CD82 visible-grid deterministic policy、轨迹扰动器、固定 seed
explorer、corpus manifest/coverage/validation 和逐帧环境 replay。正式审查入口、96 条 episode 的
构成和验收数字见 [ARC 可靠轨迹语料](arc_trajectory_corpus.md)。

下一阶段把冻结 corpus 投影为 `MemoryRecord`，仍不调用 Agent API。现有 ARC runner 只有在离线
Memory -> Schema 路径稳定后才接入同一个 recorder。
