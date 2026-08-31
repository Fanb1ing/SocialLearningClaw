# Gold Schema 构建方案（精简版）

状态：ARC-AGI-3 按当前版本冻结。ContextMATH 已生成 6 个题组的第一批审核稿，因粒度仍需调整暂缓扩展。IntPhys2 已生成覆盖四种 condition 的四场景 pilot。

## 1. 定义

Gold Schema 是根据 benchmark ground truth 构建的特权评测标注，不要求来源于
`MemoryRecord`。它与 learned Schema 分开保存，普通 runner 和学习方法不可读取。

每条 Gold Schema 必须满足：

1. **正确**：符合标准答案、固定源码或可执行环境；
2. **完备**：在声明范围内覆盖得到正确结果所需的规则；
3. **原子化**：一条节点只表达一个可独立验证的机制；
4. **不泄漏**：只供离线评测和明确标记的 privileged 实验使用。

只记录对通关规划有用、可跨具体关卡复用的机制。逐关颜色集合、初始摆放、具体步数等
原始配置属于游戏事实，不单独作为 Schema；只有它们背后的选择、移动、失败等功能规则才保留。

标准答案本身不是 Schema。例如 `WIN` 只说明结果，不说明怎样的状态变化会触发它。

## 2. 最小数据格式

```json
{
  "schema_id": "arc_agi3:cd82-fb555c5d:L2:<hash>",
  "game_id": "cd82-fb555c5d",
  "level_scope": [1, 2, 3, 4, 5, 6],
  "abstraction_level": 2,
  "kind": "action_effect",
  "trigger": "Agent 可观察条件或必要的源码机制条件",
  "action_sequence": [{"action": "ACTION5", "arguments": {}}],
  "expectation": "预期的可观察变化或状态转移",
  "constraints": [],
  "exceptions": [],
  "relations": {"parents": [], "requires": []},
  "source_evidence": [],
  "runtime_evidence": [],
  "verification": {"static": "passed", "runtime": "passed"}
}
```

`source_evidence` 直接引用固定源码 hash 和 AST/函数分支，不经过 memory。

## 3. ARC-AGI-3 批次

当前批次以 `third_party/arc_agi3_games/inventory.json` 为唯一游戏清单：25 个游戏、183 个 levels。前三个审核样本为 CD82、SK48、TU93；剩余 22 个使用相同格式一次性生成。

源码变量名经过混淆，因此只依据控制流、sprite 属性、action 分支、状态修改和
`next_level()` 条件判断规则，不根据变量名猜测语义。

## 4. 精简生成流程

### 步骤一：提取源码事实

从固定源码中提取：

- available actions；
- 可观察物体及 UI；
- action 前置条件和 effect；
- 状态修改、约束和 hazard；
- 每个 level 的 `next_level()` 条件。

### 步骤二：生成原子 Schema

把源码事实转换为 game-wide level 2 规则和 level-specific level 3 规则。自然语言只使用
源码和 grid 能支持的描述；无法可靠命名的物体使用颜色、形状、尺寸和位置描述。

Gold Schema 描述环境机制和目标，不提供完整获胜 action sequence。Agent 仍需自行规划。

### 步骤三：验证与 coverage

- 静态验证：每条节点能定位到对应源码分支；
- 运行验证：已审核游戏对关键 action-effect 做局部离线执行检查；待审核游戏先做全关卡加载和可用 action 烟雾检查，不能把烟雾检查当作逐条语义证明；
- 完备性：每个 available action、相关状态分支和 `next_level()` dependency 都映射到
  Schema ID；
- 生成 `coverage.json`，存在必需缺口时不得发布。

## 5. ARC v1 产物

```text
gold/arc_agi3/v1/
  README.md
  schema_spec.json
  manifest.json
  cross_game/
    schemas.json
    validation.json
    review.md
  games/<game-id>/
    schemas.json
    runtime_cases.json
    coverage.json
    validation.json
    review.md
```

当前共有 198 条 Schema。CD82 已按意见从 26 条收紧为 18 条；SK48、TU93 的人工意见也已写回产物。其余 22 个游戏先保持 `pending`，不在人工审核前标记为 accepted。

## 6. ARC v1 验收条件

- 所有正式节点通过源码验证；
- 已审核游戏的关键 transition 通过局部运行验证；待审核游戏通过运行烟雾检查并保留人工审核门；
- 25 个游戏、183 个 level 的目标条件均被覆盖；
- coverage 中没有未解决的必需项；
- 不包含完整获胜动作序列；
- 普通实验默认无法加载 Gold 产物。

## 7. 跨游戏抽象

在 198 条 Level 2 单游戏 Schema 之上，生成了一版 provisional 跨游戏层：

- 12 条 Level 1 机制族：四向移动、活动对象选择、操作参数选择、关系传播、原子回滚、快照撤销、自主单位更新、结构解释、匹配接触归约、世界状态切换、复合目标、有限资源失败；
- 3 条 Level 0 系统结构：条件状态算子系统、关系状态系统、有限资源目标规划系统；
- 每条跨游戏节点都显式保存 `member_schema_ids`、`game_scope`、成员审核状态和由成员继承的源码证据；
- Level 1 至少需要三个成员游戏；Level 0 至少需要两个 Level 1 机制族。

这 15 条节点覆盖 25 个游戏，但由于 22 个成员游戏仍是 `pending`，全部标记为
`provisional`，不能当作正式 Gold 发布。人工审核应优先检查原子回滚和结构解释是否过宽，以及有限资源失败是否需要拆成 GAME_OVER 与关卡内重置。

## 8. ContextMATH

范围是四个 AIME 测试 split：60 道原题，每题各有 SG/CS 两种改写，共 120 条样本。
同一原题的两种改写共享数学推导，只分别保存上下文到原题语义的对齐。

当前 pilot 位于 `gold/contextmath/v1/`，包含 6 个题组、12 个改写、23 条任务级 Level 3
Schema 和 30 个通过的可执行检查。跨题 Level 2 机制暂不生成，等 60 个题组齐全后统一归并。

生成流程：

1. 以 `(year, id)` 合并 SG/CS，检查 `ori_question` 和标准答案一致；
2. 从 `ori_question` 重建精确推导，把上下文解码、约束建立、中间引理、计算和独立验算拆成原子节点；
3. 用整数/有理数运算、枚举或 SymPy 为每题保存可执行 witness，必须复算出标准答案；
4. 由任务级节点归纳数学题型级 Schema，但最终答案本身不作为 Schema；
5. 先审核一个覆盖两年、SG/CS 和多种题型的小批次，再生成全部 60 个题组。

发布前要求：两种改写的语义均被覆盖；推导 DAG 的每个叶节点来自题面或明确的上下文解码；
每个中间结论都有依赖和可执行检查；最终计算与数据集答案一致。`aime_2025` 的带单位答案
需要先规范化再比较。

## 9. IntPhys2

范围是 Debug 60 和固定 Main 300。它们合计 90 个 split 内场景，其中一个场景重复，实际为
89 个唯一 `SceneIndex`、356 个唯一视频。重复场景只标注一次，再由两个 split 引用。

当前 pilot 位于 `gold/intphys2/v1/`：四种 condition 各选一个完整四视频场景，共 4 个唯一
场景、16 个视频、8 个 pair assessments、4 条 provisional Level 1 condition invariant 和
4 条 provisional Level 3 scene discriminant。每个场景附带可直接审核的对照帧图。

生成流程：

1. 每个 `SceneIndex` 联合 `1/2 × Possible/Impossible` 四个视频；
2. 分别对齐两组 Possible/Impossible，定位最早的持续分歧区间，并保存时间戳、帧 hash 和对照帧；
3. 标注对象、状态变化、应保持的物理不变量及实际违规，把它们绑定到 permanence、immutability、continuity 或 solidity；
4. 从两个独立 pair 归纳 scene 级机制，再跨 scene 归纳 condition 级 Schema；Possible 视频保存不变量得到保持的正证据，不能只给 Impossible 写标签；
5. 先审核四种 condition 各一个场景，再扩展到 89 个唯一场景。

`type` 和 `condition` 只用于生成后的验收，不能直接改写成 Schema。自动帧差只能提出候选时间段；
最终事件语义必须经过人工审核。发布前要求四视频配对完整、证据时间有效、两个 Impossible
实例都支持同一 condition、Possible 对照不含所声明的违规，并且所有证据可由固定视频 hash 复现。

## 10. 执行顺序

ContextMATH 暂停在 6 题 pilot，待重新确认 Schema 粒度后再扩展。当前优先审核 IntPhys2
四场景 pilot，尤其是 Solidity 的碰撞响应解释；审核通过后再批量处理剩余场景。两者都沿用
ARC 的 `manifest + schemas + coverage + validation + review` 思路，但不复用 ARC 专属字段。
