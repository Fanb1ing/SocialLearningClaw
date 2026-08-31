# Ground Truth Schema 本轮工作总结与下次升级指南

更新日期：2026-08-13

本文集中记录本轮关于 Ground Truth / Gold Schema 的讨论、试做结果、失败经验和后续升级路线。
它是下一次升级的首要交接入口，但不是事实本身的替代品；开始工作时仍需重新核对源码、数据、
manifest、审核状态和验证报告。

相关入口：

- 总体方案：[`docs/gold_schema_generation.md`](gold_schema_generation.md)
- ARC-AGI-3：[`gold/arc_agi3/v1/README.md`](../gold/arc_agi3/v1/README.md)
- ContextMATH：[`gold/contextmath/v1/README.md`](../gold/contextmath/v1/README.md)
- IntPhys2：[`gold/intphys2/v1/README.md`](../gold/intphys2/v1/README.md)
- 项目长期记忆：[`docs/project_memory.md`](project_memory.md)

## 1. 本轮最终共识

### 1.1 Gold Schema 的定位

Gold Schema 是独立、特权的评测标注，用于判断 learned Schema 的质量。它不属于普通学习流程，
也不能在正常 benchmark 推理时被方法读取。

Gold Schema 与项目中的 learned Schema 有一个有意保留的区别：

- learned Schema 仍采用 `MemoryRecord -> SchemaNode` 架构，节点必须引用 durable memory ID；
- Gold Schema **不要求**由 memory 生成，可以直接引用固定源码、benchmark 数据、视频帧、
  可执行 witness 和人工审核记录。

这里“不要求 memory”只适用于 `gold/` 下的特权标注，不能反向修改 active learned Schema 的
memory-grounded 约束。

### 1.2 四条发布标准

一份 Gold Schema 必须同时满足：

1. **正确**：描述受到源码、环境行为、严格推导或可审核视觉证据支持；
2. **范围内完备**：声明范围内，完成任务所需的重要机制没有缺失；
3. **原子化**：一个节点只表达一个可独立验证、可复用的机制；
4. **不泄漏**：普通 runner 和学习方法不能读取 Gold；Schema 也不应直接提供完整解题或通关答案。

本轮曾讨论过第五条“必须来自 memory”，用户明确删除了这一条。不要在下一版重新加回。

### 1.3 标准答案不等于 Schema

这是本轮最重要的原则：benchmark 的 ground truth 通常只给结果，不给产生结果的机制。

- ARC 的 `WIN` 不说明怎样的状态变化触发过关；
- ContextMATH 的数字答案不说明必要推导；
- IntPhys2 的 possible/impossible 标签和 condition 不说明哪一帧、哪个物体发生了什么。

因此，不能把标签或答案换一种自然语言表述后称为 Gold Schema。标准答案只适合用作生成后的
一致性检查或验收信号。

### 1.4 Schema、事实、证据和 witness 必须分开

后续版本应明确区分四类内容：

- **Schema**：可复用的机制、不变量、条件—结果规则；
- **task/scene facts**：本题数字、初始布局、颜色、具体对象和具体帧；
- **evidence**：源码位置、文件 hash、帧 hash、对照观察；
- **witness/solution trace**：证明答案或判定正确的执行轨迹。

事实和 witness 可以绑定 Schema，但不应被包装成 Schema。CD82 的人工审核已经证明，这一分离会
显著改善粒度：逐关颜色、初始 UI、具体步数等被删除，真正改变规划或目标判定的机制被保留。

### 1.5 验证必须分轴报告

以后不要只输出一个笼统的 `passed`。至少分开记录：

- 数据库存与 hash 是否固定；
- 引用、依赖和 coverage 是否结构正确；
- 静态源码或可执行 witness 是否通过；
- 行为/视觉语义是否得到证明；
- 人工审核是 `pending`、`accepted` 还是 `rejected`。

自动测试通过只说明相应的机器可检验条件成立，不等于自然语言语义已经成为正式 Gold。

## 2. 本轮工作过程与状态

| Benchmark | 当前产物 | 审核状态 | 本轮决定 |
|---|---|---|---|
| ARC-AGI-3 | 25 games、183 levels、198 个单游戏节点；另有 12 个 Level 1 和 3 个 Level 0 跨游戏节点 | CD82、SK48、TU93 accepted；其余 22 个游戏 pending；跨游戏层全部 provisional | 冻结在当前 v1，不继续扩展 |
| ContextMATH | 6 个原题组、12 个 SG/CS 改写、23 个任务级节点、30 个 witness checks | 自动检查通过，但用户认为整体写得不够好；未接受 | 暂停，下一次先重做设计，不得直接扩到 60 题 |
| IntPhys2 | 四类 condition 各 1 个场景，共 4 场景、16 视频、8 个 pair assessments、8 个节点 | metadata/hash/配对通过；视觉语义全部 provisional | 先审核 pilot，尤其 Solidity，再决定是否扩到 89 个唯一场景 |

本轮开始时还收尾了 benchmark 扩充：

- ARC 使用固定的 25 游戏本地 inventory；
- ContextMATH 四个 AIME 测试 split 共 120 条，实际为 60 道原题各有 SG/CS 两种改写；
- IntPhys2 使用完整 Debug 60 和固定 Main 300，按四视频场景分组；两个 split 有一个完整场景重叠，
  因而是 89 个唯一场景、356 个唯一视频。

## 3. ARC-AGI-3 的有效经验

### 3.1 为什么 ARC 这一轮最成功

ARC 有固定源码和可执行离线环境，机制可以从 action 分支、状态修改、sprite 属性和
`next_level()` 条件直接追溯。这使“规则正确性”和“任务答案”之间存在清晰的证据链。

有效流程是：

1. 从固定 inventory 和源码提取 action、前置条件、effect、hazard 与 goal dependency；
2. 只把对规划有用的机制转成原子节点；
3. 用源码 hash 和行锚点做静态证据；
4. 对关键 transition 做局部运行 probe；
5. 以 coverage 检查 action、重要状态分支和所有 level goal；
6. 先做单游戏人工审核，再进行跨游戏归纳。

### 3.2 ARC 粒度准则

保留：

- action 的适用条件和状态变化；
- 会改变规划的约束、失败机制和资源限制；
- 真正改变胜利判定的例外；
- 多关共享、可参数化合并的机制。

删除或只作为 evidence/fact 保存：

- 每关颜色枚举、初始摆放和普通 UI 外观；
- 不影响规划的可视化进度条；
- 具体解法步数；
- 完整获胜 action sequence。

方向不同但机制完全相同的规则可以参数化合并；语义后果不同的碰撞或敌人行为应分开。

### 3.3 ARC 当前遗留风险

- 22 个批量生成游戏尚未逐游戏人工审核，运行 smoke 只证明能加载并接受 advertised actions，
  不证明每条自然语言解释正确；
- 15 个跨游戏节点依赖大量 pending 成员，因此不能作为 accepted Gold；
- 跨游戏审核应重点看“原子回滚”“结构解释”是否过宽，以及“有限资源失败”是否应拆分为
  hard GAME_OVER 与关卡内 reset；
- 下一次若升级，应创建 `v2`，不要直接覆盖已经可复现的 `v1`。

## 4. ContextMATH 的失败经验与重做方向

### 4.1 当前 pilot 做了什么

当前版本把同一原题的 SG/CS 合并为一个题组，将 surface alignment、任务级推导节点、可执行
witness、coverage 和 validation 分开。答案只出现在 witness/validation 中。这些工程边界是对的。

### 4.2 为什么当前质量仍不够好

用户明确反馈“ContextMATH 写得不是特别好”。虽然没有给出逐条修改意见，但从产物结构可以确认
下一次不能把当前 pilot 直接机械扩充，原因至少包括：

- 很多节点更像一份题目解法的步骤拆分，而不是可跨题复用的数学 Schema；
- `representation / derivation / calculation` 的命名只是步骤类型，还没有形成稳定的数学机制本体；
- task-specific 数字、解码、定理/策略、计算 witness 之间的边界仍不够清楚；
- 自动复算出正确答案只能验证解法结果，不能证明节点粒度适合当 Gold Schema；
- 仅凭 6 道题无法可靠设计跨题 Level 2 分类，但等全部 60 题写完再分类又会放大返工成本。

因此当前 6 题只能视为格式实验，不是下一版的模板，也不能标记 accepted。

### 4.3 下次重做 ContextMATH 前必须先回答的问题

1. 评测对象到底是“上下文解码能力”“数学策略/定理”“完整推导结构”，还是三者的组合？
2. learned Schema 与 Gold 的匹配单位是什么：节点、推导 DAG、题型机制集合，还是最终覆盖率？
3. 哪些内容属于题面事实，哪些是通用数学规则，哪些只是本题 witness？
4. 一个通用规则至少需要多少不同原题支持，才能从 Level 3 提升到 Level 2？
5. 正确性由官方解答、独立证明、符号执行、穷举，还是多种证据共同保证？

### 4.4 推荐的新结构

下一版至少拆成四层，不要把它们都叫 Schema：

1. `surface_alignment`：SG/CS 叙事片段到 canonical problem facts 的映射；
2. `problem_facts`：变量、给定条件、目标，全部 task-specific；
3. `gold_schemas`：定理、表示转换、求解策略、约束传播等真正可复用机制；
4. `solution_witness`：本题如何实例化这些机制并得到答案。

应先横向分析更多题目，建立小型数学机制词表，再选若干题反向标注验证；不要先逐题自由写一份
完整解答，再把每个步骤命名为 Schema。

## 5. IntPhys2 的有效经验与风险

### 5.1 四视频配对是正确的基本单位

每个 `SceneIndex` 有：

- `1_Possible` / `1_Impossible`；
- `2_Possible` / `2_Impossible`。

两组 counterfactual pair 可以互相校验：Possible 提供物理不变量成立的正对照，Impossible 提供
违规证据。不能只看 Impossible，也不能把 `condition=solidity` 直接改写成“发生了穿透”。

### 5.2 当前四类 pilot 的语义

- Permanence：物体经过遮挡后消失，或遮挡后凭空出现；
- Immutability：同一球体遮挡前后由蓝变红或由红变蓝；
- Continuity：蓝色方块没有可见路径却在左右杯位间换位；
- Solidity：下落箱体的倾转/直立响应与落点是否存在黄色障碍物不一致。

每个场景保存视频 hash、抽取帧 hash、时间/帧编号、四行 contact sheet 和中文审核稿。

### 5.3 IntPhys2 的关键风险

- 自动帧差只能定位候选时间段，不能命名对象或解释物理违规；
- uniform sampling 可能错过瞬时接触、传送或变形，应保存事件前、事件时、事件后的密集窗口；
- camera motion、遮挡和视觉相似性可能使简单像素差失效；
- condition-level Level 1 规则只由一个 pilot scene 支持时证据太弱，应保持 provisional；
- Solidity pilot 最难解释：它依赖时序碰撞响应，而不是单帧是否重叠，必须优先人工审核；
- 扩展到全量需要控制视觉标注质量，不能用 metadata 批量自动生成自然语言事件。

### 5.4 推荐的扩展方式

审核当前四个 pilot 后，先按 `condition × game family × camera` 分层选第二批，而不是直接处理剩余
85 个场景。每个 scene 应记录：

1. tracked object 与稳定属性；
2. event 前/中/后的时间窗口；
3. 两个 Possible 的正证据；
4. 两个 Impossible 的具体 violation predicate；
5. pair 间共享的不变量；
6. camera/occlusion 等替代解释是否已排除；
7. 人工审核人和状态。

只有多个不同 scene/family 均支持的规则，才提升为正式 condition-level Schema。

## 6. 跨 benchmark 的反模式

下一次升级时应主动避免：

- **答案重述**：把 gold label 或最终数字换句话说；
- **解法步骤膨胀**：把每个代数运算或观察动作都称为 Schema；
- **原始配置膨胀**：把颜色、初始位置、具体数值、帧号当成可复用规则；
- **烟雾验证冒充语义验证**：能加载、能运行、答案吻合不等于规则文本正确；
- **用一个格式强套所有 benchmark**：ARC 是 action/state，数学是定理/推导，视频是不变量/事件；
- **过早跨任务抽象**：成员节点未审核时，父节点只能 provisional；
- **先全量生成再审核**：pilot 粒度不对会导致全量返工，ContextMATH 已经给出这一教训；
- **隐式泄漏**：完整获胜序列、逐题答案或标签条件不能进入普通推理上下文；
- **覆盖范围不声明**：必须说明是单题、单场景、单游戏、condition、game family 还是全 benchmark。

## 7. 尚未完成的核心工作

本轮生成了 Gold 数据，但**没有完成 Gold evaluator**。下一次升级前必须定义 learned Schema 与
Gold Schema 如何比较，否则“Gold”只能人工浏览，无法用于正式实验指标。

至少需要决定：

- 节点匹配使用严格字段、语义蕴含、人工对齐还是混合方法；
- 是否分别报告 precision、recall、coverage、contradiction 和层级一致性；
- learned 节点比 Gold 更抽象或更具体时如何计分；
- 多个 learned 节点合起来覆盖一个 Gold 节点、或反过来时如何计分；
- 错误 Schema、缺失 Schema 与多余但正确 Schema 的惩罚；
- evaluator 是否能读取 privileged evidence，但绝不能把 Gold 注入被评方法。

推荐先人工建立一小批 learned-to-Gold 对齐案例，再冻结 scoring protocol；不要在看到模型结果后
反向调整匹配规则。

## 8. 下次升级的建议顺序

### 阶段 0：保护当前版本

1. 阅读本文、`docs/project_memory.md` 和目标 benchmark 的 README；
2. 检查当前源码、数据 hash、manifest 和 git 状态；
3. 保留 v1，创建新的 `v2` 或明确的 draft 目录；
4. 不覆盖人工写入的 `review.md`。

### 阶段 1：先定义评测合同

1. 明确 Gold 的用途和匹配指标；
2. 明确 annotation unit、抽象层级和 scope；
3. 为“事实 / Schema / evidence / witness”定义独立字段；
4. 定义 `pending / provisional / accepted / rejected` 的状态迁移；
5. 写出 leakage boundary。

### 阶段 2：审核而不是继续生成

优先级建议：

1. 审核 IntPhys2 四场景，首先判断 Solidity；
2. 审核 ARC 剩余游戏与跨游戏节点，若 ARC 仍保持冻结则只记录意见、不扩产；
3. 重新设计 ContextMATH 本体，废弃“逐解法步骤即 Schema”的默认思路。

### 阶段 3：第二批小规模验证

- IntPhys2：按 condition、family、camera 分层补充场景，检验规则能否跨 scene；
- ContextMATH：先建立机制词表，再挑跨题共享同一机制的题组做反向验证；
- 每批先完成 human review，再批量生成。

### 阶段 4：规模化与跨任务抽象

1. 仅用 accepted 的具体节点归纳上层 Schema；
2. 自动生成 coverage 和 dependency audit；
3. 上层节点继承并汇总成员证据和审核状态；
4. 任一关键成员 pending 时，父节点不能 accepted；
5. 最后才接入 evaluator 和正式实验。

## 9. 当前产物和复现命令

生成脚本：

```text
scripts/generate_arc_gold_batch.py
scripts/generate_remaining_arc_gold.py
scripts/generate_cross_game_gold.py
scripts/generate_contextmath_gold.py
scripts/generate_intphys2_gold.py
```

对应测试：

```text
tests/test_gold_schema.py
tests/test_contextmath_gold.py
tests/test_intphys2_gold.py
```

复现与检查：

```bash
.venv/bin/python scripts/generate_contextmath_gold.py
.venv/bin/python scripts/generate_intphys2_gold.py
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q socialclaw scripts tests
git diff --check
```

注意：生成脚本可能重写其管理的 review 文档。下一次执行前先检查脚本行为，并保护已经加入的人工
意见。ARC SDK 的正式支持环境是 Python 3.12+；当前 `.venv` 的历史验证环境可能不同。

## 10. 一句话交接

ARC 证明了“固定机制证据 + 小批人工定粒度 + coverage”是可行路线；ContextMATH 证明了“答案可复算”
并不意味着“Schema 写得好”；IntPhys2 证明了必须以 counterfactual 视频 pair 和人工事件语义为核心。
下一次不要先追求数量，应先冻结评测合同、修正 ContextMATH 本体、审核 IntPhys2 视觉语义，再创建
新的版本扩展。
