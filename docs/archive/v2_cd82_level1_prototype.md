# V2 通用视觉 Agent：CD82 Level 1 真实测试

状态：游戏无关代码和第一次真实模型运行已完成。20 步内没有完成 Level 1；结果按失败保留。

## 信息边界

认知系统从空 EFPS graph 开始。三个 Agent 共用同一个视觉模型边界，Main/Update 返回结构化结果，
Exploration 返回给 Main 的一段纯文本建议：

- Exploration 子 Agent从当前原始图片、公开动作参数合同、公开历史和只读 EFPS 生成一段探索建议；
- Main Agent自己生成暂定 goal hypothesis，是唯一选择环境动作的组件；
- Update 子 Agent从初始图片或公开 before/action/after 证据提出图更新；
- runtime 只校验动作、类型、引用、证据闭包和原子事务，不生成游戏语义。

认知 Agent没有收到游戏目标、对象标签、动作含义、CD82 坐标、goal mask、Gold、游戏源码或预制
路线。`game_id` 只由评测 harness 用来创建环境。SDK 的公开 action schema 会告诉 Agent 哪些动作
接受哪些参数；这是接口合同，不是游戏规则。

## 实际结果

运行配置：

```bash
.venv/bin/python -u scripts/run_v2_arc_online.py \
  --game-id cd82-fb555c5d \
  --model anthropic/claude-opus-4.8 \
  --output-dir outputs/review/v2_generic_cd82_level1_opus48 \
  --max-steps 20 --stop-after-levels 1 \
  --max-tokens 3000 --temperature 0
```

- Level 1 未完成，`public_levels_completed=0`；
- 20 个动作全部通过全新环境 replay；
- 61 次视觉模型调用；
- 最终形成 21 Evidence、19 Entity、35 Feature definition、50 assertion、2 Prototype、10 Schema、
  117 typed relation；
- 所有 Schema 都引用持久 transition Evidence；
- 模型累计约 1,285,957 token，说明完整图和多张证据图反复注入的成本过高。代码随后已把 acting
  view 去掉冗余 relation，并把 supporting evidence 图片限制为一张；本次结果不伪造为重跑结果。

## 中间过程

| 阶段 | Step | 实际现象 | 认知意义 |
|---|---:|---|---|
| 初始观察 | 0 | 模型从原始画面提出 6 Entity、1 Prototype，Schema=0 | 没有预制实体或动作知识 |
| 全动作覆盖 | 1–7 | 依次尝试 ACTION1、2、两次坐标点击、3、4、5 | 假设全部来自模型；首次 ACTION5 改变 50 cells |
| Schema 使用 | 8–10 | Main 开始引用 ACTION5/3/4 Schema | 有 Schema 时才出现 `schema_prediction` |
| 目标/坐标探索 | 11–13 | 尝试 ACTION2 和顶部两个点击位置 | 点击没有找到有效按钮中心，坐标视觉理解失败 |
| 重复利用 | 14–20 | 多次复用 ACTION3/4/5 Schema | 学到了局部动作相关性，但没有形成完成目标的规划 |

动作序列是：

```text
ACTION1, ACTION2, ACTION6(28,28), ACTION6(32,36), ACTION3,
ACTION4, ACTION5, ACTION5, ACTION3, ACTION4, ACTION2,
ACTION6(40,4), ACTION6(32,4), ACTION4, ACTION4, ACTION4,
ACTION3, ACTION5, ACTION5, ACTION3
```

主要正确学习包括：ACTION3/4 会在部分状态下造成约 200 cells 的大范围对象变化；ACTION5 首次在
某状态下改变 50 cells；相同动作在边界或已经生效后可能没有主体变化。主要错误包括：模型把画面
解释成“红框与黑白 cluster 的对齐/旋转问题”，没有识别真正的 target/canvas/palette 结构；坐标点击
没有命中有效中心；Main 对已知低信息动作重复利用过多；Update 产生了一些重复甚至互相冲突的
state-specific Schema。

## Timeline 怎么读

日常审查先读同目录的 `process.md`。它按 Step 0、Step 1……展开，并在每一步明确写出触发原因、
Explore/Main 收到的画面与认知摘要、两个 Agent的输出、实际动作、动作前后 PNG、公开结果、Update
输入/输出和 validator 图事务。顶部还有带前后图片链接的完整时间线总表。

其中“最近公开 transition”是当前动作以前最多 8 条公开交互摘要。除了动作、像素差分、环境状态和
level delta，它还必须包含 Update 对受影响 Entity 的语义分析。每个 Evidence ID 同时附带可解引用
内容：动作、结果、Entity 变化、前后 observation 指纹和 artifact IDs。它是 Explore/Main 的短期
工作记忆，不含目标、Gold、源码或事后解释，也不会自行触发某个动作。

当前保存的 `opus48` 运行发生在 Entity 级 transition 字段加入以前，因此旧 Step 会明确显示“旧版记录
缺失”，不会把事后分析伪装成当时 Agent 的输入。下一次运行开始，Update 缺少
`transition_analysis` 会直接校验失败；有像素变化时必须列出 `entity_changes`，或明确列入
`unassigned_visual_changes`。

2026-08-29 根据人工标注进一步发现：旧运行把每 8 cell 带定位线的 512×512 人类审查图误送给了
Agent，这导致模型把连续黑色区域误读成 2×2 blocks。最初改为无辅助线 64×64 原图；后续审查
发现该图过小，又调整为 512×512 最近邻放大但仍不加辅助线。人类单独看带线审查图；Schema
Evidence 也改成 before/after 原图对。旧 `process.md` 的 22 处标注
旁已直接补写解释，不改写历史输出。

同日的新代码重跑在完成 8 个 action 后，于 Step 9 Main 请求收到 OpenRouter `402 Payment Required`。
该目录只保留失败说明和未完成 trajectory，不作为新实验结果：
`outputs/review/v2_generic_cd82_level1_opus48_semantic_20260829/failure.md`。补充额度后必须从 Step 0
用新目录重跑；runner 现已增加逐步 partial checkpoint，未来 provider 中断不会再丢失模型/EFPS 过程。

`timeline.json` 顶层保存四类共享目录：

- `instruction_profiles`：三个 Agent的固定、无游戏语义职责；
- `input_catalog.observations`：Agent实际看到的公开 observation 和 PNG/grid artifact；
- `input_catalog.cognition_views`：expanded view hash、revision、发送字段、节点/Schema/证据 ID 收据。
- `input_catalog.evidence`：由 Evidence ID/引用可还原的动作、结果、Entity 变化和 artifact IDs。

每个 step 按以下顺序记录：

1. `shared_decision_input`：Explore/Main 共同收到哪些 observation、action contract、EFPS revision 和历史；
2. `agent_calls.exploration_agent`：模型给 Main 的一段纯文本探索建议；
3. `agent_calls.main_agent` / `decision`：goal hypotheses、最终动作、Schema prediction 或 exploration hypothesis；
4. `environment_transition`：公开动作结果、Entity 语义变化及可解引用 Evidence；
5. `update_input` / `agent_calls.update_agent`：Update 实际收到的前后图片、动作、结果和认知；
6. `cognitive_update`：validator 接受的事务、前后计数和警告。

## 只保留的审查文件

1. `report.md`：最短结果摘要；
2. `process.md`：按时间顺序的人类可读完整过程；
3. `timeline.json`：必要的机器可读 Agent输入/输出审计；
4. `cognition/graph.json`：唯一最终 EFPS 和 audit log；
5. `trajectory/episodes/` 与 `trajectory/assets/`：replay 和视觉 Evidence。

不再另存重复的 `summary.json`、`evidence.json`、`manifest.json` 或每个 revision 的 graph snapshot。

## 下一步

当前最需要解决的不是增加 CD82 规则，而是三个通用问题：

1. 用模型生成的候选区域配合通用坐标校准，而不是游戏专用坐标；
2. 让 Exploration 评分真正抑制已证实的 no-op/repeated probe；
3. 对同 action 的条件分支做冲突检测和 Schema consolidation，再让 Main 做可执行多步规划。
