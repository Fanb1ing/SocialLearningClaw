## 项目定位:
Proactive Social learning Agent
### Motivation:
当前的self-evolve agent架构往往只强调如何从过往trace中总结经验-->书读百遍其意自现。
但trace的来源都是被动的,自己试错的, 没有主动学习,没有别人的经验。
我们希望让agent自发地询问获取人类经验，通过人类回答获得举一反三能力-->类比传道授业解惑。

### Keyword:  
Social 社会进化（能够询问别人的智能体，通过非参数方法（如skill/schema等方式）优化）  
Proactive 主动（自行挑选query，主动提问/探索, 有针对性、有意识地“补短”）  
Efficient 有效（相比于自进化架构，有针对性的提问将实现更少样本学习，降低token消耗）  
Transferable 可迁移（agent在知识/skill层面进行自进化，学习总结的是能泛化的、抽象出来的rule/insight，而非具体某些细节,具备跨benchmark的能力提升）  

### 定位：
提出agent学习的新范式，从神经网络层token层面上升到涌现的agent层concept的推理认知。
对标Agent层面工作(Openclaw/Hermes Agent) 
不是对标GEPA等优化算法：因为仅仅优化一个文本不能实现举一反三, 得评测有skill加持的agent能力上升了。项目中可以调GEPA,比如问了问题之后怎么优化skill

## 项目方案: 

### 评测问题:
（1）抽象逻辑推理	ARC-AGI-3	135个人类手工设计的全新交互推理环境,测试模型面对未知的能力  
需要在探索环境的反馈中推断规则,并迁移到下一关  
最接近"人类智能本质"的测试,最优模型回答率仅1  
（2）上下文学习 	CL-bench	500个复杂上下文,1899个任务,测试检索和阅读理解长上下文能力.  
测试模型从新颖、复杂且非公开的上下文中提取信息、学习新知识，并将其应用于解决实际问题   
无法仅依赖预训练知识。当前解决率仅为17.2%,出现"上下文忽略","上下文误用"	混元 2026.3	长上下文的文本召回器  
（3）世界模型	Cosmos-Reason1	评估模型的物理常识推理和具身推理方面问题 但AV尚未开源	NVIDIA 2025.3	真实物理世界规律  
PBench	模型需要根据输入的图像/视频和文本提示，判断某个物理场景或预测是否成立	NVIDIA 2025.6  
CLEVRER	视频问答VQA，描述性、解释性、预测性、反事实问题	ICLR 2020 Oral cite707  
考虑换成Blink(“微观视觉感知（Perception）”,两张极其相似的图里有几处不同（Spot the difference）、3D 形状的旋转匹配、物体的相对远近深度判断、视错觉图形的辨别、线段长度的微小对比。)  
考虑换成
（备注：考虑增加一个违背现实的图片benchmark）  

### 技术框图: 

0. 基于端侧个性化Agent进行，目前只需要直接调用LLM API
1. 给定一个问题到Agent
2. schema匹配：Agent先用LLM提取该问题所需的关键concept名称，再逐个与schema库中的concept做embedding相似度匹配。若有concept未匹配上（missing），认为缺少必要知识，启动提问；否则进入第三步。
2.5 提问：因缺乏必要信息的主动提问，通过UI向人类提问，描述问题和未知的概念。获取答案后，将答案更新到schema库中
3. 解答问题：Agent在schema库的帮助下回答该问题，包括读取Schema的信息，在schema的知识网络中reasoning等方式
4. 评估：判断回答是否正确，若正确进入Schema巩固环节，若错误进入schema纠错环节
5. schema巩固：回答正确的题目和答案可以作为正信号，加强Schema中相关concept之间的概率
6. schema纠错：回答错误的题目和答案可以作为负信号，更正schema中相关concept之间的概率。如果在schema reasoning过程中概率比较低，出现负面信号后，只需要agent自行更新；但若schema reasoning过程中的概率很高，那么需要向人类提问，帮助找到schema中的问题。

### 技术框图（ARC-AGI-3）：【TAB：这里的流程和实际实现有出入，重新调整】

0. 基于 ARC-AGI-3 交互式环境（`arc_agi.Arcade`），LLM 作为 Agent 观察 grid（文本编码）并选择 action
1. 创建环境，reset() 获取初始 observation（grid + available_actions + state），初始化 Schema 存储（按 run 隔离，`runs/arc_agi3/{model}/{timestamp}/schema/`）
2. 每步循环（step 0 → max_steps）：
   2a. Grid 感知：BFS 连通区域算法从 grid 提取 objects（颜色、位置、面积、质心）  
   2b. Objects → Concepts + Spatial Relations：每个 object 映射为一个 Concept（含坐标/面积描述，confidence 0.6），objects 之间按质心最近邻（nearest 3）生成空间 Relation（above/below/left_of/right_of，weight 0.5）  
   2c. Transformation Rules：若上一步有 pre objects，通过颜色+质心最近邻匹配前后帧 objects，生成 `transformed_by_<action>` relation  
   2d. 注入 Schema：将当前步 concepts/relations 加入 SchemaGraph，计算 embedding；按当前步 concept ids 提取 depth=1 子图作为 prompt context  
   2e. Agent 决策：LLM 阅读 grid 文本 + schema 子图（"Known objects/rules from schema"），输出 JSON（action + data + reasoning），解析并选择 action  
   2f. 执行 action：env.step(action, data)，获取下一帧 observation  
   2g. 检查关卡结果：WIN/GAME_OVER/TIMEOUT → 进入对应流程  
3. 关卡结束后：
   - WIN → Schema 巩固：相关 concept confidence +0.05（上限 0.95），推进到下一关
   - GAME_OVER → Schema 纠错：相关 concept confidence -0.05（下限 0.1），重试同一关（schema 已被修正）
   - TIMEOUT → 推进到下一关
4. Schema 持久化：每一步 action 执行后立即 save schema（concepts.jsonl + relations.jsonl + embeddings），不再等到 level 结束


### 技术细节及优化方向：
1. schema库的形式
2. schema库embedding模型的训练
3. agent如何利用schema库，如何评估在schema上Reasoning的自信度
4. 怎么初始化schema库，怎么根据反馈信号更新schema库
5. 怎么把人类的非结构化输入整理为schema

### 目标:
可学习进化：随着轮数上升,   
Social交互更高效更有前景：对比baseline/人类/our method学习曲线(1)样本效率：指标性能提升效率大于自动优化方法,接近甚至超过人；（2）学习天花板：“脚手架效应”，在人类帮助下可以提高学习的天花板  
可泛化性强：少量样本上学习即可实现指标提升；同类型不同benchmark、不同基座模型之间、不同任务（如编程python语言和C语言）之间具有可迁移性  
安装简单：整个项目包装成skill/plugin，适配不同的agent架构，下载即用

## 开发阶段[TAB:这里有没有和实际实现不统一的地方？]

> **总体原则**：先跑通主实验（Schema 辅助答题端到端），baseline 后续补充；ARC / CL-bench 数据接口优先准备。

### Stage 1：Schema 基础设施 + 静态 Schema 辅助答题（端到端验证）✅

目标：把旧版「文本 Skill」循环替换为「结构化 Schema 网络」循环，~~验证静态 Schema 就能带来答题提升~~。**已实现并跑通。**

1. **Schema 数据建模与存储**
   - 定义 `Concept`（id, name, description, category, embedding, confidence, source, **neighbors**）
   - 定义 `Relation`（source, target, relation_type, weight, evidence）
   - 实现 `SchemaGraph`：增删查改、子图提取、confidence 计算、**邻居动态计算**
   - **Schema 按 run 隔离**：不再使用全局共享目录，而是像 episode 一样作为日志保存在 `runs/{benchmark}/{model}/{timestamp}/schema/` 下，不同 context / 游戏天然隔离

2. **Embedding 检索与「Concept 充足度」判断**
   - 接入开源 Embedding 模型（`BAAI/bge-small-en-v1.5`，可替换）
   - **先由 LLM 提取问题所需 concept 名称列表，再逐个与 schema concept 做 embedding 相似度匹配**
   - `is_sufficient` 基于「未匹配到的 concept（missing）是否为空」判断，无硬编码阈值
   - 若 insufficient，标记该问题需要「主动提问」

3. **Schema 初始化（Agent 自动生成）**
   - **不做旧 Skill 迁移**；Agent 读题后自主输出结构化 concept + relation
   - 支持 `--auto-yes` 模式（测试时自动跳过人类提问，由 Agent 生成 schema）
   - 支持 CLI 向人类提问，解析自由文本回答为结构化 Concept/Relation

4. **Agent + Schema 答题 Pipeline**
   - 改造 prompt：注入检索到的 Concept（name + description）和 Relation（source → target）
   - Agent 在 Schema 网络上 reasoning，输出 reasoning_trace
   - **Confidence 由 SchemaGraph 基于几何平均计算，不交给 LLM 输出**

5. **评测框架扩展**
   - 抽象 `Problem` 基类（增加 `retrieval_query` 字段）与 `Evaluator` 接口
   - **PBench**（MCQ，选择题字母匹配）
   - **CL-bench**（长上下文阅读理解，已跑通，增加 LLM-as-judge 评估）
   - **ARC-AGI-3**（交互式环境已跑通，每关生成标准 Episode，支持 schema-based object/rule 学习）

6. **Prompt 全英文化**
   - `prompt_builder.py`、`human_io.py`、`evaluator.py`、`schema/initializer.py`、`schema/retriever.py`、`agent/openai_compatible.py` 中所有中文 prompt / UI 文本 / 解析标记已改为英文
   - 解析标记从 `[推理过程]` / `[最终答案]` 改为 `[Reasoning Process]` / `[Final Answer]`

7. **Episode 格式标准化**
   - `HOW_TO_READ_EPISODES.md` 已更新，与实际的 `episode.json` 结构对齐
   - `problem.meta` 按 `problem_type` 分层（mcq / long_context / arc_grid）
   - `attempts[].raw` 字段补充说明（response / meta / messages）

8. **主实验**
   - 「静态 Schema → 答题」完整链路已跑通（PBench + CL-bench）
   - 记录正确率、token 消耗、Schema 命中率、Concept 覆盖率
   - **Baseline 留到 Stage 3 补充**

### Stage 2：动态更新 + 主动提问 ✅（已实现并跑通）

1. **Schema 巩固/纠错 ✅**
   - 正反馈（答对）：相关 concept confidence +0.05，relation weight +0.05（上限 0.95）
   - 负反馈（答错）：相关 concept confidence -0.05，relation weight -0.05（下限 0.1）
   - **trace 中的 concept/relation 名称支持模糊匹配**（精确 → 大小写不敏感 → 子串包含 → difflib 相似度），解决 LLM 输出自由文本的匹配问题
   - 高自信错误（confidence > 0.8 但答错）：触发 CLI 向人类提问纠错

2. **主动提问 UI ✅**
   - 缺失 concept 时：CLI 展示 missing concepts，向人类提问，解析回答写入 Schema
   - 高自信错误时：CLI 展示推理路径 + confidence，向人类确认 concept / relation 是否正确

3. **ARC-AGI-3 交互式环境 ✅（已实现并跑通）**
   - `dataset/arc_agi3.py`：封装 `arc_agi.Arcade`，`reset()` 返回初始 observation，`get_available_actions(obs)` 从 observation 读取可用 actions
   - `schema/arc_agi3_parser.py`：Grid -> Object 提取（连通区域 BFS 或 LLM Vision）-> Schema Concept/Relation；新增 `compute_grid_diff`（逐像素对比 pre/post grid）和 `build_action_effect_concepts_and_relations`（生成 Action concept + `no_effect`/`affected` relation）
   - `arc_runner.py`（核心逻辑）+ `run_arc_agi3.py`（CLI 入口）：多轮 action/observation 循环，每关生成**标准 Episode**（problem/attempts/evals/flags/stop_reason）；Prompt 中 schema 注入限制为 confidence top 10 concepts + 前 10 条 transformation rules；**执行 action 后提取 post-action grid 并更新 schema**，供下一步复用
   - 修复 double-stepping：移除 loop 内多余的 `env.step()`，每 agent step 只执行一次 chosen action
   - Schema 表示：Object（颜色块/形状）+ Spatial Relation（above/below/left_of/right_of，限 nearest 3 邻居）+ Transformation Rule（action 触发的 object 变化）+ **Action-Effect Relation（`no_effect` / `affected`）**
   - **所有 prompt 已统一为英文**（包括 ARC-AGI-3、CL-bench、PBench 的系统指令和 UI）
   - **ARC-AGI-3 现已集成主动提问**：每个 level 开始时检查 schema 充足度，不足时通过 CLI 向人类提问（或 `--auto-yes` 自动跳过）；关卡失败且 reasoning_confidence 高时触发人类纠错
   - **增强健壮性**：LLM/API 调用异常时自动 fallback 到默认 action，避免进程 crash；支持 `--always-ask-correction` 调试模式
   - **Prompt 与 reasoning_trace 兼容**：ARC-AGI-3 的 JSON 输出格式中嵌入 `concepts_used` / `reasoning_path` / `explanation`，可被正确解析为结构化 trace，用于 confidence 计算和 schema 动态更新
   - **修复 grid 显示 bug**：`grid_to_text` 中心裁剪从 16×16 扩大到 32×32，避免 LLM 只看到单一颜色（如 sk48 中心全为 Yellow）
   - **修复 transformation rule 生成逻辑**：`diff_objects_to_rules` 从笛卡尔积改为颜色+质心最近邻匹配，消除无意义的 1-to-many 规则
   - **精简 Episode 体积**：`AttemptRecord` 移除 `raw` 字段，不再保存完整的 LLM 原始响应
   - **Per-step Schema 持久化**：每一步 action 执行并更新 schema 后，立即调用 `storage.save()` 落盘，不再等到关卡结束
   - **Action-Effect 反馈学习（防循环）**：
     - 每一步记录 `pre_grid`，执行 action 后与 `post_grid` 逐像素对比（`compute_grid_diff`）
     - 若 grid 无变化：生成 `no_effect` relation（weight=0.9），LLM prompt 中显示 `ACTION6 at (x,y) had no effect on grid`
     - 若 grid 有变化：生成 `affected` relation，将 action concept 与受影响的后置 object concept 关联
     - 该反馈直接注入 prompt 的 `Learned action effects:` 区块，防止模型重复点击无效坐标
   - **Trajectory JSON**：每关结束后生成 `trajectory.json`，直观记录每一步的 action、坐标、state、`grid_changed`、新增的 schema concepts
   - **修复 LLM Vision 坐标尺度 bug**：概念提取 prompt 原为静态文本，导致 LLM 返回 pixel 坐标（如 200~600）。改为 template 动态填入 grid 尺寸，明确要求返回 grid-cell 索引（0~h-1, 0~w-1），并校验边界、丢弃越界概念
   - **Episode 记录模型名称**：`Episode` 新增 `model` 字段，保存每次运行使用的 LLM 模型名
   - **Correction 门槛下调**：`correction_conf_threshold` 默认值从 `0.6` 改为 `-1.0`，确保 timeout/失败后必定触发人类纠错提问（便于调试）

4. **CL-bench 多轮对话修复 ✅**
   - **数据预处理修复（`scripts/download_clbench.py`）**：原 `prepare()` 仅保留最后一条 user/assistant 消息，中间轮次的对话历史（含 API 文档、前轮回答）全部丢失。修复后所有非最后轮次的 system/user/assistant 消息均以 `[role]: content` 格式拼接为 context，确保每一轮模型都能看到完整对话历史。
   - **Task 排序修复（`pipeline.py`）**：原按 UUID 字母序排序，违反对话轮次顺序（e59912e9 应先于 b06df7c8）。修复后按 `meta.msg_count` 排序（消息数少 = 早期轮次），保证 context-aware 模式下 schema 按正确对话顺序累积。
   - **验证结果**：修复后 context 1（B2B Sales）3/3 correct（修复前 0/3）；context 2（Travel/API）1/3 correct（修复前 0/3，b06df7c8 首次正确识别了 API 限制 + 用户 profile）。合计 4/6 vs 修复前 0/6。
   - **Schema 按 context 目录隔离**：`group_by_context` 模式下每个 context 的 schema 保存到 `run_dir/<context_id>/schema/`，不再共用一个全局目录导致互相覆盖。

### Stage 3：全量评测与迁移

1. **Baseline 补充**
   - Baseline A：无 Schema（裸 LLM）
   - Baseline B：旧版文本 Skill（Stage 1 历史实现）
   - 对比指标：正确率、样本效率、token 消耗、学习曲线

2. **跨 Benchmark 迁移**
   - 同类型任务间 Schema 迁移（如 PBench → Cosmos-Reason1）
   - 不同基座模型间 Schema 复用

3. **Embedding 模型调整**
   - 当前 `no_effect` 反馈未能有效阻止 agent 重复点击同一坐标，需调整 embedding 检索策略（如让 action-effect concept 的 embedding 更具区分度，或引入负采样）
   - BGE 对 grid 对象摘要与人类反馈 concept 的语义匹配效果差，关键概念经常无法召回。需考虑替换为视觉-语言嵌入模型或基于属性/规则的硬匹配。

4. **已识别待修复问题（defer 到 Stage 3 或后续迭代）**
   - **人类纠错/补充的回答未正确持久化到 schema**：`parse_correction()` / `parse_human_answer()` 解析后的 concept 疑似未被正确写入 schema，或写入后被覆盖/丢失。
   - **人类回答后不应再回到 schema 充足度询问**：当前 `ask()` / `ask_correction()` 结束后会重新进入 `retriever.is_sufficient()` 检查，导致二次提问。应直接跳过 sufficiency 检查，回到主答题循环。
   - **每步重复抽取 grid concepts 造成冗余**：即使 `grid_changed=False`，代码仍会生成新的 post-action concepts，导致 schema 中堆积大量完全相同的 object。应仅在 grid 变化或首次观察时抽取，否则复用上一帧 concepts。
   - **关卡重试时 step 文件被覆盖**：timeout/失败后重试关卡时，`write_step()` 的固定路径 `step_{step:03d}.json` 导致前一次尝试的 step 记录被覆盖，丢失完整失败轨迹。需为每次 level 尝试分配独立子目录（如 `attempt_1/step_001.json`）。
   - **Action-effect 反馈未注入 prompt（category 不匹配）**：`build_action_effect_concepts_and_relations()` 存入 schema 的 concept category 为 `"action"` / `"action_effect"`，但检索注入时（`run_arc_agi3.py`）的过滤条件是 `category == f"level_{current_level}"`（如 `"level_0"`），二者永不匹配，导致 action-effect concept/relation 无法被检索到并注入 prompt 的 "Learned action effects" 区块。修复方向：统一 category 命名，或直接按 `source == "action_effect"` 检索。

### Stage 4: 打包
1.  **Plugin 化打包**
   - 封装为可插拔 skill / plugin，适配不同 Agent 架构