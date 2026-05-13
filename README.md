## 项目定位:
Proactive Social-evolve Agent
### Motivation:
当前的self-evolve agent架构往往只强调如何从过往trace中总结经验-->书读百遍其意自现。
但trace的来源都是被动的,自己试错的, 没有主动学习,没有别人的经验。
我们希望让agent自发地询问获取人类经验，通过人类回答获得举一反三能力-->类比传道授业解惑。

### Keyword:  
Social-evolve 社会进化（能够询问别人的智能体，通过非参数方法（如skill/schema等方式）优化）
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


### 技术细节及优化方向：
1. schema库的形式
2. schema库embedding模型的训练
3. agent如何利用schema库，如何评估在schema上Reasoning的自信度
4. 怎么初始化schema库，怎么根据反馈信号更新schema库


### 目标:
可学习进化：随着轮数上升,指标性能提升
Social交互更高效更有前景：对比baseline/人类/our method学习曲线(1)样本效率：指标性能提升效率大于自动优化方法,接近甚至超过人；（2）学习天花板：“脚手架效应”，在人类帮助下可以提高学习的天花板
可泛化性强：少量样本上学习即可实现指标提升；同类型不同benchmark、不同基座模型之间、不同任务（如编程python语言和C语言)之间具有可迁移性
安装简单：整个项目包装成skill/plugin，适配不同的agent架构，下载即用

## 开发阶段

> **总体原则**：先跑通主实验（Schema 辅助答题端到端），baseline 后续补充；ARC / CL-bench 数据接口优先准备。

### Stage 1：Schema 基础设施 + 静态 Schema 辅助答题（端到端验证）✅

目标：把旧版「文本 Skill」循环替换为「结构化 Schema 网络」循环，验证静态 Schema 就能带来答题提升。**已实现并跑通。**

1. **Schema 数据建模与存储**
   - 定义 `Concept`（id, name, description, category, embedding, confidence, source, **neighbors**）
   - 定义 `Relation`（source, target, type, weight, evidence）
   - 实现 `SchemaGraph`：增删查改、子图提取、confidence 计算、**邻居动态计算**
   - **Schema 按 run 隔离**：不再使用全局共享目录，而是像 episode 一样作为日志保存在 `runs/<run_id>/schema/` 下，不同 context / 游戏天然隔离

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
   - **ARC-AGI**（接口抽象已完成）

6. **主实验**
   - 「静态 Schema → 答题」完整链路已跑通（PBench + CL-bench）
   - 记录正确率、token 消耗、Schema 命中率、Concept 覆盖率
   - **Baseline 留到 Stage 3 补充**

### Stage 2：动态更新 + 主动提问 ✅（部分实现）

1. **Schema 巩固/纠错 ✅**
   - 正反馈（答对）：相关 concept confidence +0.05，relation weight +0.05（上限 0.95）
   - 负反馈（答错）：相关 concept confidence -0.05，relation weight -0.05（下限 0.1）
   - **trace 中的 concept/relation 名称支持模糊匹配**（精确 → 大小写不敏感 → 子串包含 → difflib 相似度），解决 LLM 输出自由文本的匹配问题
   - 高自信错误（confidence > 0.8 但答错）：触发 CLI 向人类提问纠错

2. **主动提问 UI ✅**
   - 缺失 concept 时：CLI 展示 missing concepts，向人类提问，解析回答写入 Schema
   - 高自信错误时：CLI 展示推理路径 + confidence，向人类确认 concept / relation 是否正确

3. **ARC-AGI-3 交互式环境 ✅（接口已实现）**
   - `dataset/arc_agi3.py`：封装 `arc_agi.Arcade`，管理多 level 游戏循环
   - `schema/arc_agi3_parser.py`：Grid -> Object 提取（连通区域）-> Schema Concept/Relation
   - `run_arc_agi3.py`：多轮 action/observation 循环，每关通关后强化/修正 schema rules
   - Schema 表示：Object（颜色块/形状）+ Spatial Relation（above/below/left_of/right_of）+ Transformation Rule（action 触发的 object 变化）

### Stage 3：全量评测与迁移

1. **Baseline 补充**
   - Baseline A：无 Schema（裸 LLM）
   - Baseline B：旧版文本 Skill（Stage 1 历史实现）
   - 对比指标：正确率、样本效率、token 消耗、学习曲线

2. **跨 Benchmark 迁移**
   - 同类型任务间 Schema 迁移（如 PBench → Cosmos-Reason1）
   - 不同基座模型间 Schema 复用

3. **Plugin 化打包**
   - 封装为可插拔 skill / plugin，适配不同 Agent 架构