#### 项目哲学
现有agent学习方式从上到下分为四种形式，如下图。当前已有的下三层学习仍有不足缺陷，因此我们提出schema learning
学习定义为：基于经验优化任务性能的方法
学习方式：
金字塔式	修改内容	优化方法
第四层	Schema	抽象的、原则性的、可泛化的世界认知
第三层	Memory	Experience
第二层	Workflow/Harness	Cognitive Architecture
第一层	Model Weights	Agentic RL/SFT

#### Schema:  自动生成的、自动管理的、多层的、图结构
**（0)Schema 数据结构--定位在“对世界的认知”**

Class SchemaNode():

Index: 编号

Level：层次，越小说明越通用，越大说明越具体

Desription：自然语言形式的Schema描述，内容形式“[触发条件 (Context/Perception) + 动作序列 (Action/Execution) + 预期结果 (Expectation)]”

Memory Index：列表形式，记录该Schema的来源memory、增强该Schema的正向memory与削弱该Schema的负向memory

RelatedSchema Index： 列表形式，记录该Schema上下游的邻居Schema以及同级的相似Schema Index

Relability Weight： 可靠性权重，可随反馈

**（1) 生成: 自动生成，而非人类手动设定**

从episode记忆中归纳而来，自动判断是否新增或和已有结点进行融合。新增则自动生成内容以及对应的邻居schema。

**（2) 管理：更新、遗忘、整合去重**

a. 更新权重：根据环境反馈更新相关schema的权重，不同层次的delta weight有所区别

b. 遗忘：设置mask，当任务完全不相关时隐藏底层schema；设计遗忘曲线，考虑邻居较少的结点，对偶发性事件进行适当删除/降级

c. 整合去重：定时任务进行结点整合



多层例如（仅做示例）：

底层:  玩游戏的原则，点击Action-->游戏环境触发某个变化，多次尝试同一个Action没有变化-->选择其他Action-->解决

中层：ARC-AGI的不同游戏之间的通用的rules，Action1-->某个东西向上

高层:  ARC-AGI-3中sk48游戏的rules, Action1-->左侧操作杆往上



###### 来自memory调研的经验：
按照遗忘曲线衰减:(H-MEM),去重(G-memory)

根据反馈调整权重(H-MEM, M3-Agent) 

生成链接,用具体的实例链接两个schema(G-memory)

上下两层带positional index用来定位sub memories(H-MEM)

不一定要抽取重要信息做graph(memanto/storage is not memeory),我们可以全部保留过往轨迹往上加东西

要保留memory(和skill)层,不能只靠最顶层

```plain
从多条 episode / skill 中归纳 rule
为 rule 保留 evidence log
验证 rule 是否真的提升 downstream performance
发现 rule 过时或冲突时降级/废弃
在具体场景中决定是否应用 rule
```

**与memory系统的关系：**

memory系统依旧存在，用来存储知识点和过往发生的历史记录

Schema从memory系统中获得'观察',用来评估预测错误并更新schema







###### 
##### **分工：**
+ Schema生成与管理: 

侧重于Schema的生成与管理。Schema部分更关注对世界的认识，类似“world model”，区别于memory系统中的具体的事实和情景

输入：当前的schema+**<font style="color:#DF2A3F;">任务日志</font>**（原任务，llm action, 结果）

输出：更新的schema

指标：Schema的密度、Schema的命中率（**<font style="color:#DF2A3F;">以ground truth schema为准</font>**）、预测错误的下降速度、新任务命中已有schema的比例、抽象规则的凝练能力（各层级schema的比例）；

技术点:

（1）如何根据轨迹中的人类回答内容/问题反馈建立schema，比如：判断是否需要新建schema，是否需要修改已有schema内容，是否需要调整weight

（2）类比人类的重放和遗忘机制，实现Schema的自进化，比如合并类似的概念，抽象高层级的schema，遗忘不重要的细节。触发时机可以每个游戏level总结一次，完成x次任务总结一次



+ Agent Action部分：exploit 

侧重于Agent使用全量Schema解决任务的能力

输入：全量groundtruth schema+原任务

输出：完成该任务

指标：任务完成率、任务完成效率、选择schema的准确率、schema的召回率

技术点：

（1）如何在适当的时机使用适当的Schema：为了保证召回的覆盖度，尝试将关键词+query一同召回; LLM应自行控制何时进行记忆召回,而非简单的单次注入，（甚至不限于在回答任务时才可以使用）；

（2）如何基于Schema进行推理：如何在Schema的帮助下完成推理任务，怎么使用schema指导当前任务，如何规划任务等



+ Agent proactive部分：explore

侧重于LLM在有限schema下进行exploration能力

输入：删除部分关键schema的部分groundtruth schema+任务轨迹集

输出：缺失的schema（下一步探索方向），探索行为（向人提问/自行探索）

指标：判断当前schema是否不足的正确率，schema exploration的方向准确度

技术点：

（1）如何根据当前的schema和任务轨迹判断是否schema缺失：之前是根据reasoning链条的confidence连乘，但不是所有的问题reasoning链条都清晰，并且连乘的方法会受到推理链条长度的影响，如何设计算法能够真实可靠地反馈当前schema的confidence（充足且可靠）

（2） schema不足以完成任务时，需要找到增益最高的探索的方向【可能不能和Action部分解的太开？】

(3) 生成什么样的向人提问的问题，以补充缺失的Schema



##### 时间规划:ICLR投稿【原计划】
+ 7.8-7.21 Benchmark构建+训练数据集构建：构建任务轨迹集和ground truth schema，拆分schema获取与schema使用两部分
+ 7.22-8.12 分模块进行技术迭代，提升各自指标
+ 8.12-8.26 整体系统合并，根据测试结构迭代debug
+ 8.26-9月截稿：收尾所有实验，绘制图表，撰写论文 

