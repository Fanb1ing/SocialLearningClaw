Schema:  自动生成的、自动管理的、多层的、图结构
（0)Schema 数据结构--定位在“对世界的认知”
Class SchemaNode():
Index: 编号
Level：层次，越小说明越通用，越大说明越具体
Desription：自然语言形式的Schema描述，内容形式“[触发条件 (Context/Perception) + 动作序列 (Action/Execution) + 预期结果 (Expectation)]”
Memory Index：列表形式，记录该Schema的来源memory、增强该Schema的正向memory与削弱该Schema的负向memory
RelatedSchema Index： 列表形式，记录该Schema上下游的邻居Schema以及同级的相似Schema Index
Relability Weight： 可靠性权重，可随反馈
（1) 生成: 自动生成，而非人类手动设定
从episode记忆中归纳而来，自动判断是否新增或和已有结点进行融合。新增则自动生成内容以及对应的邻居schema。
（2) 管理：更新、遗忘、整合去重
a. 更新权重：根据环境反馈更新相关schema的权重，不同层次的delta weight有所区别
b. 遗忘：设置mask，当任务完全不相关时隐藏底层schema；设计遗忘曲线，考虑邻居较少的结点，对偶发性事件进行适当删除/降级
c. 整合去重：定时任务进行结点整合

多层例如（仅做示例）：
底层:  玩游戏的原则，点击Action-->游戏环境触发某个变化，多次尝试同一个Action没有变化-->选择其他Action-->解决
中层：ARC-AGI的不同游戏之间的通用的rules，Action1-->某个东西向上
高层:  ARC-AGI-3中sk48游戏的rules, Action1-->左侧操作杆往上