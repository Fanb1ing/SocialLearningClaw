# 三游戏 Schema 原型审查

这是由真实 CD82、SK48、TU93 轨迹离线生成的 Phase C 输出，不是手写 Schema。

- 输入：144 episodes / 2068 public-action steps；
- Memory：2068 transition + 333 window + 144 episode = 2545；
- Schema：40（CD82 17、SK48 11、TU93 12）；
- LLM/API 调用：0；Gold Schema 读取：0。

先看 `report.json`，再打开 `schema.json`。每个 node 的 `memory_index.source` 都是至少两条真实
transition memory ID；在 `memory.json` 搜索该 ID，可以找到原 episode/step、corpus 根目录和
pre/post 的 grid/PNG 引用。

示例 `schema_25fcaea6696918fb` 总结 CD82 的 ACTION1 有效状态变化，引用 47 条 transition；
`schema_d7d29be476a1670b` 则单独总结 ACTION1 的 no-effect 情况，引用 124 条反向/边界 evidence。
这说明原型不会把同一 action 的有效和无效结果粗暴合成一条规则。

注意：`transition_bucket_v1` 是管线 baseline，不是最终语义算法。其 trigger 目前只包含 game、
level scope、action 和 target role，尚未从视觉中命名具体对象或自动发现精细前置条件；这属于
Phase D 的 keyframe/window semantic induction 工作。
