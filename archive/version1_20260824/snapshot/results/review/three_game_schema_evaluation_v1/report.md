# Learned Schema vs Gold Schema

## 结论

当前 learned Schema 能识别一部分动作/区域/终局相关性，但没有任何规则达到严格语义等价；分级分数只表示结构化代理相似度，不能作为正式论文主分数。

- strict learned precision: 0.000
- strict Gold recall: 0.000
- graded learned precision: 0.562
- graded Gold recall: 0.415
- graded semantic F1: 0.478
- partially covered Gold: 24/37
- action-signature recall: 1.000
- evidence traceability: 1.000

## 分游戏

| game | learned | Gold | graded precision | graded recall | partial coverage |
|---|---:|---:|---:|---:|---:|
| cd82-fb555c5d | 18 | 18 | 0.531 | 0.533 | 0.833 |
| sk48-d8078629 | 16 | 10 | 0.520 | 0.448 | 0.700 |
| tu93-0768757b | 16 | 9 | 0.640 | 0.142 | 0.222 |

## 按 Gold 类型

| kind | Gold | graded recall | partial coverage |
|---|---:|---:|---:|
| action_effect | 23 | 0.584 | 0.913 |
| goal | 3 | 0.640 | 1.000 |
| hazard | 7 | 0.000 | 0.000 |
| observation_semantics | 4 | 0.000 | 0.000 |

## 未覆盖 Gold

- `cd82-fb555c5d` — 左上 10×10 图案是当前关卡的目标图案
- `cd82-fb555c5d` — 中央偏下的黑色 10×10 方块是可绘制画布
- `cd82-fb555c5d` — 达到 100 次 action 时游戏失败
- `sk48-d8078629` — 场外同色链条给出各目标链条需要覆盖的颜色序列
- `sk48-d8078629` — ACTION6 点击可控链头可切换当前活动链条
- `sk48-d8078629` — 方向输入消耗有限移动预算，耗尽时失败
- `tu93-0768757b` — 箭头角色需沿蓝色通路到达黄色出口
- `tu93-0768757b` — 橙色单位正对一格外角色时激活并直线追击
- `tu93-0768757b` — 青色单位每回合沿通路前进，前方无路时反向
- `tu93-0768757b` — 品红单位正对两格外角色时激活，并以两步延迟模仿转向
- `tu93-0768757b` — 角色主动移入敌对单位所在格会逐阶段将其清除
- `tu93-0768757b` — 敌对单位主动移入角色所在格会逐阶段清除角色
- `tu93-0768757b` — 每次方向尝试消耗一步，步数耗尽时失败

## 未获 Gold 支持的 learned 节点

- `cd82-fb555c5d` — `schema_35ff352355b61d96` (ACTION6; no_effect)
- `cd82-fb555c5d` — `schema_41985149d3410205` (ACTION6; no_effect)
- `cd82-fb555c5d` — `schema_bee376c67bf45d49` (ACTION6; no_effect)
- `sk48-d8078629` — `schema_ea1e2112dc4454cf` (ACTION6; no_effect)
- `sk48-d8078629` — `schema_6c290ea6d312f1cf` (ACTION7; no_effect)
- `sk48-d8078629` — `schema_447cb1c8c4a9318f` (ACTION6; no_effect)

## 解释边界

`structured_arc_proxy_v1` 只使用 game、level、action/role、Schema kind、视觉区域和双语概念标签。
凡是只写“grid changes/no change/level completes”的 learned Schema 都被限制在 partial 以下，不能因
action 名相同判为 equivalent。该报告适合定位生成算法缺什么；正式主指标还需要冻结人工 alignment
fixture，并用人工或独立语义 judge 校准。
