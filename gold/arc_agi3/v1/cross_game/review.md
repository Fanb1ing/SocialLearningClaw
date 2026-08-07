# ARC-AGI-3 跨游戏 Gold Schema — 审核稿

> 状态：provisional。当前抽象覆盖全部 25 个游戏，但成员中含 22 个尚未人工审核的游戏，因此不能发布为正式 Gold。

## 分层

- Level 2：单游戏机制，保存在 `games/<game-id>/schemas.json`；
- Level 1：至少三个游戏共同支持的机制族；
- Level 0：由多个 Level 1 机制族支持的任务系统结构。

跨游戏节点不靠关键词相似度自动合并。每条节点显式保存 `member_schema_ids`、`game_scope`、成员审核状态和全部源码证据；具体游戏的例外仍留在 Level 2。

## 节点

| 层级 | 抽象 | 游戏数 | 直接成员数 | 状态 |
|---|---|---:|---:|---|
| L0 | 交互环境由带前置条件的状态变换算子组成 | 20 | 5 | provisional |
| L0 | 局部操作可通过对象关系图产生非局部状态变化 | 14 | 4 | provisional |
| L0 | 规划是在有限资源内满足复合目标，并可利用有限恢复操作 | 23 | 3 | provisional |
| L1 | 四向输入提出受可通行条件约束的空间移动 | 9 | 9 | provisional |
| L1 | 点击选择决定后续操作作用的活动实体 | 7 | 7 | provisional |
| L1 | 先选择操作参数或程序，再执行后续状态变换 | 3 | 3 | provisional |
| L1 | 对象变换沿推挤、附着、层级或分组关系传播 | 6 | 6 | provisional |
| L1 | 非法复合变换不留下部分成功的中间状态 | 6 | 6 | provisional |
| L1 | 撤销从最近快照恢复可逆状态 | 6 | 6 | provisional |
| L1 | 玩家操作会推进自主单位的策略更新 | 5 | 7 | provisional |
| L1 | 可见结构被解释为图案、程序或重写序列 | 4 | 4 | provisional |
| L1 | 满足匹配关系的接触会消除、合并或完成配对 | 4 | 4 | provisional |
| L1 | 局部触发器切换影响后续可达性的世界状态 | 5 | 7 | provisional |
| L1 | 过关要求一组目标谓词同时成立 | 11 | 11 | provisional |
| L1 | 有限动作资源在目标完成前耗尽会触发失败 | 16 | 16 | provisional |

## 建议审核顺序

1. 先检查 `atomic_rejection`、`structured_interpretation` 是否抽象过宽；
2. 再检查 `finite_resource_failure` 是否需要拆成 GAME_OVER 与关卡内重置两类；
3. 单游戏成员审核完成后重新生成，只有成员状态不再 pending 的节点才可进入正式审核。
