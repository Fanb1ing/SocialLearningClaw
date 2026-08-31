# ARC-AGI-3 Gold Schema v1

固定版本 inventory 中的 25 个游戏已全部生成，共 183 个 levels、198 条 Schema。

- `accepted` / `accepted_and_revised`：已完成人工审核；
- `pending`：已通过源码锚定、结构验证、全关卡加载与 action 烟雾检查，等待人工语义审核；
- 每个游戏目录均含 `schemas.json`、`runtime_cases.json`、`coverage.json`、`validation.json` 和中文 `review.md`。

| 游戏 | Levels | Schema | 审核 | 核心任务 |
|---|---:|---:|---|---|
| [ar25-0c556536](games/ar25-0c556536/review.md) | 8 | 8 | pending | 移动实物及其镜像组合，使所有目标标记被组合图形覆盖 |
| [bp35-0a0ad940](games/bp35-0a0ad940/review.md) | 9 | 7 | pending | 在可翻转重力的平台空间中到达宝石/出口 |
| [cd82-fb555c5d](games/cd82-fb555c5d/review.md) | 6 | 18 | accepted_and_revised | 按参考图案绘制 10×10 画布 |
| [cn04-2fe56bfb](games/cn04-2fe56bfb/review.md) | 6 | 7 | pending | 移动和旋转彩色图形，使所有标记像素按同色两两重合 |
| [dc22-fdcac232](games/dc22-fdcac232/review.md) | 6 | 9 | pending | 操纵角色与起重机搭桥、开路并到达终点 |
| [ft09-0d8bbf25](games/ft09-0d8bbf25/review.md) | 6 | 6 | pending | 点击局部翻色，使每个线索周围满足相同/不同约束 |
| [g50t-5849a774](games/g50t-5849a774/review.md) | 7 | 7 | pending | 记录路径并生成延迟重放分身，协作到达阶段终点 |
| [ka59-38d34dbb](games/ka59-38d34dbb/review.md) | 7 | 8 | pending | 移动可选房间并利用推挤与爆炸，把两类对象送入对应容器 |
| [lf52-271a04aa](games/lf52-271a04aa/review.md) | 10 | 8 | pending | 选择同类物体并按可用方向跳跃消除，清空目标集合 |
| [lp85-305b61c3](games/lp85-305b61c3/review.md) | 8 | 5 | pending | 点击方向按钮移动关联部件，使两类部件落到各自目标 |
| [ls20-9607627b](games/ls20-9607627b/review.md) | 7 | 7 | pending | 在迷宫中修改携带物的形状、颜色和朝向，逐个匹配目标 |
| [m0r0-492f87ba](games/m0r0-492f87ba/review.md) | 6 | 8 | pending | 用对称控制或单独控制让四类图形两两重合配对 |
| [r11l-495a7899](games/r11l-495a7899/review.md) | 6 | 8 | pending | 拖动组成图形并放到颜色集合匹配的目标上 |
| [re86-8af5384d](games/re86-8af5384d/review.md) | 8 | 7 | pending | 移动并借助边界改变多个轮廓，使合成像素匹配目标模板 |
| [s5i5-18d95033](games/s5i5-18d95033/review.md) | 8 | 6 | pending | 按颜色成组旋转和伸缩线段，使所有端点覆盖目标 |
| [sb26-7fbdac44](games/sb26-7fbdac44/review.md) | 8 | 7 | pending | 重排嵌套序列节点，使递归求值结果与输出序列一致 |
| [sc25-635fd71a](games/sc25-635fd71a/review.md) | 6 | 9 | pending | 移动角色并在 3×3 面板输入法术图案，解除障碍后到达出口 |
| [sk48-d8078629](games/sk48-d8078629/review.md) | 8 | 10 | accepted | 伸缩链条并按参考颜色序列推动方块 |
| [sp80-589a99af](games/sp80-589a99af/review.md) | 6 | 7 | pending | 摆放容器并倾倒液体，让液体到达所有目标且避开危险区 |
| [su15-1944f8ab](games/su15-1944f8ab/review.md) | 9 | 8 | pending | 点击产生收缩圆形脉冲，移动和合并对象以满足目标区计数 |
| [tn36-ef4dde99](games/tn36-ef4dde99/review.md) | 7 | 8 | pending | 编辑二进制程序并运行，使对象的位置、尺寸、旋转和颜色匹配目标 |
| [tr87-cd924810](games/tr87-cd924810/review.md) | 6 | 7 | pending | 根据上方示例翻译规则，调整下方输出序列使其成为输入的正确翻译 |
| [tu93-0768757b](games/tu93-0768757b/review.md) | 9 | 9 | accepted | 沿通路避开/处理移动单位并到达出口 |
| [vc33-5430563c](games/vc33-5430563c/review.md) | 7 | 7 | pending | 点击隔板转移空间或交换两侧内容，使彩色物体到达匹配通道 |
| [wa30-ee6fef47](games/wa30-ee6fef47/review.md) | 9 | 7 | pending | 搬运绿色对象并与自主载具协作，把它们送到匹配目标区 |

## 跨游戏抽象

已生成 15 条 provisional 跨游戏 Schema：Level 0 为 3 条，Level 1 为 12 条，覆盖 25 个游戏。详见 [跨游戏审核稿](cross_game/review.md)。
