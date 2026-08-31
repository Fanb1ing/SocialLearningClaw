# IntPhys2 Gold Schema v1 — 四类物理规则 pilot

> 当前为四个场景的审核稿，不代表 89 个唯一场景已经完成；所有视觉语义均为 provisional。

本批每种 condition 选择一个四视频场景。Metadata、视频 hash、帧范围和 possible/impossible
配对由程序验证；具体物体事件由成对帧判读，并保留人工审核门。标签只用于生成后的
一致性检查，不作为 Schema 的推导依据。

## 清单

| Condition | Scene | Family | 审核 |
|---|---|---|---|
| `permanence` | `debug:3` | `HotAirBallonTwoDist` | [review.md](scenes/debug_3/review.md) |
| `immutability` | `main_300:21` | `HotAirBallon` | [review.md](scenes/main_300_21/review.md) |
| `continuity` | `main_300:19` | `RotatingCup` | [review.md](scenes/main_300_19/review.md) |
| `solidity` | `debug:1` | `SolidityFallingFlat` | [review.md](scenes/debug_1/review.md) |

## 汇总

- 唯一场景：4 / 89
- 唯一视频：16
- Pair assessments：8
- Schema：8（4 条 Level 1 condition invariant + 4 条 Level 3 scene discriminant）
- 自动验证：passed；视觉语义：provisional_review_pending。

## 复现

```bash
.venv/bin/python scripts/generate_intphys2_gold.py
```
