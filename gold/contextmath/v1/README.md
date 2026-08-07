# ContextMATH Gold Schema v1 — 第一批审核稿

> 当前仅为 6 个题组的 pilot，不代表 60 个题组已经完成。所有节点均等待人工审核。

本批将同一 AIME 原题的 SG/CS 两种改写合并：叙事对齐保存在
`surface_alignment.json`，共享数学推理保存在 `schemas.json`，标准答案和精确复算只保存在
`witness.json` 与 `validation.json`。因此最终答案不会被包装成 Schema。
当前只生成任务级 Level 3 节点；跨题的 Level 2 数学机制将在 60 个题组完成后统一归并，
避免依据少量样本过早建立题型分类。

## 清单

| 题组 | Schema | Witness | 状态 | 审核 |
|---|---:|---:|---|---|
| `aime_2024:60` | 5 | 7 | pending | [review.md](problems/aime_2024_60/review.md) |
| `aime_2024:67` | 3 | 4 | pending | [review.md](problems/aime_2024_67/review.md) |
| `aime_2024:75` | 3 | 6 | pending | [review.md](problems/aime_2024_75/review.md) |
| `aime_2024:84` | 4 | 4 | pending | [review.md](problems/aime_2024_84/review.md) |
| `aime_2025:0` | 4 | 4 | pending | [review.md](problems/aime_2025_0/review.md) |
| `aime_2025:3` | 4 | 5 | pending | [review.md](problems/aime_2025_3/review.md) |

## 汇总

- 题组：6 / 60
- 样本改写：12 / 120
- 原子 Schema：23
- 可执行检查：30/30 通过
- 自动验证只能证明数据配对、推导依赖、精确复算和 coverage；语义粒度仍需人工确认。

## 复现

```bash
.venv/bin/python scripts/generate_contextmath_gold.py
```
