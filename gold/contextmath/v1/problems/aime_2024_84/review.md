# aime_2024:84 Gold Schema v1 — 人工审核稿

> 状态：自动验证通过，等待人工审核。标准答案只存在于 witness/validation，不作为 Schema 节点。

## 问题结构

三个正变量的商具有给定二进制对数，要求一个加权乘积的对数绝对值。

## 原子 Schema

| 顺序 | 类型 | Schema | 依赖数 |
|---:|---|---|---:|
| 1 | `representation` | 用对数变量把乘除约束线性化 | 0 |
| 2 | `derivation` | 两两相加线性方程隔离单个对数变量 | 1 |
| 3 | `derivation` | 把幂乘积的对数转成指数加权和 | 1 |
| 4 | `calculation` | 取绝对值并将有理数化为最简分数 | 1 |

## 证据与验算

- SG/CS 语义对齐：2/2 完整；详见 [`surface_alignment.json`](surface_alignment.json)。
- 可执行检查：4/4 通过；详见 [`witness.json`](witness.json)。
- 推导 coverage：上下文对齐、必要推理链和可执行答案检查均完整。

## 请重点审核

1. 每条节点是否只表达一个可复用的推理机制；
2. 是否存在为了得到正确答案仍不可缺少、但当前推导链遗漏的步骤；
3. SG/CS 的叙事语义是否都正确映射到同一原题结构；
4. 是否有只属于本题数字事实、不应称为 Schema 的内容。
