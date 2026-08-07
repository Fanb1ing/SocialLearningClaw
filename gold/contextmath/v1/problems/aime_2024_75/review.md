# aime_2024:75 Gold Schema v1 — 人工审核稿

> 状态：自动验证通过，等待人工审核。标准答案只存在于 witness/validation，不作为 Schema 节点。

## 问题结构

每人至少拥有一个共同物品，并给出另外三类物品人数及恰有二、三件的人数。

## 原子 Schema

| 顺序 | 类型 | Schema | 依赖数 |
|---:|---|---|---:|
| 1 | `representation` | 用逐物品计数得到总拥有次数 | 0 |
| 2 | `derivation` | 扣除每人必有的一件以计数额外拥有次数 | 1 |
| 3 | `calculation` | 从额外拥有次数方程隔离恰有四件者 | 1 |

## 证据与验算

- SG/CS 语义对齐：2/2 完整；详见 [`surface_alignment.json`](surface_alignment.json)。
- 可执行检查：6/6 通过；详见 [`witness.json`](witness.json)。
- 推导 coverage：上下文对齐、必要推理链和可执行答案检查均完整。

## 请重点审核

1. 每条节点是否只表达一个可复用的推理机制；
2. 是否存在为了得到正确答案仍不可缺少、但当前推导链遗漏的步骤；
3. SG/CS 的叙事语义是否都正确映射到同一原题结构；
4. 是否有只属于本题数字事实、不应称为 Schema 的内容。
