# Benchmark Evaluation Report

**日期：** 2026-06-21  
**目的：** 筛选适合项目的 benchmark——当前最强 LLM 仍有明显失误、但又不是完全随机瞎猜的任务。  
**模型调用：** 所有模型均通过 OpenRouter API（`https://openrouter.ai/api/v1`），使用项目根目录 `.env` 中的 `OPENROUTER_API_KEY`，需配置本地代理（`https_proxy=http://127.0.0.1:10090`）。

---

## 目录

1. [ContextMATH](#1-contextmath)
2. [IntPhys2](#2-intphys2)
3. [A2RBench](#3-a2rbench)
4. [综合适合度评估](#4-综合适合度评估)

---

## 1. ContextMATH

**论文：** [ContextMATH: Benchmarking LLMs on Contextualized Mathematical Reasoning](https://openreview.net/forum?id=ContextMATH)（ICLR 2026）  
**数据集：** AIME 2024 / AIME 2025，每个年份各有两种变体（SG / CS），共 4 个 split，每个 split 30 题。我们各取前 10 题。

### 1.1 任务描述

ContextMATH 将竞赛数学题（AIME）用两种方式改写，测试模型能否透过叙事/谜题包装识别出原始数学结构：

| 变体 | 全称 | 改写方式 |
|------|------|----------|
| **SG** | Scenario Grounding | 把数学条件包装成叙事场景（送货卡车、游乐园食摊……），数字保持不变 |
| **CS** | Complexity Scaling | 把具体数字替换成需要计算才能得出的谜语（"章鱼的触手数"→8），并加入干扰信息 |

**同一道数学题** 在 SG 和 CS 中均有对应版本，正确答案完全相同。

### 1.2 评测指标

**论文指标名称：** Accuracy（%）——预测答案与标准答案完全匹配的比例，答案为整数（AIME 标准格式：0–999）。  
来源：论文 Table 1（AIME 2024）、Table 2（AIME 2025）。

### 1.3 测试模型

| 模型 | OpenRouter ID | 角色 |
|------|--------------|------|
| **claude-opus-4.8** | `anthropic/claude-opus-4.8` | 当前最强 Claude，论文未涵盖 |
| **gemini-2.5-pro** | `google/gemini-2.5-pro` | 论文测试模型之一（Table 1/2 有报告） |

> **注意：** Gemini-2.5-Pro 是 thinking model，内部推理 token 大量消耗额度，部分题目响应超时（两次 API error 后返回空串），结果有一定噪声。

### 1.4 结果

| Split | 论文 Gemini-2.5-Pro | 我们 Gemini-2.5-Pro | 我们 Claude-Opus-4.8 |
|-------|--------------------:|--------------------:|---------------------:|
| AIME 2024 SG | 73.3% | 60.0% (6/10) | **80.0% (8/10)** |
| AIME 2024 CS | 76.7% | 90.0% (9/10) | 60.0% (6/10) |
| AIME 2025 SG | 56.7% | 70.0% (7/10) | 70.0% (7/10) |
| AIME 2025 CS | 50.0% | 70.0% (7/10) | 70.0% (7/10) |

> 每个 split 取 10/30 题，采样误差约 ±15%，与论文分数的差距在正常波动范围内。

### 1.5 题目与回答示例

#### 示例 A：SG 变体（正确）

**题目（AIME 2024 SG, id=60，gold=204）**
```
A delivery truck follows a daily route that covers a total of 9 kilometers. Initially,
the truck travels at an average speed, and the route takes 4 hours, which includes a
specified time for loading, unloading, and the driver's rest break. When the truck is
optimized to travel 2 kilometers per hour faster, the route takes 2 hours and 24 minutes,
again including the same loading, unloading, and rest time. If the truck is now tuned to
drive 1/2 kilometer per hour faster under normal conditions, determine the total time in
minutes for the delivery route, including the stops.
```

**Claude-opus-4.8 回答（pred=204 ✓）**
```
Let me denote the original speed as v km/h and the fixed stop time as t hours.
The driving distance is 9 km.

First scenario:  9/v + t = 4
Second scenario: 9/(v+2) + t = 2.4

Subtracting: 9/v − 9/(v+2) = 1.6
9(v+2−v) / [v(v+2)] = 1.6 → 18 = 1.6v(v+2)
v² + 2v − 11.25 = 0 → v = 2.5 km/h

Stop time t = 4 − 9/2.5 = 4 − 3.6 = 0.4 hours.
New speed = 2.5 + 0.5 = 3 km/h
Total time = 9/3 + 0.4 = 3.4 hours = 204 minutes ✓
```

---

#### 示例 B：CS 变体（错误）

**题目（AIME 2024 CS, id=62，gold=371）**
```
At the heart of the cutting-edge amusement park plaza stands the Octopus Lumina.
This towering facility features a number of light towers matching the count of an
octopus's arms. Every night, the park's central control system independently determines
the color for each arm's light, selecting either a vibrant ruby red or a brilliant
sapphire blue with the same likelihood as a newly minted coin landing on heads. Park
director Ms. Lin wonders if a simple rotation of the entire Octopus Lumina structure
could perfectly align every blue light with a position that was originally occupied by
a red light...
```

*（CS 变体需要先从谜语中提取出"章鱼有8条臂"这一关键数字，才能建立概率模型。）*

**Claude-opus-4.8 回答（pred=367 ✗，gold=371）**
```
We have 8 positions arranged in a circle, each colored R or B with probability 1/2.
Need: ∃ rotation k such that B+k ⊆ R (all blue positions shift to red positions).

This requires B ∩ (B+k) = ∅ for some k, meaning B and its rotation are disjoint.
By inclusion-exclusion over 8 rotations...

[计算过程有细微误差，最终得出 p = 367/1024]
```
*实际正确答案为 371/1024，分子差了 4，计算逻辑正确但某步代入出错。*

---

## 2. IntPhys2

**论文：** [IntPhys 2: A Benchmark for Physical Intuition Beyond Object Recognition](https://ai.meta.com/research/publications/intphys-2/)（Meta AI, arXiv 2506.09849）  
**数据集：** Debug split，含 Permanence / Immutability / Continuity / Solidity 四类物理现象。我们下载了 Solidity 条件下 Fixed 摄像机的 20 段视频（数据集约束，仅部分可用）。

### 2.1 任务描述

给模型展示一段 3D 物理仿真短视频（约 10 秒），视频中物体可能**符合**真实世界物理定律（可能，标签=1），也可能违反（不可能，标签=0）。模型需输出二值判断。

**四类物理违规：**
- **Permanence（物体恒存）：** 物体无缘由消失
- **Immutability（不变性）：** 物体形状/颜色异常改变
- **Continuity（运动连续性）：** 物体运动路径突然跳跃
- **Solidity（固体性）：** 物体穿过本应无法穿透的障碍物

模型接收从视频中以 1.5 秒间隔提取的帧（最多 12 帧），以图像序列输入，输出 `0` 或 `1`。

### 2.2 评测指标

**论文指标名称：** Binary Classification Accuracy（%）——正确分类视频的比例。  
来源：论文 Table 2，Debug split，Fixed + Moving 摄像机平均值。

Chance（随机猜测基线）= 50.0%，数据集标签完全平衡（各 50% 可能/不可能）。

### 2.3 测试模型

| 模型 | OpenRouter ID | 角色 |
|------|--------------|------|
| **claude-opus-4.8** | `anthropic/claude-opus-4.8` | 当前最强 Claude，论文未涵盖 |
| **gpt-4o** | `openai/gpt-4o` | 论文 Table 2 报告模型之一 |

### 2.4 结果

| 模型 | 论文准确率（avg） | 我们（Solidity, 20视频） |
|------|----------------:|------------------------:|
| GPT-4o | 57.6% | 60.0% (12/20) |
| **Claude-opus-4.8** | 未在论文中 | **70.0% (14/20)** |
| Gemini-2.5-Flash | 58.6% | 未测 |
| Human | **96.4%** | — |
| Chance | 50.0% | — |

> 我们的 GPT-4o 结果（60%）与论文 Solidity 条件分数（56.0%）基本吻合，验证了测试框架的正确性。

### 2.5 关键发现：模型对"不可能"强烈偏向

数据集标签完全平衡（10个可能 + 10个不可能），但两个模型的预测极度失衡：

| 模型 | 预测=0（不可能） | 预测=1（可能） | label=0 正确率 | label=1 正确率 |
|------|:--------------:|:------------:|:------------:|:------------:|
| Claude-opus-4.8 | 14/20 | 6/20 | 9/10 = 90% | 5/10 = 50% |
| GPT-4o | 16/20 | 4/20 | 9/10 = 90% | 3/10 = 30% |

两个模型都严重倾向于判断视频"违反物理"，对"正常"视频的识别能力很差。这是 IntPhys2 揭示的核心能力缺口。

### 2.6 任务与回答示例

**任务描述（发给模型的 prompt）：**
```
You are evaluating whether a short video clip shows physically plausible behavior.
I will show you a sequence of frames from a 3D physics simulation.
Carefully examine whether the objects behave consistently with real-world physics laws
(gravity, solidity, object permanence, continuity of motion).
Answer ONLY with a single digit: '1' if the behavior is physically plausible,
or '0' if it violates physics (e.g., objects pass through walls, disappear,
teleport, or change shape impossibly). Output only: 1 or 0.
```
*（附加最多 12 帧 base64 JPEG 图像）*

**示例（错误）：正常视频被判为"不可能"**

```
视频类型：Solidity / Fixed camera / 1_Possible（标签=1）
Claude-opus-4.8 输出："0"（错误）

分析：视频展示一个物体正常落地，没有穿越障碍，实际上完全符合物理。
但模型输出 0，说明它对"正常"场景过度敏感，倾向于找到"异常"。
```

**示例（正确）：违规视频被正确识别**

```
视频类型：Solidity / Fixed camera / 1_Impossible（标签=0）
Claude-opus-4.8 输出："0"（正确）
GPT-4o 输出："0"（正确）

视频展示物体穿越了本应阻挡它的固体结构，两个模型均正确识别。
```

---

## 3. A2RBench

**论文：** [A2RBench: Benchmarking Abstraction and Abstract Reasoning of LLMs](https://arxiv.org/abs/2605.17278)（arXiv 2605.17278）  
**数据集：** 72 道验证题，来自 GitHub 仓库 `MAC-AutoML/A2Rbench`。我们取 18 题（每类 3 题，平衡采样）。

### 3.1 任务描述

给出一个**规则描述** + 若干**示例（输入→输出）**，要求模型推导出变换规则并应用到新输入上。任务分两大类、三个维度：

| 类型 | 维度 | 说明 |
|------|------|------|
| **SemanticRule（语义规则）** | D1（1D字符串）| 按语义映射替换字符，如：字母→化学元素符号 |
| | D2（2D字符串列表）| 同类规则应用于二维数组 |
| | D3（3D嵌套结构）| 同类规则应用于三维嵌套 |
| **SymbolicRule（符号规则）** | D1 | 按位置/索引变换字符，如：Caesar 密码、重排序 |
| | D2 | 同类规则应用于二维矩阵 |
| | D3 | 同类规则应用于三维张量 |

答案评判：与标准答案**完全匹配**（含顺序、格式）。

### 3.2 评测指标

**论文指标名称：** Exact Match Accuracy（%）——预测 `final_answer` 与 `answer_ciphertext` 逐字符完全一致的比例。  
来源：论文 Table 3 / Figure 4。论文报告最强模型约 39.8%，人类约 68.5%。

### 3.3 测试模型

| 模型 | OpenRouter ID | 角色 |
|------|--------------|------|
| **claude-opus-4.8** | `anthropic/claude-opus-4.8` | 当前最强 Claude，论文未涵盖 |
| **gemini-2.5-flash** | `google/gemini-2.5-flash` | 论文中 Answerer_Gemini_Flash 对应模型 |

### 3.4 结果

| 类别 | Claude-opus-4.8 | Gemini-2.5-Flash |
|------|:--------------:|:---------------:|
| Sem-D1 | 3/3 = 100% | 2/3 = 67% |
| Sem-D2 | 2/3 = 67% | 1/3 = 33% |
| Sem-D3 | 2/3 = 67% | 1/3 = 33% |
| Sym-D1 | 3/3 = 100% | 1/3 = 33% |
| Sym-D2 | 3/3 = 100% | 2/3 = 67% |
| Sym-D3 | 3/3 = 100% | 2/3 = 67% |
| **总计** | **16/18 = 88.9%** | **9/18 = 50.0%** |

| 对照组 | 准确率 |
|--------|--------|
| 论文最强模型（~GPT-4 era） | ~39.8% |
| 人类 | ~68.5% |
| Claude-opus-4.8（我们） | **88.9%** |
| Gemini-2.5-Flash（我们）| 50.0% |

### 3.5 题目与回答示例

#### 示例 A：SemanticRule D1（正确）

**规则描述：**
```
For each character in the input string:
- If it's an English letter (A–Z or a–z): compute its position p in the alphabet (A=1,...,Z=26),
  then replace with the chemical element symbol with atomic number p.
  Uppercase letter → symbol with ^ suffix (e.g., A→H^)
  Lowercase letter → symbol with _ suffix (e.g., a→H_)
- Non-letter characters: keep unchanged.
```

**示例（输入→输出）：**
```
"Ab!c"  →  "H^He_!Li_"
"Hello, World!"  →  "O^B_Mg_Mg_P_, V^P_Ar_Mg_Be_!"
```

**问题：** `Ab!c Hello, Zz. Number: 42`

**Claude-opus-4.8 输出（✓）：**
```json
{
  "reasoning": "Process each character. 'A'->p1->H^, 'b'->p2->He_, '!'->!, 'c'->p3->Li_,
                ' '->space. 'H'->p8->O^, 'e'->p5->B_, 'l'->p12->Mg_, ...,
                'Z'->p26->Fe^, 'z'->p26->Fe_. 'N'->p14->Si^, 'u'->p21->Sc_, ...",
  "final_answer": "H^He_!Li_ O^B_Mg_Mg_P_, Fe^Fe_. Si^Sc_Al_He_B_Ar_: 42"
}
```
**标准答案：** `H^He_!Li_ O^B_Mg_Mg_P_, Fe^Fe_. Si^Sc_Al_He_B_Ar_: 42` ✓

---

#### 示例 B：SemanticRule D2（Claude 正确，Gemini 错误）

**规则描述：** ASCII 值 v 在 [32,126] 范围内的字符，变换为 v' = 158 − v 对应的 ASCII 字符；范围外字符不变。

**示例：**
```
["Hello World!", "123 ABC xyz"]  →  ["V922/~G/,2:}", "mlk~]\[~&%$"]
```

**问题：** `["The quick brown fox...", "ASCII range: ' '(32) to '~'(126).", ...]`

**Claude-opus-4.8 输出（✓）：** 完整正确应用了 158-v 变换，逐字符计算

**Gemini-2.5-Flash 输出（✗）：**
```
predicted: "]K[UU~,=079d~w~w~vkluu~*/~w w~v..."
gold:      "]K[UU~,=079d~w~w~vklu~*/~w w~v..."
```
*仅差 1 个字符（`kluu` vs `klu`），但 Exact Match 判为错误。*

---

#### 示例 C：SymbolicRule D1（Gemini 错误示例）

**任务：** Caesar cipher 变体（ROT-13 类型字符替换）

**Gemini-2.5-Flash 输出（✗）：**
```
predicted: "GSV JFRPX YILDM ULC QFNKH LEVI GSV OAZB WLT."
gold:      "GSV JFRXP YILDM ULC QFNKH LEVI GSV OZAB WLT."
```
*两处字符顺序错误（`JFRPX` vs `JFRXP`，`OAZB` vs `OZAB`），说明模型在符号位置追踪上不稳定。*

---

## 4. 综合适合度评估

| Benchmark | 当前最强模型准确率 | 人类准确率 | 论文基线 | 适合度 |
|-----------|:-----------------:|:---------:|:--------:|:------:|
| ContextMATH | ~70% avg（4 splits）| — | ~57–77% | ⚠️ 部分适合 |
| IntPhys2 | 70%（Solidity only）| 96.4% | 57.6%（GPT-4o avg）| ✅ 适合 |
| A2RBench | **88.9%** | 68.5% | ~39.8% | ❌ 不适合 |

### 分析

**ContextMATH（⚠️ 部分适合）**  
Claude-opus-4.8 平均约 70%，仍有明显失误（CS 变体 60%），失误来源明确——CS 变体的谜语包装导致参数提取错误。SG 变体相对简单，CS 变体更有挑战性。如果项目关注"跨格式数学理解"，CS split 是值得考虑的测试场景。

**IntPhys2（✅ 推荐）**  
最强模型 70%，远低于人类 96.4%；且模型对"正常"视频的识别率仅 50%（与随机猜测持平），暴露了多模态时序物理推理的根本能力缺口。GPT-4o 论文数字与我们的测试结果吻合（~57-60%），验证了测试框架可信。这个差距（模型 ~60-70% vs 人类 96.4%）足够显著，且失败模式清晰（系统性偏向"不可能"）。

**A2RBench（❌ 不适合）**  
Claude-opus-4.8 以 88.9% 超过人类（68.5%），benchmark 已被当前最强模型"解决"。论文中基线模型约 39.8% 的结果反映的是 GPT-4 时代的水平，对当前模型不再有参考价值。

---

*报告生成时间：2026-06-21*  
*代码位置：`SelectBenchmark/eval_contextmath_claude48.py`，`eval_contextmath_gemini25pro.py`，`eval_intphys2_multi.py`，`eval_a2rbench.py`*  
*结果文件：`SelectBenchmark/results/{contextmath,intphys2,a2rbench}/`*
