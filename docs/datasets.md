# 数据集说明

> 本文档说明 SocialLearningClaw 项目中所有评测数据集的来源、原始结构、处理流程、最终格式和使用方法。

---

## 数据集总览

| 数据集 | 类型 | 数量 | 脚本 | 状态 |
|--------|------|------|------|------|
| **PBench** | MCQ（图像/视频 + 文本） | - | 已有 | 待接入 |
| **CL-bench** | 长上下文阅读理解 | 1,899 任务 | `scripts/download_clbench.py` | 已准备 |
| **CL-bench-Life** | 长上下文阅读理解（生活类） | 405 任务 | `scripts/download_clbench.py` | 已准备 |
| **ARC-AGI-1** | 静态网格推理 | 800 任务 | `scripts/download_arc.py` | 已准备 |
| **ARC-AGI-2** | 静态网格推理 | 1,120 任务 | `scripts/download_arc.py` | 已准备 |
| **ARC-AGI-3** | 交互式推理环境 | 135 环境 | `scripts/download_arc.py` | 需 API key |

---

## CL-bench / CL-bench-Life

### 来源
- 官方 GitHub：https://github.com/Tencent-Hunyuan/CL-bench
- HuggingFace：`tencent/CL-bench`、`tencent/CL-bench-Life`
- 论文：[arXiv:2602.03587](https://arxiv.org/abs/2602.03587)

### 原始数据结构
从 HuggingFace 下载后，原始数据为 `datasets.Dataset` 格式，每行包含：

| 字段 | 类型 | 说明 |
|------|------|------|
| `messages` | `List[{"role": str, "content": str}]` | 对话格式：`system`（上下文/规则）、`user`（问题）、`assistant`（参考答案） |
| `rubrics` | `List[str]` | 评分细则，每条为二进制 pass/fail 标准 |
| `metadata` | `dict` | `task_id`, `context_id`, `context_category`, `sub_category` |

**示例**：
```python
{
  "messages": [
    {"role": "system", "content": "You are an AI designed to play the Twisted Cryptids board Game..."},
    {"role": "user", "content": "RULE BOOK\nTM\nIt’s so hard to feel seen as a Cryptid..."},
    {"role": "assistant", "content": "Sighting cards are revealed to award/penalize..."}
  ],
  "rubrics": [
    "The response should define what a Sighting card is...",
    "The response should explain that Sighting cards are revealed..."
  ],
  "metadata": {
    "task_id": "2bbe2e03-972d-45d5-8e54-0654115fddd1",
    "context_id": "71a2cd92-6978-4ea8-a37f-d99728129d89",
    "context_category": "Rule System Application",
    "sub_category": "Game Rules"
  }
}
```

### 处理流程
1. **下载**：使用 `datasets.load_dataset("tencent/CL-bench")` 下载到 `data/clbench/raw/hf_dataset/`
2. **解析 messages**：
   - `system` → `context`（规则/背景知识，通常很长，平均 ~63K tokens）
   - `user` → `question`（具体任务/问题）
   - `assistant` → `answer`（参考答案）
3. **保留元数据**：`task_id`, `context_id`, `context_category`, `rubrics`
4. **输出**：每行一个 JSON object，保存为 `.jsonl`

### 最终格式
```json
{
  "id": "2bbe2e03-972d-45d5-8e54-0654115fddd1",
  "context": "You are an AI designed to play...",
  "question": "RULE BOOK\nTM\nIt’s so hard...",
  "answer": "Sighting cards are revealed...",
  "rubrics": ["The response should define...", "..."],
  "meta": {
    "context_id": "71a2cd92-6978-4ea8-a37f-d99728129d89",
    "context_category": "Rule System Application",
    "sub_category": "Game Rules"
  }
}
```

### 使用方法
```python
import json
from pathlib import Path

for line in open("data/clbench/prepared/clbench.jsonl"):
    record = json.loads(line)
    print(record["id"], record["meta"]["context_category"])
    # context 很长，question 是具体任务
```

### 关键特点
- **长上下文**：平均输入长度 ~63K tokens，测试模型从上下文中学习新知识的能力
- **防污染**：使用虚构内容、修改后的知识或小众专业知识，防止模型依赖预训练知识
- **顺序依赖**：51.1% 的任务是多轮且依赖前面答案的
- **严格评估**：使用 GPT-5.1 作为 judge，instance-level 二进制 rubrics

---

## ARC-AGI-1 / ARC-AGI-2

### 来源
- ARC-AGI-1：https://github.com/fchollet/ARC-AGI
- ARC-AGI-2：https://github.com/arcprize/ARC-AGI-2

### 原始数据结构
从 GitHub clone 后，原始数据为 JSON 文件，按 `training/` 和 `evaluation/` 目录存放。每个 JSON 文件包含：

| 字段 | 类型 | 说明 |
|------|------|------|
| `train` | `List[{input: List[List[int]], output: List[List[int]]}]` | 训练样本（输入输出网格对） |
| `test` | `List[{input: List[List[int]], output: List[List[int]]}]` | 测试样本（需要模型预测的） |

每个网格是一个二维整数数组（0~9 代表不同颜色），尺寸通常在 1x1 到 30x30 之间。

**示例**（`007bbfb7.json`）：
```json
{
  "train": [
    {"input": [[0,7,7],[7,7,7],[0,7,7]], "output": [[0,0,0,0,7,7,0,7,7], ...]},
    {"input": [[4,0,4],[0,0,0],[0,4,0]], "output": [[4,0,4,0,0,0,4,0,4], ...]}
  ],
  "test": [
    {"input": [[7,0,7],[7,0,7],[7,7,0]], "output": [[7,0,7,0,0,0,7,0,7], ...]}
  ]
}
```

### 处理流程
1. **下载**：通过 `git clone --depth 1` 从 GitHub 镜像下载到 `data/arc/raw/arc1/` 和 `arc2/`
2. **遍历**：遍历 `training/` 和 `evaluation/` 目录下的所有 `.json` 文件
3. **统一格式**：保留原始 `train`/`test` 结构，附加 `id`（文件名）、`split`、`version`
4. **输出**：每行一个 JSON object，保存为 `.jsonl`

### 最终格式
```json
{
  "id": "007bbfb7",
  "split": "training",
  "train": [
    {"input": [[0,7,7],[7,7,7],[0,7,7]], "output": [[...]]},
    ...
  ],
  "test": [
    {"input": [[7,0,7],[7,0,7],[7,7,0]], "output": [[...]]}
  ],
  "version": "arc1"
}
```

### 使用方法
```python
import json

for line in open("data/arc/prepared/arc1.jsonl"):
    task = json.loads(line)
    print(task["id"], len(task["train"]), "train pairs,", len(task["test"]), "test pairs")
    # grid 是 List[List[int]]，0~9 代表颜色
```

### 关键特点
- **抽象推理**：没有预训练知识可以依赖，必须从少量示例中推断变换规则
- **网格变换**：输入是一个彩色网格，输出是应用某种规则后的网格
- **ARC-AGI-2 难度更高**：更多样化的变换规则和更大的网格尺寸

---

## ARC-AGI-3

### 来源
- 官方 Agents 仓库：https://github.com/arcprize/ARC-AGI-3-Agents
- 官网：https://arcprize.org/

### 为什么需要 API key？

ARC-AGI-3 与前两代有本质区别：
- **ARC-AGI-1/2** 是**静态数据集**（JSON 文件，本地即可评测）
- **ARC-AGI-3** 是**交互式在线环境**（135 个人类手工设计的推理游戏），必须通过 API 连接官方服务器运行

官方为了防止数据泄露和确保公平评测，所有交互都必须通过认证 API 进行。

### 已准备的内容
`scripts/download_arc.py` 已经通过 `ghfast.top` 镜像 clone 了官方 Agents 仓库到：
```
data/arc/raw/arc3/agents/
```

### 获取 API key 的步骤

1. **访问官网**：https://arcprize.org/
2. **注册/登录账号**
3. **申请 API key**（通常在账户设置或开发者页面）
4. **配置环境变量**：
   ```bash
   cd data/arc/raw/arc3/agents
   cp .env.example .env
   # 编辑 .env，填入你的 ARC_API_KEY
   ```
5. **测试运行**：
   ```bash
   cd data/arc/raw/arc3/agents
   uv run main.py --agent=random --game=ls20
   ```

### 技术说明
- ARC-AGI-3 每个环境都是一个小游戏，Agent 需要发送 action（如移动、选择、修改网格等），服务器返回 observation（新的状态）
- 这是**多轮交互**过程，不是单轮问答
- 本项目在 **Stage 2** 才会实现 ARC-AGI-3 的适配器（将 Schema reasoning 与多轮 action/observation 循环结合）
- Stage 1 中 ARC-AGI-3 仅作为接口预留，主实验先用 ARC-AGI-1/2 的静态数据

---

## 数据集加载器接口设计

所有数据集统一实现以下接口：

```python
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class Problem:
    id: str
    prompt: str           # 给 LLM 看的完整问题文本
    problem_type: str     # "mcq" | "long_context" | "arc_grid"
    gold: Any             # 标准答案
    meta: Dict[str, Any]  # 原始元数据

class DatasetLoader:
    def load(self, split: str = "train") -> list[Problem]: ...
```

| 数据集 | problem_type | prompt 构造方式 | gold 类型 |
|--------|-------------|----------------|----------|
| PBench | `mcq` | 图像/视频描述 + 选择题文本 | str（选项字母） |
| CL-bench | `long_context` | `context + "\n\n" + question` | str（参考答案） |
| ARC-AGI | `arc_grid` | `train` 样本序列化为文本 + `test` input | List[List[int]]（输出网格） |

---

## 重新下载/更新数据

```bash
# CL-bench（已有 raw 时只跑 prepare）
.venv/bin/python scripts/download_clbench.py all

# ARC（包含 clone + prepare + inspect）
.venv/bin/python scripts/download_arc.py all

# 单独步骤
.venv/bin/python scripts/download_clbench.py download   # 只下载
.venv/bin/python scripts/download_clbench.py prepare    # 只预处理
.venv/bin/python scripts/download_clbench.py inspect    # 只查看统计
```
