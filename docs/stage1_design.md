# Stage 1 设计与实现计划（SocialLearningClaw）

> 目标：实现 README 中 Stage 1 的闭环流程：
> - 给定 Cosmos-Reason1 部分问题 → 交给 **Claude Code Agent (CC Agent)** 回答（返回 answer + confidence + token/tool_calls 统计）
> - 依据门限停止 → 评估对错
> - 错误则 **向人类主动提问** 获取关键知识点 → 注入上下文再答
> - 满足门槛则总结为 **Skill** 并入库，同时保存全量日志/轨迹。

本文包含：模块设计、文件结构、接口定义、CC Agent 接入方式对比（含 Autosota 的方式）、Skill 机制（渐进式披露）、Cosmos-Reason1 数据准备方案与环境配置。

---

## 0. 非目标（Stage 1 不做什么）

- 不做复杂 UI（先 CLI / 可替换的 HumanIO）。
- 不强依赖 auto-pipeline-ab（只参考风格）。
- 不追求所有 Cosmos-Reason1 子任务覆盖；先支持 **二元/多选**（你已确认该数据应为二元或多选）。
- 不在 Stage 1 做训练/优化算法（GEPA 等）。

---

## 1. 总体闭环（与 README 1~5 对齐）

### 1.1 状态对象：`Episode`

每道题一次运行形成一个 `Episode`（可序列化 JSON）：

- `problem`: `{id, split, domain, prompt, choices?, answer_key?}`
- `attempts[]`: 每次对 CC Agent 的一次调用
  - `input_prompt`: 最终给 CC Agent 的 prompt（含 skills/human kp）
  - `answer`: 结构化答案（对 choice 任务建议规范化为 `{"choice": "A"}`）
  - `confidence`: `0~1` float（由 **CC Agent 自己输出**；Prompt 里强制其给出置信度；pipeline 侧只做格式校验与缺省回退）
  - `usage`: `{input_tokens, output_tokens, total_tokens}`（必须有）
  - `tool_calls`: `{count, tools: [...]}`（必须有；至少统计次数）
  - `raw`: 原始返回（用于复盘）
- `evals[]`: 与 attempts 对齐
  - `correct`: bool
  - `pred`: 预测（规范化）
  - `gold`: 标准答案（规范化）
  - `details`: 误差说明
- `human_feedback`:
  - `knowledge_points[]`: 人类输入的关键知识点
- `stop_reason`: `confidence|max_iters|max_tokens|...`
- `skill`: 若触发 skill，总结后的 skill 文本 + 元数据

所有 episode 都会落盘：`runs/YYYYmmdd_HHMMSS/<problem_id>/episode.json`。

### 1.2 运行流程（伪代码）

```text
for problem in dataset:
  episode = init(problem)
  skills = skill_retriever.retrieve(problem)

  for iter in range(max_iters):
    prompt = prompt_builder.build(problem, skills, episode.human_feedback)
    attempt = cc_agent.answer(prompt)
    episode.attempts.append(attempt)

    eval = evaluator.evaluate(problem, attempt)
    episode.evals.append(eval)

    if stop_policy.should_stop(episode):
      break

    if eval.correct:
      break

    # wrong -> proactive ask human
    knowledge_points = human_io.ask_key_points(problem, attempt, eval)
    episode.human_feedback.knowledge_points.extend(knowledge_points)

  # skill threshold
  if skill_gate.should_summarize(episode):
    skill = skill_summarizer.summarize(episode)
    skill_writer.write(skill)

  logger.write(episode)
```

---

## 2. 文件结构（建议）

在仓库根目录新增一个独立包，避免和 `auto-pipeline-ab/` 混杂：

```text
SocialLearningClaw/
  socialclaw/
    __init__.py
    stage1/
      run_stage1.py
      config_schema.py
      types.py
      pipeline.py
      prompt_builder.py
      stop_policy.py
      evaluator.py
      human_io.py
      logging.py
      skill/
        store.py
        retrieve.py
        gate.py
        summarize.py
      cc_agent/
        base.py
        adapters/
          claude_code_cli.py
          claude_code_sdk.py
          anthropic_api.py
  scripts/
    stage1_download_cosmos_reason1.py
    stage1_prepare_cosmos_reason1.py
  data/
    cosmos_reason1/
      raw/        # huggingface 下载原始
      prepared/   # 处理后 jsonl
  skills_db/
    index.jsonl
    skills/
      <skill_id>.md
  runs/
    ...
  docs/
    stage1_design.md
    stage1_env.md
```

说明：
- `socialclaw/`：核心代码。
- `scripts/`：一次性数据准备脚本。
- `data/`：数据落盘位置（可通过 config 覆盖）。
- `skills_db/`：skill 库（与 autosota 的 `Skills/` 目录概念一致，但我们实现更结构化的索引）。

---

## 3. 模块接口定义（关键类/函数）

### 3.1 CC Agent 接口（必须返回 token/tool_calls）

`socialclaw/stage1/cc_agent/base.py`

```python
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass
class Usage:
    input_tokens: int
    output_tokens: int
    total_tokens: int

@dataclass
class ToolCallStats:
    count: int
    tools: List[str]  # tool names

@dataclass
class CCAgentResult:
    answer_text: str
    confidence: Optional[float]
    usage: Usage
    tool_calls: ToolCallStats
    raw: Dict[str, Any]

class CCAgent:
    def answer(self, *, prompt: str, meta: Dict[str, Any]) -> CCAgentResult:
        ...
```

**硬性约束（你提出的 #4）：** `usage` 与 `tool_calls` 不能为空（至少能统计 `count`，tokens 至少 `total_tokens`）。

### 3.2 DatasetProvider

`socialclaw/stage1/types.py`

```python
@dataclass
class MCQProblem:
    id: str
    prompt: str
    choices: List[str]  # e.g. ["A. ...", "B. ..."]
    answer_key: str     # "A"/"B"/"C"/...
    meta: Dict[str, Any]
```

`socialclaw/stage1/dataset_cosmos_reason1.py`（后续实现）

- `load_prepared_jsonl(path) -> Iterable[MCQProblem]`

### 3.3 PromptBuilder（含 Skill 渐进式披露）

核心职责：把 `problem + retrieved skills + human knowledge points` 组装成对 CC Agent 友好的提示词。

`socialclaw/stage1/prompt_builder.py`

- 输入：`problem, skills, knowledge_points, attempt_index`
- 输出：最终 prompt 字符串

渐进式披露（你提出的 #5）：
- `attempt=0`：只注入 topK skills 的 **标题 + 适用范围 + checklist（短）**
- `attempt>=1` 且仍错误：升级披露更多细节（例如“反例/注意事项/更长解释”）
- 如果 token 压力大：只保留最短 checklist，丢弃长解释

### 3.4 StopPolicy

`socialclaw/stage1/stop_policy.py`

- `should_stop(episode) -> (bool, reason)`
- 按：conf / iters / token 三个门限

### 3.5 Evaluator（Choice/二元）

`socialclaw/stage1/evaluator.py`

- `normalize_choice(text) -> "A"|"B"|...`
- `evaluate(problem, result) -> EvalResult(correct, pred, gold, details)`

Stage1 只做 **严格 choice**（并具备一定鲁棒性：支持 “答案：A” / “I choose B”）。

### 3.6 HumanIO（主动提问）

`socialclaw/stage1/human_io.py`

- `ask_key_points(problem, attempt, eval) -> List[str]`

Stage1 默认实现：CLI 交互；后续可加 Webhook/IM。

### 3.7 Skill 子系统

#### Skill 数据模型

`socialclaw/stage1/skill/store.py`

- Skill 文件：`skills_db/skills/<skill_id>.md`
- index：`skills_db/index.jsonl` 每行一个 skill 元数据

Skill Markdown 建议格式：

```markdown
---
id: skill_2026xxxx_xxxx
created_at: ...
tags: ["cosmos", "mcq", "physics"]
trigger: ["wrong_then_human_help", "tool_calls_gt_3", "self_fix"]
---

# <Skill Title>

## When to use
...

## Checklist
- ...

## Common pitfalls
- ...

## Minimal example
...
```

#### Skill 检索（RAG-lite）

Stage1 为了简单可靠，先做两种策略：
1) **关键词/标签规则检索**（基于 metadata + prompt 的关键词）
2) 可选：本地 embedding（后续再加）

接口：`retrieve(problem) -> List[SkillDoc]`

#### SkillGate（门槛）

`socialclaw/stage1/skill/gate.py`

- `should_summarize(episode) -> bool`
  - wrong_then_human_help（必须实现）
  - tool_calls > 3（依赖 CC Agent trace，必须实现）
  - self_fix（定义：某次 attempt eval 从错→对，且 attempts 中出现 “error/fix” 标记；Stage1 可先用启发式正则）

#### SkillSummarizer

`socialclaw/stage1/skill/summarize.py`

Stage1 可先用模板总结（不用 LLM），保证可离线；也支持用 CC Agent 做总结（可控开关）。

---

## 4. Cosmos-Reason1 数据：下载与预处理

你要求从：`https://huggingface.co/datasets/nvidia/Cosmos-Reason1-Benchmark` 下载。

### 4.1 下载方式（建议用 huggingface_hub）

实现脚本：`scripts/stage1_download_cosmos_reason1.py`

- 使用 `huggingface_hub.snapshot_download(repo_id=...)`
- 下载到：`data/cosmos_reason1/raw/`

### 4.2 预处理为统一格式

实现脚本：`scripts/stage1_prepare_cosmos_reason1.py`

职责：
- 遍历 raw 内的 json/jsonl/parquet（以实际文件为准）
- 抽取字段：`id, prompt, choices, answer_key`
- 输出：`data/cosmos_reason1/prepared/<split>.jsonl`

**假设与约束：**
- 任务为二元或多选；我们统一成 choices = ["A. ...", ...]。
- 标准答案统一成 `"A"|"B"|...`。

> 具体字段名需要在 Stage1 开工时读取数据文件确认；预处理脚本会做 schema 探测并打印统计。

---

## 5. CC Agent 接入方法对比（含 Autosota 用的是什么）

你要求：
- **必须拿到 token/tool_calls**
- 目前你没有 CC Agent 入口，需要分析几种介入方式优缺点
- 问：Autosota 用的是什么方法？

### 5.1 方法 A：直接驱动 Claude Code CLI（推荐“最像 CC Agent”的方式）

**思路**：使用 `claude`（或 Claude Code 的官方 CLI）在一个受控工作目录运行，开启输出 JSON/日志，把 tool use 过程解析出来。

- 优点：
  - 真实 Claude Code，会产生 tool calls（读写文件、运行命令等）
  - 更符合你 README 中“CC Agent 自行回答 + 工具调用统计”的设定
- 缺点：
  - 依赖你本机安装 Claude Code + 登录态
  - 获取 tokens 不一定稳定：需要 CLI 支持 usage 输出或从 session log 推断
  - 需要处理会话文件/缓存路径

**tokens/tool_calls 获取**：
- tool_calls：可从 Claude Code session transcript（通常是 JSON event stream）统计
- tokens：若 transcript 包含 usage 字段则直接取；否则只能估算（你要求必须保证拿到 token，因此需要确认 CLI 是否提供 usage）

### 5.2 方法 B：使用 Anthropic API（最稳拿 token；tool_calls 需要“定义成工具调用”）

**思路**：用 Anthropic Messages API 直接调用模型。

- 优点：
  - usage tokens 官方返回，稳定
  - 可控性强，易部署
- 缺点：
  - 严格意义上不是“Claude Code Agent”（不自带代码执行/文件工具）
  - 要想有 tool_calls，需要我们自己实现 tool calling：例如定义 `run_shell`, `read_file`, `write_file` 等工具，并在 agent loop 中执行——这会把你项目变成“自研 agent”，而不是 Claude Code

**是否符合你 #4？**
- 可以保证 token
- tool_calls 可以保证（因为是我们自己定义的工具调用并统计），但不等同于 Claude Code 内置工具

### 5.3 方法 C：通过 OpenRouter 等代理（快，但不推荐作为主路径）

- 优点：接入快
- 缺点：usage/tool_calls 字段各家不一致；长期稳定性差

### 5.4 方法 D：复用 Autosota/Repo2Run 的“agent 框架”

从你仓库内的 `auto-pipeline-ab/Autosota/Repo2Run/build_agent/main.py` 可见：
- 它们用的是一个自研的 agent 框架（`build_agent/agents/*`），运行在 Docker sandbox 里
- CLI 参数 `--llm anthropic/claude-sonnet-4.6`
- 其 agent 通过 `Configuration(...).run(...)` 与 sandbox 交互，产生轨迹 `track.json`

**结论：Autosota 并不是直接用 Claude Code 客户端，而是“自研 Agent + 选用 Claude 模型”**。
- 优点：可以严格控制工具执行（docker sandbox），并能记录工具调用
- tokens：取决于他们的 LLM client 是否保存 usage（需要进一步读 `build_agent/agents/*` 才能确认）
- 缺点：与 Claude Code 行为不完全一致

> 由于 `SocialLearningClaw/.gitignore` 把 `auto-pipeline-ab/` 忽略掉，主工程不应依赖它的代码；但可以借鉴其“sandbox + agent loop + track.json”的日志形态。

### 5.5 我建议的落地选择（Stage1）

由于你当前**没有 Claude/Anthropic 账号**，且希望“下载 agent 框架后调用不同厂家的 API”，Stage1 主路径调整为：

- **主路径：自研 Agent + 可插拔 LLM Provider（Multi-Provider）**
  - 统一通过 `CCAgent` 抽象封装不同厂商 API（OpenAI-compatible / Anthropic / 其它）
  - **tokens**：从各厂商 API 的 usage 字段读取（适配层做归一化，保证 `input/output/total`）
  - **tool_calls**：来自“自研工具调用框架”的执行轨迹（即模型触发 tool call → 我们执行 → 记录），从而稳定满足门槛统计
  - **confidence**：要求模型在最终回答中显式输出 `confidence: 0~1`（Prompt 强制），pipeline 只做格式校验/缺省回退
  - **skills**：以 system prompt 为主注入，配合渐进式披露与 token 裁剪

- **可选适配器：Claude Code CLI**（非 Stage1 必需项）
  - 作为“对齐 Claude Code 行为”的后续工作，等你未来具备账号/授权后再接入

> 备注：这里的“tool_calls”语义是“Agent 框架内的工具调用”，不等同于 Claude Code 内置工具事件；但对 Stage1 的 stop/skill 门槛、日志与可复现性更稳定。

### 5.6 Autosota 的方法（结论仍然成立）

Autosota 更接近“自研 Agent + 选用 Claude 模型 + sandbox + track.json”。我们会借鉴其日志形态（轨迹可回放、可统计 tool calls），但不依赖其代码。

---

## 6. 环境配置（你要求“写完代码告诉你怎么配环境”）

我会单独写 `docs/stage1_env.md`，包含：
- Python 版本（建议 3.11/3.12）
- venv 创建
- 依赖安装（huggingface_hub、datasets、pyyaml、rich、pydantic 等）
- Anthropic key / HuggingFace token 环境变量
- 运行命令示例（从下载数据到跑 Stage1）

---

## 7. 下一步（我建议按这个顺序实现）

1) 建好 `socialclaw/` 包骨架 + types + logger（可先无外部依赖）
2) 写 Cosmos 下载/预处理脚本（先下载并统计 schema）
3) 写 evaluator（MCQ）
4) 写 HumanIO（CLI）
5) 写 Skill store/retrieve（先关键词）
6) 写 **自研 Agent 框架**（最小工具集 + tool call 轨迹 + usage 归一化）
7) 拼 pipeline + stage1_run 入口

---

## 8. 需要你确认的最小信息（开始敲代码前）

1) CC Agent 主路径：已确定为 **自研 Agent + 可插拔 LLM Provider**。

2) Skill 注入位置：默认 **system prompt**。

3) 人类交互形态：默认 **终端输入（CLI）**。
