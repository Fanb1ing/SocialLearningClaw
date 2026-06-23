确认benchmark当前还有提升空间：
测试opus 4.8和一个文章中的model(contextmath qwq32b, intphys gpt4o)


- ContextMATH：我报告的是 Accuracy (%)，即每个 split 中回答正确的题目比例。对应论文 Table 1/Table 2 中的列名 Accuracy (%) 。我每个 split 采样 10/30 题，论文用全部 30 题。
- IntPhys2：我报告的是 Binary Classification Accuracy (%)，即正确判断视频为 plausible(1)/impossible(0) 的比例。对应论文 Table 2 中按 Condition × Camera 分列的 Accuracy，以及最后一列的 Avg（Fixed+Moving camera 平均）。我目前只测了 Fixed camera 下的 solidity condition，论文对每种 condition（Permanence/Immutability/Continuity/Solidity）× camera（Fixed/Moving）分别报告。


可能的baseline:

来自ARC-AGI-3：
    非LLM用不了：
    1. StochasticGoose(Preview 赛第 1 名)— CNN+RL； 
    2. Graph-Based Exploration / "Just Explore"(Preview 第 3 名,有 AAAI 2026 workshop 论文)— training-free、无 LLM

    LLM：
    1. Arcgentica:36.08分
    symbolica/arcgentica 分支含 agents/、main.py、scripts/、tests/ 等完整实现(不只是 README)
    https://github.com/symbolica-ai/ARC-AGI-3-Agents/tree/symbolica/arcgentica
    https://github.com/Quriosity-agent/articles/blob/main/2026-03-28/arc-agi-3-agentica-analysis.md

    orchestrator-->
    - "Explorer" + bound_submit_action:探索网格、测试操作、记录发现; 写入 Memories
    - "Theorist": 读 Memories、分析 Explorer 报告; 形成/验证假说
    - "Solver" + bound_submit_action:读 Memories 获取已知规则; 执行通关序列

    可以复用他们的游戏工具包


    2. 官方 4 个 LLM agent：
    https://github.com/arcprize/ARC-AGI-3-Agents

    通用机制(所有 LLM agent 共用):
- 状态编码:每帧是 0–15 整数的 3D 网格,以 Grid {i}: {row} 文本形式喂给模型,附带当前 score(已通关数)和 game state(WIN / GAME_OVER 等);由 build_user_prompt() / build_func_resp_prompt() 构造。
- 动作:6 种 GameAction(RESET、ACTION1–6);新模型走 OpenAI tool_calls(结构化 JSON),老模型走 legacy function_call,再用 GameAction.from_name() 转成动作。
- 历史管理:messages 列表用 FIFO 维护,MESSAGE_LIMIT=10(只保留最近 10 条),保证 tool 响应顺序。

四个具体 agent:

┌──────────────┬─────────────────┬────────────────────────────────────────────────────────────────────┐
│      类      │      模型       │                                特点                                │
├──────────────┼─────────────────┼────────────────────────────────────────────────────────────────────┤
│ LLM(基类)    │ gpt-4o-mini     │ 标准版:DO_OBSERVATION=True,每次行动前先让模型写一段"观察",再选动作 │
├──────────────┼─────────────────┼────────────────────────────────────────────────────────────────────┤
│ FastLLM      │ gpt-4o-mini     │ DO_OBSERVATION=False,跳过观察步骤,省 token/降延迟,牺牲准确度       │
├──────────────┼─────────────────┼────────────────────────────────────────────────────────────────────┤
│ ReasoningLLM │ o4-mini         │ 用 OpenAI reasoning 模型,记录 reasoning tokens / 思考过程          │
├──────────────┼─────────────────┼────────────────────────────────────────────────────────────────────┤
│ GuidedLLM    │ o3(high effort) │ 注入人工写好的"游戏规则"提示;官方注明仅教学、不泛化到其他游戏      │
└──────────────┴─────────────────┴────────────────────────────────────────────────────────────────────┘

ReasoningLLM / reasoning_agent 的额外设计(比基础版更接近"真 agent"):
- 输出是结构化的 ReasoningActionResponse(Pydantic):动作名 + reasoning(10–2000 字理由)+ hypothesis(当前对机制的猜想)+ aggregated findings(累积发现)+ 简短描述。
- 维护 history(动作历史)和 screen_history(最近 10 帧),把网格转成带 zone 坐标的图像提供视觉上下文。
- 以"提出假设 → 验证"为主线组织探索,每过一关清空历史,聚焦当前关卡学习。
- 记录 _last_reasoning_tokens / _total_reasoning_tokens 等元数据。

    （3. Duke Harness）:Duke 团队做的是围绕大型 reasoning model 的 agentic harness(效果惊人——TR87 变体上 Opus 4.6 裸跑 0.0%,加 Duke harness 到 97.1%),但目前找不到公开 GitHub 仓库。它可能只在技术报告里被引用、未单独开源。我可以晚点再深挖一次,但现在没有可复现代码,先不纳入。
    

来自IntPhys2：
    视频/物理理解类的模型：V-JEPA 2不能用

来自ContextMATH/A2RBench： 没有