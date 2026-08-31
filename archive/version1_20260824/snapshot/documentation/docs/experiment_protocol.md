# Experiment Protocol

## Required controls

同一张比较表中的方法必须使用相同的：

- model provider and model ID；
- temperature；
- sample IDs and order；
- max input/output token policy；
- per-sample attempts；
- ARC per-level step and restart budget；
- IntPhys2 frame sampling；
- feedback permission。

这些字段写入每个 run 的 `manifest.json`。

## Feedback

允许两种模式：

- `none`：方法不接收评测反馈，也不更新 memory/RAG。
- `binary`：只接收正确/错误信号。

禁止把 gold answer、gold label 或包含 gold 的字符串传给 Reflexion、ExPeL、A-MEM、TGM、RAG 或 Schema update。统一 memory update API 刻意没有 `gold` 参数。

## Attempts and retries

结果同时报告：

- `first_attempt_accuracy`：任何 retry 之前的表现；
- `accuracy`：统一 attempt budget 下的最终表现。

不能只给 Reflexion 或 Schema 额外 retry。ARC 默认 `max_attempts=1`；当前 prompt/memory ARC runners 也只支持一个不中断的环境 attempt。

## ICL demonstrations

- ContextMATH 默认从 `math_500_sg` 取 demonstration，不使用 AIME test answer。
- IntPhys2 从本地视频序列开头划出固定保留集。所有方法都排除该保留集，只有 ICL
  能看到其中的样本，从而保证不同方法的 evaluated sample IDs 完全相同。
- ARC 使用固定的 game-specific demonstration 文档。

所有静态实验把 demonstration IDs 写入 manifest。

## Usage accounting

`results.json` 记录任务回答调用的 token usage。Reflexion、ExPeL、A-MEM、TGM 等方法的
辅助 memory/reflection 调用尚未由所有后端统一返回 usage，因此当前结果不能直接用于宣称
不同方法具有相同的总 token 成本；正式做效率比较前需要补齐端到端调用计量。

## Historical results

`archive/results/benchmark_selection` 和 `outputs/legacy` 中的结果来自旧协议，可能具有不同模型、采样、帧数、retry 或 gold feedback 设置。它们只能作为开发历史，不能与新结果自动合并。
