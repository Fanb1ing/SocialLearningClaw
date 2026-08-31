# V2 三游戏正式实验：可校验复现包

> 历史合同说明（2026-08-31）：本包冻结的是重构前的 role-binding Schema 实现。当前主线已经把
> Schema 收紧为 `Prototype → Action → Output` 三元组并新增全局 Insight，因此当前代码会在冻结
> prompt/payload 哈希校验处拒绝这批旧响应。保留本目录用于审计旧结果，不能把它当作当前实现的新
> 实验；需要在新合同上重新在线运行后建立新的转录和 manifest。

这个目录冻结 2026-08-30 的 CD82、SK48、TU93 三次正式实验。每关最多 30 个
Agent action，不限制关数；如果出现公开 `GAME_OVER`，runtime 会重开当前关，
但不会重置该关的 30 步预算。三次实验都在 Level 1 用完 30 步，因此本批没有
实际触发 reset，也没有进入 Level 2。

## 原合同下的复现命令（需要对应的旧源码版本）

对应的 contract-v2 旧源码要求 Python 3.12+。切到该源码版本后：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m pip install -r experiments/v2_formal_20260830/requirements-reproduction.txt
.venv/bin/python scripts/reproduce_v2_formal_20260830.py
```

在 contract-v2 源码下，最后一条命令不读取 `.env`、不需要 API key、也不访问网络。它会：

1. 校验三份冻结模型转录与仓库内游戏环境源码指纹；
2. 重新执行真实 ARC 环境、EFPS 更新、validator、trajectory replay 和报告生成；
3. 校验 summary，并对 `process.md`、`report.md`、`timeline.json`、
   `token_usage.json/.md`、`cognition/graph.json` 做逐文件 SHA-256 比较；
4. 写出 `outputs/reproduced/v2_formal_20260830/reproduction_verification.json`。

输出目录已存在时脚本会拒绝覆盖。需要换位置时使用：

```bash
.venv/bin/python scripts/reproduce_v2_formal_20260830.py \
  --output-root /tmp/socialclaw-v2-reproduction
```

## 复现结果

| 游戏 | 通过关数 / 尝试关数 | Level 通过率 | Agent actions | tokens | 终止原因 |
|---|---:|---:|---:|---:|---|
| CD82 | 0 / 1 | 0% | 30 | 2,748,352 | Level 1 步数耗尽 |
| SK48 | 0 / 1 | 0% | 30 | 2,871,771 | Level 1 步数耗尽 |
| TU93 | 0 / 1 | 0% | 30 | 2,612,454 | Level 1 步数耗尽 |
| 合计 | 0 / 3 | 0% | 90 | 8,232,577 | — |

`transcripts/*.json.gz` 保存的是已审计的逻辑模型调用：模型输出、provider usage、
工具调用轨迹及输入/图片哈希。回放时只返还这些冻结响应；每个调用前仍校验当前
prompt、instructions 和图片序列完全一致。因此这是“同一次实验 artifact 的确定性
复现”，不是一次新的 Opus 在线采样。

即使 `temperature=0`，重新请求远程模型也不承诺字节级确定性；模型版本、服务路由、
重试和工具调用都可能改变。需要研究新的独立 trial 时应使用在线命令，并为每个 trial
建立新输出目录；不能把这里的转录回放当成新的样本。

## 原始在线参数

以下是正式批次的三条在线命令。`--reset-on-game-over` 现在对所有游戏默认开启，这里仍
显式写出，避免命令含义依赖默认值。运行它们需要 `.env` 中的模型 API key；它们用于新增
trial，不能保证重新产生冻结日志的模型文本或 token 数。

```bash
.venv/bin/python -u scripts/run_v2_arc_online.py \
  --game-id cd82-fb555c5d --model anthropic/claude-opus-4.8 \
  --output-dir outputs/review/v2_formal_cd82_all_levels_opus48_step30_20260830 \
  --max-step 30 --stop-after-levels all --reset-on-game-over --compact-process

.venv/bin/python -u scripts/run_v2_arc_online.py \
  --game-id sk48-d8078629 --model anthropic/claude-opus-4.8 \
  --output-dir outputs/review/v2_formal_sk48_all_levels_opus48_step30_20260830 \
  --max-step 30 --stop-after-levels all --reset-on-game-over --compact-process

.venv/bin/python -u scripts/run_v2_arc_online.py \
  --game-id tu93-0768757b --model anthropic/claude-opus-4.8 \
  --output-dir outputs/review/v2_formal_tu93_all_levels_opus48_step30_20260830 \
  --max-step 30 --stop-after-levels all --reset-on-game-over --compact-process
```

## 为什么正式实验之前出现过重跑

第一次曾并行发起三游戏在线调用。CD82 在 Step 19、TU93 在 Step 17 附近连续收到空或
非法 JSON，provider 调用中断；这两个目录只是带 `partial` 标记的非最终 checkpoint。
为了不把 provider 故障后的残缺轨迹和最终结果混在一起，正式统计采用从空认知图开始、
逐游戏串行执行的干净运行。SK48 也跟随同一串行批次重新执行，使三游戏具有一致的运行
条件。失败的 partial 运行没有进入上表，也没有被包装成额外 trial。

文件职责：

- `manifest.json`：实验参数、环境/转录指纹、期望 summary 与结果日志哈希；
- `transcripts/*.json.gz`：冻结模型逻辑调用；
- `requirements-reproduction.txt`：会影响环境、图和日志字节结果的关键版本；
- `scripts/reproduce_v2_formal_20260830.py`：仓库根目录下的统一复现入口；
- `scripts/freeze_v2_model_transcript.py`：维护者从已完成在线 run 生成转录的工具。
