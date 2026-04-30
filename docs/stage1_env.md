# Stage 1 环境配置与运行

## 1. Python 环境（建议单独 venv）

在仓库根目录：

- 建议 Python 3.11 或 3.12
- 创建 venv：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

## 2. 安装依赖 

```bash
pip install -e .
```

PBench 原始数据为 `.parquet`，预处理需要 `pyarrow`（推荐）：

```bash
pip install pyarrow
```

（可选）如你更习惯用 pandas：

```bash
pip install pandas
```

## 3. 准备数据（PBench）

> Cosmos-Reason1 在当前 snapshot 下可能缺失 AV 资源（视频/音频文件），因此 Stage1 默认改用 **nvidia/PBench**。

### 3.1 下载

```bash
python scripts/stage1_download_pbench.py --out data/pbench/raw
```

如 HuggingFace 需要 token：

```bash
export HUGGINGFACE_HUB_TOKEN=...   # 或 HF_TOKEN
```

### 3.2 预处理

```bash
python scripts/stage1_prepare_pbench.py --raw data/pbench/raw --out data/pbench/prepared
```

输出为：`data/pbench/prepared/all.jsonl`

> 备注（重要）：Cosmos-Reason1 的样本通常包含 `video` 字段,但是目前没有上传AV数据，换用PBench

## 4. 配置可插拔 API（OpenAI-compatible）

本项目 Stage1 的 Agent 目前使用 OpenAI-compatible 的 `/chat/completions`。

你可以使用：
- OpenRouter（base_url: `https://openrouter.ai/api/v1`）
- 硅基流动（使用其 OpenAI 兼容地址）
- 阿里云（若你使用其 OpenAI 兼容网关/第三方代理）

## 5. 运行 Stage1

```bash
python -m socialclaw.stage1.run_stage1 \
  --prepared data/pbench/prepared/all.jsonl \
  --base-url https://openrouter.ai/api/v1 \
  --api-key $OPENROUTER_API_KEY \
  --model <model-name> \
  --max-problems 5 \
  --max-iters 2
```

> Stage1 默认只启用 **安全工具**（`calculator`/`noop`），不会执行 `run_shell`。

运行产物：
- `runs/<run_id>/<problem_id>/episode.json`
- `skills_db/skills/<skill_id>.md`（当触发总结门槛时）