# Result Format

每个新实验目录至少包含：

## `manifest.json`

- `format_version`
- `created_at`
- `git_commit`
- 完整 `config` 与统一 `budget`
- evaluated `sample_ids`
- ICL `demonstration_ids`
- `dataset_fingerprint`

## `results.json`

Static benchmarks 写入：

- `metrics.accuracy`
- `metrics.first_attempt_accuracy`
- `metrics.correct`
- `metrics.total`
- 每个 sample 的 prediction、attempts、response、token usage 和非答案 metadata

## ARC artifacts

ARC 继续保存每关 `episode.json`、`trajectory.json`、step JSON/PNG，以及 method-specific `memory.json` 或 `schema/`。Layered Schema 的 `schema/` 内包含 `memory.json` 和 `schema.json`；step artifact 记录注入/新学到的 Schema ID 与来源 memory ID。统一 `manifest.json` 用于识别 method 与协议；`scripts/eval_arc_summary.py` 从这些文件汇总。

两个汇总脚本默认校验 model、预算、feedback、dataset fingerprint 和 evaluated sample IDs；
static 汇总还校验帧采样。不一致的运行不会被静默放进同一比较表。

## Trajectory corpus artifacts

Phase A–D 新增了尚未接入正式 runner 的通用 trajectory/corpus 和离线归纳输出：

```text
<trajectory-root>/
  episodes/<episode-id>.json
  assets/grids/<logical-sha256>.npy
  assets/images/<content-sha256>.png
  manifest.json
  coverage.json
  validation.json
  replay_validation.json
  splits/<split>.json
```

episode JSON 使用 `format_version: 1`，包含 actor、evidence tier、provenance、初始 observation、
连续 steps 和 terminal outcome。每个 observation 只引用内容寻址 asset；相邻 step 的相同状态
复用 artifact ID。CD82 v1 的正式固定输入还保存生成脚本/policy/environment hash、coverage gate
和逐帧环境 replay 结果；Phase A 的 synthetic review demo 仍不能当作实验结果。详见
[通用任务轨迹合同](trajectory_contract.md)和 [ARC 可靠轨迹语料](arc_trajectory_corpus.md)。

离线 Memory/Schema review snapshot 使用：

```text
<schema-review-root>/
  memory.json                 # Phase C，可单独作为冻结输入
  schema.json                 # Memory-grounded SchemaNode graph
  report.json                 # 数量、coverage、零网络/Gold 声明
  keyframes.json              # Phase D 选中的 pre/post 视觉引用
  audit.json                  # 每个窗口的 proposal、validator 判定和受影响节点
```

Phase D 可以只引用同级 Phase C `memory.json`，不必复制 2545 条记录；Schema 的每个 evidence ID
仍必须能在该冻结 Memory snapshot 解析。详见
[Window/keyframe Schema 归纳](window_schema_induction.md)。

离线 learned-vs-Gold evaluator 另写入独立目录，不得放回 learned state：

```text
<evaluation-root>/
  config.json
  metrics.json
  alignments.json
  unmatched_learned.json
  unmatched_gold.json
  judge_cache.jsonl
  report.md
```

`config.json` 固定 learned/Memory hash、Gold 根目录和 judge 版本。详见
[Learned Schema vs Gold Schema 评测](schema_evaluation.md)。
