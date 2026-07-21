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

ARC 继续保存每关 `episode.json`、`trajectory.json`、step JSON/PNG，以及 method-specific `memory.json` 或 `schema/`。统一 `manifest.json` 用于识别 method 与协议；`scripts/eval_arc_summary.py` 从这些文件汇总。

两个汇总脚本默认校验 model、预算、feedback、dataset fingerprint 和 evaluated sample IDs；
static 汇总还校验帧采样。不一致的运行不会被静默放进同一比较表。
