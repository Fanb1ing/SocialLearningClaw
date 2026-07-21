# Benchmarks

## ARC-AGI-3

- 类型：交互式 grid environment。
- 当前默认游戏：`cd82-fb555c5d`、`sk48-d8078629`、`tu93-0768757b`。
- 主指标：完成关卡数、成功率、总 actions/steps。
- 本地 game source：`third_party/arc_agi3_games/`。
- 环境通过 `arc_agi.Arcade` 创建；运行还需要有效 ARC API key。

SC25 环境源码保留在 third-party 中，但不属于默认三游戏实验集合。

## ContextMATH

本地路径：`data/contextmath/`。

支持：

- `aime_2024_sg`
- `aime_2024_cs`
- `aime_2025_sg`
- `aime_2025_cs`
- `math_500_sg`，用作默认 ICL demonstration pool

指标为 exact numeric accuracy。解析最后一个 `\boxed{...}`，数字比较允许前导零和逗号差异。

## IntPhys2

本地路径：`data/intphys2/`。

当前只准备了 Debug split 的 20 个本地视频；metadata 中没有本地视频的行不会进入实验。默认每 1.5 秒抽一帧，最多 12 帧。指标为 plausible/impossible binary accuracy，并保留 condition 与 camera metadata。

默认将开头 3 个视频固定为 ICL 保留集，因此各方法在当前本地数据上评测余下 17 个相同视频；
可用 `--num-demos` 修改，但同一张比较表必须保持该值一致。

论文级实验前需要下载完整 split，并在 manifest 中固定样本 ID 和 dataset fingerprint。

## Legacy data

CL-bench、PBench、Cosmos-Reason1、ARC-1/2 和 A2RBench 不再是当前项目 benchmark。它们的数据移入 `data/legacy/`，旧调研与结果位于 `docs/archive/` 和 `archive/results/`。
