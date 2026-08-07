# Benchmarks

## ARC-AGI-3

- 类型：交互式 grid environment。
- 批处理默认游戏：`cd82-fb555c5d`、`sk48-d8078629`、`tu93-0768757b`。
- 主指标：完成关卡数、成功率、总 actions/steps。
- 本地 game source：`third_party/arc_agi3_games/`。
- 当前本地固定 25 个带完整 version ID 的游戏，清单及哈希见
  `third_party/arc_agi3_games/inventory.json`。
- 环境通过 `arc_agi.Arcade` 的 offline mode 创建；普通实验不从 ARC API
  下载环境。只有刷新本地库存时需要有效 ARC API key。

SC25 与其余 21 个环境源码保留在 third-party 中，但不属于默认三游戏批处理集合。

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

支持两个固定本地 split：完整 Debug 60 视频，以及从 Main 按 scene 配对、按
condition/camera/difficulty 分层抽样的 `main_300`（75 scenes、每个 scene 四种
possible/impossible type，共 300 视频）。默认每 1.5 秒抽一帧，最多 12 帧。
指标为 plausible/impossible binary accuracy，并保留 condition 与 camera metadata。

默认将所选 split 开头 3 个视频固定为 ICL 保留集，因此各方法评测相同的剩余样本；
可用 `--num-demos` 修改，但同一张比较表必须保持该值一致。

`scripts/prepare_intphys2_data.py` 固定上游 revision、抽样 seed 和分层算法，并
生成 `sample_300_manifest.json`。Adapter 会拒绝缺少该 manifest 或视频不完整的
`main_300`；每次实验仍会在 manifest 中固定样本 ID 和 dataset fingerprint。

## Legacy data

CL-bench、PBench、Cosmos-Reason1、ARC-1/2 和 A2RBench 不再是当前项目 benchmark。它们的数据移入 `data/legacy/`，旧调研与结果位于 `docs/archive/` 和 `archive/results/`。
