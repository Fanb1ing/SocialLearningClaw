# SocialLearningClaw V1 冻结归档（2026-08-24）

本目录是在 EFPS Version 2 开发开始前创建的只读回退快照。V1 业务代码没有被移动或删除；
`snapshot/` 是额外复制，当前项目仍可按原路径运行。

## 冻结点

- 日期：2026-08-24（Asia/Shanghai）
- Git branch：`main`
- 基准 commit：`7c785777775e0b379235d0719ae177c01c5698ef`
- 运行基线：`.venv/bin/python -m unittest discover -s tests -v`，79/79 通过
- V1 架构：`MemoryRecord -> layered SchemaNode`
- V2 转换设计：`snapshot/documentation/docs/version2_efps_development_plan.md`

冻结时业务代码相对基准 commit 没有未提交修改。工作树中另有本轮按仓库约定更新的
`docs/project_memory.md`、用户提供且未跟踪的 `0824version2plan.md`，以及新整理的 V2 设计文档；
它们均已纳入本快照。`.env`、API key、`.venv`、cache 和用户/主机私有配置没有归档。

## 目录说明

```text
snapshot/root/                根级配置、README、AGENTS 与原始 0824 设计输入
snapshot/code/                socialclaw、tests、scripts、configs
snapshot/documentation/       当前全部 docs，包括项目记忆和 V2 方案
snapshot/reference/           Gold artifacts 与 25 个本地 ARC 游戏版本
snapshot/historical_archive/  冻结前已有的 legacy code/results 归档
snapshot/data/                ContextMATH、IntPhys2 metadata、trajectory corpora
snapshot/results/             Phase A-D/review/evaluator 与 smoke retest 输出
```

实际复制的逐文件内容、大小、SHA-256 和用途见 `FILE_INVENTORY.md`；机器校验清单见
`MANIFEST.sha256`。未复制的大型原始数据和历史输出见 `EXTERNAL_ASSETS.md` 与
`EXTERNAL_ASSETS.sha256`。

## 数据与结果边界

本归档实际保存约 49 MiB、1739 个 V1/转换期文件，其中包括：

- 完整当前业务代码、测试、脚本和配置；
- 当前全部 docs、Gold artifacts 和 vendored ARC 游戏源码；
- CD82/SK48/TU93 三份可 replay trajectory corpora；
- Phase A-D、learned-vs-Gold review 输出和 2026-07-24 smoke retest；
- ContextMATH parquet；
- IntPhys2 metadata、固定 Main 300 sample CSV 与 provenance manifest。

未重复复制约 1.4 GiB 的 IntPhys2 视频、`data/legacy/` 与 `outputs/legacy/`。它们不是 V2
代码回退的必要组成，仍保留在仓库原位置；外部资产清单记录路径、大小与 hash，便于之后验证。

## 回退方法

建议先把当前工作复制到新 branch 或新目录，再从 `snapshot/` 选择性恢复。示例映射：

```text
snapshot/code/socialclaw/       -> socialclaw/
snapshot/code/tests/            -> tests/
snapshot/code/scripts/          -> scripts/
snapshot/code/configs/          -> configs/
snapshot/documentation/docs/    -> docs/
snapshot/reference/gold/        -> gold/
snapshot/reference/third_party/ -> third_party/
snapshot/data/                  -> data/
snapshot/results/               -> outputs/
```

恢复后执行：

```bash
(cd archive/version1_20260824 && sha256sum -c MANIFEST.sha256)
.venv/bin/python -m unittest discover -s tests -v
```

`MANIFEST.sha256` 中的路径相对于本归档目录。归档本身不应被 V2 包 import，也不进入测试发现路径。
从仓库根目录执行
`sha256sum -c archive/version1_20260824/EXTERNAL_ASSETS.sha256` 可以核验未复制的大型原始资产。
