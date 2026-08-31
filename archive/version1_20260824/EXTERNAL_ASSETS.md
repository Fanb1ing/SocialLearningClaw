# 未复制的大型外部资产

以下内容保留在仓库原位置，没有再次复制到 V1 snapshot：

| 原路径 | 冻结时规模 | 内容 | 回退意义 |
|---|---:|---|---|
| `data/intphys2/Debug/Videos/` | IntPhys2 396 MiB 总量的一部分 | Debug 视频 | 仅旧静态 benchmark 运行需要 |
| `data/intphys2/Main/Videos/` | IntPhys2 396 MiB 总量的一部分 | 固定 Main 300 视频及残留下载文件 | 仅旧静态 benchmark 运行需要 |
| `data/legacy/` | 757 MiB | 早期 benchmark、DiscoveryWorld 与废弃数据 | V1 当前 runner 不依赖 |
| `outputs/legacy/` | 264 MiB | 2026-05/06 的旧运行和逐步 artifacts | 历史研究记录，不是代码恢复依赖 |

对应每个文件的 SHA-256、字节数和原路径保存在 `EXTERNAL_ASSETS.sha256`。该清单只用于核验；
它不包含文件内容，也不包含凭据。IntPhys2 metadata/sample manifest 已实际复制到
`snapshot/data/intphys2/`，当前关键 review/smoke 结果已复制到 `snapshot/results/`。
