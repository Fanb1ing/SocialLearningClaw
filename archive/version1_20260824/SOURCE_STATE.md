# V1 来源状态

## 版本信息

- 冻结日期：2026-08-24
- 分支：`main`
- Git commit：`7c785777775e0b379235d0719ae177c01c5698ef`
- Python 项目版本：`0.1.0`
- Python 要求：3.12+

## 冻结前工作树

开始归档前只有以下状态：

```text
 M docs/project_memory.md
?? 0824version2plan.md
```

其中 `docs/project_memory.md` 的修改是本轮按 `AGENTS.md` 写入的 V2 复用审计；
`0824version2plan.md` 是用户提供的设计输入，未被修改。随后新增的
`docs/version2_efps_development_plan.md` 是根据用户确认意见整理的完整 V2 方案，也已纳入 snapshot。

业务代码、测试、脚本、配置、Gold 和现有生成结果在冻结前没有已知未提交修改。

## 明确排除

- `.env` 与任何 API key；
- `.venv/`；
- `__pycache__/`、`*.pyc` 和下载 cache；
- `.claude/settings.local.json` 等主机/用户私有配置；
- `.git/` 对象数据库；
- 大型外部数据/历史输出的实体副本，改由外部资产 hash 清单引用。
