# Methods and Baselines

| Name | Definition | State carried forward |
|---|---|---|
| `naive` | Task input only | None |
| `icl` | Fixed labeled demonstrations from a disjoint demonstration set | Fixed prompt context |
| `rag` | Retrieve similar previous experiences from the current run | Experience buffer + embeddings |
| `withrule` | Inject human-written or fixed expert rules | Fixed prompt context |
| `reflexion` | LLM-generated verbal reflection after binary failure | Reflection list |
| `expel` | Experience pool with periodic generalizable insight extraction | Experiences + insights |
| `amem` | Structured, linked, embedding-retrieved notes | Notes + links + embeddings |
| `tgm` | Query/path/meta-cognition graph with reward-weighted retrieval | Graph nodes and edges |
| `schema` | Current structured object/action-effect schema method | Schema graph |

## Naming

旧名称 `zero_shot` 和 `few_shot` 分别统一为 `naive` 和 `icl`；新入口不再接受旧名称，避免同一方法产生两套结果目录。

正确拼写统一为 `Reflexion`；CLI 使用小写 `reflexion`。

## RAG

- Static benchmarks：跨已完成样本检索相似任务、历史响应和 binary outcome。
- ARC-AGI-3：在同一次环境运行中检索相似 grid experience。
- RAG 不读取 gold answer。

## WithRule

`withrule` 使用人工提供的规则，因此属于 privileged baseline/upper bound，不能和 `naive` 描述为完全相同的信息设置。

- ARC 规则位于 `configs/arc_agi3/rules/`。
- ContextMATH 与 IntPhys2 使用 adapter 中固定、与具体答案无关的任务原则。

## Schema availability

当前 `schema` 只支持 ARC-AGI-3。ContextMATH/IntPhys2 的 Schema 接入留到分层 Schema 架构重写，避免把旧 CL-bench/PBench pipeline 误当成正式实现。
