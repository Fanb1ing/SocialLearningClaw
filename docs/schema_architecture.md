# Memory-grounded layered Schema architecture

## Boundary

The new architecture separates concrete experience from learned world rules:

```text
task / observation / action / result
                 |
                 v
       MemoryRecord (durable evidence)
                 |
          automatic induction
                 v
    SchemaNode (generalized world rule)
                 |
        retrieval and feedback
                 v
              agent
```

Memory is the source of truth. A schema contains references to memory IDs,
not copied trajectories. This makes every generated, reinforced, weakened, or
merged rule auditable against its original evidence.

The legacy `Concept`/`Relation` modules are retained as source history under
`archive/code/legacy_schema/`; they are no longer importable from the active
`socialclaw` package. No current `schema` runner uses them.

## Benchmark integration

Benchmark-neutral trajectory foundation 和三个 ARC 示例 corpus 已实现数据合同、事件流、视觉
资产去重、原子 recording、环境 replay 及离线 `MemoryRecord -> SchemaNode` 窗口归纳，但尚未替换
本节描述的 active runner 路径。当前 `semantic_window_v1` 通过 benchmark profiler 提取视觉 diff，
选择关键帧并审计 create/support/revise/contradict/skip；下一阶段加入定时管理，再把在线 ARC loop
接到同一 recorder。详见
[`trajectory_contract.md`](trajectory_contract.md) 和
[`arc_trajectory_corpus.md`](arc_trajectory_corpus.md) 以及
[`window_schema_induction.md`](window_schema_induction.md) 及
[`arc_learned_schema_pipeline_plan.md`](arc_learned_schema_pipeline_plan.md)。

- ContextMATH stores the full problem, model response, and binary evaluation.
- IntPhys2 retrieves by the task plus non-label condition/camera/scene metadata;
  neither `gold` nor label-bearing `type` metadata enters Schema.
- ARC-AGI-3 stores every observation/action/environment-result transition and
  learns action-effect rules online. Terminal environment outcomes reinforce or
  weaken the schemas injected during that level.

All three write `schema/memory.json` and `schema/schema.json` inside the run
directory. Auxiliary induction calls are separate from answer-call token usage.

## Data models

`MemoryRecord` stores a concrete episode, knowledge item, or skill. An episode
contains the original task, context, ordered `MemoryEvent` transitions,
outcome, binary success when available, feedback, tags, and metadata.

`SchemaNode` maps directly to the structure in `temp.md`:

| Requirement | Field |
|---|---|
| Index | `index` |
| Level | `level` |
| Description | `description`, plus structured `trigger`, `action_sequence`, `expectation` |
| Memory Index | `memory_index.source/positive/negative` |
| Related Schema Index | `related_schema_index.parents/children/similar` |
| Reliability Weight | `reliability_weight` |
| Forgetting mask | `status` (`active`, `masked`, `deprecated`) |

Level `0` is the most general. Larger levels are increasingly task-specific.
Parent edges therefore always point from a smaller level to a larger level;
the graph rejects inverted edges and dangling references.
Every graph link can also retain the concrete memory IDs that justified that
relationship, so graph structure is auditable as well as node content.

## Lifecycle

`SchemaManager` is the application API:

1. `remember_and_learn(record)` persists an episode and induces a rule.
2. `learn(memory_id)` retrieves nearby rules and asks `LLMSchemaGenerator` to
   choose `create`, `merge`, or `skip`.
3. `retrieve(query)` combines optional embedding similarity, lexical overlap,
   reliability, and graph-neighbor expansion.
   `context_block(query)` renders those rules for direct prompt injection.
4. `apply_feedback(...)` records positive/negative evidence and updates
   reliability. Specific rules react faster; general rules change more slowly.
5. `update_task_mask(...)` hides task-irrelevant rules without deleting their
   memory evidence.
6. `run_maintenance()` applies evidence-aware decay and consolidates duplicate
   same-level nodes. It is safe to call from an external periodic scheduler.

Weak rules are masked first and only later deprecated. Raw memories are never
deleted by schema forgetting.

## Construction and LLM integration

```python
from sentence_transformers import SentenceTransformer

from socialclaw.llm import OpenAIChatClient
from socialclaw.memory import MemoryRecord
from socialclaw.schema import build_schema_system

llm = OpenAIChatClient(
    base_url="https://openrouter.ai/api/v1",
    api_key="...",
    model="your-model",
)
embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
system = build_schema_system(
    "outputs/my_run/learned_state",
    llm=llm,
    embedder=embedder,
)

episode = MemoryRecord(
    task="Explore the current game state",
    context="A lever is visible on the left",
    outcome="The target moved upward",
    success=True,
)
episode.add_event(
    observation="Lever at rest",
    action="ACTION1",
    result="Target moved upward",
)
node = system.remember_and_learn(episode)
```

The LLM is replaceable through the small `ChatModel` protocol, and the
embedder through the `Embedder` protocol. If no LLM is supplied, a conservative
deterministic fallback only creates a rule when an episode contains complete
observation/action/result evidence. No network call is made during unit tests.

The state directory contains atomic snapshots:

```text
learned_state/
  memory.json
  schema.json
```

Learned-vs-Gold evaluation is deliberately outside this runtime lifecycle.
`scripts/evaluate_learned_schema.py` reads an immutable learned snapshot and an
accepted Gold version into canonical read-only views, writes a separate report,
and never updates Memory/Schema state. See
[`schema_evaluation.md`](schema_evaluation.md).

## Main implementation files

- `socialclaw/memory/models.py`: episode and event classes.
- `socialclaw/memory/store.py`: in-memory and atomic JSON persistence.
- `socialclaw/memory/bank.py`: memory CRUD and retrieval.
- `socialclaw/schema/node.py`: layered node and evidence indexes.
- `socialclaw/schema/layered_graph.py`: graph invariants and links.
- `socialclaw/schema/induction.py`: structured LLM generation/fusion.
- `socialclaw/schema/trajectory_pipeline.py`: frozen trajectory -> transition/window/episode Memory projection.
- `socialclaw/schema/window_induction.py`: profiler/keyframe/proposal/validator/applier/audit pipeline.
- `socialclaw/schema/evaluation.py`: evaluator-only canonical matching and metrics.
- `socialclaw/schema/gold_loader.py`: accepted Gold loader, never imported by induction or runners.
- `socialclaw/schema/manager.py`: learning, feedback, masking, decay, merge.
- `socialclaw/schema/system.py`: complete stack factory.
- `socialclaw/methods/schema.py`: benchmark-neutral binary-feedback lifecycle.
- `socialclaw/arc_runner.py`: migrated online ARC transition loop.
