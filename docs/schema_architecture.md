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

The legacy `Concept`/`Relation` schema remains in place for existing ARC
runners. The new implementation uses distinct `LayeredSchema*` names so the
runner can be migrated deliberately instead of mixing two data models.

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

## Main implementation files

- `socialclaw/memory/models.py`: episode and event classes.
- `socialclaw/memory/store.py`: in-memory and atomic JSON persistence.
- `socialclaw/memory/bank.py`: memory CRUD and retrieval.
- `socialclaw/schema/node.py`: layered node and evidence indexes.
- `socialclaw/schema/layered_graph.py`: graph invariants and links.
- `socialclaw/schema/induction.py`: structured LLM generation/fusion.
- `socialclaw/schema/manager.py`: learning, feedback, masking, decay, merge.
- `socialclaw/schema/system.py`: complete stack factory.
