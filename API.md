# memex-python — API reference

Complete reference for the public surface re-exported from the top-level `memex`
package. Everything below is importable directly:

```python
from memex import create_graph_state, apply_command, get_scored_items, MemexStore
```

This document covers the **functional core** (pure `state -> (new_state, events)`
functions), the typed **Pydantic models**, and the optional **`MemexStore`**
facade. For the conceptual overview see [`README.md`](README.md); for design
rationale see [`PLAN.md`](PLAN.md).

## Contents

- [Conventions](#conventions)
- [Core concepts](#core-concepts)
- [Graph state](#graph-state)
- [Entities & factories](#entities--factories)
- [Commands & the reducer](#commands--the-reducer)
- [Querying & filtering](#querying--filtering)
- [Scoring, decay & sorting](#scoring-decay--sorting)
- [Retrieval](#retrieval)
- [Integrity: contradictions, aliases, stale, cascade](#integrity)
- [Bulk operations](#bulk-operations)
- [Intent graph](#intent-graph)
- [Task graph](#task-graph)
- [Statistics](#statistics)
- [Replay](#replay)
- [Serialization](#serialization)
- [Event envelopes](#event-envelopes)
- [Transplant: export / import](#transplant)
- [Validation](#validation)
- [`MemexStore` facade](#memexstore-facade)
- [UUID helpers](#uuid-helpers)
- [Errors](#errors)
- [Type aliases](#type-aliases)

---

## Conventions

**Pure & immutable.** Every mutation has the shape
`f(state, ...) -> (new_state, events)`. The input `state` is never modified; a
fresh `GraphState` is returned with the relevant dict cloned. Entities
(`MemoryItem`, `Edge`, `Intent`, `Task`) are **frozen** Pydantic models — an
"edit" produces a new instance via `model_copy(update=...)`.

**Dicts or models.** Public functions that take a filter, options, or weights
accept either a plain `dict` or the typed model; dicts are validated through the
model. The examples mix both styles freely.

**Validation is always on.** Constructing any model with an out-of-range score
(`authority`/`conviction`/`importance`/`weight`/`priority` must be `0..1`) raises
`pydantic.ValidationError`. Use `Model.model_construct(...)` to deliberately
bypass validation (e.g. test fixtures).

**Keyword-only constructors.** All `create_*` factories take keyword arguments
only.

**Reserved-word fields.** `from`, `not`, and `or` are Python keywords, so the
attributes are `from_`, `not_`, and `or_`; each serializes to/accepts its bare
JSON alias (`"from"`, `"not"`, `"or"`).

**Wire compatibility.** Command tags (`"memory.create"`), enum values
(`"derived_deterministic"`, `"DERIVED_FROM"`), and JSON keys are byte-identical
to the TypeScript `@ai2070/memex`, so a Python and a TS service can share one
event log.

---

## Core concepts

Three independent graphs, each driven by the same `commands -> reducer ->
lifecycle events` pattern:

| Graph  | State        | Core type    | Reducer                  | Namespace  |
|--------|--------------|--------------|--------------------------|------------|
| Memory | `GraphState` | `MemoryItem` | `apply_command`          | `"memory"` |
| Intent | `IntentState`| `Intent`     | `apply_intent_command`   | `"intent"` |
| Task   | `TaskState`  | `Task`       | `apply_task_command`     | `"task"`   |

Each `MemoryItem` carries three orthogonal `0..1` scores — **authority** (trust),
**conviction** (author confidence), **importance** (current salience) — plus
provenance (`parents`) and typed `edges`.

```python
from memex import create_graph_state, create_memory_item, apply_command, get_scored_items

state = create_graph_state()
item = create_memory_item(
    scope="user:laz/general", kind="observation",
    content={"key": "login_count", "value": 42},
    author="agent:monitor", source_kind="observed",
    authority=0.9, importance=0.7,
)
state, events = apply_command(state, {"type": "memory.create", "item": item})
top = get_scored_items(state, {"authority": 1.0, "importance": 0.5})
```

---

## Graph state

### `class GraphState`

Frozen dataclass holding the memory graph. Not a Pydantic model (re-validating
every item per command would be unusably slow).

| Field   | Type                    | Description            |
|---------|-------------------------|------------------------|
| `items` | `dict[str, MemoryItem]` | id → item (insertion-ordered) |
| `edges` | `dict[str, Edge]`       | id → edge (insertion-ordered) |

### `create_graph_state() -> GraphState`

A new empty memory graph.

### `clone_graph_state(state: GraphState) -> GraphState`

Shallow clone — new dicts, shared (immutable) item/edge instances.

---

## Entities & factories

### `class MemoryItem`

Frozen. The core memory node.

| Field         | Type                  | Notes                                   |
|---------------|-----------------------|-----------------------------------------|
| `id`          | `str`                 | usually a UUIDv7                         |
| `scope`       | `str`                 | namespacing key, e.g. `"user:laz/general"` |
| `kind`        | `str`                 | open union; see [`KnownMemoryKind`](#type-aliases) |
| `content`     | `dict[str, Any]`      | arbitrary payload                       |
| `author`      | `str`                 |                                         |
| `source_kind` | `str`                 | open union; see [`KnownSourceKind`](#type-aliases) |
| `parents`     | `list[str] \| None`   | provenance ids                          |
| `authority`   | `float` (0..1)        | required                                |
| `conviction`  | `float \| None` (0..1)|                                         |
| `importance`  | `float \| None` (0..1)|                                         |
| `created_at`  | `int \| None`         | ms epoch; falls back to the UUIDv7 ts   |
| `intent_id`   | `str \| None`         | cross-graph link                        |
| `task_id`     | `str \| None`         | cross-graph link                        |
| `meta`        | `dict[str, Any] \| None` |                                      |

### `class Edge`

Frozen, `populate_by_name=True`.

| Field         | Type                  | Notes                                   |
|---------------|-----------------------|-----------------------------------------|
| `edge_id`     | `str`                 |                                         |
| `from_`       | `str`                 | JSON alias `"from"`                      |
| `to`          | `str`                 |                                         |
| `kind`        | `str`                 | see [`KnownEdgeKind`](#type-aliases)    |
| `weight`      | `float \| None` (0..1)|                                         |
| `author`      | `str`                 |                                         |
| `source_kind` | `str`                 |                                         |
| `authority`   | `float` (0..1)        | required                                |
| `active`      | `bool`                |                                         |
| `meta`        | `dict[str, Any] \| None` |                                      |

### `create_memory_item(...) -> MemoryItem`

```python
create_memory_item(*, scope, kind, content, author, source_kind, authority,
                   id=None, parents=None, conviction=None, importance=None,
                   created_at=None, intent_id=None, task_id=None, meta=None)
```

Mints `id` (UUIDv7) when omitted. `created_at` is the explicit value, else the
id's UUIDv7 timestamp, else the wall clock. Validates score bounds.

### `create_edge(...) -> Edge`

```python
create_edge(*, from_, to, kind, author, source_kind, authority,
            edge_id=None, active=None, weight=None, meta=None)
```

`active` defaults to `True`, `edge_id` to a fresh UUIDv7.

---

## Commands & the reducer

Commands are a Pydantic discriminated union keyed on `type`. `apply_command`
accepts a command **model** or a plain **dict** (validated via
`MemoryCommandAdapter`).

### Memory commands

| Model           | `type`             | Fields (besides `type`)                       |
|-----------------|--------------------|-----------------------------------------------|
| `MemoryCreate`  | `"memory.create"`  | `item: MemoryItem`                            |
| `MemoryUpdate`  | `"memory.update"`  | `item_id`, `partial: dict`, `author`, `reason?`, `basis?` |
| `MemoryRetract` | `"memory.retract"` | `item_id`, `author`, `reason?`               |
| `EdgeCreate`    | `"edge.create"`    | `edge: Edge`                                  |
| `EdgeUpdate`    | `"edge.update"`    | `edge_id`, `partial: dict`, `author`, `reason?` |
| `EdgeRetract`   | `"edge.retract"`   | `edge_id`, `author`, `reason?`               |

- `MemoryCommand` — the `Annotated[Union[...], discriminator="type"]` alias.
- `MemoryCommandAdapter` — `TypeAdapter[MemoryCommand]` for validating raw dicts.

### `apply_command(state, cmd) -> CommandResult`

`cmd: MemoryCommand | dict`. Returns a `CommandResult` NamedTuple
`(state: GraphState, events: list[MemoryLifecycleEvent])`. The result is
iterable, so `state, events = apply_command(...)` works.

Behavior per command:

- **create** — raises `DuplicateMemoryError` / `DuplicateEdgeError` on id clash.
- **update** — shallow-merges `partial` (for `content`/`meta`, merges keys; keys
  cannot be deleted via update); `id`/`created_at` are immutable for items,
  `edge_id`/`from`/`to` for edges. Raises `MemoryNotFoundError` /
  `EdgeNotFoundError` if absent. Updates do **not** re-validate scores.
- **retract (memory)** — removes the item *and* every incident edge, emitting one
  `edge.retracted` per removed edge.

Emits [`MemoryLifecycleEvent`](#class-memorylifecycleevent)s.

### `merge_item(existing: MemoryItem, partial: dict) -> MemoryItem`
### `merge_edge(existing: Edge, partial: dict) -> Edge`

The merge primitives used by the reducer (no validation, mirroring the TS
"updates don't validate" guarantee). Exposed for advanced/bulk callers.

### `class MemoryLifecycleEvent`

| Field        | Type                          | Notes                         |
|--------------|-------------------------------|-------------------------------|
| `namespace`  | `Literal["memory"]`           | always `"memory"`             |
| `type`       | [`LifecycleEventType`](#type-aliases) | e.g. `"memory.created"` |
| `item`       | `MemoryItem \| None`          | set for memory events         |
| `edge`       | `Edge \| None`                | set for edge events           |
| `cause_type` | `str \| None`                 | the command tag that caused it|

---

## Querying & filtering

### `get_items(state, filter=None, options=None) -> list[MemoryItem]`

Filter, then sort, then page. `filter: MemoryFilter | dict | None`,
`options: QueryOptions | dict | None`. With no filter, returns all items in
insertion order.

### `get_item_by_id(state, id) -> MemoryItem | None`

### `matches_filter(item, f: MemoryFilter) -> bool`

The predicate behind `get_items` — useful for ad-hoc filtering.

### `class MemoryFilter`

All present conditions are AND-combined (except `or_`). `populate_by_name=True`.

| Field          | Type                       | Matches when…                                    |
|----------------|----------------------------|--------------------------------------------------|
| `ids`          | `list[str]`                | `item.id` is in the list                         |
| `scope`        | `str`                      | exact scope                                      |
| `scope_prefix` | `str`                      | `item.scope.startswith(...)`                     |
| `author`       | `str`                      | exact                                            |
| `kind`         | `str`                      | exact                                            |
| `source_kind`  | `str`                      | exact                                            |
| `range`        | `ScoreRanges`              | each score within its `Range`                    |
| `intent_id`    | `str`                      | exact                                            |
| `intent_ids`   | `list[str]`                | `item.intent_id` in the list                     |
| `task_id`      | `str`                      | exact                                            |
| `task_ids`     | `list[str]`                | `item.task_id` in the list                       |
| `has_parent`   | `str`                      | id is in `item.parents`                          |
| `is_root`      | `bool`                     | `True`: no parents; `False`: has parents         |
| `parents`      | `ParentsFilter`            | see below                                        |
| `decay`        | `DecayFilter`              | decay multiplier ≥ `min`                         |
| `created`      | `CreatedFilter`            | timestamp window                                 |
| `not_`         | `MemoryFilter` (alias `not`) | sub-filter does **not** match                  |
| `meta`         | `dict[str, Any]`           | each dotted path equals the value                |
| `meta_has`     | `list[str]`                | each dotted path exists                          |
| `or_`          | `list[MemoryFilter]` (alias `or`) | at least one sub-filter matches           |

Supporting models:

- **`Range`** — `min: float | None`, `max: float | None` (inclusive bounds).
- **`ScoreRanges`** — `authority`, `conviction`, `importance`, each a `Range`.
- **`ParentsFilter`** — `includes: str`, `includes_any: list[str]`,
  `includes_all: list[str]`, `count: Range` (range over the number of parents).
- **`DecayFilter`** — `config: DecayConfig`, `min: float` (0..1).
- **`CreatedFilter`** — `before: int | None` (exclusive upper, `ts < before`),
  `after: int | None` (inclusive lower, `ts >= after`); both ms epoch.

`meta`/`meta_has` paths are dotted (`"a.b.c"`) and walk nested dicts.

### `class QueryOptions`

| Field    | Type                              | Notes                          |
|----------|-----------------------------------|--------------------------------|
| `sort`   | `SortOption \| list[SortOption] \| None` | multi-key, applied in order |
| `limit`  | `int \| None` (≥0)                | applied after sort             |
| `offset` | `int \| None` (≥0)                | applied after sort             |

### `class SortOption`

`field: "authority" | "conviction" | "importance" | "recency"`,
`order: "asc" | "desc"`. `recency` sorts by item timestamp. Ties preserve
insertion order (stable). An unknown field raises `ValueError`.

### Edges

- **`get_edges(state, filter=None) -> list[Edge]`** — `filter: EdgeFilter | dict | None`.
  When no filter is given, only **active** edges are returned.
- **`get_edge_by_id(state, edge_id) -> Edge | None`**

#### `class EdgeFilter`

| Field        | Type            | Notes                                |
|--------------|-----------------|--------------------------------------|
| `from_`      | `str` (alias `from`) | exact source                    |
| `to`         | `str`           | exact target                         |
| `kind`       | `str`           | exact                                |
| `min_weight` | `float`         | `edge.weight >= min_weight`          |
| `active_only`| `bool \| None`  | default `True`; `False` includes retracted edges |

### Navigation

- **`get_parents(state, item_id) -> list[MemoryItem]`** — resolves `item.parents`
  to the items that exist.
- **`get_children(state, item_id) -> list[MemoryItem]`** — items listing `item_id`
  in their `parents`.
- **`get_related_items(state, item_id, direction="both") -> list[MemoryItem]`** —
  items connected by active edges. `direction: "from" | "to" | "both"`. Returns
  a deduplicated, insertion-ordered list excluding the item itself.

---

## Scoring, decay & sorting

### `class ScoreWeights`

Multipliers (intentionally unbounded — not `0..1`).

| Field        | Type                  | Notes                          |
|--------------|-----------------------|--------------------------------|
| `authority`  | `float \| None`       | weight on `item.authority`     |
| `conviction` | `float \| None`       | weight on `item.conviction`    |
| `importance` | `float \| None`       | weight on `item.importance`    |
| `decay`      | `DecayConfig \| None` | if set, multiplies the score   |

### `class DecayConfig`

| Field      | Type           | Notes                                   |
|------------|----------------|-----------------------------------------|
| `rate`     | `float` (0..1) | per-interval decay rate                 |
| `interval` | `str`          | `"hour"` \| `"day"` \| `"week"`         |
| `type`     | `str`          | `"exponential"` \| `"linear"` \| `"step"` |

### `compute_decay_multiplier(item, decay: DecayConfig) -> float`

`intervals = age_ms / interval_ms`, where `age_ms = now - item_timestamp`.

- **exponential** → `(1 - rate) ** intervals`
- **linear** → `max(0, 1 - rate * intervals)`
- **step** → `(1 - rate) ** floor(intervals)`

Future-dated items (age ≤ 0, clock skew) return `1.0`. Unknown `interval`/`type`
raises `ValueError`.

### `compute_score(item, weights: ScoreWeights) -> float`

`authority*w.authority + conviction*w.conviction + importance*w.importance`,
then `* compute_decay_multiplier(...)` if `weights.decay` is set. Missing weights
and missing scores are treated as `0`.

### `get_scored_items(state, weights, options=None) -> list[ScoredItem]`

`weights: ScoreWeights | dict`, `options: ScoredQueryOptions | dict | None`.
Pipeline: `pre`-filter → score → sort by score descending → `min_score` →
`post`-filter → `offset`/`limit`.

#### `class ScoredQueryOptions`

| Field       | Type                 | Notes                              |
|-------------|----------------------|------------------------------------|
| `pre`       | `MemoryFilter \| None` | filter applied before scoring    |
| `post`      | `MemoryFilter \| None` | filter applied after scoring     |
| `min_score` | `float \| None`      | drop items scoring below this      |
| `limit`     | `int \| None`        |                                    |
| `offset`    | `int \| None`        |                                    |

#### `class ScoredItem`

`item: MemoryItem`, `score: float`, `contradicted_by: list[MemoryItem] | None`.
Not frozen — `surface_contradictions` annotates `contradicted_by` in place.

### `extract_timestamp(uuid_id: str) -> int`

Extract the ms timestamp from a UUIDv7 id. Raises `InvalidTimestampError` on
anything that isn't a valid v7 UUID.

---

## Retrieval

### Provenance walks

- **`get_support_tree(state, item_id) -> SupportNode | None`** — full provenance
  tree, deduplicating on cycles. Returns `None` if the item is absent.
- **`get_support_set(state, item_id) -> list[MemoryItem]`** — flattened set of
  items that justify a claim.
- **`class SupportNode`** — dataclass `item: MemoryItem`, `parents: list[SupportNode]`.

### Contradiction policies

Both remove **superseded** items (targets of active `SUPERSEDES` edges).

- **`filter_contradictions(state, scored) -> list[ScoredItem]`** — drops the
  lower-scoring side of each unresolved `CONTRADICTS` pair (deterministic
  tie-breaks by score then `edge_id`).
- **`surface_contradictions(state, scored) -> list[ScoredItem]`** — keeps both
  sides, annotating each via `contradicted_by`. Self-edges are ignored.

### Diversity

- **`class DiversityOptions`** — `author_penalty`, `parent_penalty`,
  `source_penalty` (each `float | None`).
- **`apply_diversity(scored, options) -> list[ScoredItem]`** — subtracts a
  per-duplicate penalty (cumulative per repeated author / parent / source),
  clamps at `0`, and re-sorts by score descending.

### `smart_retrieve(...) -> list[ScoredItem]`

```python
smart_retrieve(state, *, budget, cost_fn, weights,
               filter=None, contradictions=None, diversity=None)
```

Score → contradiction policy → diversity → greedy budget pack. `cost_fn:
Callable[[MemoryItem], float]` must return a finite, non-negative number
(otherwise `ValueError`). `contradictions: "filter" | "surface" | None`.
`diversity: DiversityOptions | dict | None`. Greedily appends items whose cost
fits the remaining `budget`.

```python
packed = smart_retrieve(
    state, budget=2000, cost_fn=lambda i: len(str(i.content)),
    weights={"authority": 0.6, "importance": 0.4},
    contradictions="surface", diversity={"author_penalty": 0.2},
)
```

### `get_items_by_budget(...) -> list[ScoredItem]`

```python
get_items_by_budget(state, *, budget, cost_fn, weights, filter=None)
```

The budget-pack step without contradiction/diversity passes.

---

## Integrity

Contradiction & alias management, stale detection, and cascade retraction.

### Contradictions

- **`get_contradictions(state) -> list[Contradiction]`** — active `CONTRADICTS`
  pairs whose endpoints both exist.
- **`mark_contradiction(state, item_id_a, item_id_b, author, meta=None) -> CommandResult`**
  — creates a `CONTRADICTS` edge.
- **`resolve_contradiction(state, winner_id, loser_id, author, reason=None) -> CommandResult`**
  — retracts the `CONTRADICTS` edge(s) between the pair, adds a `SUPERSEDES`
  edge (winner → loser), and drops the loser's `authority` to 10%. A stale call
  with no matching edge is a no-op.
- **`class Contradiction`** — `a: MemoryItem`, `b: MemoryItem`, `edge: Edge | None`.

### Stale items & dependents

- **`get_stale_items(state) -> list[StaleItem]`** — items whose `parents`
  reference ids no longer present.
- **`class StaleItem`** — `item: MemoryItem`, `missing_parents: list[str]`.
- **`get_dependents(state, item_id, transitive=False) -> list[MemoryItem]`** —
  direct children, or the whole dependent subtree when `transitive=True`
  (cycle-safe).

### Cascade retraction

- **`cascade_retract(state, item_id, author, reason=None) -> CascadeResult`** —
  retracts an item and all transitive dependents in post-order (leaves first),
  cleaning incident edges. Cycle- and DAG-safe; iterative (no recursion limit).
- **`class CascadeResult`** — `state: GraphState`,
  `events: list[MemoryLifecycleEvent]`, `retracted: list[str]` (ids in
  retraction order).

### Aliases (identity)

- **`mark_alias(state, item_id_a, item_id_b, author, meta=None) -> CommandResult`**
  — creates bidirectional `ALIAS` edges. Self-alias is a no-op.
- **`get_aliases(state, item_id) -> list[MemoryItem]`** — direct alias targets.
- **`get_alias_group(state, item_id) -> list[MemoryItem]`** — the full connected
  alias component (transitive closure), including the item itself.

---

## Bulk operations

Single-pass transforms that clone the state once instead of per command.

### `apply_many(state, filter, transform, author, reason=None, options=None) -> CommandResult`

Applies `transform: Callable[[MemoryItem], dict | None]` to every item matching
`filter` (with optional `QueryOptions`). The transform returns:

- `None` → **retract** the item (and clean incident edges),
- an **empty dict** → skip,
- a **partial dict** → update.

```python
ItemTransform = Callable[[MemoryItem], dict[str, Any] | None]
```

### `bulk_adjust_scores(state, criteria, delta, author, reason=None) -> CommandResult`

Adds a `ScoreAdjustment` to matching items, clamping each result to `0..1`.

- **`class ScoreAdjustment`** — `authority`, `conviction`, `importance` (each
  `float | None`); only the provided deltas are applied.

### `decay_importance(state, older_than_ms, factor, author, reason=None) -> CommandResult`

Multiplies `importance` by `factor` for items created before
`now - older_than_ms`. Items with zero/absent importance are skipped.

---

## Intent graph

Active goals with a status machine: `active ⇄ paused → completed / cancelled`.

### `class Intent`

Frozen. `id`, `parent_id?`, `label`, `description?`, `priority` (0..1), `owner`,
`status: IntentStatus`, `context?`, `root_memory_ids?`, `meta?`.

- **`IntentStatus`** — `"active" | "paused" | "completed" | "cancelled"`.
- **`class IntentState`** — frozen dataclass `intents: dict[str, Intent]`.

### State & factories

- **`create_intent_state() -> IntentState`**
- **`create_intent(*, label, priority, owner, id=None, parent_id=None,
  description=None, status=None, context=None, root_memory_ids=None, meta=None)
  -> Intent`** — `status` defaults to `"active"`.

### `apply_intent_command(state, cmd) -> IntentResult`

`cmd: IntentCommand | dict`. `IntentResult` is `(state: IntentState,
events: list[IntentLifecycleEvent])`.

| `type`              | Fields                                   | Effect                          |
|---------------------|------------------------------------------|---------------------------------|
| `"intent.create"`   | `intent: Intent`                         | add (dup → `DuplicateIntentError`) |
| `"intent.update"`   | `intent_id`, `partial`, `author`, `reason?` | merge (`id`/`status` ignored) |
| `"intent.complete"` | `intent_id`, `author`, `reason?`         | → `completed` (from active/paused) |
| `"intent.cancel"`   | `intent_id`, `author`, `reason?`         | → `cancelled` (from active/paused) |
| `"intent.pause"`    | `intent_id`, `author`, `reason?`         | → `paused` (from active)        |
| `"intent.resume"`   | `intent_id`, `author`, `reason?`         | → `active` (from paused)        |

Invalid transitions raise `InvalidIntentTransitionError`; missing id raises
`IntentNotFoundError`. `IntentCommand` is the discriminated-union alias.

### Queries

- **`get_intents(state, filter=None) -> list[Intent]`** — `filter: IntentFilter | dict | None`.
- **`get_intent_by_id(state, id) -> Intent | None`**
- **`get_child_intents(state, parent_id) -> list[Intent]`**

#### `class IntentFilter`

`owner`, `status`, `statuses: list[IntentStatus]`, `min_priority`,
`has_memory_id` (in `root_memory_ids`), `parent_id`, `is_root` (bool).

### Events

- **`class IntentLifecycleEvent`** — `namespace="intent"`, `type`
  (`"intent.created"` …), `intent: Intent`, `cause_type: str`.

---

## Task graph

Units of work tied to intents: `pending → running → completed`, with
`running → failed → running` retry and `cancel` from any non-terminal state.

### `class Task`

Frozen. `id`, `intent_id`, `parent_id?`, `action`, `label?`, `status:
TaskStatus`, `priority` (0..1), `context?`, `result?`, `error?`,
`input_memory_ids?`, `output_memory_ids?`, `agent_id?`, `attempt?`, `meta?`.

- **`TaskStatus`** — `"pending" | "running" | "completed" | "failed" | "cancelled"`.
- **`class TaskState`** — frozen dataclass `tasks: dict[str, Task]`.

### State & factories

- **`create_task_state() -> TaskState`**
- **`create_task(*, intent_id, action, priority, id=None, parent_id=None,
  label=None, status=None, context=None, result=None, error=None,
  input_memory_ids=None, output_memory_ids=None, agent_id=None, attempt=None,
  meta=None) -> Task`** — `status` defaults to `"pending"`, `attempt` to `0`.

### `apply_task_command(state, cmd) -> TaskResult`

`cmd: TaskCommand | dict`. `TaskResult` is `(state: TaskState,
events: list[TaskLifecycleEvent])`.

| `type`            | Fields                                   | Effect                              |
|-------------------|------------------------------------------|-------------------------------------|
| `"task.create"`   | `task: Task`                             | add (dup → `DuplicateTaskError`)    |
| `"task.update"`   | `task_id`, `partial`, `author`           | merge (`id`/`status` ignored)       |
| `"task.start"`    | `task_id`, `agent_id?`                    | → `running` (from pending/failed), `attempt++` |
| `"task.complete"` | `task_id`, `result?`, `output_memory_ids?` | → `completed` (from running)      |
| `"task.fail"`     | `task_id`, `error`, `retryable?`         | → `failed` (from running)           |
| `"task.cancel"`   | `task_id`, `reason?`                      | → `cancelled` (from non-terminal)   |

Invalid transitions raise `InvalidTaskTransitionError`; missing id raises
`TaskNotFoundError`. `TaskCommand` is the discriminated-union alias.

### Queries

- **`get_tasks(state, filter=None) -> list[Task]`** — `filter: TaskFilter | dict | None`.
- **`get_task_by_id(state, id) -> Task | None`**
- **`get_tasks_by_intent(state, intent_id) -> list[Task]`**
- **`get_child_tasks(state, parent_id) -> list[Task]`**

#### `class TaskFilter`

`intent_id`, `action`, `status`, `statuses: list[TaskStatus]`, `agent_id`,
`min_priority`, `has_input_memory_id`, `has_output_memory_id`, `parent_id`,
`is_root`.

### Events

- **`class TaskLifecycleEvent`** — `namespace="task"`, `type` (`"task.created"`
  …), `task: Task`, `cause_type: str`.

---

## Statistics

### `get_stats(state) -> GraphStats`

Aggregate counts over a `GraphState`.

- **`class GraphStats`** — `items: ItemStats`, `edges: EdgeStats`.
- **`class ItemStats`** — `total`, `by_kind`, `by_source_kind`, `by_author`,
  `by_scope` (each `dict[str, int]`), `with_parents: int`, `root: int`.
- **`class EdgeStats`** — `total: int`, `active: int`, `by_kind: dict[str, int]`.

---

## Replay

Rebuild a `GraphState` from an event/command log. Integrity-tolerant: per-item
failures are collected, not raised.

### `replay_commands(commands: list) -> ReplayResult`

Fold a list of commands (models or dicts) in order.

### `replay_from_envelopes(envelopes: list) -> ReplayResult`

Sort envelopes by their `ts` (strict ISO-8601, ms precision, explicit offset or
`Z`) and fold their payloads. Each envelope may be a dict or an `EventEnvelope`.

- **`class ReplayResult`** — `state: GraphState`,
  `events: list[MemoryLifecycleEvent]`, `skipped: list[ReplayFailure]`.
- **`class ReplayFailure`** — dataclass `index: int`, `error: Exception`,
  `command=None`, `envelope=None`.

---

## Serialization

On-disk shape matches the TS library —
`{"items": [[id, item], ...], "edges": [[id, edge], ...]}` — with unset
optionals omitted and edge `from` under its alias.

- **`to_json(state) -> SerializedGraphState`** — `dict[str, list[list[Any]]]`.
- **`from_json(data) -> GraphState`** — tolerates missing `items`/`edges` keys.
- **`stringify(state, pretty=False) -> str`** — compact, or 2-space indented.
- **`parse(json_str) -> GraphState`**
- **`SerializedGraphState`** — the serialized-dict type alias.

```python
from memex import stringify, parse
snapshot = stringify(state, pretty=True)
state = parse(snapshot)
```

---

## Event envelopes

### `class EventEnvelope` (generic over `payload`)

`id`, `namespace`, `type`, `ts` (ISO string), `trace_id: str | None`, `payload: T`.

### `create_event_envelope(type, payload, *, trace_id=None, namespace="memory") -> EventEnvelope[Any]`

Mints `id` (UUIDv7) and `ts` (`now_iso`).

### Wrappers

Build envelopes from reducer output for an append-only log:

- **`wrap_lifecycle_event(event, cause_id, trace_id=None) -> EventEnvelope[dict]`**
  — wraps a `MemoryLifecycleEvent`; payload carries the set fields plus `cause_id`.
- **`wrap_state_event(item, cause_id, trace_id=None) -> EventEnvelope[dict]`**
  — a `"state.memory"` snapshot of an item.
- **`wrap_edge_state_event(edge, cause_id, trace_id=None) -> EventEnvelope[dict]`**
  — a `"state.edge"` snapshot of an edge.

---

## Transplant

Export a slice of all three graphs and import it elsewhere, optionally re-id'ing
on collision.

### `export_slice(mem_state, intent_state, task_state, *, ...) -> MemexExport`

Keyword options: `memory_ids`, `intent_ids`, `task_ids` (seed sets);
`include_parents`, `include_children`, `include_aliases`,
`include_related_tasks`, `include_related_intents` (all `bool`, default
`False`). Walks the requested relationships and collects edges between included
memories.

- **`class MemexExport`** — `memories: list[MemoryItem]`, `edges: list[Edge]`,
  `intents: list[Intent]`, `tasks: list[Task]`.
- **`class ExportOptions`** — the same options as a model (for callers that
  prefer to pass a struct).

### `import_slice(mem_state, intent_state, task_state, slice, *, ...) -> ImportResult`

`slice: MemexExport | dict`. Options: `skip_existing_ids=True`,
`shallow_compare_existing=False`, `re_id_on_difference=False`. When re-id'ing,
colliding-but-different entities get fresh UUIDv7-shaped ids (1ms after the
original), and all cross-references (`parents`, `parent_id`, `intent_id`,
memory-id lists) are remapped via a per-graph pre-pass.

- **`class ImportResult`** — `mem_state`, `intent_state`, `task_state`,
  `report: ImportReport`.
- **`class ImportReport`** — `created`, `updated`, `skipped`, `conflicts`, each
  an `ImportBucket`.
- **`class ImportBucket`** — `memories`, `intents`, `tasks`, `edges` (each
  `list[str]` of ids).
- **`class ImportOptions`** — the options as a model.

---

## Validation

`memex.schemas` is the validation entry point (the parity shim for
`@ai2070/memex/schemas`). In Pydantic the models *are* the schema.

```python
from memex.schemas import validate_command
from memex import apply_command

cmd = validate_command(raw)          # raises pydantic.ValidationError on bad shape
state = apply_command(state, cmd).state
```

| Function                          | Returns        |
|-----------------------------------|----------------|
| `validate_command(raw)`           | `MemoryCommand`|
| `validate_intent_command(raw)`    | `IntentCommand`|
| `validate_task_command(raw)`      | `TaskCommand`  |
| `validate_memory_item(raw)`       | `MemoryItem`   |
| `validate_edge(raw)`              | `Edge`         |

Schema aliases (the model is the schema): `MemoryItemSchema`, `EdgeSchema`,
`IntentSchema`, `TaskSchema`. Adapters: `MemoryCommandAdapter`,
`IntentCommandAdapter`, `TaskCommandAdapter`.

---

## `MemexStore` facade

A mutable, object-oriented container over the three graphs. It holds the states,
rebinds them on each mutation, and returns the emitted events — convenient for
agents and daemons that don't want to thread `state =` through every call. The
functional API remains the backbone; the facade just wraps it.

```python
from memex import MemexStore

store = MemexStore()
a = store.create(scope="user:laz", kind="observation", content={"v": 1},
                 author="agent:x", source_kind="observed", authority=0.9)
store.mark_contradiction(a.id, b_id, "system:detector")
snapshot = store.dumps(pretty=True)
restored = MemexStore.loads(snapshot)
```

### Constructor

`MemexStore(mem=None, intents=None, tasks=None)` — start empty or from existing
states. Attributes `store.mem`, `store.intents`, `store.tasks` expose the live
states.

### Memory

| Method | Returns | Notes |
|--------|---------|-------|
| `apply(cmd)` | `list[MemoryLifecycleEvent]` | apply any memory command |
| `create(**kwargs)` | `MemoryItem` | forwards to `create_memory_item`, then creates |
| `add(item)` | `MemoryItem` | create from an existing item |
| `update(item_id, partial, author, reason=None)` | events | |
| `retract(item_id, author, reason=None)` | events | |
| `add_edge(**kwargs)` | `Edge` | forwards to `create_edge`, then creates |
| `items(filter=None, options=None)` | `list[MemoryItem]` | |
| `item(id)` | `MemoryItem \| None` | |
| `scored(weights, options=None)` | `list[ScoredItem]` | |
| `edges(filter=None)` | `list[Edge]` | |
| `parents(item_id)` / `children(item_id)` | `list[MemoryItem]` | |
| `related(item_id, direction="both")` | `list[MemoryItem]` | |
| `smart_retrieve(**kwargs)` | `list[ScoredItem]` | |
| `support_tree(item_id)` | `SupportNode \| None` | |
| `support_set(item_id)` | `list[MemoryItem]` | |
| `stats()` | `GraphStats` | |

### Integrity

| Method | Returns |
|--------|---------|
| `mark_contradiction(a, b, author, meta=None)` | events |
| `resolve_contradiction(winner, loser, author, reason=None)` | events |
| `mark_alias(a, b, author, meta=None)` | events |
| `cascade_retract(item_id, author, reason=None)` | `list[str]` (retracted ids) |
| `contradictions()` | `list[Contradiction]` |
| `stale_items()` | `list[StaleItem]` |
| `aliases(item_id)` / `alias_group(item_id)` | `list[MemoryItem]` |

### Bulk

`apply_many(filter, transform, author, reason=None, options=None)`,
`bulk_adjust_scores(criteria, delta, author, reason=None)`,
`decay_importance(older_than_ms, factor, author, reason=None)` — each returns
events.

### Intent / Task

`apply_intent(cmd)`, `create_intent(**kwargs)`, `get_intents(filter=None)`;
`apply_task(cmd)`, `create_task(**kwargs)`, `get_tasks(filter=None)`.

### Transplant & serialization

`export_slice(**kwargs)` → `MemexExport`; `import_slice(slice, **kwargs)` →
`ImportReport` (mutates the store's states); `to_json()` →
`SerializedGraphState`; `dumps(pretty=False)` → `str`; `MemexStore.loads(json_str)`
→ `MemexStore` (classmethod, restores the memory graph).

---

## UUID helpers

- **`uuid7(ms=None) -> str`** — generate a UUIDv7 string (RFC 9562) for `ms`
  (defaults to now). Encodes a 48-bit big-endian ms timestamp in the first six
  bytes.
- **`safe_extract_timestamp(value: str) -> int | None`** — decode the ms
  timestamp from a UUIDv7; returns `None` for non-v7 input or a non-positive
  timestamp (unlike [`extract_timestamp`](#extract_timestampuuid_id-str---int),
  which raises).

> Install the optional `fast-uuid` extra (`uuid-utils`) for faster generation.

---

## Errors

All domain errors derive from `MemexError`.

| Exception | Raised by | Attributes |
|-----------|-----------|------------|
| `MemexError` | base class | — |
| `MemoryNotFoundError` | update/retract of an absent item | `item_id` |
| `EdgeNotFoundError` | update/retract of an absent edge | `edge_id` |
| `DuplicateMemoryError` | create with an existing id | `item_id` |
| `DuplicateEdgeError` | create with an existing edge id | `edge_id` |
| `InvalidTimestampError` | bad UUIDv7 / envelope timestamp | — |
| `IntentNotFoundError` | intent reducer | `intent_id` |
| `DuplicateIntentError` | intent reducer | `intent_id` |
| `InvalidIntentTransitionError` | intent reducer | `intent_id`, `from_status`, `to_status` |
| `TaskNotFoundError` | task reducer | `task_id` |
| `DuplicateTaskError` | task reducer | `task_id` |
| `InvalidTaskTransitionError` | task reducer | `task_id`, `from_status`, `to_status` |

Out-of-range scores and malformed command shapes surface as
`pydantic.ValidationError`. `cost_fn` contract violations and unknown
sort/decay enums surface as `ValueError`.

---

## Type aliases

Open string unions documented as `Literal` aliases — fields accept any `str`,
but these are the canonical values.

| Alias | Values |
|-------|--------|
| `KnownMemoryKind` | `observation`, `assertion`, `assumption`, `hypothesis`, `derivation`, `simulation`, `policy`, `trait` |
| `KnownSourceKind` | `user_explicit`, `observed`, `derived_deterministic`, `agent_inferred`, `simulated`, `imported` |
| `KnownEdgeKind` | `DERIVED_FROM`, `CONTRADICTS`, `SUPPORTS`, `ABOUT`, `SUPERSEDES`, `ALIAS` |
| `KnownNamespace` | `memory`, `task`, `agent`, `tool`, `net`, `app`, `chat`, `system`, `debug` |
| `LifecycleEventType` | `memory.created`, `memory.updated`, `memory.retracted`, `edge.created`, `edge.updated`, `edge.retracted` |
| `SortField` | `authority`, `conviction`, `importance`, `recency` |
| `DecayInterval` | `hour`, `day`, `week` |
| `DecayType` | `exponential`, `linear`, `step` |
