# memex-python

**Structured, provenance-tracked memory for AI agents — a faithful Pydantic port of [`@ai2070/memex`](https://www.npmjs.com/package/@ai2070/memex).**

MemEX stores beliefs, evidence, conflicts, and updates — not just retrieved text. It separates three graphs (what is *believed* / *wanted* / *done*) and makes retrieval, contradiction, decay, and identity first-class. This is the Python implementation: every typed structure is a Pydantic v2 model, and behavior matches the TypeScript original (the full upstream test suite is ported — **575 tests**).

- **Pure & immutable** — every mutation is `apply_command(state, cmd) -> (new_state, events)`.
- **Pydantic-native** — models validate on construction; the discriminated-union commands *are* the schema.
- **Wire-compatible** — command tags, enum values, and JSON keys are byte-identical to the TS library, so a Python service and a TS service can share one event log.

## Install

```bash
pip install memex-python      # import name: `memex`
```

Only runtime dependency: `pydantic>=2.6`.

## Quickstart (functional core)

```python
from memex import create_graph_state, create_memory_item, apply_command, get_scored_items

state = create_graph_state()

obs = create_memory_item(
    scope="user:laz/general",
    kind="observation",
    content={"key": "login_count", "value": 42},
    author="agent:monitor",
    source_kind="observed",
    authority=0.9,
    importance=0.7,
)
state, events = apply_command(state, {"type": "memory.create", "item": obs})

top = get_scored_items(state, {"authority": 1.0, "importance": 0.5})
```

Filters, options, and weights accept plain dicts *or* the typed models:

```python
from memex import get_items, smart_retrieve

recent = get_items(state, {"kind": "observation", "range": {"authority": {"min": 0.5}}})

packed = smart_retrieve(
    state,
    budget=2000,
    cost_fn=lambda item: len(str(item.content)),
    weights={"authority": 0.6, "importance": 0.4},
    contradictions="surface",   # or "filter"
    diversity={"author_penalty": 0.2},
)
```

## Quickstart (`MemexStore` facade)

For a stateful, object-oriented surface that rebinds state for you:

```python
from memex import MemexStore

store = MemexStore()
a = store.create(scope="user:laz", kind="observation", content={"v": 1},
                 author="agent:x", source_kind="observed", authority=0.9)
b = store.create(scope="user:laz", kind="assertion", content={"v": 2},
                 author="agent:y", source_kind="user_explicit", authority=0.4)

store.mark_contradiction(a.id, b.id, "system:detector")
store.resolve_contradiction(a.id, b.id, "system:resolver")   # b's authority drops

intent = store.create_intent(label="find target", priority=0.9, owner="user:laz")
task = store.create_task(intent_id=intent.id, action="search", priority=0.8)

snapshot = store.dumps(pretty=True)        # JSON; MemexStore.loads(...) restores
```

## The three graphs

| Graph  | Reducer                 | Core type    | Holds                                   |
|--------|-------------------------|--------------|-----------------------------------------|
| Memory | `apply_command`         | `MemoryItem` | beliefs, evidence, contradictions, edges|
| Intent | `apply_intent_command`  | `Intent`     | active goals with a status machine      |
| Task   | `apply_task_command`    | `Task`       | units of work tied to intents           |

Each item carries three orthogonal `0..1` scores — **authority** (trust), **conviction** (author confidence), **importance** (current salience) — plus `kind`, `source_kind`, `parents` (provenance), and typed `edges` (`DERIVED_FROM`, `CONTRADICTS`, `SUPPORTS`, `ABOUT`, `SUPERSEDES`, `ALIAS`).

## Validating external input

```python
from memex.schemas import validate_command
from memex import apply_command

cmd = validate_command(raw)          # raises pydantic.ValidationError on a bad shape
state = apply_command(state, cmd).state
```

## Intentional divergences from the TS library

1. **Always-on validation** — constructing any model with an out-of-range score raises `pydantic.ValidationError` (TS only validated inside the factories). Use `Model.model_construct(...)` to bypass for deliberately-invalid fixtures.
2. **Error types** — score-bound violations surface as `ValidationError`; `cost_fn` contract violations and unknown sort/decay enums as `ValueError` (the TS `RangeError`).
3. **Frozen entities** — `MemoryItem` / `Edge` / `Intent` / `Task` are immutable; "edits" produce new instances (`model_copy(update=...)`).
4. **`from_` field** — `from` is a Python keyword, so `Edge.from_` is the attribute; it serializes to `"from"` via its alias.

Everything else — command tags, JSON shape, scoring/decay math, contradiction determinism, replay tolerance, transplant re-id semantics — matches the TS library exactly.

## Development

```bash
pip install -e ".[dev]"
pytest            # 575 tests
ruff check .
mypy
```

## License

Apache-2.0 (matching upstream).
