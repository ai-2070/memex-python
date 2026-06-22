# memex-python — Implementation Plan

A faithful Python/Pydantic port of [`@ai2070/memex`](https://www.npmjs.com/package/@ai2070/memex) — a typed, immutable, provenance-tracked memory graph for AI agents.

- **Source of truth:** the TypeScript library at `../memex` (v0.13.0, ~4,350 LOC source / ~10,200 LOC tests, 18 modules).
- **PyPI/distribution name:** `memex-python` · **import name:** `memex`
- **Status:** ✅ **complete.** All 8 phases shipped — 18 source modules + the `MemexStore` facade, **582 tests passing** (all 27 upstream test files ported + facade/store tests), `ruff` and `mypy --strict` both clean.

---

## 1. What we're porting

MemEX is pure, functional, and immutable. Every mutation is `apply_command(state, cmd) -> (new_state, events)`, and the same `commands → reducer → lifecycle events` pattern is applied across three graphs:

| Graph  | Core type    | Holds                                    | Namespace  |
|--------|--------------|------------------------------------------|------------|
| Memory | `MemoryItem` | beliefs, evidence, contradictions, edges | `"memory"` |
| Intent | `Intent`     | active goals, priorities, status         | `"intent"` |
| Task   | `Task`       | units of work tied to intents            | `"task"`   |

On top of the three reducers sit: retrieval/scoring/decay, contradiction & alias integrity, bulk operations, event-log replay, transplant (export/import slices), serialization, and stats.

**Wire compatibility is a design goal.** MemEX's data field names are already snake_case (`source_kind`, `created_at`, `intent_id`), and command tags / enum values / JSON keys are plain strings (`"memory.create"`, `"derived_deterministic"`, `"DERIVED_FROM"`). We keep these **byte-identical** so a Python service and a TS service can share an append-only event log.

## 2. Interpretation of "Pydantic-like API and conventions"

A faithful behavioral port where every typed structure is a Pydantic v2 model, validation lives in the models, and the public surface follows Python conventions — **not** a redesign of semantics.

- **Entities → `pydantic.BaseModel`** (`MemoryItem`, `Edge`, `Intent`, `Task`, `EventEnvelope`, filters, options, lifecycle events).
- **Score bounds → `Field(ge=0, le=1)`** instead of hand-written `validateScore` throwing `RangeError`.
- **Commands → a discriminated union** (`Annotated[Union[...], Field(discriminator="type")]`) of small models — this **replaces the entire `schemas.ts`/Zod layer for free**, because in Pydantic the model *is* the schema.
- **Functions → snake_case** (`create_memory_item`, `apply_command`, `get_items`, `smart_retrieve`, `cascade_retract`, `export_slice`, `replay_from_envelopes`).
- **Command tags, enum values, and JSON keys stay byte-identical** to TS for cross-language event-log interop.

## 3. Locked decisions

| # | Decision | Resolution |
|---|----------|-----------|
| **D1** | Entity mutability | `model_config = ConfigDict(frozen=True)` on `MemoryItem`/`Edge`/`Intent`/`Task`; "merge" = `existing.model_copy(update=...)`. `model_copy(update=)` skips validation, exactly mirroring TS: *factory validates scores, `*.update` does not.* |
| **D2** | Validation strictness | **EMBRACE.** Scores validated on every construction via Pydantic. There is no "raw literal bypasses the factory" path as in TS. `model_construct()` is the documented escape hatch for the handful of tolerance tests that must inject deliberately-invalid data. This is the one intentional semantic divergence from TS. |
| **D3** | `GraphState`/`IntentState`/`TaskState` | Lightweight `@dataclass(frozen=True, slots=True)` holding plain `dict[str, Model]`, **not** Pydantic models. The reducer clones the dict on every command (`dict(state.items)` ≡ TS `new Map(...)`); validated-model state would re-validate every item per command (O(n)) and is unacceptable. |
| **D4** | Result bundles | `NamedTuple`s: `CommandResult(state, events)`, `ReplayResult(state, events, skipped)`, `ImportResult(mem_state, intent_state, task_state, report)`. Supports both `s = apply_command(...).state` and `s, events = apply_command(...)`. |
| **D5** | `Partial<T>` updates | `partial` is a `dict[str, Any]`. Faithful JS→Python mapping: JS `undefined` ≡ key absent; JS `null` ≡ Python `None`. Key present → applied (even if `None`); `content`/`meta` shallow-merge present keys; `id`/`created_at` stripped. |
| **D6** | UUIDv7 source | Internal `_uuid.py` (generate v7 from `time` + `os.urandom`; decode the 48-bit big-endian ms timestamp). Keeps runtime deps to **just `pydantic`**, mirroring TS's minimal-deps ethos. Implementation checks for a stdlib `uuid7` (Py 3.14+) and uses it if present. |
| **D7** | Errors | Custom exceptions for domain errors (`MemoryNotFoundError`, `EdgeNotFoundError`, `DuplicateMemoryError`, `DuplicateEdgeError`, `InvalidTimestampError`, `IntentNotFoundError`, `DuplicateIntentError`, `InvalidIntentTransitionError`, `TaskNotFoundError`, `DuplicateTaskError`, `InvalidTaskTransitionError`). `RangeError` → `pydantic.ValidationError` (scores) and `ValueError` (e.g. `cost_fn` non-finite/negative). |
| **D8** | Strict ISO-8601 replay parser | Port `replay.ts`'s regex-based parser **verbatim** (rejects sub-ms precision, validates calendar fields incl. leap years, requires `Z` or explicit offset). Do **not** use `datetime.fromisoformat` — it is too lenient and would change accept/reject behavior and break replay ordering determinism. |
| **D9** | OO facade | **INCLUDE.** Functional core is the backbone (1:1 with TS, test-portable). A `MemexStore` facade (`.create()/.update()/.retract()/.query()/.scored()/.smart_retrieve()` plus intent/task helpers) wraps the functional core for ergonomic Pydantic-style use. Added after the core is green (Phase 8). |
| **D10** | Naming | Distribution `memex-python`; import `memex`. |
| **Scope** | — | **Full port including the complete test suite.** All 27 vitest files ported to pytest; parity on all ~540 assertions before sign-off. |
| **Speed** | — | **Sequential.** No parallel agent fan-out. |

## 4. Package layout

```
memex-python/
├── pyproject.toml            # PEP 621, hatchling; runtime dep: pydantic>=2.6 ; dev: ruff, mypy, pytest, pytest-cov
├── PLAN.md                   # this file
├── README.md  API.md  WHITEPAPER.md   # ported docs (Phase 8)
├── .github/workflows/ci.yml  # matrix 3.10–3.14, mirrors ../memex CI
├── src/memex/
│   ├── __init__.py           # barrel exports (mirrors index.ts)
│   ├── _uuid.py              # uuid7 gen + timestamp extraction            (D6)
│   ├── models.py             # MemoryItem, Edge, EventEnvelope, MemoryFilter, EdgeFilter,
│   │                         #   SortOption, QueryOptions, DecayConfig, ScoreWeights,
│   │                         #   ScoredItem, MemoryLifecycleEvent
│   ├── commands.py           # MemoryCommand discriminated union (replaces schemas.ts)
│   ├── errors.py             # all domain exceptions                       (D7)
│   ├── graph.py              # GraphState dataclass + create/clone          (D3)
│   ├── factories.py          # create_memory_item / create_edge / create_event_envelope
│   ├── reducer.py            # apply_command + merge_item (match statement)
│   ├── query.py              # get_items, get_scored_items, get_edges, get_item_by_id,
│   │                         #   get_related_items, get_parents, get_children,
│   │                         #   extract_timestamp, decay/score internals
│   ├── retrieval.py          # get_support_tree/set, filter/surface_contradictions,
│   │                         #   apply_diversity, smart_retrieve
│   ├── integrity.py          # contradictions, aliases, stale, dependents,
│   │                         #   cascade_retract, get_items_by_budget
│   ├── bulk.py               # apply_many, bulk_adjust_scores, decay_importance
│   ├── replay.py             # replay_commands, replay_from_envelopes + strict ISO parser  (D8)
│   ├── envelope.py           # wrap_lifecycle_event / wrap_state_event / wrap_edge_state_event
│   ├── serialization.py      # to_json/from_json/dumps/loads (wire-compatible shape)
│   ├── stats.py              # get_stats
│   ├── intent.py             # Intent model + IntentState + reducer + queries + errors
│   ├── task.py               # Task model + TaskState + reducer + queries + errors
│   ├── transplant.py         # export_slice / import_slice (+ re-id logic)
│   ├── schemas.py            # optional parity shim: re-export models + validate_command()
│   └── store.py              # MemexStore OO facade                         (D9)
└── tests/                    # pytest port of all 27 vitest files
```

## 5. Module port map & per-module gotchas

| TS module | Python module | Notes / traps to preserve |
|-----------|---------------|---------------------------|
| `types.ts` | `models.py` + `commands.py` | `MemoryKind`/`SourceKind`/`EdgeKind`/`Namespace` = open `str` (known values documented via `Literal`/constants, **not** closed enums). |
| `helpers.ts` | `factories.py` + `_uuid.py` | `created_at = input.created_at ?? extract_ts(id) ?? now_ms()`. Reproduce the 48-bit big-endian ms decode precisely. Factories validate scores (D2). |
| `graph.ts` | `graph.py` | `create_graph_state()`, `clone_graph_state()` = shallow dict copies. |
| `errors.ts` | `errors.py` | Match class names/messages (D7). |
| `reducer.ts` | `reducer.py` | `match cmd:` over command models. `merge_item` strips `id`/`created_at`, shallow-merges `content`/`meta` (D5). `memory.retract` deletes all incident edges and emits an `edge.retracted` per edge. `merge_edge` strips `edge_id`/`from`/`to`. |
| `query.ts` | `query.py` | `matches_filter` recursion (`not`/`or`), dot-path `meta` resolution, range matching where a **missing value fails any min/max**, decay multiplier math, multi-sort via `functools.cmp_to_key` to match the JS comparator exactly (incl. `recency` = item timestamp). `extract_timestamp` throws `InvalidTimestampError` on non-UUIDv7. |
| `retrieval.ts` | `retrieval.py` | Deterministic contradiction-edge sort (max-score, then min-score, then `edge_id` lexicographic) and self-edge / dedup handling in `surface_contradictions`. `smart_retrieve` pipeline: score → contradiction policy → diversity → greedy budget pack; `cost_fn` non-finite/negative raises `ValueError`. |
| `integrity.ts` | `integrity.py` | `cascade_retract` = iterative post-order DFS (no recursion), root pre-marked visited; cycle-safe, DAG-safe. `resolve_contradiction` = no-op when no active `CONTRADICTS` edge; creates `SUPERSEDES`, lowers loser authority ×0.1. `mark_alias(a, a)` = silent no-op. `mark_contradiction(a, a)` records a self-edge deliberately. |
| `bulk.ts` | `bulk.py` | `apply_many` lazy edge-clone + lazy reverse-index; `transform → None` retracts **and** cascades edge cleanup; `{}` = skip. `clamp` to [0,1] in score adjusters. |
| `replay.ts` | `replay.py` | Strict ISO parser (D8). Both replay fns **never throw**; failures → `skipped`. Envelope `ts` parse errors land in `skipped`, not the call site. Sort by parsed ts before folding. |
| `envelope.ts` | `envelope.py` | `wrap_*` helpers attach `cause_id` and ISO `ts`. |
| `serialization.ts` | `serialization.py` | Keep the `{items: [[id, item], …], edges: [[id, edge], …]}` tuple-pair shape; `model_dump(mode="json", exclude_none=True)` to omit unset optionals as TS omits `undefined`. Round-trip pinned by tests. |
| `stats.ts` | `stats.py` | `get_stats` counts by kind/source_kind/author/scope; root vs with_parents; active edges. |
| `intent.ts` | `intent.py` | Status machine `active ⇄ paused → completed/cancelled`; typed `InvalidIntentTransitionError`. `create_intent` defaults `status="active"`. |
| `task.ts` | `task.py` | Status machine; `start` valid from `pending`/`failed` (retry) and increments `attempt`; `complete`/`fail` only from `running`; `cancel` from any non-terminal. `create_task` defaults `status="pending"`, `attempt=0`. |
| `transplant.ts` | `transplant.py` | **Re-id pre-pass** populates id maps *before* processing so cross-refs (parents, `parent_id`) remap regardless of slice order; `deep_value_equal`/`shallow_equal`; second pass remaps `intent_id`/`task_id` on imported memories. Honor `bugfix-reid-ordering` tests. |
| `schemas.ts` | `schemas.py` (parity shim) | Folded into models. Provide `validate_command()` / `TypeAdapter`-based entry points for parity with the `@ai2070/memex/schemas` export. |
| `index.ts` | `__init__.py` | Barrel export of the full public surface. |

### Anchor snippets

```python
# commands.py — discriminated union replaces all of schemas.ts
class MemoryCreate(BaseModel):
    type: Literal["memory.create"]; item: MemoryItem
class MemoryUpdate(BaseModel):
    type: Literal["memory.update"]; item_id: str
    partial: dict[str, Any]; author: str
    reason: str | None = None; basis: dict[str, Any] | None = None
# ... memory.retract, edge.create/update/retract ...
MemoryCommand = Annotated[Union[MemoryCreate, MemoryUpdate, ...], Field(discriminator="type")]

# reducer.py — the switch becomes a match
def apply_command(state: GraphState, cmd: MemoryCommand) -> CommandResult:
    match cmd:
        case MemoryCreate(item=item):
            if item.id in state.items:
                raise DuplicateMemoryError(item.id)
            items = {**state.items, item.id: item}
            return CommandResult(GraphState(items, state.edges), [created_event(item)])
        # ...

# models.py — immutability + bounded scores (D1, D2)
class MemoryItem(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str; scope: str; kind: str; content: dict[str, Any]
    author: str; source_kind: str; parents: list[str] | None = None
    authority: float = Field(ge=0, le=1)
    conviction: float | None = Field(default=None, ge=0, le=1)
    importance: float | None = Field(default=None, ge=0, le=1)
    created_at: int | None = None
    intent_id: str | None = None; task_id: str | None = None
    meta: dict[str, Any] | None = None
```

## 6. Testing strategy (the parity guarantee)

The 27 vitest files — especially the `bugfix-*` suite (re-id ordering, surface-contradiction dedup, bulk-retract edge cleanup, cascade on DAGs, the "holes"/"sweep" coverage) — encode the hard-won invariants. Porting them is what *proves* behavioral parity, so test porting is a first-class deliverable.

- Mechanical map: `describe/it` → pytest `class`/`def test_`; `expect(x).toEqual(y)` → `assert x == y`; `expect(fn).toThrow(X)` → `pytest.raises(X)`.
- Tests that inject out-of-range scores via raw literals (relying on the TS factory-bypass path) are reworked to `Model.model_construct(...)` per D2.
- Add Python-specific tests beyond the TS suite: serialization round-trip, frozen-model enforcement, discriminated-union validation, `match` exhaustiveness.
- **Definition of done:** parity on all ported assertions; `ruff`, `mypy --strict`, and `pytest` all green.

Per-file targets (vitest → pytest): `reducer`, `query`, `query-advanced`, `retrieval`, `integrity`, `bulk`, `replay`, `envelope`, `serialization`, `stats`, `intent`, `task`, `transplant`, `cross-graph-fields`, `edge-cases`, `edge-cases-v2`, `types`, `graph`, `helpers`, `setup`, and the `bugfix-*` set (`and-coverage`, `holes`, `regressions`, `reid-ordering`, `surface-contradictions-dedup`, `sweep`, `bulk-retract-cascade`).

## 7. Tooling & packaging

- **Build:** `hatchling`, `src/` layout, PEP 621 `pyproject.toml`.
- **Runtime dep:** `pydantic>=2.6` only (internal uuid7; optional `uuid-utils` extra if a faster backend is wanted).
- **Quality gates:** `ruff` (lint + format), `mypy --strict` (or `pyright`), `pytest` + `pytest-cov`.
- **Python target:** 3.10+ (for `X | Y` unions and `match`); CI matrix through 3.14; use stdlib `uuid7` when available.
- **CI:** GitHub Actions matrix mirroring `../memex/.github`.
- **License:** Apache-2.0 (match upstream).

## 8. Phased roadmap (dependency-ordered, sequential)

1. **Foundation** — `pyproject.toml`, `_uuid.py`, `errors.py`, `models.py`, `commands.py`, `graph.py`, `factories.py`. *(unblocks everything)*
2. **Core reducer + query** — `reducer.py`, `query.py`. Port `reducer`, `query`, `query-advanced`, `graph`, `helpers`, `types` tests.
3. **Retrieval & integrity** — `retrieval.py`, `integrity.py`. Port `retrieval`, `integrity`, and contradiction/cascade bugfix tests.
4. **Bulk / replay / envelope / serialization / stats** — port `bulk`, `replay`, `envelope`, `serialization`, `stats` tests (incl. strict-ISO and round-trip).
5. **Intent & Task graphs** — `intent.py`, `task.py`. Port `intent`, `task`, `cross-graph-fields` tests.
6. **Transplant** — `transplant.py`. Port `transplant` + `bugfix-reid-ordering` tests.
7. **Full sweep & parity** — remaining `edge-cases*` and `bugfix-*` files; close all gaps; quality gates green.
8. **Facade & docs** — `store.py` (`MemexStore`), `schemas.py` parity shim, ported `README.md`/`API.md`, Pydantic-flavored quickstart, CI workflow.

First vertical slice to land: **Phases 1 + 2 with their ported tests green**, proving the `commands → reducer → events` pattern end-to-end before building outward.

## 9. Intentional divergences from TS (call-outs for users)

1. **Always-on validation (D2):** constructing any model with an out-of-range score raises `ValidationError`; TS only validated inside factories. Use `model_construct()` to bypass.
2. **`RangeError` → `ValidationError`/`ValueError` (D7):** score errors surface as `pydantic.ValidationError`; `cost_fn` contract violations as `ValueError`.
3. **Frozen entities (D1):** `MemoryItem`/`Edge`/`Intent`/`Task` are immutable; "edits" produce new instances. Matches MemEX's conceptual immutability, now enforced by the type system.

Everything else — command tags, JSON shape, scoring/decay math, contradiction determinism, replay tolerance, transplant re-id semantics — is preserved exactly, including wire compatibility with the TS event log.
