# memex-python: Epistemic Memory for AI Agents

**Memory that stores what an agent *believes* — with provenance, trust, contradiction, and time — not just text it can retrieve.** A faithful Pydantic v2 port of [`@ai2070/memex`](https://www.npmjs.com/package/@ai2070/memex).

Most "AI memory" answers *"what does the corpus say about X?"* — embed text, retrieve the top‑k, paste it into a prompt. That works until the question becomes *epistemic*: **what should I believe about X**, given that my inputs contain rumors, retractions, partisan sources, stale facts, and outright contradictions?

memex models memory as a **typed, scored, provenance‑tracked graph over an append‑only command log**. Every belief records *who said it, why we believe it, what it conflicts with, when it was true, and how confident we are* — the structure that high‑stakes analytical work (finance, law, geopolitics) depends on and that vector stores collapse away.

> Vector search tells you what is **similar**. memex tells you what you **believe**.

```bash
pip install memex-python                  # import name: `memex` — one runtime dep: pydantic>=2.6
pip install "memex-python[fast-uuid]"     # optional: faster uuidv7 generation via uuid-utils
```

Every typed structure is a Pydantic model, the discriminated‑union commands *are* the schema, and command tags / enum values / JSON keys are **byte‑identical** to the TypeScript library — so a Python service and a TS service can share one event log. Behavior matches the original, with the full upstream test suite ported.

---

## 60‑second tour

Create a graph, record an observation, derive a belief from it, then ask *why* you believe it.

```python
from memex import (
    create_graph_state,
    create_memory_item,
    apply_command,
    get_scored_items,
    get_support_tree,
)

state = create_graph_state()

# A legal-research agent investigating the famous "hot coffee" case.

# 1. Facts from the primary source — the trial record. High authority.
record = create_memory_item(
    scope="case:liebeck-v-mcdonalds",
    kind="observation",
    content={
        "source": "trial record (Liebeck v. McDonald's Restaurants, 1994)",
        "finding": "coffee served at 180-190°F; third-degree burns requiring "
                   "skin grafts; ~700 prior burn complaints on file",
    },
    author="source:trial-record-1994",
    source_kind="imported",
    authority=0.95,
    conviction=0.9,
    importance=0.7,
)
state, events = apply_command(state, {"type": "memory.create", "item": record})

# 2. The agent's legal read *derived from* the record. It's an interpretation,
#    not a recorded fact, so it carries less authority — but it's the answer to
#    the research question, so it's highly salient.
assessment = create_memory_item(
    scope="case:liebeck-v-mcdonalds",
    kind="hypothesis",
    content={"claim": "the verdict rested on documented gross negligence, not a frivolous claim"},
    author="agent:legal-researcher",
    source_kind="agent_inferred",
    parents=[record.id],          # <- provenance edge
    authority=0.55,
    conviction=0.75,
    importance=0.85,
)
state, events = apply_command(state, {"type": "memory.create", "item": assessment})

# 3. Rank beliefs by a weighted blend of the three scores.
ranked = get_scored_items(state, {"authority": 0.5, "conviction": 0.2, "importance": 0.3})

# 4. "What backs this assessment?" -> walk the provenance tree.
tree = get_support_tree(state, assessment.id)
# tree.item is assessment; tree.parents[0].item is record (the trial record)
```

Nothing was mutated in place: `apply_command` returns a **new** `GraphState` plus the lifecycle events it produced (it's a `NamedTuple`, so unpack `state, events = ...` or use `.state`). State is always a fold over the command log — every belief change is replayable and auditable.

> **Dicts or models, everywhere.** Filters, options, weights, and commands accept a plain `dict` *or* the typed Pydantic model. The dict form (shown above) keeps the TS object‑literal style portable; the model form gives you validation and autocomplete. Prefer a stateful object over threading `state =`? See the [`MemexStore` facade](#prefer-an-object-the-memexstore-facade).

---

## Why memex exists

Four concerns drive the design. They aren't a spec every "epistemic memory" must meet — they're the things practitioners in finance, law, and geopolitics reliably run into, and the reason memex has the primitives it has.

### 1. Differential trust — *not all sources are equal*

A 10‑K and a Twitter rumor are not semantic peers. memex gives every item three **orthogonal** scores, so you can say "trust this a lot, but the author wasn't sure, and it barely matters right now" — or any other combination.

| Score | Question it answers | Range |
|-------|--------------------|-------|
| `authority` | How much should the **system** trust this, regardless of who said it? | 0..1 |
| `conviction` | How sure was the **author** when they said it? | 0..1 |
| `importance` | How much should we be **thinking about this right now**? (salience) | 0..1 |

```python
# Same event, two sources — the trust topology is explicit, not flattened.
audited = create_memory_item(
    scope="10K:ACME-2025", kind="observation",
    content={"line": "revenue", "value": 9.84e9, "period": "FY2025"},
    author="filing:ACME-10K-2025", source_kind="user_explicit",
    authority=0.98, importance=0.85,              # audited statement
)

rumor = create_memory_item(
    scope="10K:ACME-2025", kind="hypothesis",
    content={"claim": "revenue will be restated downward"},
    author="social:anon-tip", source_kind="agent_inferred",
    authority=0.2,                                 # barely trusted...
    importance=0.9,                                # ...but worth checking
)
```

`importance` decoupled from `authority` is the cell single‑score systems can't represent: *worth checking, not worth trusting.*

### 2. Preserved disagreement — *show both sides*

When two credible sources conflict, the disagreement is itself a signal. memex records it as a `CONTRADICTS` **edge** and lets retrieval either *surface* both sides (annotated) or *filter* to the higher‑scoring one.

```python
from memex import mark_contradiction, smart_retrieve

# Early hours of a contested event: official narratives disagree; an OSINT
# rumor is low-trust but high-attention.
s = create_graph_state()
dod = create_memory_item(
    scope="geo:event-2023-balloon", kind="assertion",
    content={"claim": "PRC surveillance platform"},
    author="agency:US-DOD", source_kind="user_explicit",
    authority=0.85, conviction=0.85, importance=0.95,
)
mfa = create_memory_item(
    scope="geo:event-2023-balloon", kind="assertion",
    content={"claim": "civilian weather balloon, off-course"},
    author="agency:PRC-MFA", source_kind="user_explicit",
    authority=0.7, conviction=0.8, importance=0.95,
)
for item in (dod, mfa):
    s = apply_command(s, {"type": "memory.create", "item": item}).state
s = mark_contradiction(s, dod.id, mfa.id, "agent:event-router").state

# "surface" keeps both sides and flags each with `contradicted_by`.
briefing = smart_retrieve(
    s,
    budget=4000,
    cost_fn=lambda i: len(str(i.content)),
    weights={"authority": 0.4, "importance": 0.6},
    contradictions="surface",            # or "filter" for a single clean answer
    diversity={"source_penalty": 0.4},   # don't return 20 paraphrases of one wire
)
# briefing[i].contradicted_by lists the items each one conflicts with.
```

### 3. Causal traceability — *answer "what justifies this?"*

Every derivation carries `parents`. `get_support_tree` / `get_support_set` reconstruct the evidence chain back to root observations — a generated citation graph, not a narration.

```python
from memex import get_support_set

get_support_set(state, rating_id)
# -> [rating, leverage_ratio, debt, ebitda, footnote12, ...]
#    every item that justifies the rating, deduped, cycle-safe
```

### 4. Temporal honesty — *distinguish "true now" from "true then" without rewriting history*

Time decay is computed **at query time** from each item's `uuidv7` timestamp; stored scores are never mutated. The same graph answers "what do we know now?" and "what did we know in March?" — both first‑class.

```python
# A query that down-weights stale items, configured per call.
get_scored_items(state, {
    "authority": 0.5,
    "importance": 0.5,
    "decay": {"rate": 0.1, "interval": "day", "type": "exponential"},
})
```

---

## The core model

### Items and edges

A **`MemoryItem`** is a node. Its `kind` says what it *is*; its `source_kind` says how it *got here*.

- **kinds:** `observation` · `assertion` · `assumption` · `hypothesis` · `derivation` · `simulation` · `policy` · `trait`
- **source kinds:** `user_explicit` · `observed` · `derived_deterministic` · `agent_inferred` · `simulated` · `imported`

**Edges** are first‑class objects with their **own** author and authority — because *"case A overrules case B"* or *"filing X supports thesis Y"* is itself a claim someone made with some confidence.

| Edge | Meaning |
|------|---------|
| `DERIVED_FROM` | A relationship discovered after creation |
| `SUPPORTS` | Evidence for another item |
| `CONTRADICTS` | Two items assert conflicting things |
| `SUPERSEDES` | Replaces another item (conflict resolution) |
| `ALIAS` | Same entity, different observations |
| `ABOUT` | References another item |

> `parents` on an item is the fast path for provenance set at creation time; an edge is the general form, added any time, with its own trust score. Both feed `get_support_tree`. (`from` is a Python keyword, so the edge attribute is `from_` and serializes to `"from"`.)

### Event sourcing and immutability

memex is a pure reducer over commands:

```python
apply_command(state, cmd)  # -> CommandResult(state: GraphState, events: list[MemoryLifecycleEvent])
```

Commands (`memory.create | update | retract`, `edge.create | update | retract`) are the only way to change state, and they're meant to be stored append‑only. This buys three properties that matter in regulated settings:

- **Auditability** — every belief change traces to the command that caused it.
- **Time travel** — fold the log up to any point to reconstruct historical state.
- **Branching** — multiple worldlines fork from one checkpoint without contention.

```python
from memex import replay_from_envelopes

# Rebuild state on restart from a persisted, timestamp-ordered event log.
result = replay_from_envelopes(envelopes)
state, events, skipped = result        # ReplayResult is a NamedTuple
# Replay is integrity-tolerant: bad records land in `skipped`, the batch keeps
# going — a long-running daemon doesn't die on one malformed event.
for f in skipped:
    logger.warning("replay skipped index %s: %s", f.index, f.error)
```

Entities (`MemoryItem`, `Edge`, `Intent`, `Task`) are **frozen** — an "edit" produces a new instance. Scores are validated on construction (out‑of‑range raises `pydantic.ValidationError`); `Model.model_construct(...)` is the documented escape hatch for deliberately‑invalid fixtures.

---

## Working with beliefs — recipes

### Provenance: explain a conclusion

```python
from memex import get_support_tree

tree = get_support_tree(state, rating_id)
# SupportNode(item, parents: list[SupportNode]) — recursive, dedupes cycles.
# "Why do we rate this BBB?" -> the tree walks back through the calculated
# ratios to the audited line items and footnotes that conditioned them.
```

### Supersession: replace without deleting

Being restated (finance) or overruled (law) is **not** the same as being wrong. `resolve_contradiction` adds a `SUPERSEDES` edge, lowers the loser's authority, and retracts the open `CONTRADICTS` edge — but keeps the old item queryable.

```python
from memex import mark_contradiction, resolve_contradiction, replay_commands

# Record the conflict, then resolve it: Brown v. Board supersedes Plessy for the
# segregation doctrine. (resolve_contradiction acts on an existing CONTRADICTS
# edge — it lowers the loser's authority and adds SUPERSEDES.)
state = mark_contradiction(state, brown.id, plessy.id, "court:SCOTUS").state
state = resolve_contradiction(
    state, brown.id, plessy.id,
    "court:SCOTUS", "Brown v. Board of Education, 347 U.S. 483",
).state

# `plessy` still exists (reduced authority). A query over the modern scope
# filters it out; a query over the 1953 worldline — rebuilt from the log with
# replay_commands — returns Plessy as live, controlling law.
```

### Staleness and cascade retraction

When evidence is pulled, find what depended on it — and optionally invalidate the whole chain.

```python
from memex import get_stale_items, cascade_retract

get_stale_items(state)                # items whose parents are now missing
# -> [StaleItem(item=..., missing_parents=[...]), ...]

next_state, events, retracted = cascade_retract(
    state, restated_filing_id, "system:restatement",
)
# Retracts the item and every transitive dependent (leverage ratios, covenant
# headroom, growth rates...) in DFS post-order — cycle-safe.
```

### Identity: two names, one entity

```python
from memex import mark_alias, get_alias_group

state = mark_alias(state, the_company.id, acme_corp.id, "agent:resolver").state
state = mark_alias(state, acme_corp.id, acme_industries.id, "agent:resolver").state

get_alias_group(state, the_company.id)   # transitive closure -> all three
```

### Smart retrieval: the whole pipeline in one call

`smart_retrieve` composes scoring → contradiction handling → diversity → budget packing.

```python
from memex import smart_retrieve

context = smart_retrieve(
    state,
    budget=16000,                                    # e.g. a token budget
    cost_fn=lambda i: len(str(i.content)),
    weights={
        "authority": 0.85, "importance": 0.15,
        "decay": {"rate": 0.05, "interval": "day", "type": "exponential"},
    },
    filter={
        "scope_prefix": "doctrine:1A/",
        "range": {"authority": {"min": 0.7}},        # exclude low-authority noise
        "or": [{"source_kind": "user_explicit"}, {"source_kind": "imported"}],
    },
    contradictions="surface",
    diversity={"source_penalty": 0.4},               # span across courts/sources
)
```

Diversity penalties matter: naive ranking returns five paraphrases of the same report. Penalizing duplicate authors / shared parents / source kinds forces genuinely independent sources into the context.

### Querying: the filter algebra

`get_items(state, filter=None, options=None)` supports `and` (implicit), `or`, `not`, `range`, `ids`, `scope` / `scope_prefix`, `has_parent` / `is_root` / `parents` (`includes` / `includes_any` / `includes_all` / `count`), `intent_id` / `task_id`, `meta` (dot‑path) / `meta_has`, `created` (time range), and `decay` (freshness floor), plus multi‑field sort.

```python
from memex import get_items

# Low resolution: trusted, recent, no speculation.
get_items(state, {
    "range": {"authority": {"min": 0.7}},
    "not": {"or": [{"kind": "hypothesis"}, {"kind": "simulation"}]},
    "decay": {"config": {"rate": 0.3, "interval": "day", "type": "exponential"}, "min": 0.5},
}, {"sort": {"field": "importance", "order": "desc"}, "limit": 20})

# The attention queue: high-importance, low-trust items worth thinking about.
get_items(state, {"range": {"authority": {"max": 0.5}, "importance": {"min": 0.7}}})
```

### Bulk operations

Sweep the graph in a single pass — for periodic re‑weighting, decay, or rule‑based cleanup.

```python
from memex import bulk_adjust_scores, decay_importance, apply_many

# Boost a whole episode's importance when an analog gains traction.
state = bulk_adjust_scores(
    state, {"scope": "macro:history/1995"}, {"importance": 0.4},
    "system:rebalance", "1995 analog conviction crossed 0.7",
).state

# Age out importance on everything older than a week.
state = decay_importance(state, 7 * 86_400_000, 0.5, "system:nightly").state

# Conditional transform: return a partial to update, None to retract, {} to skip.
state = apply_many(
    state,
    {"scope_prefix": "tmp:", "range": {"importance": {"max": 0.05}}},
    lambda item: None,                    # retract everything matching
    "system:gc",
).state
```

---

## The three graphs: memory, intent, task

memex coordinates three graphs under one event‑envelope pattern. Use only what you need; they cross‑reference by id.

| Graph | Holds | Core type | Reducer | Question |
|-------|-------|-----------|---------|----------|
| **Memory** | beliefs, evidence, contradictions | `MemoryItem` | `apply_command` | What is believed? |
| **Intent** | goals & objectives | `Intent` | `apply_intent_command` | What is wanted? |
| **Task** | units of work tied to intents | `Task` | `apply_task_command` | What is done? |

This closes the loop a flat store can only fake — **beliefs → goals → tasks → new beliefs**, with provenance running all the way back:

```
 Memory ───▶ Intent ───▶ Task ───┐
 (belief)    (direction) (work)   │ produces new memory
    ▲                             │ (results, observations)
    └─────────────────────────────┘  with parents + intent_id + task_id
```

```python
from memex import (
    create_intent_state, create_intent, apply_intent_command,
    create_task_state, create_task, apply_task_command,
)

intents = create_intent_state()
tasks = create_task_state()

# A goal, anchored to the belief that motivated it.
intent = create_intent(
    label="determine whether Liebeck was a frivolous lawsuit",
    owner="agent:legal-researcher",
    priority=0.8,
    root_memory_ids=[assessment.id],
)
intents = apply_intent_command(intents, {"type": "intent.create", "intent": intent}).state

# An executable unit under that intent, consuming the evidence it weighs.
task = create_task(
    intent_id=intent.id,
    action="review_primary_record",
    priority=0.8,
    input_memory_ids=[record.id],
)
tasks = apply_task_command(tasks, {"type": "task.create", "task": task}).state
tasks = apply_task_command(tasks, {"type": "task.start", "task_id": task.id}).state
tasks = apply_task_command(
    tasks, {"type": "task.complete", "task_id": task.id, "output_memory_ids": []},
).state

# The new belief the task produced links back to its origins.
synthesis = create_memory_item(
    scope="case:liebeck-v-mcdonalds", kind="derivation",
    content={"synthesis": "the popular 'frivolous lawsuit' framing is contradicted by the record"},
    author="agent:legal-researcher", source_kind="derived_deterministic",
    parents=[record.id], intent_id=intent.id, task_id=task.id,
    authority=0.8,
)
```

Intents run a status machine (`active ↔ paused → completed / cancelled`); tasks run (`pending → running → completed / failed`, with `failed → running` retry). Invalid transitions raise typed errors (`InvalidIntentTransitionError`, `InvalidTaskTransitionError`).

---

## Prefer an object? The `MemexStore` facade

The functional API is the backbone and stays pure. `MemexStore` is a Python‑side convenience: it holds the three graph states, rebinds them on each mutation, and returns the emitted events — so you don't thread `state =` through every call.

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

top = store.scored({"authority": 1.0, "importance": 0.5})
snapshot = store.dumps(pretty=True)        # JSON; MemexStore.loads(...) restores
```

Every functional operation has a method (`items`, `scored`, `edges`, `smart_retrieve`, `support_tree`, `cascade_retract`, `apply_many`, `export_slice`, `stats`, …). The live states remain on `store.mem` / `store.intents` / `store.tasks` if you need to drop back to the functional API.

---

## Transplant: portable belief, sandboxed sub‑agents

`export_slice` pulls a self‑contained sub‑graph (optionally walking up parents, down children, across aliases and related intents/tasks). `import_slice` merges it back **append‑only** with a per‑entity report. Memory becomes a *value you can move* — between agents, processes, or machines.

```python
from memex import export_slice, import_slice, get_items

# Pick the entities to hand off (export is by id), then walk up to their
# dependencies so the slice is self-contained.
ids = [i.id for i in get_items(mem_state, {"scope_prefix": "deal:reorg-2026/"})]
slice_ = export_slice(
    mem_state, intent_state, task_state,
    memory_ids=ids,
    include_parents=True,         # also: include_children, include_aliases,
                                  #       include_related_intents / _tasks
)

# ... sub-agent reasons over its OWN copy, adding derivations ...

# Merge back. Existing items are untouched by default (append-only); with
# shallow compare + re-id, a divergent edit to an existing id is minted as a
# fresh uuidv7 instead of clobbering the consensus graph.
result = import_slice(
    mem_state, intent_state, task_state, sub_agent_slice,
    shallow_compare_existing=True, re_id_on_difference=True,
)
merged_mem = result.mem_state
report = result.report
# report.created / updated / skipped / conflicts — what the sub-agent did.
```

This makes multi‑agent patterns fall out of the design rather than requiring bespoke sync code:

| Pattern | How it works |
|---------|--------------|
| **Crews** | Each member works a `scope_prefix` slice; a partner agent queries by `author` / `meta.agent_id` and reconciles. Coordination is data, not chatter. |
| **Swarms** | Fan out N sub‑agents on one baseline slice; merge with conflict detection. Branching scenarios (route IP through Lux vs. SG; Anglo‑German vs. US‑Soviet analog) are this primitive at coarse grain. |
| **Cross‑session memory** | One graph spans all conversations. Continuity is the default; *forgetting* is the explicit operation (`memory.retract`). |
| **Background thinking** | Pick low‑authority/high‑importance items (`importance × (1 − authority)`), open tasks under an intent, write results back with full provenance. |

```python
# Soft isolation: one shared graph, segmented by query — no per-agent stores.
get_items(state, {"meta": {"agent_id": "agent:researcher"}})  # just my work
get_items(state, {"scope_prefix": "project:cyberdeck/"})      # a project view
```

---

## Persistence

The library is pure (no I/O). Persistence is JSON plus your own store; the on‑disk shape is byte‑compatible with the TypeScript library.

```python
from memex import stringify, parse, to_json, from_json, get_stats

blob = stringify(state, pretty=True)   # -> save anywhere
restored = parse(blob)                  # -> GraphState

get_stats(state)  # counts by kind / source_kind / author / scope / edge kind
```

For event‑sourced persistence, store the lifecycle events (wrapped in envelopes via `create_event_envelope` / `wrap_lifecycle_event`) and rebuild with `replay_commands` / `replay_from_envelopes` on startup.

---

## Validating external input

Untrusted input (a webhook, a queue message) should be validated before it's folded in. In Pydantic the models *are* the schema:

```python
from memex.schemas import validate_command
from memex import apply_command

cmd = validate_command(raw)            # raises pydantic.ValidationError on a bad shape
state = apply_command(state, cmd).state
```

`memex.schemas` also exposes `validate_intent_command`, `validate_task_command`, `validate_memory_item`, `validate_edge`, and the schema aliases `MemoryItemSchema` / `EdgeSchema` / `IntentSchema` / `TaskSchema`.

---

## Choosing parameters

Starting points, not prescriptions — calibrate to your domain.

**Decay**

| Scenario | Recommendation |
|----------|----------------|
| Chat context, ephemeral | `{"rate": 0.3, "interval": "hour", "type": "linear"}` |
| Project / working memory | `{"rate": 0.1, "interval": "day", "type": "exponential"}` |
| Policies, traits, foundational docs | No decay — a 1972 communiqué can be critical to a 2024 briefing |

**Score weights**

| Goal | Weights |
|------|---------|
| High‑trust retrieval | `{"authority": 0.8, "importance": 0.2}` |
| Attention queue (what needs thinking?) | `{"importance": 0.8, "authority": 0.2}` |
| Balanced | `{"authority": 0.4, "conviction": 0.3, "importance": 0.3}` |

**Diversity penalties**

| Goal | Recommendation |
|------|----------------|
| Exploration ("what do we know?") | High `author_penalty` (0.3–0.5) — spread across sources |
| Verification ("is this true?") | Low/zero — you *want* corroborating evidence |
| Audit / debugging | Zero — show everything |

**Contradictions**

| Audience | Mode |
|----------|------|
| User‑facing context | `contradictions="filter"` (one clean answer) |
| Agent reasoning | `contradictions="surface"` (see the disagreement) |
| Audit | Neither — call `get_contradictions(state)` directly |

---

## The same primitives, across domains

The three target domains exercise the *same* small set of primitives:

| Need | Finance | Law | Geopolitics | memex primitive |
|------|---------|-----|-------------|-----------------|
| Differential trust | 10‑K vs. tweet | SCOTUS vs. blog | wire vs. troll | `authority` |
| Author confidence ≠ system trust | analyst conviction | dictum vs. holding | source caveats | `conviction` ⟂ `authority` |
| Salience without endorsement | rumor worth checking | unsettled doctrine | unverified field report | `importance` ⟂ `authority` |
| Provenance | audit trail to filings | brief citations | OSINT chains | `parents` + `get_support_tree` |
| Disagreement preserved | bull/bear theses | conflicting clauses | contradicting OSINT | `CONTRADICTS` + `surface` |
| Supersession without deletion | restatements | overruled cases | retracted reports | `SUPERSEDES` |
| Temporal honesty | point‑in‑time | "as of" doctrine | scenario timing | query‑time `DecayConfig` |
| Branching | shadow portfolios | alternative arguments | scenario worldlines | `export_slice` / `import_slice` |
| Goal‑tracked work | thesis verification | brief drafting | verification ops | `Intent` + `Task` |
| Avoiding source collapse | correlated funds | over‑citing one circuit | wire‑service echo | diversity penalties |

The point isn't that memex is uniquely capable in any single dimension — it's that the *same small library* exposes all of these as composable primitives. A sovereign‑debt analyst combining geopolitics, regulatory law, and credit can use one substrate.

---

## What memex deliberately is *not*

memex is a **substrate**, not a thinking system. It makes belief structure cheap to represent; it does not do the reasoning. Known boundaries:

- **It does not assign authority for you.** Importing every source at `0.7` defeats the purpose — calibration is the application's job.
- **It does not detect contradictions.** It makes them easy to *represent*; detecting them is a domain‑specific NLP problem upstream.
- **It is not a probabilistic graphical model.** The three scores are heuristics, not a posterior. Where the math matters, put memex *beside* a Bayesian engine, not in place of one.
- **State is in‑memory.** Serialization and replay make persistence straightforward but external; very large or distributed graphs need a partitioning story the library doesn't provide.
- **The three decay curves are coarse.** Real claim half‑lives vary enormously; per‑class decay config is on you.

It is intended to sit **underneath** vector and text search and **above** event logs and persistence — providing the epistemic semantics those layers lack.

```
        Agent / Cognition layer        (thinking, prioritization)
                  │
               ┌──▼──┐
               │memex│   ← belief state: trust, conflict, provenance, time
               └──┬──┘
     ┌────────────┼────────────┐
 Vector search  Text search  Event store   (recall + durability)
```

---

## Intentional divergences from the TS library

The behavior is a faithful port; the surface follows Python conventions.

1. **Always‑on validation** — constructing any model with an out‑of‑range score raises `pydantic.ValidationError` (TS only validated inside the factories). Use `Model.model_construct(...)` to bypass for deliberately‑invalid fixtures.
2. **Error types** — score‑bound violations surface as `ValidationError`; `cost_fn` contract violations and unknown sort/decay enums as `ValueError` (the TS `RangeError`).
3. **Frozen entities** — `MemoryItem` / `Edge` / `Intent` / `Task` are immutable; "edits" produce new instances (`model_copy(update=...)`).
4. **`from_` field** — `from` is a Python keyword, so `Edge.from_` is the attribute; it serializes to `"from"` via its alias.

Everything else — command tags, JSON shape, scoring/decay math, contradiction determinism, replay tolerance, transplant re‑id semantics — matches the TS library exactly.

---

## Development

CI uses [uv](https://docs.astral.sh/uv/) (see `.github/workflows/`):

```bash
uv sync --all-extras
uv run ruff check .
uv run mypy
uv run pytest
```

Or with plain pip:

```bash
pip install -e ".[dev]"
pytest
```

Releases publish to PyPI via `release.yml` (trusted publishing) on a published GitHub release, or manually via `workflow_dispatch`.

---

## Documentation

- **[API.md](API.md)** — full public API reference (every exported symbol, with signatures and field tables).
- **[PLAN.md](PLAN.md)** — design rationale and the port's locked decisions.

## License

Apache‑2.0 — see [LICENSE](LICENSE).
