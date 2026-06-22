"""Pydantic models for the memory graph.

Design notes (see PLAN.md):
- Entities (`MemoryItem`, `Edge`) are ``frozen`` — immutability is enforced by
  the type system; a "merge" produces a new instance via ``model_copy(update=)``.
- Numeric score bounds use ``Field(ge=0, le=1)`` so construction validates them
  (D2: validation is always on; use ``Model.model_construct`` to bypass).
- Open string unions (kind / source_kind / edge kind / namespace) are plain
  ``str``; the ``Known*`` ``Literal`` aliases document the canonical values.
- ``from`` / ``not`` / ``or`` are Python keywords, so the corresponding fields
  are ``from_`` / ``not_`` / ``or_`` with JSON aliases. ``populate_by_name``
  lets you pass either the field name or the alias.
"""

from __future__ import annotations

from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Open-string type families (known values documented via Literal aliases)
# ---------------------------------------------------------------------------

KnownMemoryKind = Literal[
    "observation",
    "assertion",
    "assumption",
    "hypothesis",
    "derivation",
    "simulation",
    "policy",
    "trait",
]

KnownSourceKind = Literal[
    "user_explicit",
    "observed",
    "derived_deterministic",
    "agent_inferred",
    "simulated",
    "imported",
]

KnownEdgeKind = Literal[
    "DERIVED_FROM",
    "CONTRADICTS",
    "SUPPORTS",
    "ABOUT",
    "SUPERSEDES",
    "ALIAS",
]

KnownNamespace = Literal[
    "memory",
    "task",
    "agent",
    "tool",
    "net",
    "app",
    "chat",
    "system",
    "debug",
]

LifecycleEventType = Literal[
    "memory.created",
    "memory.updated",
    "memory.retracted",
    "edge.created",
    "edge.updated",
    "edge.retracted",
]

SortField = Literal["authority", "conviction", "importance", "recency"]
DecayInterval = Literal["hour", "day", "week"]
DecayType = Literal["exponential", "linear", "step"]


# ---------------------------------------------------------------------------
# MemoryItem (core node)
# ---------------------------------------------------------------------------


class MemoryItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    scope: str
    kind: str
    content: dict[str, Any]

    author: str
    source_kind: str
    parents: list[str] | None = None

    authority: float = Field(ge=0, le=1)
    conviction: float | None = Field(default=None, ge=0, le=1)
    importance: float | None = Field(default=None, ge=0, le=1)

    created_at: int | None = None

    intent_id: str | None = None
    task_id: str | None = None

    meta: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Edge
# ---------------------------------------------------------------------------


class Edge(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)

    edge_id: str
    from_: str = Field(alias="from")
    to: str
    kind: str

    weight: float | None = Field(default=None, ge=0, le=1)

    author: str
    source_kind: str
    authority: float = Field(ge=0, le=1)
    active: bool

    meta: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Event envelope (generic over its payload)
# ---------------------------------------------------------------------------

T = TypeVar("T")


class EventEnvelope(BaseModel, Generic[T]):
    id: str
    namespace: str
    type: str
    ts: str
    trace_id: str | None = None
    payload: T


# ---------------------------------------------------------------------------
# Scoring / decay configuration
# ---------------------------------------------------------------------------


class DecayConfig(BaseModel):
    rate: float = Field(ge=0, le=1)
    interval: str  # DecayInterval; runtime-checked in query.compute_decay_multiplier
    type: str  # DecayType; runtime-checked in query.compute_decay_multiplier


class ScoreWeights(BaseModel):
    # Weights are multipliers, intentionally unbounded.
    authority: float | None = None
    conviction: float | None = None
    importance: float | None = None
    decay: DecayConfig | None = None


class ScoredItem(BaseModel):
    # Not frozen: surface_contradictions annotates `contradicted_by` in place.
    item: MemoryItem
    score: float
    contradicted_by: list[MemoryItem] | None = None


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


class Range(BaseModel):
    min: float | None = None
    max: float | None = None


class ScoreRanges(BaseModel):
    authority: Range | None = None
    conviction: Range | None = None
    importance: Range | None = None


class ParentsFilter(BaseModel):
    includes: str | None = None
    includes_any: list[str] | None = None
    includes_all: list[str] | None = None
    count: Range | None = None


class DecayFilter(BaseModel):
    config: DecayConfig
    min: float = Field(ge=0, le=1)


class CreatedFilter(BaseModel):
    before: int | None = None
    after: int | None = None


class MemoryFilter(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    ids: list[str] | None = None
    scope: str | None = None
    scope_prefix: str | None = None
    author: str | None = None
    kind: str | None = None
    source_kind: str | None = None

    range: ScoreRanges | None = None

    intent_id: str | None = None
    intent_ids: list[str] | None = None
    task_id: str | None = None
    task_ids: list[str] | None = None

    has_parent: str | None = None
    is_root: bool | None = None
    parents: ParentsFilter | None = None

    decay: DecayFilter | None = None
    created: CreatedFilter | None = None

    not_: MemoryFilter | None = Field(default=None, alias="not")
    meta: dict[str, Any] | None = None
    meta_has: list[str] | None = None
    or_: list[MemoryFilter] | None = Field(default=None, alias="or")


class SortOption(BaseModel):
    field: str  # SortField; runtime-checked in query.get_sort_value
    order: Literal["asc", "desc"]


class QueryOptions(BaseModel):
    sort: SortOption | list[SortOption] | None = None
    limit: int | None = Field(default=None, ge=0)
    offset: int | None = Field(default=None, ge=0)


class EdgeFilter(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    from_: str | None = Field(default=None, alias="from")
    to: str | None = None
    kind: str | None = None
    min_weight: float | None = None
    active_only: bool | None = None


# ---------------------------------------------------------------------------
# Memory lifecycle event (emitted by the reducer)
# ---------------------------------------------------------------------------


class MemoryLifecycleEvent(BaseModel):
    namespace: Literal["memory"] = "memory"
    type: LifecycleEventType
    item: MemoryItem | None = None
    edge: Edge | None = None
    cause_type: str | None = None


MemoryFilter.model_rebuild()
