"""Filtering, scoring, decay, sorting, and neighborhood navigation.

Iteration order follows insertion order everywhere (Python ``dict`` preserves it,
matching JS ``Map``), so result ordering is identical to the TS library. The
multi-sort comparator is ported via ``functools.cmp_to_key`` to reproduce the JS
comparator exactly, including its stable tie-breaking.
"""

from __future__ import annotations

import math
from functools import cmp_to_key
from typing import Any

from pydantic import BaseModel

from . import _time
from ._uuid import safe_extract_timestamp
from .errors import InvalidTimestampError
from .graph import GraphState
from .models import (
    DecayConfig,
    Edge,
    EdgeFilter,
    MemoryFilter,
    MemoryItem,
    QueryOptions,
    Range,
    ScoredItem,
    ScoreWeights,
    SortOption,
)

__all__ = [
    "ScoredQueryOptions",
    "matches_filter",
    "extract_timestamp",
    "get_items",
    "get_scored_items",
    "get_edges",
    "get_item_by_id",
    "get_edge_by_id",
    "get_related_items",
    "get_parents",
    "get_children",
    "compute_decay_multiplier",
    "compute_score",
    "get_sort_value",
]

_MISSING = object()

INTERVAL_MS: dict[str, int] = {
    "hour": 3_600_000,
    "day": 86_400_000,
    "week": 604_800_000,
}


# ---------------------------------------------------------------------------
# Coercion helpers — public functions accept models OR plain dicts (D5/ergonomics)
# ---------------------------------------------------------------------------


def _coerce_filter(f: MemoryFilter | dict[str, Any] | None) -> MemoryFilter | None:
    if f is None or isinstance(f, MemoryFilter):
        return f
    return MemoryFilter.model_validate(f)


def _coerce_options(o: QueryOptions | dict[str, Any] | None) -> QueryOptions | None:
    if o is None or isinstance(o, QueryOptions):
        return o
    return QueryOptions.model_validate(o)


def _coerce_weights(w: ScoreWeights | dict[str, Any]) -> ScoreWeights:
    if isinstance(w, ScoreWeights):
        return w
    return ScoreWeights.model_validate(w)


def _coerce_edge_filter(f: EdgeFilter | dict[str, Any] | None) -> EdgeFilter | None:
    if f is None or isinstance(f, EdgeFilter):
        return f
    return EdgeFilter.model_validate(f)


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------


def extract_timestamp(uuid_id: str) -> int:
    """Extract the ms timestamp from a UUIDv7 id, raising on anything else."""
    ts = safe_extract_timestamp(uuid_id)
    if ts is None:
        raise InvalidTimestampError(
            f'Cannot extract timestamp: "{uuid_id}" is not a valid UUIDv7'
        )
    return ts


def _item_timestamp(item: MemoryItem) -> int:
    ts = item.created_at if item.created_at is not None else safe_extract_timestamp(item.id)
    if ts is None:
        raise InvalidTimestampError(
            f'Cannot determine timestamp for item "{item.id}": '
            "set created_at or use a UUIDv7 id"
        )
    return ts


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def _resolve_path(obj: Any, path: str) -> Any:
    current = obj
    for segment in path.split("."):
        if not isinstance(current, dict) or segment not in current:
            return _MISSING
        current = current[segment]
    return current


def _matches_range(value: float | None, rng: Range | None) -> bool:
    if rng is None:
        return True
    if rng.min is not None and (value is None or value < rng.min):
        return False
    if rng.max is not None and (value is None or value > rng.max):
        return False
    return True


def matches_filter(item: MemoryItem, f: MemoryFilter) -> bool:
    if f.ids is not None and item.id not in f.ids:
        return False

    if f.scope is not None and item.scope != f.scope:
        return False
    if f.scope_prefix is not None and not item.scope.startswith(f.scope_prefix):
        return False

    if f.author is not None and item.author != f.author:
        return False
    if f.kind is not None and item.kind != f.kind:
        return False
    if f.source_kind is not None and item.source_kind != f.source_kind:
        return False

    if f.intent_id is not None and item.intent_id != f.intent_id:
        return False
    if f.intent_ids is not None and (
        item.intent_id is None or item.intent_id not in f.intent_ids
    ):
        return False
    if f.task_id is not None and item.task_id != f.task_id:
        return False
    if f.task_ids is not None and (
        item.task_id is None or item.task_id not in f.task_ids
    ):
        return False

    if f.range is not None:
        if not _matches_range(item.authority, f.range.authority):
            return False
        if not _matches_range(item.conviction, f.range.conviction):
            return False
        if not _matches_range(item.importance, f.range.importance):
            return False

    if f.has_parent is not None:
        if item.parents is None or f.has_parent not in item.parents:
            return False
    if f.is_root is not None:
        has_parents = item.parents is not None and len(item.parents) > 0
        if f.is_root and has_parents:
            return False
        if not f.is_root and not has_parents:
            return False

    if f.parents is not None:
        p = item.parents or []
        if f.parents.includes is not None and f.parents.includes not in p:
            return False
        if f.parents.includes_any is not None and not any(i in p for i in f.parents.includes_any):
            return False
        if f.parents.includes_all is not None and not all(i in p for i in f.parents.includes_all):
            return False
        if f.parents.count is not None and not _matches_range(len(p), f.parents.count):
            return False

    if f.decay is not None:
        multiplier = compute_decay_multiplier(item, f.decay.config)
        if multiplier < f.decay.min:
            return False

    if f.created is not None:
        ts = _item_timestamp(item)
        if f.created.before is not None and ts >= f.created.before:
            return False
        if f.created.after is not None and ts < f.created.after:
            return False

    if f.not_ is not None and matches_filter(item, f.not_):
        return False
    if f.meta is not None:
        for path, value in f.meta.items():
            if _resolve_path(item.meta, path) != value:
                return False
    if f.meta_has is not None:
        for path in f.meta_has:
            if _resolve_path(item.meta, path) is _MISSING:
                return False
    if f.or_:
        if not any(matches_filter(item, sub) for sub in f.or_):
            return False
    return True


# ---------------------------------------------------------------------------
# Decay & scoring
# ---------------------------------------------------------------------------


def compute_decay_multiplier(item: MemoryItem, decay: DecayConfig) -> float:
    age_ms = _time.now_ms() - _item_timestamp(item)
    if age_ms <= 0:
        return 1.0  # future item (clock skew) — no decay
    interval_ms = INTERVAL_MS.get(decay.interval)
    if interval_ms is None:
        raise ValueError(
            f'Unknown decay interval: "{decay.interval}". Expected "hour", "day", or "week".'
        )
    intervals = age_ms / interval_ms

    if decay.type == "exponential":
        return float((1 - decay.rate) ** intervals)
    if decay.type == "linear":
        return max(0.0, 1 - decay.rate * intervals)
    if decay.type == "step":
        return float((1 - decay.rate) ** math.floor(intervals))
    raise ValueError(
        f'Unknown decay type: "{decay.type}". Expected "exponential", "linear", or "step".'
    )


def _n(value: float | None) -> float:
    return value if value is not None else 0.0


def compute_score(item: MemoryItem, weights: ScoreWeights) -> float:
    base = (
        _n(weights.authority) * item.authority
        + _n(weights.conviction) * _n(item.conviction)
        + _n(weights.importance) * _n(item.importance)
    )
    if weights.decay is None:
        return base
    return base * compute_decay_multiplier(item, weights.decay)


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------


def get_sort_value(item: MemoryItem, field: str) -> float:
    if field == "authority":
        return item.authority
    if field == "conviction":
        return _n(item.conviction)
    if field == "importance":
        return _n(item.importance)
    if field == "recency":
        return _item_timestamp(item)
    raise ValueError(
        f'Unknown sort field: "{field}". '
        'Expected "authority", "conviction", "importance", or "recency".'
    )


def _multi_sort(items: list[MemoryItem], sorts: list[SortOption]) -> list[MemoryItem]:
    def _cmp(a: MemoryItem, b: MemoryItem) -> int:
        for s in sorts:
            va = get_sort_value(a, s.field)
            vb = get_sort_value(b, s.field)
            if va < vb:
                return -1 if s.order == "asc" else 1
            if va > vb:
                return 1 if s.order == "asc" else -1
        return 0

    return sorted(items, key=cmp_to_key(_cmp))


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------


def get_items(
    state: GraphState,
    filter: MemoryFilter | dict[str, Any] | None = None,
    options: QueryOptions | dict[str, Any] | None = None,
) -> list[MemoryItem]:
    f = _coerce_filter(filter)
    if f is None:
        results = list(state.items.values())
    else:
        results = [item for item in state.items.values() if matches_filter(item, f)]

    opts = _coerce_options(options)
    if opts is not None and opts.sort is not None:
        sorts = opts.sort if isinstance(opts.sort, list) else [opts.sort]
        results = _multi_sort(results, sorts)

    if opts is not None and (opts.offset is not None or opts.limit is not None):
        start = opts.offset or 0
        end = start + opts.limit if opts.limit is not None else None
        results = results[start:end]

    return results


class ScoredQueryOptions(BaseModel):
    pre: MemoryFilter | None = None
    post: MemoryFilter | None = None
    min_score: float | None = None
    limit: int | None = None
    offset: int | None = None


def _coerce_scored_options(
    o: ScoredQueryOptions | dict[str, Any] | None,
) -> ScoredQueryOptions | None:
    if o is None or isinstance(o, ScoredQueryOptions):
        return o
    return ScoredQueryOptions.model_validate(o)


def get_scored_items(
    state: GraphState,
    weights: ScoreWeights | dict[str, Any],
    options: ScoredQueryOptions | dict[str, Any] | None = None,
) -> list[ScoredItem]:
    w = _coerce_weights(weights)
    opts = _coerce_scored_options(options)

    items = get_items(state, opts.pre if opts else None)
    scored = [ScoredItem(item=item, score=compute_score(item, w)) for item in items]
    scored.sort(key=lambda s: s.score, reverse=True)

    if opts is not None and opts.min_score is not None:
        scored = [s for s in scored if s.score >= opts.min_score]
    if opts is not None and opts.post is not None:
        scored = [s for s in scored if matches_filter(s.item, opts.post)]
    if opts is not None and (opts.offset is not None or opts.limit is not None):
        start = opts.offset or 0
        end = start + opts.limit if opts.limit is not None else None
        scored = scored[start:end]

    return scored


def get_edges(
    state: GraphState,
    filter: EdgeFilter | dict[str, Any] | None = None,
) -> list[Edge]:
    f = _coerce_edge_filter(filter)
    active_only = True if f is None or f.active_only is None else f.active_only
    results: list[Edge] = []
    for edge in state.edges.values():
        if active_only and not edge.active:
            continue
        if f is not None:
            if f.from_ is not None and edge.from_ != f.from_:
                continue
            if f.to is not None and edge.to != f.to:
                continue
            if f.kind is not None and edge.kind != f.kind:
                continue
            if f.min_weight is not None and (edge.weight is None or edge.weight < f.min_weight):
                continue
        results.append(edge)
    return results


def get_item_by_id(state: GraphState, id: str) -> MemoryItem | None:
    return state.items.get(id)


def get_edge_by_id(state: GraphState, edge_id: str) -> Edge | None:
    return state.edges.get(edge_id)


def get_related_items(
    state: GraphState,
    item_id: str,
    direction: str = "both",
) -> list[MemoryItem]:
    related_ids: dict[str, None] = {}  # insertion-ordered set
    for edge in state.edges.values():
        if not edge.active:
            continue
        if direction in ("from", "both") and edge.from_ == item_id:
            related_ids[edge.to] = None
        if direction in ("to", "both") and edge.to == item_id:
            related_ids[edge.from_] = None
    related_ids.pop(item_id, None)

    results: list[MemoryItem] = []
    for rid in related_ids:
        item = state.items.get(rid)
        if item is not None:
            results.append(item)
    return results


def get_parents(state: GraphState, item_id: str) -> list[MemoryItem]:
    item = state.items.get(item_id)
    if item is None or not item.parents:
        return []
    return [state.items[pid] for pid in item.parents if pid in state.items]


def get_children(state: GraphState, item_id: str) -> list[MemoryItem]:
    return [
        item
        for item in state.items.values()
        if item.parents and item_id in item.parents
    ]
