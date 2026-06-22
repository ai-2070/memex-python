"""Bulk operations: single-pass transforms, score adjustments, importance decay."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from . import _time
from .graph import GraphState
from .models import Edge, MemoryFilter, MemoryItem, MemoryLifecycleEvent, QueryOptions
from .query import get_items
from .reducer import CommandResult, merge_item

__all__ = ["ScoreAdjustment", "ItemTransform", "apply_many", "bulk_adjust_scores", "decay_importance"]

ItemTransform = Callable[[MemoryItem], "dict[str, Any] | None"]


class ScoreAdjustment(BaseModel):
    authority: float | None = None
    conviction: float | None = None
    importance: float | None = None


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def apply_many(
    state: GraphState,
    filter: MemoryFilter | dict[str, Any] | None,
    transform: ItemTransform,
    author: str,
    reason: str | None = None,
    options: QueryOptions | dict[str, Any] | None = None,
) -> CommandResult:
    """Apply ``transform`` to every matching item in a single pass.

    ``transform`` returns ``None`` to retract (also cleaning up incident edges),
    an empty dict to skip, or a partial to update. The items dict is cloned once;
    the edges dict and a reverse index are built lazily on the first retract.
    """
    matched = get_items(state, filter, options)
    if not matched:
        return CommandResult(state, [])

    items = dict(state.items)
    edges: dict[str, Edge] | None = None
    edges_by_endpoint: dict[str, list[str]] | None = None
    all_events: list[MemoryLifecycleEvent] = []
    changed = False

    for item in matched:
        if item.id not in items:
            continue

        partial = transform(item)

        if partial is None:
            del items[item.id]
            all_events.append(
                MemoryLifecycleEvent(type="memory.retracted", item=item, cause_type="memory.retract")
            )
            changed = True
            if state.edges:
                if edges is None:
                    edges = dict(state.edges)
                if edges_by_endpoint is None:
                    edges_by_endpoint = {}
                    for edge_id, edge in state.edges.items():
                        edges_by_endpoint.setdefault(edge.from_, []).append(edge_id)
                        if edge.from_ != edge.to:
                            edges_by_endpoint.setdefault(edge.to, []).append(edge_id)
                incident_ids = edges_by_endpoint.get(item.id)
                if incident_ids:
                    for edge_id in incident_ids:
                        incident_edge = edges.get(edge_id)
                        if incident_edge is None:
                            continue  # already cleaned by a prior retract
                        del edges[edge_id]
                        all_events.append(
                            MemoryLifecycleEvent(type="edge.retracted", edge=incident_edge, cause_type="memory.retract")
                        )
        elif len(partial) > 0:
            merged = merge_item(item, partial)
            items[item.id] = merged
            all_events.append(
                MemoryLifecycleEvent(type="memory.updated", item=merged, cause_type="memory.update")
            )
            changed = True

    if not changed:
        return CommandResult(state, [])

    return CommandResult(GraphState(items, edges if edges is not None else state.edges), all_events)


def bulk_adjust_scores(
    state: GraphState,
    criteria: MemoryFilter | dict[str, Any],
    delta: ScoreAdjustment | dict[str, Any],
    author: str,
    reason: str | None = None,
) -> CommandResult:
    d = delta if isinstance(delta, ScoreAdjustment) else ScoreAdjustment.model_validate(delta)

    def transform(item: MemoryItem) -> dict[str, Any]:
        partial: dict[str, Any] = {}
        if d.authority is not None:
            partial["authority"] = _clamp(item.authority + d.authority)
        if d.conviction is not None:
            partial["conviction"] = _clamp((item.conviction or 0) + d.conviction)
        if d.importance is not None:
            partial["importance"] = _clamp((item.importance or 0) + d.importance)
        return partial

    return apply_many(state, criteria, transform, author, reason)


def decay_importance(
    state: GraphState,
    older_than_ms: int,
    factor: float,
    author: str,
    reason: str | None = None,
) -> CommandResult:
    """Decay importance on items created before a cutoff time."""
    cutoff = _time.now_ms() - older_than_ms

    def transform(item: MemoryItem) -> dict[str, Any]:
        current = item.importance if item.importance is not None else 0
        if current == 0:
            return {}
        return {"importance": _clamp(current * factor)}

    return apply_many(state, {"created": {"before": cutoff}}, transform, author, reason or "time-based importance decay")
