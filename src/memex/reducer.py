"""The memory-graph reducer: ``apply_command(state, cmd) -> (new_state, events)``.

Pure and immutable — every branch returns a fresh :class:`GraphState` (the
relevant dict is cloned) and never mutates the input. ``merge_item`` /
``merge_edge`` use ``model_copy(update=...)``, which does NOT re-validate — this
exactly mirrors the TS guarantee that *factories validate scores, updates do not*.
"""

from __future__ import annotations

from typing import Any, NamedTuple

from pydantic import BaseModel

from .commands import (
    EdgeCreate,
    EdgeRetract,
    EdgeUpdate,
    MemoryCommand,
    MemoryCommandAdapter,
    MemoryCreate,
    MemoryRetract,
    MemoryUpdate,
)
from .errors import (
    DuplicateEdgeError,
    DuplicateMemoryError,
    EdgeNotFoundError,
    MemoryNotFoundError,
)
from .graph import GraphState
from .models import Edge, MemoryItem, MemoryLifecycleEvent

__all__ = ["CommandResult", "apply_command", "merge_item", "merge_edge"]


class CommandResult(NamedTuple):
    state: GraphState
    events: list[MemoryLifecycleEvent]


_EDGE_IMMUTABLE = frozenset({"edge_id", "from", "from_", "to"})


def _merge_and_prune(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Shallow-merge ``patch`` onto ``base``.

    The TS version strips ``undefined`` entries before and after merging.
    Python has no ``undefined``; JS ``null`` maps to ``None`` and is kept. So
    this is a plain shallow merge — content/meta keys cannot be *deleted* via an
    update, matching the TS behavior.
    """
    return {**base, **patch}


def merge_item(existing: MemoryItem, partial: dict[str, Any]) -> MemoryItem:
    """Merge a partial onto an item. ``id`` and ``created_at`` are never changed."""
    update: dict[str, Any] = {}
    for key, value in partial.items():
        if key in ("id", "created_at"):
            continue
        if key == "content":
            update["content"] = _merge_and_prune(existing.content, value)
        elif key == "meta":
            update["meta"] = _merge_and_prune(existing.meta or {}, value)
        else:
            update[key] = value
    return existing.model_copy(update=update)


def merge_edge(existing: Edge, partial: dict[str, Any]) -> Edge:
    """Merge a partial onto an edge. ``edge_id`` / ``from`` / ``to`` are fixed."""
    update = {k: v for k, v in partial.items() if k not in _EDGE_IMMUTABLE}
    return existing.model_copy(update=update)


def apply_command(state: GraphState, cmd: MemoryCommand | dict[str, Any]) -> CommandResult:
    command = cmd if isinstance(cmd, BaseModel) else MemoryCommandAdapter.validate_python(cmd)

    match command:
        case MemoryCreate(item=item):
            if item.id in state.items:
                raise DuplicateMemoryError(item.id)
            items = {**state.items, item.id: item}
            return CommandResult(
                GraphState(items, state.edges),
                [MemoryLifecycleEvent(type="memory.created", item=item, cause_type="memory.create")],
            )

        case MemoryUpdate(item_id=item_id, partial=partial):
            existing = state.items.get(item_id)
            if existing is None:
                raise MemoryNotFoundError(item_id)
            merged = merge_item(existing, partial)
            items = {**state.items, item_id: merged}
            return CommandResult(
                GraphState(items, state.edges),
                [MemoryLifecycleEvent(type="memory.updated", item=merged, cause_type="memory.update")],
            )

        case MemoryRetract(item_id=item_id):
            existing = state.items.get(item_id)
            if existing is None:
                raise MemoryNotFoundError(item_id)
            items = dict(state.items)
            del items[item_id]
            edges = dict(state.edges)
            events: list[MemoryLifecycleEvent] = [
                MemoryLifecycleEvent(type="memory.retracted", item=existing, cause_type="memory.retract")
            ]
            for edge_id, edge in state.edges.items():
                if edge.from_ == item_id or edge.to == item_id:
                    del edges[edge_id]
                    events.append(
                        MemoryLifecycleEvent(type="edge.retracted", edge=edge, cause_type="memory.retract")
                    )
            return CommandResult(GraphState(items, edges), events)

        case EdgeCreate(edge=edge):
            if edge.edge_id in state.edges:
                raise DuplicateEdgeError(edge.edge_id)
            edges = {**state.edges, edge.edge_id: edge}
            return CommandResult(
                GraphState(state.items, edges),
                [MemoryLifecycleEvent(type="edge.created", edge=edge, cause_type="edge.create")],
            )

        case EdgeUpdate(edge_id=edge_id, partial=partial):
            existing_edge = state.edges.get(edge_id)
            if existing_edge is None:
                raise EdgeNotFoundError(edge_id)
            merged_edge = merge_edge(existing_edge, partial)
            edges = {**state.edges, edge_id: merged_edge}
            return CommandResult(
                GraphState(state.items, edges),
                [MemoryLifecycleEvent(type="edge.updated", edge=merged_edge, cause_type="edge.update")],
            )

        case EdgeRetract(edge_id=edge_id):
            existing_edge = state.edges.get(edge_id)
            if existing_edge is None:
                raise EdgeNotFoundError(edge_id)
            edges = dict(state.edges)
            del edges[edge_id]
            return CommandResult(
                GraphState(state.items, edges),
                [MemoryLifecycleEvent(type="edge.retracted", edge=existing_edge, cause_type="edge.retract")],
            )

        case _:  # pragma: no cover - defensive
            raise TypeError(f"Unknown memory command: {command!r}")
