"""Graph state container.

``GraphState`` is a lightweight frozen dataclass holding plain dicts — NOT a
Pydantic model (D3). The reducer clones the relevant dict on every command
(``dict(state.items)`` mirrors the TS ``new Map(state.items)``); a validated
model here would re-validate every item per command and be unusably slow.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import Edge, MemoryItem

__all__ = ["GraphState", "create_graph_state", "clone_graph_state"]


@dataclass(frozen=True, slots=True)
class GraphState:
    items: dict[str, MemoryItem] = field(default_factory=dict)
    edges: dict[str, Edge] = field(default_factory=dict)


def create_graph_state() -> GraphState:
    return GraphState(items={}, edges={})


def clone_graph_state(state: GraphState) -> GraphState:
    """Shallow clone — new dicts, shared (immutable) item/edge instances."""
    return GraphState(items=dict(state.items), edges=dict(state.edges))
