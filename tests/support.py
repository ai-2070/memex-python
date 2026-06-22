"""Shared test builders mirroring the ``makeItem`` / ``makeEdge`` / ``stateWith``
fixtures used across the TypeScript test suite."""

from __future__ import annotations

from typing import Any

from memex import Edge, GraphState, MemoryItem


def make_item(**overrides: Any) -> MemoryItem:
    base: dict[str, Any] = {
        "id": "m1",
        "scope": "test",
        "kind": "observation",
        "content": {"key": "value", "nested": 1},
        "author": "user:laz",
        "source_kind": "observed",
        "authority": 0.9,
    }
    base.update(overrides)
    return MemoryItem(**base)


def make_edge(**overrides: Any) -> Edge:
    base: dict[str, Any] = {
        "edge_id": "e1",
        "from_": "m1",
        "to": "m2",
        "kind": "SUPPORTS",
        "author": "system:rule",
        "source_kind": "derived_deterministic",
        "authority": 0.8,
        "active": True,
    }
    base.update(overrides)
    return Edge(**base)


def state_with(
    items: list[MemoryItem] | tuple[MemoryItem, ...] = (),
    edges: list[Edge] | tuple[Edge, ...] = (),
) -> GraphState:
    return GraphState(
        items={i.id: i for i in items},
        edges={e.edge_id: e for e in edges},
    )
