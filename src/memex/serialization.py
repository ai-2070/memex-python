"""Serialize / restore a GraphState.

The on-disk shape matches the TS library — ``{"items": [[id, item], ...],
"edges": [[id, edge], ...]}`` with unset optionals omitted and edge ``from``
emitted under its alias — so a Python event store stays wire-compatible with a
TypeScript one.
"""

from __future__ import annotations

import json
from typing import Any

from .graph import GraphState
from .models import Edge, MemoryItem

__all__ = ["SerializedGraphState", "to_json", "from_json", "stringify", "parse"]

SerializedGraphState = dict[str, list[list[Any]]]


def _dump(model: MemoryItem | Edge) -> dict[str, Any]:
    return model.model_dump(by_alias=True, exclude_none=True)


def to_json(state: GraphState) -> SerializedGraphState:
    return {
        "items": [[id_, _dump(item)] for id_, item in state.items.items()],
        "edges": [[id_, _dump(edge)] for id_, edge in state.edges.items()],
    }


def from_json(data: SerializedGraphState) -> GraphState:
    # Tolerate a missing items/edges key (mirrors JS `new Map(undefined)`).
    items = {id_: MemoryItem.model_validate(d) for id_, d in data.get("items") or []}
    edges = {id_: Edge.model_validate(d) for id_, d in data.get("edges") or []}
    return GraphState(items=items, edges=edges)


def stringify(state: GraphState, pretty: bool = False) -> str:
    if pretty:
        return json.dumps(to_json(state), indent=2)
    return json.dumps(to_json(state), separators=(",", ":"))


def parse(json_str: str) -> GraphState:
    return from_json(json.loads(json_str))
