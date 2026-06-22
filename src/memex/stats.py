"""Aggregate counts over a GraphState."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import NamedTuple, TypeVar

from .graph import GraphState

__all__ = ["ItemStats", "EdgeStats", "GraphStats", "get_stats"]

T = TypeVar("T")


class ItemStats(NamedTuple):
    total: int
    by_kind: dict[str, int]
    by_source_kind: dict[str, int]
    by_author: dict[str, int]
    by_scope: dict[str, int]
    with_parents: int
    root: int


class EdgeStats(NamedTuple):
    total: int
    active: int
    by_kind: dict[str, int]


class GraphStats(NamedTuple):
    items: ItemStats
    edges: EdgeStats


def _count_by(values: Iterable[T], key_fn: Callable[[T], str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for v in values:
        key = key_fn(v)
        counts[key] = counts.get(key, 0) + 1
    return counts


def get_stats(state: GraphState) -> GraphStats:
    items = list(state.items.values())
    edges = list(state.edges.values())

    with_parents = 0
    root = 0
    for item in items:
        if item.parents:
            with_parents += 1
        else:
            root += 1

    return GraphStats(
        items=ItemStats(
            total=len(items),
            by_kind=_count_by(items, lambda i: i.kind),
            by_source_kind=_count_by(items, lambda i: i.source_kind),
            by_author=_count_by(items, lambda i: i.author),
            by_scope=_count_by(items, lambda i: i.scope),
            with_parents=with_parents,
            root=root,
        ),
        edges=EdgeStats(
            total=len(edges),
            active=sum(1 for e in edges if e.active),
            by_kind=_count_by(edges, lambda e: e.kind),
        ),
    )
