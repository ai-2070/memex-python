"""Contradiction & alias integrity, stale detection, cascade retraction, budget packing."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, NamedTuple

from .commands import EdgeCreate
from .factories import create_edge
from .graph import GraphState
from .models import Edge, MemoryFilter, MemoryItem, MemoryLifecycleEvent, ScoredItem, ScoreWeights
from .query import build_children_index, get_children, get_edges, get_scored_items
from .reducer import CommandResult, apply_command

__all__ = [
    "Contradiction",
    "StaleItem",
    "CascadeResult",
    "get_contradictions",
    "mark_contradiction",
    "resolve_contradiction",
    "get_stale_items",
    "get_dependents",
    "cascade_retract",
    "mark_alias",
    "get_aliases",
    "get_alias_group",
    "get_items_by_budget",
]


# ---------------------------------------------------------------------------
# 1. Contradiction detection & resolution
# ---------------------------------------------------------------------------


class Contradiction(NamedTuple):
    a: MemoryItem
    b: MemoryItem
    edge: Edge | None = None


def get_contradictions(state: GraphState) -> list[Contradiction]:
    contradict_edges = get_edges(state, {"kind": "CONTRADICTS", "active_only": True})
    results: list[Contradiction] = []
    for edge in contradict_edges:
        a = state.items.get(edge.from_)
        b = state.items.get(edge.to)
        if a is not None and b is not None:
            results.append(Contradiction(a=a, b=b, edge=edge))
    return results


def mark_contradiction(
    state: GraphState,
    item_id_a: str,
    item_id_b: str,
    author: str,
    meta: dict[str, Any] | None = None,
) -> CommandResult:
    # A self-CONTRADICTS edge is meaningful (an internally inconsistent item);
    # downstream annotation already skips self-edges, so it's safe to record.
    edge = create_edge(
        from_=item_id_a, to=item_id_b, kind="CONTRADICTS", author=author,
        source_kind="derived_deterministic", authority=1, meta=meta,
    )
    return apply_command(state, EdgeCreate(edge=edge))


def resolve_contradiction(
    state: GraphState,
    winner_id: str,
    loser_id: str,
    author: str,
    reason: str | None = None,
) -> CommandResult:
    current = state
    all_events: list[MemoryLifecycleEvent] = []

    to_retract: list[str] = []
    for edge in current.edges.values():
        if (
            edge.kind == "CONTRADICTS"
            and edge.active
            and (
                (edge.from_ == winner_id and edge.to == loser_id)
                or (edge.from_ == loser_id and edge.to == winner_id)
            )
        ):
            to_retract.append(edge.edge_id)

    for edge_id in to_retract:
        r = apply_command(
            current, {"type": "edge.retract", "edge_id": edge_id, "author": author, "reason": reason}
        )
        current = r.state
        all_events.extend(r.events)

    if not to_retract:
        # Stale/duplicate call — no-op rather than crash the fold.
        return CommandResult(current, all_events)

    supersedes = create_edge(
        from_=winner_id, to=loser_id, kind="SUPERSEDES", author=author,
        source_kind="derived_deterministic", authority=1,
        meta={"reason": reason} if reason else None,
    )
    r1 = apply_command(current, EdgeCreate(edge=supersedes))
    current = r1.state
    all_events.extend(r1.events)

    loser = current.items.get(loser_id)
    if loser is not None:
        r2 = apply_command(
            current,
            {"type": "memory.update", "item_id": loser_id,
             "partial": {"authority": loser.authority * 0.1}, "author": author, "reason": reason},
        )
        current = r2.state
        all_events.extend(r2.events)

    return CommandResult(current, all_events)


# ---------------------------------------------------------------------------
# 2. Stale detection & cascade
# ---------------------------------------------------------------------------


class StaleItem(NamedTuple):
    item: MemoryItem
    missing_parents: list[str]


def get_stale_items(state: GraphState) -> list[StaleItem]:
    results: list[StaleItem] = []
    for item in state.items.values():
        if not item.parents:
            continue
        missing = [pid for pid in item.parents if pid not in state.items]
        if missing:
            results.append(StaleItem(item=item, missing_parents=missing))
    return results


def get_dependents(state: GraphState, item_id: str, transitive: bool = False) -> list[MemoryItem]:
    direct = get_children(state, item_id)
    if not transitive:
        return direct

    # Walk the whole dependent subtree off one children index instead of
    # re-scanning every item per node (O(N + dependents), not O(dependents x N)).
    children_index = build_children_index(state)
    visited: set[str] = set()
    result: list[MemoryItem] = []
    queue = list(direct)
    while queue:
        item = queue.pop()
        if item.id in visited:
            continue
        visited.add(item.id)
        result.append(item)
        queue.extend(children_index.get(item.id, []))
    return result


class CascadeResult(NamedTuple):
    state: GraphState
    events: list[MemoryLifecycleEvent]
    retracted: list[str]


def cascade_retract(
    state: GraphState,
    item_id: str,
    author: str,
    reason: str | None = None,
) -> CascadeResult:
    """Retract an item and all transitive dependents in post-order (leaves first).

    Iterative post-order DFS over a one-shot children index: cycle-safe, DAG-safe
    (shared children), and does not consume the call stack on deep dependency
    chains. The root is pre-marked visited so a cycle pointing back to it is
    ignored — it's retracted last.

    The retraction itself is done in a single pass — the items dict is cloned
    once and a reverse edge index is built lazily on the first retract — instead
    of issuing one ``apply_command`` per dependent (each of which would re-clone
    the whole state and re-scan every edge, making a deep cascade quadratic).
    ``author``/``reason`` carry no information into the emitted events or state,
    so the events produced here are byte-identical to the per-command path.
    """
    children_index = build_children_index(state)

    visited: set[str] = {item_id}
    order: list[str] = []

    stack: list[tuple[str, str]] = [
        (child.id, "enter") for child in children_index.get(item_id, [])
    ]
    while stack:
        frame_id, phase = stack.pop()
        if phase == "exit":
            order.append(frame_id)
            continue
        if frame_id in visited:
            continue
        visited.add(frame_id)
        stack.append((frame_id, "exit"))  # processed after all children (post-order)
        for child in children_index.get(frame_id, []):
            if child.id not in visited:
                stack.append((child.id, "enter"))

    # Dependents first (post-order), then the root.
    items = dict(state.items)
    edges: dict[str, Edge] | None = None
    edges_by_endpoint: dict[str, list[str]] | None = None
    all_events: list[MemoryLifecycleEvent] = []
    retracted: list[str] = []

    for rid in (*order, item_id):
        existing = items.pop(rid, None)
        if existing is None:
            continue
        all_events.append(
            MemoryLifecycleEvent(type="memory.retracted", item=existing, cause_type="memory.retract")
        )
        retracted.append(rid)
        if state.edges:
            if edges is None:
                edges = dict(state.edges)
            if edges_by_endpoint is None:
                edges_by_endpoint = {}
                for edge_id, edge in state.edges.items():
                    edges_by_endpoint.setdefault(edge.from_, []).append(edge_id)
                    if edge.from_ != edge.to:
                        edges_by_endpoint.setdefault(edge.to, []).append(edge_id)
            incident_ids = edges_by_endpoint.get(rid)
            if incident_ids:
                for edge_id in incident_ids:
                    incident_edge = edges.get(edge_id)
                    if incident_edge is None:
                        continue  # already removed via its other endpoint
                    del edges[edge_id]
                    all_events.append(
                        MemoryLifecycleEvent(type="edge.retracted", edge=incident_edge, cause_type="memory.retract")
                    )

    if not retracted:
        return CascadeResult(state, all_events, retracted)

    return CascadeResult(
        GraphState(items, edges if edges is not None else state.edges),
        all_events,
        retracted,
    )


# ---------------------------------------------------------------------------
# 3. Identity / aliasing
# ---------------------------------------------------------------------------


def mark_alias(
    state: GraphState,
    item_id_a: str,
    item_id_b: str,
    author: str,
    meta: dict[str, Any] | None = None,
) -> CommandResult:
    if item_id_a == item_id_b:
        # Self-alias is redundant — no-op rather than throw.
        return CommandResult(state, [])

    current = state
    all_events: list[MemoryLifecycleEvent] = []

    e1 = create_edge(
        from_=item_id_a, to=item_id_b, kind="ALIAS", author=author,
        source_kind="derived_deterministic", authority=1, meta=meta,
    )
    r1 = apply_command(current, EdgeCreate(edge=e1))
    current = r1.state
    all_events.extend(r1.events)

    e2 = create_edge(
        from_=item_id_b, to=item_id_a, kind="ALIAS", author=author,
        source_kind="derived_deterministic", authority=1, meta=meta,
    )
    r2 = apply_command(current, EdgeCreate(edge=e2))
    current = r2.state
    all_events.extend(r2.events)

    return CommandResult(current, all_events)


def get_aliases(state: GraphState, item_id: str) -> list[MemoryItem]:
    alias_edges = get_edges(state, {"from": item_id, "kind": "ALIAS", "active_only": True})
    results: list[MemoryItem] = []
    for edge in alias_edges:
        item = state.items.get(edge.to)
        if item is not None:
            results.append(item)
    return results


def get_alias_group(state: GraphState, item_id: str) -> list[MemoryItem]:
    # Index active outbound ALIAS edges once (from_ -> [to]) so the BFS does not
    # re-scan every edge per node; per-source order matches get_edges (edge
    # insertion order), keeping the walk deterministic.
    alias_out: dict[str, list[str]] = {}
    for edge in state.edges.values():
        if edge.kind == "ALIAS" and edge.active:
            alias_out.setdefault(edge.from_, []).append(edge.to)

    visited: set[str] = set()
    result: list[MemoryItem] = []
    queue = [item_id]
    while queue:
        node_id = queue.pop()
        if node_id in visited:
            continue
        visited.add(node_id)
        item = state.items.get(node_id)
        if item is not None:
            result.append(item)
        for to in alias_out.get(node_id, []):
            queue.append(to)
    return result


# ---------------------------------------------------------------------------
# 4. Budget-aware retrieval
# ---------------------------------------------------------------------------


def get_items_by_budget(
    state: GraphState,
    *,
    budget: float,
    cost_fn: Callable[[MemoryItem], float],
    weights: ScoreWeights | dict[str, Any],
    filter: MemoryFilter | dict[str, Any] | None = None,
) -> list[ScoredItem]:
    """Retrieve the highest-scoring items that fit within a budget (greedy pack)."""
    scored = get_scored_items(state, weights, {"pre": filter})

    results: list[ScoredItem] = []
    remaining = budget
    for entry in scored:
        cost = cost_fn(entry.item)
        if cost < 0 or not math.isfinite(cost):
            raise ValueError(f"cost_fn must return a finite non-negative number, got {cost}")
        if cost <= remaining:
            results.append(entry)
            remaining -= cost
    return results
