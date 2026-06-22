"""Equivalence guards for the performance optimizations.

Each optimization replaced an O(n^2) loop (repeated full scans / per-command
re-clones) with an index-driven single pass. Behavior — result *and* ordering,
and for ``cascade_retract`` the full emitted event sequence — must stay
byte-identical to the straightforward implementation. Each test below pins the
optimized function against an inline reference that mirrors the original code.
"""

from __future__ import annotations

from functools import cmp_to_key
from typing import Any

from memex import (
    Edge,
    GraphState,
    MemoryItem,
    apply_command,
    cascade_retract,
    get_alias_group,
    get_dependents,
    get_items,
)
from memex.query import build_children_index, get_children, get_edges, get_sort_value


def make_item(id: str, **overrides: Any) -> MemoryItem:
    base: dict[str, Any] = {
        "id": id, "scope": "test", "kind": "observation", "content": {},
        "author": "user:laz", "source_kind": "observed", "authority": 0.8,
    }
    base.update(overrides)
    return MemoryItem(**base)


def make_edge(edge_id: str, from_: str, to: str, kind: str = "SUPPORTS") -> Edge:
    return Edge(
        edge_id=edge_id, from_=from_, to=to, kind=kind,
        author="system:rule", source_kind="derived_deterministic",
        authority=0.8, active=True,
    )


def state_with(items: list[MemoryItem], edges: list[Edge] | None = None) -> GraphState:
    return GraphState(
        items={i.id: i for i in items},
        edges={e.edge_id: e for e in (edges or [])},
    )


def _event_key(e: Any) -> tuple[str, str]:
    return (e.type, e.item.id if e.item is not None else e.edge.edge_id)


# ---------------------------------------------------------------------------
# Reference (pre-optimization) implementations
# ---------------------------------------------------------------------------


def _naive_cascade(state: GraphState, item_id: str, author: str) -> tuple[GraphState, list[Any], list[str]]:
    """The original per-command cascade: one apply_command per retraction."""
    visited: set[str] = {item_id}
    order: list[str] = []
    stack = [(c.id, "enter") for c in get_children(state, item_id)]
    while stack:
        fid, phase = stack.pop()
        if phase == "exit":
            order.append(fid)
            continue
        if fid in visited:
            continue
        visited.add(fid)
        stack.append((fid, "exit"))
        for c in get_children(state, fid):
            if c.id not in visited:
                stack.append((c.id, "enter"))

    current = state
    events: list[Any] = []
    retracted: list[str] = []
    for dep in order:
        if dep not in current.items:
            continue
        r = apply_command(current, {"type": "memory.retract", "item_id": dep, "author": author})
        current = r.state
        events.extend(r.events)
        retracted.append(dep)
    if item_id in current.items:
        r = apply_command(current, {"type": "memory.retract", "item_id": item_id, "author": author})
        current = r.state
        events.extend(r.events)
        retracted.append(item_id)
    return current, events, retracted


def _naive_alias_group(state: GraphState, item_id: str) -> list[MemoryItem]:
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
        for edge in get_edges(state, {"from": node_id, "kind": "ALIAS", "active_only": True}):
            queue.append(edge.to)
    return result


def _naive_dependents(state: GraphState, item_id: str) -> list[MemoryItem]:
    visited: set[str] = set()
    result: list[MemoryItem] = []
    queue = list(get_children(state, item_id))
    while queue:
        item = queue.pop()
        if item.id in visited:
            continue
        visited.add(item.id)
        result.append(item)
        queue.extend(get_children(state, item.id))
    return result


# ---------------------------------------------------------------------------
# cascade_retract — full state + event-sequence equivalence
# ---------------------------------------------------------------------------


def _diamond_state_with_edges() -> GraphState:
    # a -> {b, c} -> d (diamond), plus survivor s. Edges exercise: an edge
    # between two doomed items, an edge from a doomed item to the survivor,
    # a self-edge on the root, and an untouched survivor self-edge.
    items = [
        make_item("a"),
        make_item("b", parents=["a"]),
        make_item("c", parents=["a"]),
        make_item("d", parents=["b", "c"]),
        make_item("s"),
    ]
    edges = [
        make_edge("e1", "a", "b"),
        make_edge("e2", "b", "c", kind="CONTRADICTS"),
        make_edge("e3", "d", "s", kind="ABOUT"),
        make_edge("e4", "a", "a", kind="ALIAS"),   # self-edge on doomed root
        make_edge("e5", "s", "s", kind="ABOUT"),   # survivor self-edge, never touched
    ]
    return state_with(items, edges)


def test_cascade_retract_matches_naive_oracle_with_edges() -> None:
    state = _diamond_state_with_edges()

    new_state, new_events, new_retracted = cascade_retract(state, "a", "system:cleanup")
    ref_state, ref_events, ref_retracted = _naive_cascade(state, "a", "system:cleanup")

    # Identical retraction order, surviving items, surviving edges...
    assert new_retracted == ref_retracted
    assert list(new_state.items.keys()) == list(ref_state.items.keys())
    assert list(new_state.edges.keys()) == list(ref_state.edges.keys())
    # ...and an identical emitted event sequence (memory + edge events, in order).
    assert [_event_key(e) for e in new_events] == [_event_key(e) for e in ref_events]

    # Concretely: everything but the survivor is gone, and the only surviving
    # edge is the survivor's self-edge.
    assert set(new_state.items.keys()) == {"s"}
    assert set(new_state.edges.keys()) == {"e5"}


def test_cascade_retract_shared_edge_emitted_once() -> None:
    # e2 connects b and c, both retracted. The naive path removes it when the
    # first endpoint is retracted; the batched path must do the same (emit once).
    state = _diamond_state_with_edges()
    _, events, _ = cascade_retract(state, "a", "system:cleanup")
    edge_events = [e for e in events if e.type == "edge.retracted"]
    edge_ids = [e.edge.edge_id for e in edge_events]
    assert sorted(edge_ids) == ["e1", "e2", "e3", "e4"]
    assert len(edge_ids) == len(set(edge_ids))  # no double-emission


def test_cascade_retract_no_edges_keeps_edges_identity() -> None:
    # With no edges the optimized path must not needlessly clone the edges dict.
    state = state_with([make_item("a"), make_item("b", parents=["a"])])
    new_state, events, retracted = cascade_retract(state, "a", "system:cleanup")
    assert retracted == ["b", "a"]
    assert new_state.edges is state.edges  # untouched, same object
    assert all(e.type == "memory.retracted" for e in events)


def test_cascade_retract_orphan_root_not_retracted() -> None:
    # item_id absent but referenced as a (stale) parent: dependents are still
    # cascaded, the missing root is silently skipped — same as the naive path.
    state = state_with([make_item("child", parents=["ghost"])])
    new_state, _events, retracted = cascade_retract(state, "ghost", "system:cleanup")
    ref_state, _, ref_retracted = _naive_cascade(state, "ghost", "system:cleanup")
    assert retracted == ref_retracted == ["child"]
    assert list(new_state.items.keys()) == list(ref_state.items.keys())


# ---------------------------------------------------------------------------
# build_children_index
# ---------------------------------------------------------------------------


def test_build_children_index_matches_get_children() -> None:
    state = state_with([
        make_item("p1"),
        make_item("p2"),
        make_item("c1", parents=["p1"]),
        make_item("c2", parents=["p1", "p2"]),
        make_item("c3", parents=["p2"]),
    ])
    index = build_children_index(state)
    for pid in ("p1", "p2", "c1", "missing"):
        assert index.get(pid, []) == get_children(state, pid)


def test_build_children_index_dedupes_duplicate_parents() -> None:
    # A child listing the same parent twice still appears once (matches the
    # boolean membership test in get_children).
    state = state_with([make_item("p"), make_item("c", parents=["p", "p"])])
    index = build_children_index(state)
    assert [i.id for i in index["p"]] == ["c"]
    assert index["p"] == get_children(state, "p")


# ---------------------------------------------------------------------------
# get_dependents (transitive) + get_alias_group ordering
# ---------------------------------------------------------------------------


def test_get_dependents_transitive_matches_naive() -> None:
    state = state_with([
        make_item("r"),
        make_item("a", parents=["r"]),
        make_item("b", parents=["r"]),
        make_item("c", parents=["a", "b"]),
        make_item("d", parents=["c"]),
    ])
    assert get_dependents(state, "r", True) == _naive_dependents(state, "r")


def test_get_alias_group_order_matches_naive() -> None:
    # Branching alias network; BFS visitation order must match the reference.
    items = [make_item(x) for x in ("a", "b", "c", "d", "e")]
    edges = [
        make_edge("e1", "a", "b", kind="ALIAS"),
        make_edge("e2", "a", "c", kind="ALIAS"),
        make_edge("e3", "b", "d", kind="ALIAS"),
        make_edge("e4", "c", "e", kind="ALIAS"),
        make_edge("e5", "x", "y", kind="ALIAS"),  # unrelated component
    ]
    state = state_with(items, edges)
    assert [i.id for i in get_alias_group(state, "a")] == [i.id for i in _naive_alias_group(state, "a")]


# ---------------------------------------------------------------------------
# _multi_sort (tuple key) vs the original cmp_to_key comparator
# ---------------------------------------------------------------------------


def _cmp_reference(items: list[MemoryItem], sorts: list[dict[str, str]]) -> list[MemoryItem]:
    def _cmp(a: MemoryItem, b: MemoryItem) -> int:
        for s in sorts:
            va = get_sort_value(a, s["field"])
            vb = get_sort_value(b, s["field"])
            if va < vb:
                return -1 if s["order"] == "asc" else 1
            if va > vb:
                return 1 if s["order"] == "asc" else -1
        return 0

    return sorted(items, key=cmp_to_key(_cmp))


def test_multi_sort_matches_cmp_reference_with_ties() -> None:
    # Coarse score values force many ties so the stable insertion-order
    # tie-break is exercised; created_at drives `recency`.
    items: list[MemoryItem] = []
    for k in range(60):
        items.append(make_item(
            f"m{k:02d}",
            authority=round((k * 7 % 5) / 4, 2),
            conviction=round((k * 3 % 4) / 3, 2),
            importance=round((k % 3) / 2, 2),
            created_at=1_700_000_000_000 + (k % 6),
        ))
    state = state_with(items)

    sort_specs = [
        [{"field": "authority", "order": "desc"}],
        [{"field": "importance", "order": "asc"}, {"field": "authority", "order": "desc"}],
        [{"field": "authority", "order": "desc"}, {"field": "recency", "order": "desc"}],
        [
            {"field": "authority", "order": "desc"},
            {"field": "conviction", "order": "asc"},
            {"field": "importance", "order": "desc"},
            {"field": "recency", "order": "asc"},
        ],
    ]
    for spec in sort_specs:
        got = get_items(state, None, {"sort": spec})
        expected = _cmp_reference(items, spec)
        assert [i.id for i in got] == [i.id for i in expected], spec
