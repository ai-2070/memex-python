"""Port of tests/bugfix-bulk-retract-cascade.test.ts."""

from __future__ import annotations

from typing import Any

from memex import Edge, MemoryItem, apply_command, apply_many, create_graph_state


def make_item(id: str, **overrides: Any) -> MemoryItem:
    base: dict[str, Any] = {
        "id": id, "scope": "test", "kind": "observation", "content": {},
        "author": "agent:a", "source_kind": "observed", "authority": 0.5,
    }
    base.update(overrides)
    return MemoryItem(**base)


def make_edge(id: str, frm: str, to: str) -> Edge:
    return Edge(edge_id=id, from_=frm, to=to, kind="SUPPORTS", author="system:rule",
               source_kind="derived_deterministic", authority=0.8, active=True)


def build_state(items: list[MemoryItem], edges: list[Edge] | None = None):
    state = create_graph_state()
    for item in items:
        state = apply_command(state, {"type": "memory.create", "item": item}).state
    for edge in edges or []:
        state = apply_command(state, {"type": "edge.create", "edge": edge}).state
    return state


def test_removes_edges_where_item_is_from() -> None:
    state = build_state([make_item("m1"), make_item("m2")], [make_edge("e1", "m1", "m2")])
    nxt, _ = apply_many(state, {"ids": ["m1"]}, lambda item: None, "system:cleanup")
    assert "m1" not in nxt.items
    assert "e1" not in nxt.edges
    assert len(nxt.edges) == 0


def test_removes_edges_where_item_is_to() -> None:
    state = build_state([make_item("m1"), make_item("m2")], [make_edge("e1", "m2", "m1")])
    nxt, _ = apply_many(state, {"ids": ["m1"]}, lambda item: None, "system:cleanup")
    assert "e1" not in nxt.edges


def test_emits_edge_retracted_per_edge() -> None:
    state = build_state([make_item("m1"), make_item("m2"), make_item("m3")], [make_edge("e1", "m1", "m2"), make_edge("e2", "m3", "m1")])
    _, events = apply_many(state, {"ids": ["m1"]}, lambda item: None, "system:cleanup")
    types = [e.type for e in events]
    assert "memory.retracted" in types
    assert len([t for t in types if t == "edge.retracted"]) == 2


def test_leaves_unrelated_edges() -> None:
    state = build_state([make_item("m1"), make_item("m2"), make_item("m3")], [make_edge("e1", "m2", "m3")])
    nxt, _ = apply_many(state, {"ids": ["m1"]}, lambda item: None, "system:cleanup")
    assert "e1" in nxt.edges


def test_shared_edge_cleaned_once() -> None:
    state = build_state([make_item("m1"), make_item("m2")], [make_edge("e1", "m1", "m2")])
    nxt, events = apply_many(state, {}, lambda item: None, "system:cleanup")
    assert len(nxt.items) == 0
    assert len(nxt.edges) == 0
    assert len([e for e in events if e.type == "edge.retracted"]) == 1


def test_per_item_event_interleaving() -> None:
    state = build_state(
        [make_item("m1"), make_item("m2"), make_item("m3"), make_item("m4")],
        [make_edge("e1", "m1", "m2"), make_edge("e2", "m3", "m4")],
    )
    _, events = apply_many(state, {"ids": ["m1", "m3"]}, lambda item: None, "system:cleanup")
    assert len(events) == 4
    assert events[0].type == "memory.retracted" and events[0].item.id == "m1"
    assert events[1].type == "edge.retracted" and events[1].edge.edge_id == "e1"
    assert events[2].type == "memory.retracted" and events[2].item.id == "m3"
    assert events[3].type == "edge.retracted" and events[3].edge.edge_id == "e2"


def test_no_edge_clone_without_retractions() -> None:
    state = build_state([make_item("m1")], [])
    nxt, _ = apply_many(state, {"ids": ["m1"]}, lambda item: {"authority": 0.9}, "system:update")
    assert nxt.edges is state.edges
