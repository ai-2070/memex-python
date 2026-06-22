"""Port of tests/stats.test.ts."""

from __future__ import annotations

from typing import Any

from memex import Edge, GraphState, MemoryItem, create_graph_state, get_stats


def make_item(id: str, **overrides: Any) -> MemoryItem:
    base: dict[str, Any] = {
        "id": id, "scope": "test", "kind": "observation", "content": {},
        "author": "agent:a", "source_kind": "observed", "authority": 0.5,
    }
    base.update(overrides)
    return MemoryItem(**base)


def build_state() -> GraphState:
    items = [
        make_item("m1", kind="observation", source_kind="observed", author="agent:a", scope="project:x"),
        make_item("m2", kind="observation", source_kind="observed", author="agent:a", scope="project:x"),
        make_item("m3", kind="assertion", source_kind="user_explicit", author="user:laz", scope="project:x"),
        make_item("m4", kind="hypothesis", source_kind="agent_inferred", author="agent:b", scope="project:y", parents=["m1"]),
        make_item("m5", kind="derivation", source_kind="derived_deterministic", author="system:rule", scope="project:y", parents=["m2", "m3"]),
    ]
    edges = [
        Edge(edge_id="e1", from_="m1", to="m2", kind="SUPPORTS", author="system:rule", source_kind="derived_deterministic", authority=0.8, active=True),
        Edge(edge_id="e2", from_="m3", to="m4", kind="CONTRADICTS", author="system:detector", source_kind="derived_deterministic", authority=1, active=True),
        Edge(edge_id="e3", from_="m1", to="m3", kind="ABOUT", author="agent:a", source_kind="agent_inferred", authority=0.5, active=False),
    ]
    return GraphState(items={i.id: i for i in items}, edges={e.edge_id: e for e in edges})


def test_item_totals() -> None:
    stats = get_stats(build_state())
    assert stats.items.total == 5
    assert stats.items.root == 3
    assert stats.items.with_parents == 2


def test_by_kind() -> None:
    stats = get_stats(build_state())
    assert stats.items.by_kind == {"observation": 2, "assertion": 1, "hypothesis": 1, "derivation": 1}


def test_by_source_kind() -> None:
    stats = get_stats(build_state())
    assert stats.items.by_source_kind == {"observed": 2, "user_explicit": 1, "agent_inferred": 1, "derived_deterministic": 1}


def test_by_author() -> None:
    stats = get_stats(build_state())
    assert stats.items.by_author == {"agent:a": 2, "user:laz": 1, "agent:b": 1, "system:rule": 1}


def test_by_scope() -> None:
    stats = get_stats(build_state())
    assert stats.items.by_scope == {"project:x": 3, "project:y": 2}


def test_edge_totals() -> None:
    stats = get_stats(build_state())
    assert stats.edges.total == 3
    assert stats.edges.active == 2


def test_edges_by_kind() -> None:
    stats = get_stats(build_state())
    assert stats.edges.by_kind == {"SUPPORTS": 1, "CONTRADICTS": 1, "ABOUT": 1}


def test_empty_state() -> None:
    stats = get_stats(create_graph_state())
    assert stats.items.total == 0
    assert stats.items.root == 0
    assert stats.items.with_parents == 0
    assert stats.items.by_kind == {}
    assert stats.edges.total == 0
    assert stats.edges.active == 0
