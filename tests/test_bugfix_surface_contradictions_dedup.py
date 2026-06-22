"""Port of tests/bugfix-surface-contradictions-dedup.test.ts."""

from __future__ import annotations

from typing import Any

from memex import (
    GraphState,
    MemoryItem,
    ScoredItem,
    apply_command,
    create_graph_state,
    mark_contradiction,
    surface_contradictions,
)


def make_item(id: str, **overrides: Any) -> MemoryItem:
    base: dict[str, Any] = {
        "id": id, "scope": "test", "kind": "observation", "content": {},
        "author": "agent:a", "source_kind": "observed", "authority": 0.5,
    }
    base.update(overrides)
    return MemoryItem(**base)


def state_with(items: list[MemoryItem]) -> GraphState:
    state = create_graph_state()
    for item in items:
        state = apply_command(state, {"type": "memory.create", "item": item}).state
    return state


def to_scored(items: list[MemoryItem], scores: list[float]) -> list[ScoredItem]:
    return [ScoredItem(item=item, score=scores[i]) for i, item in enumerate(items)]


def test_bidirectional_single_entry() -> None:
    m1, m2 = make_item("m1"), make_item("m2")
    state = state_with([m1, m2])
    state = mark_contradiction(state, "m1", "m2", "system:detector-a").state
    state = mark_contradiction(state, "m2", "m1", "system:detector-b").state
    result = surface_contradictions(state, to_scored([m1, m2], [0.5, 0.5]))
    r1 = next(s for s in result if s.item.id == "m1")
    r2 = next(s for s in result if s.item.id == "m2")
    assert r1.contradicted_by is not None and len(r1.contradicted_by) == 1 and r1.contradicted_by[0].id == "m2"
    assert r2.contradicted_by is not None and len(r2.contradicted_by) == 1 and r2.contradicted_by[0].id == "m1"


def test_same_direction_two_edges_single_entry() -> None:
    m1, m2 = make_item("m1"), make_item("m2")
    state = state_with([m1, m2])
    state = mark_contradiction(state, "m1", "m2", "system:detector-a", {"reason": "a"}).state
    state = mark_contradiction(state, "m1", "m2", "system:detector-b", {"reason": "b"}).state
    result = surface_contradictions(state, to_scored([m1, m2], [0.5, 0.5]))
    r1 = next(s for s in result if s.item.id == "m1")
    r2 = next(s for s in result if s.item.id == "m2")
    assert r1.contradicted_by is not None and len(r1.contradicted_by) == 1
    assert r2.contradicted_by is not None and len(r2.contradicted_by) == 1


def test_no_self_annotation() -> None:
    m1 = make_item("m1")
    state = state_with([m1])
    state = apply_command(state, {
        "type": "edge.create",
        "edge": {"edge_id": "e-self", "from": "m1", "to": "m1", "kind": "CONTRADICTS",
                 "author": "system:detector", "source_kind": "derived_deterministic", "authority": 1, "active": True},
    }).state
    result = surface_contradictions(state, to_scored([m1], [0.5]))
    r1 = next(s for s in result if s.item.id == "m1")
    assert len(r1.contradicted_by or []) == 0


def test_normal_single_edge() -> None:
    m1, m2 = make_item("m1"), make_item("m2")
    state = state_with([m1, m2])
    state = mark_contradiction(state, "m1", "m2", "system:detector").state
    result = surface_contradictions(state, to_scored([m1, m2], [0.9, 0.5]))
    r1 = next(s for s in result if s.item.id == "m1")
    r2 = next(s for s in result if s.item.id == "m2")
    assert r1.contradicted_by is not None and len(r1.contradicted_by) == 1 and r1.contradicted_by[0].id == "m2"
    assert r2.contradicted_by is not None and len(r2.contradicted_by) == 1 and r2.contradicted_by[0].id == "m1"
