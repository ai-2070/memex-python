"""Port of tests/integrity.test.ts."""

from __future__ import annotations

import json
from typing import Any

import pytest

from memex import (
    Edge,
    GraphState,
    MemoryItem,
    cascade_retract,
    get_alias_group,
    get_aliases,
    get_contradictions,
    get_dependents,
    get_items_by_budget,
    get_stale_items,
    mark_alias,
    mark_contradiction,
    resolve_contradiction,
)


def make_item(id: str, **overrides: Any) -> MemoryItem:
    base: dict[str, Any] = {
        "id": id, "scope": "test", "kind": "observation", "content": {},
        "author": "user:laz", "source_kind": "observed", "authority": 0.8,
    }
    base.update(overrides)
    return MemoryItem(**base)


def state_with(items: list[MemoryItem], edges: list[Edge] | None = None) -> GraphState:
    return GraphState(items={i.id: i for i in items}, edges={e.edge_id: e for e in (edges or [])})


# === 1. Contradictions =====================================================


def test_mark_contradiction_creates_edge() -> None:
    state = state_with([make_item("m1"), make_item("m2")])
    nxt, events = mark_contradiction(state, "m1", "m2", "system:detector")
    edges = list(nxt.edges.values())
    assert len(edges) == 1
    assert edges[0].kind == "CONTRADICTS"
    assert edges[0].from_ == "m1"
    assert edges[0].to == "m2"
    assert events[0].type == "edge.created"


def test_get_contradictions_finds_pairs() -> None:
    state = state_with([make_item("m1"), make_item("m2")])
    nxt, _ = mark_contradiction(state, "m1", "m2", "system:detector")
    contradictions = get_contradictions(nxt)
    assert len(contradictions) == 1
    assert contradictions[0].a.id == "m1"
    assert contradictions[0].b.id == "m2"
    assert contradictions[0].edge is not None


def test_get_contradictions_empty() -> None:
    state = state_with([make_item("m1"), make_item("m2")])
    assert len(get_contradictions(state)) == 0


def test_resolve_contradiction_supersedes_and_lowers_authority() -> None:
    state = state_with([make_item("m1", authority=0.9), make_item("m2", authority=0.7)])
    marked, _ = mark_contradiction(state, "m1", "m2", "system:detector")
    resolved, events = resolve_contradiction(marked, "m1", "m2", "system:resolver", "m1 has more evidence")

    contradicts = [e for e in resolved.edges.values() if e.kind == "CONTRADICTS" and e.active]
    assert len(contradicts) == 0

    supersedes = [e for e in resolved.edges.values() if e.kind == "SUPERSEDES"]
    assert len(supersedes) == 1
    assert supersedes[0].from_ == "m1"
    assert supersedes[0].to == "m2"

    assert resolved.items["m2"].authority == pytest.approx(0.07)
    assert len(events) >= 3


# === 2. Stale detection & cascade ==========================================


def test_get_stale_items() -> None:
    state = state_with([make_item("m2", parents=["m1"]), make_item("m3")])
    stale = get_stale_items(state)
    assert len(stale) == 1
    assert stale[0].item.id == "m2"
    assert stale[0].missing_parents == ["m1"]


def test_get_stale_items_empty() -> None:
    state = state_with([make_item("m1"), make_item("m2", parents=["m1"])])
    assert len(get_stale_items(state)) == 0


def test_get_dependents_direct() -> None:
    state = state_with([
        make_item("m1"), make_item("m2", parents=["m1"]),
        make_item("m3", parents=["m1"]), make_item("m4"),
    ])
    deps = get_dependents(state, "m1")
    assert sorted(d.id for d in deps) == ["m2", "m3"]


def test_get_dependents_transitive() -> None:
    state = state_with([
        make_item("m1"), make_item("m2", parents=["m1"]),
        make_item("m3", parents=["m2"]), make_item("m4", parents=["m3"]),
    ])
    deps = get_dependents(state, "m1", True)
    assert sorted(d.id for d in deps) == ["m2", "m3", "m4"]


def test_cascade_retract_transitive() -> None:
    state = state_with([
        make_item("m1"), make_item("m2", parents=["m1"]),
        make_item("m3", parents=["m2"]), make_item("m4"),
    ])
    nxt, _events, retracted = cascade_retract(state, "m1", "system:cleanup", "invalid source")
    assert "m1" not in nxt.items
    assert "m2" not in nxt.items
    assert "m3" not in nxt.items
    assert "m4" in nxt.items
    assert sorted(retracted) == ["m1", "m2", "m3"]


def test_cascade_retract_diamond() -> None:
    state = state_with([
        make_item("m1"), make_item("m2", parents=["m1"]),
        make_item("m3", parents=["m1"]), make_item("m4", parents=["m2", "m3"]),
    ])
    nxt, _events, retracted = cascade_retract(state, "m1", "system:cleanup")
    assert len(nxt.items) == 0
    assert sorted(retracted) == ["m1", "m2", "m3", "m4"]


# === 3. Aliasing ===========================================================


def test_mark_alias_bidirectional() -> None:
    state = state_with([make_item("m1"), make_item("m2")])
    nxt, events = mark_alias(state, "m1", "m2", "system:dedup")
    alias_edges = [e for e in nxt.edges.values() if e.kind == "ALIAS"]
    assert len(alias_edges) == 2
    assert any(e.from_ == "m1" and e.to == "m2" for e in alias_edges)
    assert any(e.from_ == "m2" and e.to == "m1" for e in alias_edges)
    assert len(events) == 2


def test_get_aliases_direct() -> None:
    state = state_with([make_item("m1"), make_item("m2"), make_item("m3")])
    nxt, _ = mark_alias(state, "m1", "m2", "system:dedup")
    aliases = get_aliases(nxt, "m1")
    assert len(aliases) == 1 and aliases[0].id == "m2"


def test_get_aliases_empty() -> None:
    state = state_with([make_item("m1")])
    assert len(get_aliases(state, "m1")) == 0


def test_get_alias_group_transitive() -> None:
    state = state_with([make_item("m1"), make_item("m2"), make_item("m3")])
    nxt, _ = mark_alias(state, "m1", "m2", "system:dedup")
    nxt, _ = mark_alias(nxt, "m2", "m3", "system:dedup")
    group = get_alias_group(nxt, "m1")
    assert sorted(i.id for i in group) == ["m1", "m2", "m3"]


def test_get_alias_group_any_member() -> None:
    state = state_with([make_item("m1"), make_item("m2"), make_item("m3")])
    nxt, _ = mark_alias(state, "m1", "m2", "system:dedup")
    nxt, _ = mark_alias(nxt, "m2", "m3", "system:dedup")
    from_m3 = get_alias_group(nxt, "m3")
    assert sorted(i.id for i in from_m3) == ["m1", "m2", "m3"]


# === 4. getItemsByBudget ===================================================


def test_budget_packs_highest_scoring() -> None:
    state = state_with([
        make_item("m1", authority=0.9, importance=0.8, content={"text": "short"}),
        make_item("m2", authority=0.3, importance=0.2, content={"text": "short"}),
        make_item("m3", authority=0.7, importance=0.6, content={"text": "short"}),
    ])
    result = get_items_by_budget(state, budget=20, cost_fn=lambda i: 10, weights={"authority": 1})
    assert len(result) == 2
    assert result[0].item.id == "m1"
    assert result[1].item.id == "m3"


def test_budget_variable_cost() -> None:
    state = state_with([
        make_item("m1", authority=0.9, content={"text": "a" * 100}),
        make_item("m2", authority=0.8, content={"text": "b" * 10}),
        make_item("m3", authority=0.7, content={"text": "c" * 10}),
    ])
    result = get_items_by_budget(
        state, budget=50,
        cost_fn=lambda item: len(json.dumps(item.content, separators=(",", ":"))),
        weights={"authority": 1},
    )
    assert [r.item.id for r in result] == ["m2", "m3"]


def test_budget_zero() -> None:
    state = state_with([make_item("m1")])
    assert len(get_items_by_budget(state, budget=0, cost_fn=lambda i: 1, weights={"authority": 1})) == 0


def test_budget_applies_filter() -> None:
    state = state_with([
        make_item("m1", scope="a", authority=0.9),
        make_item("m2", scope="b", authority=0.8),
        make_item("m3", scope="a", authority=0.7),
    ])
    result = get_items_by_budget(state, budget=100, cost_fn=lambda i: 1, weights={"authority": 1}, filter={"scope": "a"})
    assert len(result) == 2 and all(r.item.scope == "a" for r in result)


def test_budget_weighted_ranking() -> None:
    state = state_with([
        make_item("m1", authority=0.9, importance=0.1),
        make_item("m2", authority=0.3, importance=0.9),
    ])
    result = get_items_by_budget(state, budget=100, cost_fn=lambda i: 1, weights={"authority": 0.1, "importance": 0.9})
    assert result[0].item.id == "m2"


def test_budget_skips_expensive() -> None:
    state = state_with([
        make_item("m1", authority=0.9), make_item("m2", authority=0.5), make_item("m3", authority=0.3),
    ])
    result = get_items_by_budget(
        state, budget=5,
        cost_fn=lambda item: 100 if item.id == "m1" else 2,
        weights={"authority": 1},
    )
    assert len(result) == 2
    assert result[0].item.id == "m2"
    assert result[1].item.id == "m3"


def test_budget_zero_cost() -> None:
    state = state_with([make_item("m1", authority=0.9)])
    assert len(get_items_by_budget(state, budget=100, cost_fn=lambda i: 0, weights={"authority": 1})) == 1


def test_budget_negative_cost_raises() -> None:
    state = state_with([make_item("m1", authority=0.9)])
    with pytest.raises(ValueError):
        get_items_by_budget(state, budget=100, cost_fn=lambda i: -1, weights={"authority": 1})


def test_budget_nan_cost_raises() -> None:
    state = state_with([make_item("m1", authority=0.9)])
    with pytest.raises(ValueError):
        get_items_by_budget(state, budget=100, cost_fn=lambda i: float("nan"), weights={"authority": 1})
