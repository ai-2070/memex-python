"""Port of tests/bugfix-holes.test.ts."""

from __future__ import annotations

from typing import Any

import pytest

from memex import (
    Edge,
    GraphState,
    MemoryItem,
    ScoredItem,
    apply_command,
    apply_diversity,
    apply_intent_command,
    apply_many,
    apply_task_command,
    create_graph_state,
    create_intent,
    create_intent_state,
    create_task,
    create_task_state,
    decay_importance,
    export_slice,
    filter_contradictions,
    get_contradictions,
    get_edges,
    get_items_by_budget,
    get_related_items,
    import_slice,
    mark_contradiction,
    replay_from_envelopes,
    resolve_contradiction,
    smart_retrieve,
    surface_contradictions,
)

_counter = 0


def fake_uuid(n: int) -> str:
    ms = format(1700000000000 + n, "x").rjust(12, "0")
    return f"{ms[0:8]}-{ms[8:12]}-7000-8000-{'0' * 11}{n}"


def fake_id(ts_ms: int) -> str:
    global _counter
    _counter += 1
    hex_ts = format(ts_ms, "x").rjust(12, "0")
    pad = format(_counter, "x").rjust(20, "0")
    return "-".join([hex_ts[0:8], hex_ts[8:12], "7" + pad[0:3], "8" + pad[3:6], pad[6:18]])


def make_item(id: str, **overrides: Any) -> MemoryItem:
    base: dict[str, Any] = {
        "id": id, "scope": "test", "kind": "observation", "content": {"text": f"item {id}"},
        "author": "agent:test", "source_kind": "observed", "authority": 0.8,
    }
    base.update(overrides)
    return MemoryItem(**base)


def make_edge(edge_id: str, frm: str, to: str, kind: str = "SUPPORTS", **overrides: Any) -> Edge:
    base: dict[str, Any] = {
        "edge_id": edge_id, "from_": frm, "to": to, "kind": kind,
        "author": "agent:test", "source_kind": "derived_deterministic", "authority": 1, "active": True,
    }
    base.update(overrides)
    return Edge(**base)


def state_with(items: list[MemoryItem], edges: list[Edge] | None = None) -> GraphState:
    state = create_graph_state()
    for item in items:
        state = apply_command(state, {"type": "memory.create", "item": item}).state
    for edge in edges or []:
        state = apply_command(state, {"type": "edge.create", "edge": edge}).state
    return state


# === 1. intent/task update strips undefined (D5: undefined ≡ absent) ========


def test_intent_update_does_not_overwrite_with_undefined() -> None:
    state = create_intent_state()
    state = apply_intent_command(state, {"type": "intent.create", "intent": create_intent(id="i1", label="find target", description="locate the target entity", priority=0.9, owner="user:laz")}).state
    state = apply_intent_command(state, {"type": "intent.update", "intent_id": "i1", "partial": {"label": "renamed"}, "author": "user:laz"}).state
    updated = state.intents["i1"]
    assert updated.label == "renamed"
    assert updated.description == "locate the target entity"


def test_intent_update_does_not_overwrite_context() -> None:
    state = create_intent_state()
    state = apply_intent_command(state, {"type": "intent.create", "intent": create_intent(id="i2", label="test", priority=0.5, owner="user:laz", context={"key": "value"})}).state
    state = apply_intent_command(state, {"type": "intent.update", "intent_id": "i2", "partial": {}, "author": "user:laz"}).state
    assert state.intents["i2"].context == {"key": "value"}


def test_task_update_does_not_overwrite_with_undefined() -> None:
    state = create_task_state()
    state = apply_task_command(state, {"type": "task.create", "task": create_task(id="t1", intent_id="i1", action="search", label="search linkedin", priority=0.7, context={"query": "test"})}).state
    state = apply_task_command(state, {"type": "task.update", "task_id": "t1", "partial": {"action": "search_v2"}, "author": "agent:test"}).state
    updated = state.tasks["t1"]
    assert updated.action == "search_v2"
    assert updated.label == "search linkedin"
    assert updated.context == {"query": "test"}


# === 2. Edge re-id on import conflict ======================================


def test_edge_reid_on_conflict() -> None:
    edge_id = fake_uuid(1)
    state = state_with([make_item("m1"), make_item("m2")])
    state = apply_command(state, {"type": "edge.create", "edge": make_edge(edge_id, "m1", "m2", weight=0.5)}).state
    slice = {"memories": [], "edges": [make_edge(edge_id, "m1", "m2", weight=0.9)], "intents": [], "tasks": []}
    result = import_slice(state, create_intent_state(), create_task_state(), slice, skip_existing_ids=True, shallow_compare_existing=True, re_id_on_difference=True)
    assert edge_id in result.mem_state.edges
    assert len(result.report.created.edges) == 1
    new_edge_id = result.report.created.edges[0]
    assert new_edge_id != edge_id
    new_edge = result.mem_state.edges[new_edge_id]
    assert new_edge.weight == 0.9
    assert new_edge.from_ == "m1"
    assert new_edge.to == "m2"


def test_edge_conflict_without_reid() -> None:
    state = state_with([make_item("m1"), make_item("m2")])
    state = apply_command(state, {"type": "edge.create", "edge": make_edge("e1", "m1", "m2", weight=0.5)}).state
    slice = {"memories": [], "edges": [make_edge("e1", "m1", "m2", weight=0.9)], "intents": [], "tasks": []}
    result = import_slice(state, create_intent_state(), create_task_state(), slice, skip_existing_ids=True, shallow_compare_existing=True, re_id_on_difference=False)
    assert result.report.conflicts.edges == ["e1"]
    assert result.report.created.edges == []


def test_edge_skips_identical() -> None:
    state = state_with([make_item("m1"), make_item("m2")])
    edge = make_edge("e1", "m1", "m2")
    state = apply_command(state, {"type": "edge.create", "edge": edge}).state
    slice = {"memories": [], "edges": [edge], "intents": [], "tasks": []}
    result = import_slice(state, create_intent_state(), create_task_state(), slice, skip_existing_ids=True, shallow_compare_existing=True, re_id_on_difference=True)
    assert result.report.skipped.edges == ["e1"]
    assert result.report.created.edges == []
    assert result.report.conflicts.edges == []


# === 3. filterContradictions chained =======================================


def test_filter_contradictions_chained() -> None:
    state = state_with(
        [make_item("a", authority=0.9), make_item("b", authority=0.5), make_item("c", authority=0.7)],
        [make_edge("e1", "a", "b", "CONTRADICTS"), make_edge("e2", "b", "c", "CONTRADICTS")],
    )
    scored = [ScoredItem(item=state.items["a"], score=0.9), ScoredItem(item=state.items["c"], score=0.7), ScoredItem(item=state.items["b"], score=0.5)]
    ids = [s.item.id for s in filter_contradictions(state, scored)]
    assert "a" in ids
    assert "b" not in ids
    assert "c" in ids


# === 4. filterContradictions equal-score tiebreak ==========================


def test_tiebreak_lexicographic() -> None:
    state = state_with([make_item("aaa", authority=0.8), make_item("zzz", authority=0.8)], [make_edge("e1", "aaa", "zzz", "CONTRADICTS")])
    scored = [ScoredItem(item=state.items["aaa"], score=0.5), ScoredItem(item=state.items["zzz"], score=0.5)]
    ids = [s.item.id for s in filter_contradictions(state, scored)]
    assert "aaa" in ids
    assert "zzz" not in ids


def test_tiebreak_deterministic_regardless_of_order() -> None:
    state = state_with([make_item("aaa", authority=0.8), make_item("zzz", authority=0.8)], [make_edge("e1", "aaa", "zzz", "CONTRADICTS")])
    scored = [ScoredItem(item=state.items["zzz"], score=0.5), ScoredItem(item=state.items["aaa"], score=0.5)]
    ids = [s.item.id for s in filter_contradictions(state, scored)]
    assert "aaa" in ids
    assert "zzz" not in ids


# === 5. smartRetrieve surface ==============================================


def test_smart_retrieve_surface_keeps_both() -> None:
    state = state_with(
        [make_item("m1", authority=0.9), make_item("m2", authority=0.6), make_item("m3", authority=0.3)],
        [make_edge("e1", "m1", "m2", "CONTRADICTS")],
    )
    result = smart_retrieve(state, budget=1000, cost_fn=lambda i: 1, weights={"authority": 1}, contradictions="surface")
    ids = [s.item.id for s in result]
    assert "m1" in ids and "m2" in ids and "m3" in ids
    m1 = next(s for s in result if s.item.id == "m1")
    m2 = next(s for s in result if s.item.id == "m2")
    assert m1.contradicted_by is not None and "m2" in [i.id for i in m1.contradicted_by]
    assert m2.contradicted_by is not None and "m1" in [i.id for i in m2.contradicted_by]


def test_smart_retrieve_surface_removes_superseded() -> None:
    state = state_with([make_item("m1", authority=0.9), make_item("m2", authority=0.4)])
    state = mark_contradiction(state, "m1", "m2", "agent:test").state
    state = resolve_contradiction(state, "m1", "m2", "agent:test").state
    result = smart_retrieve(state, budget=1000, cost_fn=lambda i: 1, weights={"authority": 1}, contradictions="surface")
    ids = [s.item.id for s in result]
    assert "m1" in ids
    assert "m2" not in ids


# === 6. surfaceContradictions bidirectional ================================


def test_surface_annotates_both_sides() -> None:
    state = state_with(
        [make_item("a", authority=0.8), make_item("b", authority=0.6), make_item("c", authority=0.4)],
        [make_edge("e1", "a", "b", "CONTRADICTS"), make_edge("e2", "b", "c", "CONTRADICTS")],
    )
    scored = [ScoredItem(item=state.items["a"], score=0.8), ScoredItem(item=state.items["b"], score=0.6), ScoredItem(item=state.items["c"], score=0.4)]
    result = surface_contradictions(state, scored)
    a = next(s for s in result if s.item.id == "a")
    b = next(s for s in result if s.item.id == "b")
    c = next(s for s in result if s.item.id == "c")
    assert [i.id for i in a.contradicted_by] == ["b"]
    assert sorted(i.id for i in b.contradicted_by) == ["a", "c"]
    assert [i.id for i in c.contradicted_by] == ["b"]


def test_surface_does_not_mutate_input() -> None:
    state = state_with([make_item("a"), make_item("b")], [make_edge("e1", "a", "b", "CONTRADICTS")])
    scored = [ScoredItem(item=state.items["a"], score=0.8), ScoredItem(item=state.items["b"], score=0.6)]
    surface_contradictions(state, scored)
    assert scored[0].contradicted_by is None
    assert scored[1].contradicted_by is None


# === 7. getItemsByBudget zero-cost =========================================


def test_budget_includes_all_zero_cost() -> None:
    state = state_with([make_item("m1", authority=0.9), make_item("m2", authority=0.8), make_item("m3", authority=0.7)])
    result = get_items_by_budget(state, budget=5, cost_fn=lambda i: 0, weights={"authority": 1})
    assert len(result) == 3


def test_budget_mixes_zero_and_positive() -> None:
    state = state_with([make_item("m1", authority=0.9), make_item("m2", authority=0.8), make_item("m3", authority=0.7)])
    result = get_items_by_budget(state, budget=2, cost_fn=lambda item: 0 if item.id == "m2" else 1, weights={"authority": 1})
    assert sorted(r.item.id for r in result) == ["m1", "m2", "m3"]


def test_budget_rejects_negative() -> None:
    state = state_with([make_item("m1", authority=0.9)])
    with pytest.raises(ValueError):
        get_items_by_budget(state, budget=5, cost_fn=lambda i: -1, weights={"authority": 1})


# === 8. applyMany empty object skip ========================================


def test_apply_many_skip_all() -> None:
    state = state_with([make_item("m1", authority=0.5), make_item("m2", authority=0.8)])
    result = apply_many(state, {}, lambda i: {}, "agent:test")
    assert result.state is state
    assert result.events == []


def test_apply_many_some_skip_some_apply() -> None:
    state = state_with([make_item("m1", authority=0.5), make_item("m2", authority=0.8)])
    result = apply_many(state, {}, lambda item: {"authority": 0.9} if item.id == "m1" else {}, "agent:test")
    assert len(result.events) == 1
    assert result.state.items["m1"].authority == 0.9
    assert result.state.items["m2"].authority == 0.8


# === 9. resolveContradiction multiple edges ================================


def test_resolve_multiple_contradicts_edges() -> None:
    state = state_with([make_item("m1", authority=0.9), make_item("m2", authority=0.5)])
    state = apply_command(state, {"type": "edge.create", "edge": make_edge("c1", "m1", "m2", "CONTRADICTS")}).state
    state = apply_command(state, {"type": "edge.create", "edge": make_edge("c2", "m2", "m1", "CONTRADICTS")}).state
    assert len(get_contradictions(state)) == 2
    state = resolve_contradiction(state, "m1", "m2", "agent:test").state
    assert len(get_edges(state, {"kind": "CONTRADICTS", "active_only": True})) == 0
    supersedes = get_edges(state, {"kind": "SUPERSEDES", "active_only": True})
    assert len(supersedes) == 1
    assert supersedes[0].from_ == "m1"
    assert supersedes[0].to == "m2"


# === 10/11. exportSlice walks intent_id / task_id ==========================


def test_export_walks_intent_id() -> None:
    mem_state = state_with([make_item("m1", intent_id="i1")])
    intent_state = apply_intent_command(create_intent_state(), {"type": "intent.create", "intent": create_intent(id="i1", label="test intent", priority=0.5, owner="agent:test")}).state
    slice = export_slice(mem_state, intent_state, create_task_state(), memory_ids=["m1"], include_related_intents=True)
    assert len(slice.intents) == 1 and slice.intents[0].id == "i1"


def test_export_walks_task_id() -> None:
    mem_state = state_with([make_item("m1", task_id="t1")])
    task_state = apply_task_command(create_task_state(), {"type": "task.create", "task": create_task(id="t1", intent_id="i1", action="search", priority=0.5)}).state
    slice = export_slice(mem_state, create_intent_state(), task_state, memory_ids=["m1"], include_related_tasks=True)
    assert len(slice.tasks) == 1 and slice.tasks[0].id == "t1"


# === 12. getEdges active_only ==============================================


def test_get_edges_active_only_false() -> None:
    state = state_with([make_item("m1"), make_item("m2")], [make_edge("e1", "m1", "m2", "SUPPORTS", active=True), make_edge("e2", "m1", "m2", "ABOUT", active=False)])
    assert len(get_edges(state, {"active_only": False})) == 2
    active = get_edges(state, {"active_only": True})
    assert len(active) == 1 and active[0].edge_id == "e1"


def test_get_edges_defaults_active_only() -> None:
    state = state_with([make_item("m1"), make_item("m2")], [make_edge("e1", "m1", "m2", "SUPPORTS", active=True), make_edge("e2", "m1", "m2", "ABOUT", active=False)])
    default = get_edges(state)
    assert len(default) == 1 and default[0].edge_id == "e1"


# === 13. decayImportance all zero ==========================================


def test_decay_importance_all_zero_noop() -> None:
    state = state_with([make_item(fake_id(1000), importance=0), make_item(fake_id(1001), importance=0)])
    result = decay_importance(state, 1, 0.5, "agent:test")
    assert result.state is state
    assert result.events == []


# === 14. replayFromEnvelopes ordering ======================================


def test_replay_sorts_by_timestamp() -> None:
    id1 = fake_id(1000)
    id2 = fake_id(2000)
    envelopes = [
        {"id": "env2", "namespace": "memory", "type": "memory.create", "ts": "2026-01-01T00:00:02.000Z", "payload": {"type": "memory.create", "item": make_item(id2, authority=0.9)}},
        {"id": "env1", "namespace": "memory", "type": "memory.create", "ts": "2026-01-01T00:00:01.000Z", "payload": {"type": "memory.create", "item": make_item(id1, authority=0.5)}},
    ]
    result = replay_from_envelopes(envelopes)
    assert len(result.state.items) == 2
    assert id1 in result.state.items and id2 in result.state.items
    assert result.events[0].item.id == id1
    assert result.events[1].item.id == id2


def test_replay_identical_timestamps() -> None:
    id1 = fake_id(3000)
    id2 = fake_id(3001)
    envelopes = [
        {"id": "env1", "namespace": "memory", "type": "memory.create", "ts": "2026-01-01T00:00:01.000Z", "payload": {"type": "memory.create", "item": make_item(id1)}},
        {"id": "env2", "namespace": "memory", "type": "memory.create", "ts": "2026-01-01T00:00:01.000Z", "payload": {"type": "memory.create", "item": make_item(id2)}},
    ]
    result = replay_from_envelopes(envelopes)
    assert len(result.state.items) == 2


# === 15. getRelatedItems inactive edges ====================================


def test_related_excludes_inactive() -> None:
    state = state_with(
        [make_item("m1"), make_item("m2"), make_item("m3")],
        [make_edge("e1", "m1", "m2", "SUPPORTS", active=True), make_edge("e2", "m1", "m3", "SUPPORTS", active=False)],
    )
    ids = [i.id for i in get_related_items(state, "m1")]
    assert "m2" in ids
    assert "m3" not in ids


def test_related_all_inactive_empty() -> None:
    state = state_with([make_item("m1"), make_item("m2")], [make_edge("e1", "m1", "m2", "SUPPORTS", active=False)])
    assert get_related_items(state, "m1") == []


# === Bonus: applyDiversity empty ===========================================


def test_apply_diversity_empty_input() -> None:
    assert apply_diversity([], {"author_penalty": 0.1}) == []
