"""Port of tests/bugfix-and-coverage.test.ts."""

from __future__ import annotations

from typing import Any

import pytest

from memex import (
    Edge,
    GraphState,
    Intent,
    MemoryItem,
    ScoredItem,
    Task,
    apply_command,
    apply_diversity,
    apply_intent_command,
    apply_task_command,
    cascade_retract,
    clone_graph_state,
    create_intent_state,
    create_task_state,
    export_slice,
    extract_timestamp,
    filter_contradictions,
    get_alias_group,
    get_contradictions,
    get_dependents,
    get_edges,
    get_items,
    get_scored_items,
    import_slice,
    mark_contradiction,
    merge_item,
    parse,
    resolve_contradiction,
    smart_retrieve,
    stringify,
    surface_contradictions,
)


def fake_uuid(n: int) -> str:
    ms = format(1700000000000 + n, "x").rjust(12, "0")
    return f"{ms[0:8]}-{ms[8:12]}-7000-8000-{'0' * 11}{n}"


def make_item(id: str, **overrides: Any) -> MemoryItem:
    base: dict[str, Any] = {
        "id": id, "scope": "test", "kind": "observation", "content": {},
        "author": "agent:a", "source_kind": "observed", "authority": 0.5,
    }
    base.update(overrides)
    return MemoryItem(**base)


def make_edge(id: str, frm: str, to: str, kind: str = "SUPPORTS", **overrides: Any) -> Edge:
    base: dict[str, Any] = {
        "edge_id": id, "from_": frm, "to": to, "kind": kind,
        "author": "system:rule", "source_kind": "derived_deterministic", "authority": 0.8, "active": True,
    }
    base.update(overrides)
    return Edge(**base)


def state_with(items: list[MemoryItem], edges: list[Edge] | None = None) -> GraphState:
    return GraphState(items={i.id: i for i in items}, edges={e.edge_id: e for e in (edges or [])})


def to_scored(items: list[MemoryItem], scores: list[float]) -> list[ScoredItem]:
    return [ScoredItem(item=item, score=scores[i]) for i, item in enumerate(items)]


def make_intent(**overrides: Any) -> Intent:
    base: dict[str, Any] = {"id": "i1", "label": "find_kati", "priority": 0.8, "owner": "user:laz", "status": "active"}
    base.update(overrides)
    return Intent(**base)


def make_task(**overrides: Any) -> Task:
    base: dict[str, Any] = {"id": "t1", "intent_id": "i1", "action": "search_linkedin", "status": "pending", "priority": 0.7, "attempt": 0}
    base.update(overrides)
    return Task(**base)


# === edge.update mergeEdge fixes ===========================================


def test_edge_update_no_undefined_overwrite() -> None:
    state = state_with([], [make_edge("e1", "m1", "m2", "SUPPORTS", weight=0.8)])
    nxt, _ = apply_command(state, {"type": "edge.update", "edge_id": "e1", "partial": {"kind": "ABOUT"}, "author": "test"})
    edge = nxt.edges["e1"]
    assert edge.weight == 0.8
    assert edge.kind == "ABOUT"


def test_edge_update_ignores_edge_id() -> None:
    state = state_with([], [make_edge("e1", "m1", "m2")])
    nxt, _ = apply_command(state, {"type": "edge.update", "edge_id": "e1", "partial": {"edge_id": "sneaky"}, "author": "test"})
    assert nxt.edges["e1"].edge_id == "e1"
    assert "sneaky" not in nxt.edges


def test_edge_update_ignores_from_to() -> None:
    state = state_with([], [make_edge("e1", "m1", "m2")])
    nxt, _ = apply_command(state, {"type": "edge.update", "edge_id": "e1", "partial": {"from": "x", "to": "y"}, "author": "test"})
    edge = nxt.edges["e1"]
    assert edge.from_ == "m1"
    assert edge.to == "m2"


# === intent.update status protection =======================================


def test_intent_update_ignores_status() -> None:
    state = apply_intent_command(create_intent_state(), {"type": "intent.create", "intent": make_intent(status="active")}).state
    nxt, _ = apply_intent_command(state, {"type": "intent.update", "intent_id": "i1", "partial": {"status": "completed"}, "author": "test"})
    assert nxt.intents["i1"].status == "active"


def test_intent_update_cannot_revive_cancelled() -> None:
    state = apply_intent_command(create_intent_state(), {"type": "intent.create", "intent": make_intent(status="active")}).state
    state = apply_intent_command(state, {"type": "intent.cancel", "intent_id": "i1", "author": "test"}).state
    nxt, _ = apply_intent_command(state, {"type": "intent.update", "intent_id": "i1", "partial": {"status": "active"}, "author": "test"})
    assert nxt.intents["i1"].status == "cancelled"


def test_intent_update_other_fields() -> None:
    state = apply_intent_command(create_intent_state(), {"type": "intent.create", "intent": make_intent()}).state
    nxt, _ = apply_intent_command(state, {"type": "intent.update", "intent_id": "i1", "partial": {"label": "new_label", "priority": 0.3}, "author": "test"})
    assert nxt.intents["i1"].label == "new_label"
    assert nxt.intents["i1"].priority == 0.3
    assert nxt.intents["i1"].status == "active"


# === task.update status protection ==========================================


def test_task_update_ignores_status() -> None:
    state = apply_task_command(create_task_state(), {"type": "task.create", "task": make_task(status="pending")}).state
    nxt, _ = apply_task_command(state, {"type": "task.update", "task_id": "t1", "partial": {"status": "completed"}, "author": "test"})
    assert nxt.tasks["t1"].status == "pending"


def test_task_update_cannot_revive_failed() -> None:
    state = apply_task_command(create_task_state(), {"type": "task.create", "task": make_task(status="pending")}).state
    state = apply_task_command(state, {"type": "task.start", "task_id": "t1"}).state
    state = apply_task_command(state, {"type": "task.fail", "task_id": "t1", "error": "oops"}).state
    nxt, _ = apply_task_command(state, {"type": "task.update", "task_id": "t1", "partial": {"status": "running"}, "author": "test"})
    assert nxt.tasks["t1"].status == "failed"


# === applyDiversity preserves contradicted_by ==============================


def test_diversity_preserves_contradicted_by() -> None:
    m1, m2, m3 = make_item("m1", author="a"), make_item("m2", author="a"), make_item("m3", author="b")
    scored = [
        ScoredItem(item=m1, score=0.9, contradicted_by=[m3]),
        ScoredItem(item=m2, score=0.8),
        ScoredItem(item=m3, score=0.7, contradicted_by=[m1]),
    ]
    result = apply_diversity(scored, {"author_penalty": 0.1})
    r1 = next(s for s in result if s.item.id == "m1")
    r3 = next(s for s in result if s.item.id == "m3")
    assert r1.contradicted_by == [m3]
    assert r3.contradicted_by == [m1]


def test_smart_retrieve_surface_plus_diversity() -> None:
    m1, m2 = make_item("m1", authority=0.9, author="a"), make_item("m2", authority=0.8, author="a")
    state = state_with([m1, m2])
    marked, _ = mark_contradiction(state, "m1", "m2", "system:detector")
    result = smart_retrieve(marked, budget=1000, cost_fn=lambda i: 1, weights={"authority": 1}, contradictions="surface", diversity={"author_penalty": 0.05})
    item1 = next(s for s in result if s.item.id == "m1")
    item2 = next(s for s in result if s.item.id == "m2")
    assert item1.contradicted_by is not None and len(item1.contradicted_by) > 0
    assert item2.contradicted_by is not None


# === decay interval validation =============================================


def test_decay_unknown_interval_raises() -> None:
    from memex import _time
    state = state_with([make_item("m1", authority=0.5, created_at=_time.now_ms() - 86_400_000)])
    with pytest.raises(ValueError):
        get_scored_items(state, {"authority": 1, "decay": {"rate": 0.1, "interval": "month", "type": "exponential"}})


def test_decay_unknown_interval_message() -> None:
    from memex import _time
    state = state_with([make_item("m1", authority=0.5, created_at=_time.now_ms() - 86_400_000)])
    with pytest.raises(ValueError, match="Unknown decay interval.*month"):
        get_scored_items(state, {"authority": 1, "decay": {"rate": 0.1, "interval": "month", "type": "exponential"}})


# === resolveContradiction no-op ============================================


def test_resolve_noop_no_edge() -> None:
    state = state_with([make_item("m1", authority=0.9), make_item("m2", authority=0.7)])
    result = resolve_contradiction(state, "m1", "m2", "system:resolver")
    assert len(result.events) == 0
    assert result.state.items["m1"].authority == 0.9
    assert result.state.items["m2"].authority == 0.7


# === filterContradictions equal scores =====================================


def test_filter_equal_scores_excludes_one() -> None:
    m1, m2 = make_item("m1"), make_item("m2")
    state = state_with([m1, m2])
    state = mark_contradiction(state, "m1", "m2", "system:detector").state
    filtered = filter_contradictions(state, to_scored([m1, m2], [0.5, 0.5]))
    assert len(filtered) == 1


def test_filter_tiebreak_independent_of_edge_direction() -> None:
    m1, m2 = make_item("aaa"), make_item("zzz")
    state1 = mark_contradiction(state_with([m1, m2]), "aaa", "zzz", "sys").state
    r1 = filter_contradictions(state1, to_scored([m1, m2], [0.5, 0.5]))
    state2 = mark_contradiction(state_with([m1, m2]), "zzz", "aaa", "sys").state
    r2 = filter_contradictions(state2, to_scored([m1, m2], [0.5, 0.5]))
    assert len(r1) == 1 and len(r2) == 1
    assert r1[0].item.id == r2[0].item.id
    assert r1[0].item.id == "aaa"


# === getContradictions retracted ============================================


def test_get_contradictions_skips_retracted() -> None:
    state = state_with([make_item("m1"), make_item("m2")])
    state = mark_contradiction(state, "m1", "m2", "system:detector").state
    state = apply_command(state, {"type": "memory.retract", "item_id": "m2", "author": "test"}).state
    assert len(get_contradictions(state)) == 0


# === getScoredItems post filter ============================================


def test_scored_post_filter_scope() -> None:
    state = state_with([make_item("m1", authority=0.9, scope="a"), make_item("m2", authority=0.8, scope="b"), make_item("m3", authority=0.7, scope="a")])
    result = get_scored_items(state, {"authority": 1}, {"post": {"scope": "a"}})
    assert len(result) == 2
    assert all(r.item.scope == "a" for r in result)
    assert result[0].item.id == "m1"
    assert result[1].item.id == "m3"


def test_scored_post_filter_range() -> None:
    state = state_with([make_item("m1", authority=0.9, importance=0.1), make_item("m2", authority=0.3, importance=0.9), make_item("m3", authority=0.1, importance=0.1)])
    result = get_scored_items(state, {"authority": 1}, {"post": {"range": {"importance": {"min": 0.5}}}})
    assert len(result) == 1 and result[0].item.id == "m2"


# === getEdges to filter ====================================================


def test_get_edges_to_filter() -> None:
    state = state_with([], [make_edge("e1", "m1", "m2"), make_edge("e2", "m1", "m3"), make_edge("e3", "m2", "m3")])
    result = get_edges(state, {"to": "m3"})
    assert len(result) == 2 and all(e.to == "m3" for e in result)


# === cascadeRetract edge cases =============================================


def test_cascade_nonexistent() -> None:
    state = state_with([make_item("m1")])
    nxt, _events, retracted = cascade_retract(state, "nonexistent", "test")
    assert len(retracted) == 0
    assert "m1" in nxt.items


def test_get_dependents_circular() -> None:
    state = state_with([make_item("m1", parents=["m3"]), make_item("m2", parents=["m1"]), make_item("m3", parents=["m2"])])
    deps = get_dependents(state, "m1", True)
    assert len(deps) >= 2
    ids = sorted(d.id for d in deps)
    assert "m2" in ids and "m3" in ids


# === getAliasGroup nonexistent =============================================


def test_alias_group_nonexistent() -> None:
    assert len(get_alias_group(state_with([make_item("m1")]), "nonexistent")) == 0


# === cloneGraphState shallow ===============================================


def test_clone_map_mutation_isolated() -> None:
    state = state_with([make_item("m1")])
    clone = clone_graph_state(state)
    del clone.items["m1"]
    assert "m1" in state.items
    assert "m1" not in clone.items


def test_clone_value_refs_shared() -> None:
    state = state_with([make_item("m1", content={"x": 1})])
    clone = clone_graph_state(state)
    assert clone.items["m1"] is state.items["m1"]


# === extractTimestamp edge cases ===========================================


def test_extract_non_uuidv7() -> None:
    from memex import InvalidTimestampError
    with pytest.raises(InvalidTimestampError, match="not a valid UUIDv7"):
        extract_timestamp("not-a-uuid")


def test_extract_valid_uuidv7() -> None:
    from memex import _time
    now = _time.now_ms()
    h = format(now, "x").rjust(12, "0")
    id_ = f"{h[0:8]}-{h[8:12]}-7000-8000-000000000000"
    assert extract_timestamp(id_) == now


# === serialization error handling ==========================================


def test_parse_malformed_json() -> None:
    with pytest.raises(ValueError):  # json.JSONDecodeError subclasses ValueError
        parse("{not valid json")


def test_parse_missing_items_field() -> None:
    state = parse('{"edges": []}')
    assert len(state.items) == 0
    assert len(state.edges) == 0


def test_serialization_round_trip() -> None:
    state = state_with([make_item("m1", content={"text": "hello"})], [make_edge("e1", "m1", "m2")])
    restored = parse(stringify(state))
    assert restored.items["m1"].content == {"text": "hello"}
    assert restored.edges["e1"].from_ == "m1"


# === importSlice skipExistingIds: false ====================================


def test_import_update_existing_skip_false() -> None:
    mem = state_with([make_item("m1")])
    slice = {"memories": [make_item("m1", content={"new": True})], "edges": [], "intents": [], "tasks": []}
    result = import_slice(mem, create_intent_state(), create_task_state(), slice, skip_existing_ids=False)
    assert result.mem_state.items["m1"].content == {"new": True}
    assert result.report.updated.memories == ["m1"]


def test_import_create_noncolliding_skip_false() -> None:
    mem = state_with([make_item("m1")])
    slice = {"memories": [make_item("m2")], "edges": [], "intents": [], "tasks": []}
    result = import_slice(mem, create_intent_state(), create_task_state(), slice, skip_existing_ids=False)
    assert "m2" in result.mem_state.items
    assert "m2" in result.report.created.memories


# === exportSlice include_aliases ===========================================


def test_export_include_aliases_chain() -> None:
    state = state_with([make_item("m1"), make_item("m2"), make_item("m3")])
    state = apply_command(state, {"type": "edge.create", "edge": make_edge("ae1", "m1", "m2", "ALIAS")}).state
    state = apply_command(state, {"type": "edge.create", "edge": make_edge("ae2", "m2", "m3", "ALIAS")}).state
    slice = export_slice(state, create_intent_state(), create_task_state(), memory_ids=["m1"], include_aliases=True)
    ids = sorted(m.id for m in slice.memories)
    assert "m1" in ids and "m2" in ids and "m3" in ids
    assert len(slice.edges) >= 2


# === smartRetrieve no contradiction handling ===============================


def test_smart_retrieve_no_contradiction_handling() -> None:
    m1, m2 = make_item("m1", authority=0.9), make_item("m2", authority=0.8)
    state = state_with([m1, m2])
    state = mark_contradiction(state, "m1", "m2", "system:detector").state
    result = smart_retrieve(state, budget=1000, cost_fn=lambda i: 1, weights={"authority": 1})
    assert len(result) == 2


# === surfaceContradictions idempotency =====================================


def test_surface_idempotent_fresh_clone() -> None:
    m1, m2 = make_item("m1"), make_item("m2")
    state = state_with([m1, m2])
    state = mark_contradiction(state, "m1", "m2", "system:detector").state
    r1 = surface_contradictions(state, to_scored([m1, m2], [0.5, 0.5]))
    r2 = surface_contradictions(state, to_scored([m1, m2], [0.5, 0.5]))
    r1m1 = next(s for s in r1 if s.item.id == "m1")
    r2m1 = next(s for s in r2 if s.item.id == "m1")
    assert len(r1m1.contradicted_by) == 1
    assert len(r2m1.contradicted_by) == 1


# === applyDiversity mixed parents ==========================================


def test_diversity_mixed_parents() -> None:
    m1, m2, m3 = make_item("m1", parents=["p1"]), make_item("m2"), make_item("m3", parents=["p1"])
    result = apply_diversity(to_scored([m1, m2, m3], [0.9, 0.8, 0.7]), {"parent_penalty": 0.1})
    m3r = next(s for s in result if s.item.id == "m3")
    assert m3r.score < 0.7
    m2r = next(s for s in result if s.item.id == "m2")
    assert m2r.score == 0.8


# === mergeItem edge cases (D5: undefined ≡ absent) =========================


def test_merge_content_undefined_preserved() -> None:
    existing = make_item("m1", content={"a": 1, "b": 2})
    merged = merge_item(existing, {"content": {"c": 3}})
    assert merged.content["a"] == 1
    assert merged.content["b"] == 2
    assert merged.content["c"] == 3


def test_merge_meta_undefined_preserved() -> None:
    existing = make_item("m1", meta={"agent_id": "bot", "x": 1})
    merged = merge_item(existing, {"meta": {"y": 2}})
    assert merged.meta["agent_id"] == "bot"
    assert merged.meta["x"] == 1
    assert merged.meta["y"] == 2


def test_merge_cannot_change_id() -> None:
    merged = merge_item(make_item("m1"), {"id": "sneaky"})
    assert merged.id == "m1"


# === importSlice re-id intents and tasks ===================================


def test_import_remaps_intent_root_memory_ids() -> None:
    mem_id = fake_uuid(1)
    intent_id = fake_uuid(2)
    mem = state_with([make_item(mem_id, content={"old": True})])
    slice = {
        "memories": [make_item(mem_id, content={"new": True})], "edges": [], "tasks": [],
        "intents": [make_intent(id=intent_id, root_memory_ids=[mem_id])],
    }
    result = import_slice(mem, create_intent_state(), create_task_state(), slice, skip_existing_ids=True, shallow_compare_existing=True, re_id_on_difference=True)
    assert len(result.report.created.memories) == 1
    new_mem_id = result.report.created.memories[0]
    assert new_mem_id != mem_id
    imported_intent = result.report.created.intents[0]
    intent = result.intent_state.intents[imported_intent]
    assert new_mem_id in intent.root_memory_ids
    assert mem_id not in intent.root_memory_ids


def test_import_remaps_task_memory_ids() -> None:
    mem_id = fake_uuid(1)
    intent_id = fake_uuid(2)
    task_id = fake_uuid(3)
    mem = state_with([make_item(mem_id, content={"old": True})])
    intents = apply_intent_command(create_intent_state(), {"type": "intent.create", "intent": make_intent(id=intent_id)}).state
    slice = {
        "memories": [make_item(mem_id, content={"new": True})], "edges": [], "intents": [],
        "tasks": [make_task(id=task_id, intent_id=intent_id, input_memory_ids=[mem_id], output_memory_ids=[mem_id])],
    }
    result = import_slice(mem, intents, create_task_state(), slice, skip_existing_ids=True, shallow_compare_existing=True, re_id_on_difference=True)
    new_mem_id = result.report.created.memories[0]
    imported_task_id = result.report.created.tasks[0]
    task = result.task_state.tasks[imported_task_id]
    assert new_mem_id in task.input_memory_ids
    assert new_mem_id in task.output_memory_ids


# === importSlice edge collision ============================================


def test_import_edge_collision_skip() -> None:
    edge = make_edge("e1", "m1", "m2")
    mem = state_with([make_item("m1"), make_item("m2")], [edge])
    slice = {"memories": [], "edges": [make_edge("e1", "m1", "m2", "ABOUT")], "intents": [], "tasks": []}
    result = import_slice(mem, create_intent_state(), create_task_state(), slice)
    assert "e1" in result.report.skipped.edges
    assert result.mem_state.edges["e1"].kind == "SUPPORTS"


# === created filter boundary semantics =====================================


def _id_at(ts: int) -> str:
    h = format(ts, "x").rjust(12, "0")
    return f"{h[0:8]}-{h[8:12]}-7000-8000-000000000000"


def test_created_before_exclusive() -> None:
    ts = 1700000000000
    state = state_with([make_item(_id_at(ts))])
    assert len(get_items(state, {"created": {"before": ts}})) == 0


def test_created_after_inclusive() -> None:
    ts = 1700000000000
    state = state_with([make_item(_id_at(ts))])
    assert len(get_items(state, {"created": {"after": ts}})) == 1


def test_created_between() -> None:
    ts = 1700000000000
    state = state_with([make_item(_id_at(ts))])
    assert len(get_items(state, {"created": {"after": ts - 1, "before": ts + 1}})) == 1
