"""Port of tests/bugfix-regressions.test.ts."""

from __future__ import annotations

from typing import Any

import pytest

from memex import (
    Edge,
    GraphState,
    MemoryItem,
    _time,
    apply_command,
    apply_intent_command,
    apply_task_command,
    create_graph_state,
    create_intent,
    create_intent_state,
    create_memory_item,
    create_task,
    create_task_state,
    filter_contradictions,
    get_items,
    get_scored_items,
    import_slice,
    merge_item,
)
from memex.models import ScoredItem


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


def to_scored(items: list[MemoryItem], scores: list[float]) -> list[ScoredItem]:
    return [ScoredItem(item=item, score=scores[i]) for i, item in enumerate(items)]


# === Bug 4: undefined content/meta keys preserved (D5: undefined ≡ absent) ==


def test_merge_preserves_content_key() -> None:
    existing = make_item("m1", content={"a": 1, "b": 2, "c": 3})
    merged = merge_item(existing, {"content": {}})
    assert merged.content == {"a": 1, "b": 2, "c": 3}


def test_merge_preserves_meta_key() -> None:
    existing = make_item("m1", meta={"agent_id": "agent:x", "session_id": "s1", "custom": "val"})
    merged = merge_item(existing, {"meta": {}})
    assert merged.meta == {"agent_id": "agent:x", "session_id": "s1", "custom": "val"}


def test_merge_keeps_content_intact_with_new_key() -> None:
    existing = make_item("m1", content={"x": 10, "y": 20})
    merged = merge_item(existing, {"content": {"z": 30}})
    assert merged.content == {"x": 10, "y": 20, "z": 30}


def test_merge_through_apply_command() -> None:
    state = create_graph_state()
    state = apply_command(state, {"type": "memory.create", "item": make_item("m1", content={"keep": 1, "remove": 2})}).state
    state = apply_command(state, {"type": "memory.update", "item_id": "m1", "partial": {"content": {}}, "author": "test"}).state
    assert state.items["m1"].content == {"keep": 1, "remove": 2}


# === Bug 5: created_at on MemoryItem ========================================


def test_created_at_from_uuidv7() -> None:
    item = create_memory_item(scope="test", kind="observation", content={}, author="agent:a", source_kind="observed", authority=0.5)
    assert item.created_at is not None
    assert abs(item.created_at - _time.now_ms()) < 1000


def test_created_at_for_non_uuid() -> None:
    item = create_memory_item(id="custom-non-uuid", scope="test", kind="observation", content={}, author="agent:a", source_kind="observed", authority=0.5)
    assert item.created_at is not None
    assert abs(item.created_at - _time.now_ms()) < 1000


def test_created_at_explicit_preserved() -> None:
    item = create_memory_item(id="custom-id", scope="test", kind="observation", content={}, author="agent:a", source_kind="observed", authority=0.5, created_at=1000000)
    assert item.created_at == 1000000


def test_recency_sort_with_created_at() -> None:
    state = create_graph_state()
    state = apply_command(state, {"type": "memory.create", "item": make_item("item-old", created_at=1000)}).state
    state = apply_command(state, {"type": "memory.create", "item": make_item("item-new", created_at=2000)}).state
    results = get_items(state, None, {"sort": {"field": "recency", "order": "desc"}})
    assert results[0].id == "item-new"
    assert results[1].id == "item-old"


def test_decay_uses_created_at() -> None:
    two_days_ago = _time.now_ms() - 2 * 86_400_000
    state = create_graph_state()
    state = apply_command(state, {"type": "memory.create", "item": make_item("custom-id", authority=1.0, created_at=two_days_ago)}).state
    scored = get_scored_items(state, {"authority": 1, "decay": {"rate": 0.5, "interval": "day", "type": "exponential"}})
    assert len(scored) == 1
    assert scored[0].score == pytest.approx(0.25, abs=0.05)


# === Bug 3: filterContradictions determinism ===============================


def _chain_state(edge_order: list[tuple[str, str, str]]) -> GraphState:
    state = create_graph_state()
    for item in [make_item("a", authority=0.9), make_item("b", authority=0.7), make_item("c", authority=0.5)]:
        state = apply_command(state, {"type": "memory.create", "item": item}).state
    for eid, frm, to in edge_order:
        state = apply_command(state, {"type": "edge.create", "edge": make_edge(eid, frm, to, "CONTRADICTS")}).state
    return state


def test_filter_contradictions_order_independent() -> None:
    scored = to_scored([make_item("a", authority=0.9), make_item("b", authority=0.7), make_item("c", authority=0.5)], [0.9, 0.7, 0.5])
    state1 = _chain_state([("e-bc", "b", "c"), ("e-ab", "a", "b")])
    state2 = _chain_state([("e-ab", "a", "b"), ("e-bc", "b", "c")])
    ids1 = sorted(s.item.id for s in filter_contradictions(state1, scored))
    ids2 = sorted(s.item.id for s in filter_contradictions(state2, scored))
    assert ids1 == ids2


def test_filter_contradictions_chain_resolution() -> None:
    scored = to_scored([make_item("a", authority=0.9), make_item("b", authority=0.7), make_item("c", authority=0.5)], [0.9, 0.7, 0.5])
    state = _chain_state([("e-ab", "a", "b"), ("e-bc", "b", "c")])
    ids = sorted(s.item.id for s in filter_contradictions(state, scored))
    assert ids == ["a", "c"]


# === Bug 2: skip_existing_ids=False updates instead of crashing ============


def test_update_existing_memory() -> None:
    target = create_graph_state()
    target = apply_command(target, {"type": "memory.create", "item": make_item("m1", authority=0.5)}).state
    slice = {"memories": [make_item("m1", authority=0.9)], "edges": [], "intents": [], "tasks": []}
    result = import_slice(target, create_intent_state(), create_task_state(), slice, skip_existing_ids=False)
    assert result.mem_state.items["m1"].authority == 0.9
    assert result.report.updated.memories == ["m1"]
    assert len(result.report.created.memories) == 0


def test_update_existing_edge() -> None:
    target = create_graph_state()
    target = apply_command(target, {"type": "memory.create", "item": make_item("m1")}).state
    target = apply_command(target, {"type": "memory.create", "item": make_item("m2")}).state
    target = apply_command(target, {"type": "edge.create", "edge": make_edge("e1", "m1", "m2")}).state
    slice = {"memories": [], "edges": [make_edge("e1", "m1", "m2", weight=0.99)], "intents": [], "tasks": []}
    result = import_slice(target, create_intent_state(), create_task_state(), slice, skip_existing_ids=False)
    assert result.mem_state.edges["e1"].weight == 0.99
    assert result.report.updated.edges == ["e1"]


def test_update_existing_intent() -> None:
    target = create_intent_state()
    target = apply_intent_command(target, {"type": "intent.create", "intent": create_intent(id="i1", label="old_label", priority=0.5, owner="user:laz")}).state
    slice = {"memories": [], "edges": [], "intents": [create_intent(id="i1", label="new_label", priority=0.9, owner="user:laz")], "tasks": []}
    result = import_slice(create_graph_state(), target, create_task_state(), slice, skip_existing_ids=False)
    assert result.intent_state.intents["i1"].label == "new_label"
    assert result.intent_state.intents["i1"].priority == 0.9
    assert result.report.updated.intents == ["i1"]


def test_update_existing_task() -> None:
    target = create_task_state()
    target = apply_task_command(target, {"type": "task.create", "task": create_task(id="t1", intent_id="i1", action="old_action", priority=0.5)}).state
    slice = {"memories": [], "edges": [], "intents": [], "tasks": [create_task(id="t1", intent_id="i1", action="new_action", priority=0.9)]}
    result = import_slice(create_graph_state(), create_intent_state(), target, slice, skip_existing_ids=False)
    assert result.task_state.tasks["t1"].action == "new_action"
    assert result.task_state.tasks["t1"].priority == 0.9
    assert result.report.updated.tasks == ["t1"]


# === Bug 1: importSlice remaps intent_id/task_id on memories ===============


def test_remaps_memory_intent_id() -> None:
    intent_id = fake_uuid(1)
    mem_id = fake_uuid(2)
    target_intents = create_intent_state()
    target_intents = apply_intent_command(target_intents, {"type": "intent.create", "intent": create_intent(id=intent_id, label="existing_intent", priority=0.5, owner="user:laz")}).state
    slice = {
        "memories": [make_item(mem_id, intent_id=intent_id)], "edges": [], "tasks": [],
        "intents": [create_intent(id=intent_id, label="imported_intent", priority=0.9, owner="user:laz")],
    }
    result = import_slice(create_graph_state(), target_intents, create_task_state(), slice, shallow_compare_existing=True, re_id_on_difference=True)
    assert len(result.report.created.intents) == 1
    new_intent_id = result.report.created.intents[0]
    assert new_intent_id != intent_id
    assert result.mem_state.items[mem_id].intent_id == new_intent_id


def test_remaps_memory_task_id() -> None:
    task_id = fake_uuid(1)
    intent_id = fake_uuid(2)
    mem_id = fake_uuid(3)
    target_tasks = create_task_state()
    target_tasks = apply_task_command(target_tasks, {"type": "task.create", "task": create_task(id=task_id, intent_id=intent_id, action="existing_action", priority=0.5)}).state
    target_intents = create_intent_state()
    target_intents = apply_intent_command(target_intents, {"type": "intent.create", "intent": create_intent(id=intent_id, label="intent", priority=0.5, owner="user:laz")}).state
    slice = {
        "memories": [make_item(mem_id, task_id=task_id)], "edges": [], "intents": [],
        "tasks": [create_task(id=task_id, intent_id=intent_id, action="imported_action", priority=0.9)],
    }
    result = import_slice(create_graph_state(), target_intents, target_tasks, slice, shallow_compare_existing=True, re_id_on_difference=True)
    assert len(result.report.created.tasks) == 1
    new_task_id = result.report.created.tasks[0]
    assert new_task_id != task_id
    assert result.mem_state.items[mem_id].task_id == new_task_id


def test_remaps_both_intent_and_task_id() -> None:
    intent_id = fake_uuid(1)
    task_id = fake_uuid(2)
    mem_id = fake_uuid(3)
    target_intents = create_intent_state()
    target_intents = apply_intent_command(target_intents, {"type": "intent.create", "intent": create_intent(id=intent_id, label="existing", priority=0.1, owner="user:laz")}).state
    target_tasks = create_task_state()
    target_tasks = apply_task_command(target_tasks, {"type": "task.create", "task": create_task(id=task_id, intent_id=intent_id, action="existing", priority=0.1)}).state
    slice = {
        "memories": [make_item(mem_id, intent_id=intent_id, task_id=task_id)], "edges": [],
        "intents": [create_intent(id=intent_id, label="imported", priority=0.9, owner="user:laz")],
        "tasks": [create_task(id=task_id, intent_id=intent_id, action="imported", priority=0.9)],
    }
    result = import_slice(create_graph_state(), target_intents, target_tasks, slice, shallow_compare_existing=True, re_id_on_difference=True)
    new_intent_id = result.report.created.intents[0]
    new_task_id = result.report.created.tasks[0]
    mem = result.mem_state.items[mem_id]
    assert mem.intent_id == new_intent_id
    assert mem.task_id == new_task_id
