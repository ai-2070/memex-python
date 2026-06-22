"""Port of tests/cross-graph-fields.test.ts (query/filter sections).

The transplant parent_id-rewriting section lives in test_transplant.py (Phase 6).
"""

from __future__ import annotations

from typing import Any

from memex import (
    GraphState,
    MemoryItem,
    apply_command,
    apply_intent_command,
    apply_task_command,
    create_graph_state,
    create_intent,
    create_intent_state,
    create_task,
    create_task_state,
    get_child_intents,
    get_child_tasks,
    get_intents,
    get_items,
    get_tasks,
)


def make_item(id: str, **overrides: Any) -> MemoryItem:
    base: dict[str, Any] = {
        "id": id, "scope": "test", "kind": "observation", "content": {"text": f"item {id}"},
        "author": "agent:test", "source_kind": "observed", "authority": 0.8,
    }
    base.update(overrides)
    return MemoryItem(**base)


def state_with(items: list[MemoryItem]) -> GraphState:
    state = create_graph_state()
    for item in items:
        state = apply_command(state, {"type": "memory.create", "item": item}).state
    return state


# === MemoryFilter intent_id / task_id ======================================


def test_filter_by_intent_id() -> None:
    state = state_with([make_item("m1", intent_id="i1"), make_item("m2", intent_id="i2"), make_item("m3")])
    assert [i.id for i in get_items(state, {"intent_id": "i1"})] == ["m1"]


def test_filter_by_task_id() -> None:
    state = state_with([make_item("m1", task_id="t1"), make_item("m2", task_id="t2"), make_item("m3")])
    assert [i.id for i in get_items(state, {"task_id": "t1"})] == ["m1"]


def test_filter_excludes_items_without_intent_id() -> None:
    state = state_with([make_item("m1"), make_item("m2", intent_id="i1")])
    results = get_items(state, {"intent_id": "i1"})
    assert len(results) == 1 and results[0].id == "m2"


def test_filter_by_intent_ids_any() -> None:
    state = state_with([
        make_item("m1", intent_id="i1"), make_item("m2", intent_id="i2"),
        make_item("m3", intent_id="i3"), make_item("m4"),
    ])
    assert sorted(i.id for i in get_items(state, {"intent_ids": ["i1", "i3"]})) == ["m1", "m3"]


def test_filter_by_task_ids_any() -> None:
    state = state_with([make_item("m1", task_id="t1"), make_item("m2", task_id="t2"), make_item("m3", task_id="t3")])
    assert sorted(i.id for i in get_items(state, {"task_ids": ["t2", "t3"]})) == ["m2", "m3"]


def test_filter_excludes_items_without_task_id() -> None:
    state = state_with([make_item("m1"), make_item("m2", task_id="t1")])
    results = get_items(state, {"task_ids": ["t1"]})
    assert len(results) == 1 and results[0].id == "m2"


def test_filter_combines_intent_and_task_id() -> None:
    state = state_with([
        make_item("m1", intent_id="i1", task_id="t1"),
        make_item("m2", intent_id="i1", task_id="t2"),
        make_item("m3", intent_id="i2", task_id="t1"),
    ])
    assert [i.id for i in get_items(state, {"intent_id": "i1", "task_id": "t1"})] == ["m1"]


# === IntentFilter parent_id / is_root ======================================


def setup_intents():
    state = create_intent_state()
    for intent in [
        create_intent(id="i1", label="investigate", priority=0.9, owner="user:laz"),
        create_intent(id="i2", parent_id="i1", label="find associates", priority=0.7, owner="user:laz"),
        create_intent(id="i3", parent_id="i1", label="map finances", priority=0.8, owner="user:laz"),
    ]:
        state = apply_intent_command(state, {"type": "intent.create", "intent": intent}).state
    return state


def test_intent_filter_by_parent_id() -> None:
    assert sorted(i.id for i in get_intents(setup_intents(), {"parent_id": "i1"})) == ["i2", "i3"]


def test_intent_is_root_true() -> None:
    assert [i.id for i in get_intents(setup_intents(), {"is_root": True})] == ["i1"]


def test_intent_is_root_false() -> None:
    assert sorted(i.id for i in get_intents(setup_intents(), {"is_root": False})) == ["i2", "i3"]


def test_get_child_intents() -> None:
    assert sorted(i.id for i in get_child_intents(setup_intents(), "i1")) == ["i2", "i3"]


def test_get_child_intents_leaf() -> None:
    assert get_child_intents(setup_intents(), "i2") == []


# === TaskFilter parent_id / is_root ========================================


def setup_tasks():
    state = create_task_state()
    for task in [
        create_task(id="t1", intent_id="i1", action="search", priority=0.9),
        create_task(id="t2", intent_id="i1", parent_id="t1", action="parse_profile", priority=0.7),
        create_task(id="t3", intent_id="i1", parent_id="t1", action="extract_contacts", priority=0.6),
    ]:
        state = apply_task_command(state, {"type": "task.create", "task": task}).state
    return state


def test_task_filter_by_parent_id() -> None:
    assert sorted(t.id for t in get_tasks(setup_tasks(), {"parent_id": "t1"})) == ["t2", "t3"]


def test_task_is_root_true() -> None:
    assert [t.id for t in get_tasks(setup_tasks(), {"is_root": True})] == ["t1"]


def test_task_is_root_false() -> None:
    assert sorted(t.id for t in get_tasks(setup_tasks(), {"is_root": False})) == ["t2", "t3"]


def test_get_child_tasks() -> None:
    assert sorted(t.id for t in get_child_tasks(setup_tasks(), "t1")) == ["t2", "t3"]


def test_get_child_tasks_leaf() -> None:
    assert get_child_tasks(setup_tasks(), "t3") == []
