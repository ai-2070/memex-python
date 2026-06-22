"""Port of tests/bugfix-reid-ordering.test.ts."""

from __future__ import annotations

from typing import Any

from memex import (
    MemoryItem,
    apply_command,
    apply_intent_command,
    apply_task_command,
    create_graph_state,
    create_intent,
    create_intent_state,
    create_task,
    create_task_state,
    import_slice,
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


def test_remaps_memory_parents_parent_after_child() -> None:
    parent_id = fake_uuid(1)
    child_id = fake_uuid(2)
    target = create_graph_state()
    target = apply_command(target, {"type": "memory.create", "item": make_item(parent_id, authority=0.99)}).state
    target = apply_command(target, {"type": "memory.create", "item": make_item(child_id, authority=0.99)}).state

    slice = {
        "memories": [
            make_item(child_id, authority=0.1, parents=[parent_id]),
            make_item(parent_id, authority=0.1),
        ],
        "edges": [], "intents": [], "tasks": [],
    }
    result = import_slice(target, create_intent_state(), create_task_state(), slice, shallow_compare_existing=True, re_id_on_difference=True)

    assert len(result.report.created.memories) == 2
    imported = [i for i in result.mem_state.items.values() if i.authority == 0.1]
    new_parent = next(i for i in imported if not i.parents)
    new_child = next(i for i in imported if i.parents)
    assert new_child.parents == [new_parent.id]
    assert new_child.parents != [parent_id]
    assert result.mem_state.items[parent_id].authority == 0.99
    assert result.mem_state.items[child_id].authority == 0.99


def test_remaps_intent_parent_id_parent_after_child() -> None:
    parent_id = fake_uuid(1)
    child_id = fake_uuid(2)
    target_intents = create_intent_state()
    target_intents = apply_intent_command(target_intents, {"type": "intent.create", "intent": create_intent(id=parent_id, label="existing-parent", priority=0.99, owner="user:laz")}).state
    target_intents = apply_intent_command(target_intents, {"type": "intent.create", "intent": create_intent(id=child_id, label="existing-child", priority=0.99, owner="user:laz")}).state

    slice = {
        "memories": [], "edges": [], "tasks": [],
        "intents": [
            create_intent(id=child_id, parent_id=parent_id, label="imported-child", priority=0.1, owner="user:laz"),
            create_intent(id=parent_id, label="imported-parent", priority=0.1, owner="user:laz"),
        ],
    }
    result = import_slice(create_graph_state(), target_intents, create_task_state(), slice, shallow_compare_existing=True, re_id_on_difference=True)

    assert len(result.report.created.intents) == 2
    all_intents = list(result.intent_state.intents.values())
    new_parent = next(i for i in all_intents if i.label == "imported-parent")
    new_child = next(i for i in all_intents if i.label == "imported-child")
    assert new_child.parent_id == new_parent.id
    assert new_child.parent_id != parent_id


def test_remaps_task_parent_id_parent_after_child() -> None:
    intent_id = fake_uuid(1)
    parent_id = fake_uuid(2)
    child_id = fake_uuid(3)
    target_intents = create_intent_state()
    target_intents = apply_intent_command(target_intents, {"type": "intent.create", "intent": create_intent(id=intent_id, label="intent", priority=0.5, owner="user:laz")}).state

    target_tasks = create_task_state()
    target_tasks = apply_task_command(target_tasks, {"type": "task.create", "task": create_task(id=parent_id, intent_id=intent_id, action="existing-parent", priority=0.99)}).state
    target_tasks = apply_task_command(target_tasks, {"type": "task.create", "task": create_task(id=child_id, intent_id=intent_id, action="existing-child", priority=0.99)}).state

    slice = {
        "memories": [], "edges": [], "intents": [],
        "tasks": [
            create_task(id=child_id, intent_id=intent_id, parent_id=parent_id, action="imported-child", priority=0.1),
            create_task(id=parent_id, intent_id=intent_id, action="imported-parent", priority=0.1),
        ],
    }
    result = import_slice(create_graph_state(), target_intents, target_tasks, slice, shallow_compare_existing=True, re_id_on_difference=True)

    assert len(result.report.created.tasks) == 2
    all_tasks = list(result.task_state.tasks.values())
    new_parent = next(t for t in all_tasks if t.action == "imported-parent")
    new_child = next(t for t in all_tasks if t.action == "imported-child")
    assert new_child.parent_id == new_parent.id
    assert new_child.parent_id != parent_id
