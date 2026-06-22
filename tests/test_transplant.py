"""Port of tests/transplant.test.ts + the transplant re-id section of
tests/cross-graph-fields.test.ts."""

from __future__ import annotations

import json
from typing import Any

import pytest

from memex import (
    Edge,
    MemoryItem,
    apply_command,
    apply_intent_command,
    apply_task_command,
    create_graph_state,
    create_intent,
    create_intent_state,
    create_task,
    create_task_state,
    export_slice,
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


def make_edge(id: str, frm: str, to: str, kind: str = "SUPPORTS") -> Edge:
    return Edge(
        edge_id=id, from_=frm, to=to, kind=kind, author="system:rule",
        source_kind="derived_deterministic", authority=0.8, active=True,
    )


def build_state():
    mem = create_graph_state()
    for item in [make_item("m1"), make_item("m2", parents=["m1"]), make_item("m3", parents=["m2"]), make_item("m4")]:
        mem = apply_command(mem, {"type": "memory.create", "item": item}).state
    mem = apply_command(mem, {"type": "edge.create", "edge": make_edge("e1", "m1", "m2")}).state
    mem = apply_command(mem, {"type": "edge.create", "edge": make_edge("e2", "m2", "m3")}).state

    intents = create_intent_state()
    intents = apply_intent_command(intents, {"type": "intent.create", "intent": create_intent(id="i1", label="find_kati", priority=0.9, owner="user:laz", root_memory_ids=["m1"])}).state

    tasks = create_task_state()
    tasks = apply_task_command(tasks, {"type": "task.create", "task": create_task(id="t1", intent_id="i1", action="search", priority=0.8, input_memory_ids=["m1"], output_memory_ids=["m2"])}).state

    return mem, intents, tasks


# === Export ================================================================


def test_export_specific_memory_ids() -> None:
    mem, intents, tasks = build_state()
    slice = export_slice(mem, intents, tasks, memory_ids=["m1"])
    assert len(slice.memories) == 1 and slice.memories[0].id == "m1"
    assert len(slice.edges) == 0
    assert len(slice.intents) == 0
    assert len(slice.tasks) == 0


def test_export_with_parents() -> None:
    mem, intents, tasks = build_state()
    slice = export_slice(mem, intents, tasks, memory_ids=["m3"], include_parents=True)
    assert sorted(m.id for m in slice.memories) == ["m1", "m2", "m3"]
    assert len(slice.edges) >= 2


def test_export_with_children() -> None:
    mem, intents, tasks = build_state()
    slice = export_slice(mem, intents, tasks, memory_ids=["m1"], include_children=True)
    assert sorted(m.id for m in slice.memories) == ["m1", "m2", "m3"]


def test_export_related_intents_and_tasks() -> None:
    mem, intents, tasks = build_state()
    slice = export_slice(mem, intents, tasks, memory_ids=["m1"], include_related_intents=True, include_related_tasks=True)
    assert len(slice.intents) == 1 and slice.intents[0].id == "i1"
    assert len(slice.tasks) == 1 and slice.tasks[0].id == "t1"


def test_export_by_intent_id_with_related_tasks() -> None:
    mem, intents, tasks = build_state()
    slice = export_slice(mem, intents, tasks, intent_ids=["i1"], include_related_tasks=True)
    assert len(slice.intents) == 1
    assert len(slice.tasks) == 1


def test_export_aliases_reverse_direction() -> None:
    mem = create_graph_state()
    mem = apply_command(mem, {"type": "memory.create", "item": make_item("m1")}).state
    mem = apply_command(mem, {"type": "memory.create", "item": make_item("m2")}).state
    mem = apply_command(mem, {"type": "edge.create", "edge": make_edge("e-alias", "m2", "m1", "ALIAS")}).state
    slice = export_slice(mem, create_intent_state(), create_task_state(), memory_ids=["m1"], include_aliases=True)
    assert sorted(m.id for m in slice.memories) == ["m1", "m2"]
    assert len(slice.edges) == 1 and slice.edges[0].edge_id == "e-alias"


def test_export_empty() -> None:
    mem, intents, tasks = build_state()
    slice = export_slice(mem, intents, tasks)
    assert len(slice.memories) == 0
    assert len(slice.edges) == 0
    assert len(slice.intents) == 0
    assert len(slice.tasks) == 0


# === Import (default: skip existing) =======================================


def test_import_into_empty() -> None:
    mem, intents, tasks = build_state()
    slice = export_slice(mem, intents, tasks, memory_ids=["m1", "m2"], include_related_intents=True, include_related_tasks=True)
    result = import_slice(create_graph_state(), create_intent_state(), create_task_state(), slice)
    assert len(result.mem_state.items) == 2
    assert len(result.intent_state.intents) == 1
    assert len(result.task_state.tasks) == 1
    assert len(result.report.created.memories) == 2
    assert len(result.report.skipped.memories) == 0


def test_import_skips_existing_ids() -> None:
    mem, intents, tasks = build_state()
    slice = export_slice(mem, intents, tasks, memory_ids=["m1", "m2"])
    target = create_graph_state()
    target = apply_command(target, {"type": "memory.create", "item": make_item("m1", authority=0.99)}).state
    result = import_slice(target, create_intent_state(), create_task_state(), slice)
    assert len(result.mem_state.items) == 2
    assert result.mem_state.items["m1"].authority == 0.99
    assert result.report.created.memories == ["m2"]
    assert result.report.skipped.memories == ["m1"]


def test_import_does_not_mutate_originals() -> None:
    empty = create_graph_state()
    slice = {"memories": [make_item("m1")], "edges": [], "intents": [], "tasks": []}
    import_slice(empty, create_intent_state(), create_task_state(), slice)
    assert len(empty.items) == 0


# === Import (shallow compare + re-id) ======================================


def test_import_detects_conflicts_without_reid() -> None:
    target = create_graph_state()
    target = apply_command(target, {"type": "memory.create", "item": make_item("m1", authority=0.99)}).state
    slice = {"memories": [make_item("m1", authority=0.1)], "edges": [], "intents": [], "tasks": []}
    result = import_slice(target, create_intent_state(), create_task_state(), slice, shallow_compare_existing=True)
    assert result.report.conflicts.memories == ["m1"]
    assert len(result.report.created.memories) == 0
    assert result.mem_state.items["m1"].authority == 0.99


def test_import_skips_when_shallow_equal() -> None:
    item = make_item("m1")
    target = create_graph_state()
    target = apply_command(target, {"type": "memory.create", "item": item}).state
    slice = {"memories": [item], "edges": [], "intents": [], "tasks": []}
    result = import_slice(target, create_intent_state(), create_task_state(), slice, shallow_compare_existing=True)
    assert result.report.skipped.memories == ["m1"]
    assert len(result.report.conflicts.memories) == 0


def test_import_reids_on_difference() -> None:
    mem_id = fake_uuid(1)
    target = create_graph_state()
    target = apply_command(target, {"type": "memory.create", "item": make_item(mem_id, authority=0.99)}).state
    slice = {"memories": [make_item(mem_id, authority=0.1)], "edges": [], "intents": [], "tasks": []}
    result = import_slice(target, create_intent_state(), create_task_state(), slice, shallow_compare_existing=True, re_id_on_difference=True)
    assert len(result.mem_state.items) == 2
    assert result.mem_state.items[mem_id].authority == 0.99
    assert len(result.report.created.memories) == 1
    new_id = result.report.created.memories[0]
    assert new_id != mem_id
    assert result.mem_state.items[new_id].authority == 0.1


# === Round-trip ============================================================


def test_roundtrip_full_chain() -> None:
    mem, intents, tasks = build_state()
    slice = export_slice(mem, intents, tasks, memory_ids=["m1"], include_children=True, include_related_intents=True, include_related_tasks=True)
    result = import_slice(create_graph_state(), create_intent_state(), create_task_state(), slice)
    assert len(result.mem_state.items) == 3
    assert len(result.mem_state.edges) == 2
    assert len(result.intent_state.intents) == 1
    assert len(result.task_state.tasks) == 1
    assert sorted(result.report.created.memories) == ["m1", "m2", "m3"]


def test_roundtrip_json_serializable() -> None:
    mem, intents, tasks = build_state()
    slice = export_slice(mem, intents, tasks, memory_ids=["m1", "m2"], include_related_intents=True, include_related_tasks=True)
    parsed = json.loads(json.dumps(slice.model_dump(by_alias=True, exclude_none=True)))
    result = import_slice(create_graph_state(), create_intent_state(), create_task_state(), parsed)
    assert len(result.mem_state.items) == 2
    assert len(result.intent_state.intents) == 1


# === Regression: reIdFor with non-UUIDv7 ids ===============================


def test_reid_non_uuidv7_with_created_at() -> None:
    non_uuid = "custom-id-not-uuidv7"
    target = create_graph_state()
    target = apply_command(target, {"type": "memory.create", "item": make_item(non_uuid, authority=0.9, created_at=1700000000000)}).state
    slice = {"memories": [make_item(non_uuid, authority=0.1, created_at=1700000000000)], "edges": [], "intents": [], "tasks": []}
    result = import_slice(target, create_intent_state(), create_task_state(), slice, shallow_compare_existing=True, re_id_on_difference=True)
    assert len(result.mem_state.items) == 2
    assert len(result.report.created.memories) == 1
    new_id = result.report.created.memories[0]
    assert new_id != non_uuid
    assert result.mem_state.items[new_id].authority == 0.1


def test_reid_non_uuidv7_without_created_at_raises() -> None:
    non_uuid = "custom-id-not-uuidv7"
    target = create_graph_state()
    target = apply_command(target, {"type": "memory.create", "item": make_item(non_uuid, authority=0.9)}).state
    slice = {"memories": [make_item(non_uuid, authority=0.1)], "edges": [], "intents": [], "tasks": []}
    with pytest.raises(ValueError, match=r"Cannot re-id.*created_at"):
        import_slice(target, create_intent_state(), create_task_state(), slice, shallow_compare_existing=True, re_id_on_difference=True)


# === Regression: shallowEqual with nested objects ==========================


def _import_compare(item_target: MemoryItem, item_slice: MemoryItem):
    target = create_graph_state()
    target = apply_command(target, {"type": "memory.create", "item": item_target}).state
    slice = {"memories": [item_slice], "edges": [], "intents": [], "tasks": []}
    return import_slice(target, create_intent_state(), create_task_state(), slice, shallow_compare_existing=True)


def test_shallow_equal_distinct_objects_same_value() -> None:
    id = fake_uuid(10)
    content = {"key": "theme", "value": "dark"}
    meta = {"source": "ui", "tags": {"env": "prod"}}
    result = _import_compare(
        make_item(id, content=dict(content), meta=dict(meta)),
        make_item(id, content=json.loads(json.dumps(content)), meta=json.loads(json.dumps(meta))),
    )
    assert result.report.skipped.memories == [id]
    assert len(result.report.conflicts.memories) == 0


def test_shallow_equal_different_key_order() -> None:
    id = fake_uuid(12)
    result = _import_compare(make_item(id, content={"a": 1, "b": 2}), make_item(id, content={"b": 2, "a": 1}))
    assert result.report.skipped.memories == [id]
    assert len(result.report.conflicts.memories) == 0


def test_shallow_equal_arrays() -> None:
    id = fake_uuid(13)
    result = _import_compare(make_item(id, parents=["p1", "p2"]), make_item(id, parents=["p1", "p2"]))
    assert result.report.skipped.memories == [id]
    assert len(result.report.conflicts.memories) == 0


def test_shallow_equal_nested_object_arrays() -> None:
    id = fake_uuid(14)
    result = _import_compare(
        make_item(id, content={"items": [{"a": 1}, {"b": 2}]}),
        make_item(id, content={"items": [{"a": 1}, {"b": 2}]}),
    )
    assert result.report.skipped.memories == [id]
    assert len(result.report.conflicts.memories) == 0


def test_shallow_equal_detects_nested_differences() -> None:
    id = fake_uuid(11)
    result = _import_compare(
        make_item(id, content={"key": "theme", "value": "dark"}),
        make_item(id, content={"key": "theme", "value": "light"}),
    )
    assert result.report.conflicts.memories == [id]
    assert len(result.report.skipped.memories) == 0


# === Cross-graph: parent_id rewriting on re-id =============================


def test_transplant_rewrites_intent_parent_id() -> None:
    parent_id = fake_uuid(1)
    child_id = fake_uuid(2)
    intent_state = create_intent_state()
    intent_state = apply_intent_command(intent_state, {"type": "intent.create", "intent": create_intent(id=parent_id, label="existing", priority=0.5, owner="user:laz")}).state

    slice = {
        "memories": [], "edges": [], "tasks": [],
        "intents": [
            create_intent(id=parent_id, label="different", priority=0.9, owner="user:laz"),
            create_intent(id=child_id, parent_id=parent_id, label="child", priority=0.7, owner="user:laz"),
        ],
    }
    result = import_slice(create_graph_state(), intent_state, create_task_state(), slice, skip_existing_ids=True, shallow_compare_existing=True, re_id_on_difference=True)

    new_parent_id = result.report.created.intents[0]
    assert new_parent_id != parent_id
    child = result.intent_state.intents[child_id]
    assert child.parent_id == new_parent_id


def test_transplant_rewrites_task_parent_id() -> None:
    parent_id = fake_uuid(1)
    child_id = fake_uuid(2)
    intent_id = fake_uuid(3)
    task_state = create_task_state()
    task_state = apply_task_command(task_state, {"type": "task.create", "task": create_task(id=parent_id, intent_id=intent_id, action="search", priority=0.5)}).state

    slice = {
        "memories": [], "edges": [], "intents": [],
        "tasks": [
            create_task(id=parent_id, intent_id=intent_id, action="different_search", priority=0.9),
            create_task(id=child_id, intent_id=intent_id, parent_id=parent_id, action="parse", priority=0.7),
        ],
    }
    result = import_slice(create_graph_state(), create_intent_state(), task_state, slice, skip_existing_ids=True, shallow_compare_existing=True, re_id_on_difference=True)

    new_parent_id = result.report.created.tasks[0]
    assert new_parent_id != parent_id
    child = result.task_state.tasks[child_id]
    assert child.parent_id == new_parent_id
