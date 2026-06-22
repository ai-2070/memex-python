"""Tests for the MemexStore OO facade (Python-specific; D9)."""

from __future__ import annotations

from memex import MemexStore
from memex.schemas import validate_command


def test_store_crud_roundtrip() -> None:
    store = MemexStore()
    a = store.create(scope="user:laz/general", kind="observation", content={"k": 1}, author="agent:x", source_kind="observed", authority=0.9)
    b = store.create(scope="user:laz/general", kind="assertion", content={"k": 2}, author="agent:y", source_kind="user_explicit", authority=0.4, parents=[a.id])
    assert len(store.items()) == 2
    assert [i.id for i in store.children(a.id)] == [b.id]
    assert [i.id for i in store.parents(b.id)] == [a.id]


def test_store_update_and_retract() -> None:
    store = MemexStore()
    item = store.create(scope="s", kind="observation", content={}, author="a", source_kind="observed", authority=0.5)
    store.update(item.id, {"authority": 0.95}, "sys")
    assert store.item(item.id).authority == 0.95
    store.retract(item.id, "sys")
    assert store.item(item.id) is None


def test_store_contradiction_flow() -> None:
    store = MemexStore()
    a = store.create(scope="s", kind="observation", content={}, author="x", source_kind="observed", authority=0.9)
    b = store.create(scope="s", kind="observation", content={}, author="y", source_kind="observed", authority=0.4)
    store.mark_contradiction(a.id, b.id, "detector")
    assert len(store.contradictions()) == 1
    store.resolve_contradiction(a.id, b.id, "resolver")
    assert len(store.contradictions()) == 0
    assert store.item(b.id).authority < 0.4  # loser authority lowered


def test_store_intent_task_graphs() -> None:
    store = MemexStore()
    intent = store.create_intent(label="find target", priority=0.9, owner="user:laz")
    task = store.create_task(intent_id=intent.id, action="search", priority=0.8)
    assert len(store.get_intents()) == 1
    assert len(store.get_tasks({"intent_id": intent.id})) == 1
    store.apply_task({"type": "task.start", "task_id": task.id})
    assert store.get_tasks()[0].status == "running"


def test_store_serialization_roundtrip() -> None:
    store = MemexStore()
    store.create(scope="s", kind="observation", content={"text": "hi"}, author="a", source_kind="observed", authority=0.5, id="m1")
    restored = MemexStore.loads(store.dumps())
    assert restored.item("m1").content == {"text": "hi"}


def test_store_transplant_roundtrip() -> None:
    src = MemexStore()
    a = src.create(scope="s", kind="observation", content={}, author="a", source_kind="observed", authority=0.5, id="m1")
    src.create(scope="s", kind="derivation", content={}, author="a", source_kind="derived_deterministic", authority=0.5, id="m2", parents=[a.id])
    slice = src.export_slice(memory_ids=["m1"], include_children=True)
    dest = MemexStore()
    report = dest.import_slice(slice)
    assert len(dest.items()) == 2
    assert sorted(report.created.memories) == ["m1", "m2"]


def test_schemas_validate_command() -> None:
    cmd = validate_command({"type": "memory.update", "item_id": "m1", "partial": {"authority": 0.5}, "author": "sys"})
    assert cmd.type == "memory.update"
    assert cmd.item_id == "m1"
