"""Port of tests/edge-cases-v2.test.ts."""

from __future__ import annotations

from typing import Any

import pytest

from memex import (
    Edge,
    GraphState,
    InvalidTaskTransitionError,
    MemoryItem,
    ScoredItem,
    _time,
    apply_command,
    apply_intent_command,
    apply_task_command,
    create_graph_state,
    create_intent,
    create_intent_state,
    create_task,
    create_task_state,
    export_slice,
    extract_timestamp,
    get_items,
    get_scored_items,
    import_slice,
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


def state_with(items: list[MemoryItem], edges: list[Edge] | None = None) -> GraphState:
    return GraphState(items={i.id: i for i in items}, edges={e.edge_id: e for e in (edges or [])})


def fake_id_at_ms(ms: int) -> str:
    h = format(ms, "x").rjust(12, "0")
    return f"{h[0:8]}-{h[8:12]}-7000-8000-000000000000"


# === 1. Export with circular parents =======================================


def test_export_circular_parents_no_loop() -> None:
    state = state_with([make_item("m1", parents=["m2"]), make_item("m2", parents=["m1"])])
    slice = export_slice(state, create_intent_state(), create_task_state(), memory_ids=["m1"], include_parents=True)
    assert sorted(m.id for m in slice.memories) == ["m1", "m2"]


def test_export_circular_children_no_loop() -> None:
    state = state_with([make_item("m1", parents=["m2"]), make_item("m2", parents=["m1"])])
    slice = export_slice(state, create_intent_state(), create_task_state(), memory_ids=["m1"], include_children=True)
    assert sorted(m.id for m in slice.memories) == ["m1", "m2"]


# === 2. Transplant reIdFor validity ========================================


def test_reid_timestamp_is_original_plus_one() -> None:
    original_ms = _time.now_ms() - 5000
    original_id = fake_id_at_ms(original_ms)
    target = create_graph_state()
    target = apply_command(target, {"type": "memory.create", "item": make_item(original_id, authority=0.1)}).state
    slice = {"memories": [make_item(original_id, authority=0.9)], "edges": [], "intents": [], "tasks": []}
    result = import_slice(target, create_intent_state(), create_task_state(), slice, shallow_compare_existing=True, re_id_on_difference=True)
    new_id = result.report.created.memories[0]
    assert extract_timestamp(new_id) == original_ms + 1


# === 3. Intent: update on terminal states ==================================


def test_update_completed_intent() -> None:
    state = create_intent_state()
    state = apply_intent_command(state, {"type": "intent.create", "intent": create_intent(id="i1", label="test", priority=0.5, owner="user:laz")}).state
    state = apply_intent_command(state, {"type": "intent.complete", "intent_id": "i1", "author": "test"}).state
    nxt, _ = apply_intent_command(state, {"type": "intent.update", "intent_id": "i1", "partial": {"description": "added after completion"}, "author": "test"})
    assert nxt.intents["i1"].description == "added after completion"
    assert nxt.intents["i1"].status == "completed"


def test_update_cancelled_intent() -> None:
    state = create_intent_state()
    state = apply_intent_command(state, {"type": "intent.create", "intent": create_intent(id="i1", label="test", priority=0.5, owner="user:laz")}).state
    state = apply_intent_command(state, {"type": "intent.cancel", "intent_id": "i1", "author": "test"}).state
    nxt, _ = apply_intent_command(state, {"type": "intent.update", "intent_id": "i1", "partial": {"meta": {"reason": "post-mortem note"}}, "author": "test"})
    assert nxt.intents["i1"].meta["reason"] == "post-mortem note"


# === 4. Task state machine edge cases ======================================


def test_task_fail_on_pending_raises() -> None:
    state = create_task_state()
    state = apply_task_command(state, {"type": "task.create", "task": create_task(id="t1", intent_id="i1", action="test", priority=0.5)}).state
    with pytest.raises(InvalidTaskTransitionError):
        apply_task_command(state, {"type": "task.fail", "task_id": "t1", "error": "oops"})


def test_task_fail_on_cancelled_raises() -> None:
    state = create_task_state()
    state = apply_task_command(state, {"type": "task.create", "task": create_task(id="t1", intent_id="i1", action="test", priority=0.5)}).state
    state = apply_task_command(state, {"type": "task.cancel", "task_id": "t1"}).state
    with pytest.raises(InvalidTaskTransitionError):
        apply_task_command(state, {"type": "task.fail", "task_id": "t1", "error": "oops"})


def test_task_update_on_cancelled() -> None:
    state = create_task_state()
    state = apply_task_command(state, {"type": "task.create", "task": create_task(id="t1", intent_id="i1", action="test", priority=0.5)}).state
    state = apply_task_command(state, {"type": "task.cancel", "task_id": "t1"}).state
    nxt, _ = apply_task_command(state, {"type": "task.update", "task_id": "t1", "partial": {"meta": {"cancelled_reason": "no longer needed"}}, "author": "test"})
    assert nxt.tasks["t1"].meta["cancelled_reason"] == "no longer needed"


# === 5. Reducer: nested undefined in content/meta (D5: undefined ≡ absent) ==


def test_content_undefined_key_preserved() -> None:
    # JS {a: undefined, c: 3} maps to Python {c: 3}; `a` is preserved.
    state = state_with([make_item("m1", content={"a": 1, "b": 2})])
    nxt, _ = apply_command(state, {"type": "memory.update", "item_id": "m1", "partial": {"content": {"c": 3}}, "author": "test"})
    content = nxt.items["m1"].content
    assert content["a"] == 1
    assert content["b"] == 2
    assert content["c"] == 3


def test_meta_undefined_key_preserved() -> None:
    state = state_with([make_item("m1", meta={"agent_id": "agent:x", "session_id": "s1"})])
    nxt, _ = apply_command(state, {"type": "memory.update", "item_id": "m1", "partial": {"meta": {"tag": "new"}}, "author": "test"})
    meta = nxt.items["m1"].meta
    assert meta["agent_id"] == "agent:x"
    assert meta["session_id"] == "s1"
    assert meta["tag"] == "new"


# === 6. Decay with future items (clock skew) ===============================


def test_future_item_no_decay_boost() -> None:
    future_id = fake_id_at_ms(_time.now_ms() + 60000)
    state = state_with([make_item(future_id, authority=1.0)])
    result = get_scored_items(state, {"authority": 1.0, "decay": {"rate": 0.5, "interval": "day", "type": "exponential"}})
    assert result[0].score == 1.0


def test_future_item_passes_decay_filter() -> None:
    future_id = fake_id_at_ms(_time.now_ms() + 60000)
    state = state_with([make_item(future_id)])
    result = get_items(state, {"decay": {"config": {"rate": 0.5, "interval": "day", "type": "exponential"}, "min": 0.5}})
    assert len(result) == 1


# === 7. surfaceContradictions immutability =================================


def test_surface_does_not_mutate_input() -> None:
    state = state_with([make_item("m1", authority=0.9), make_item("m2", authority=0.7)])
    marked, _ = mark_contradiction(state, "m1", "m2", "system:detector")
    original = [ScoredItem(item=marked.items["m1"], score=0.9), ScoredItem(item=marked.items["m2"], score=0.7)]
    surface_contradictions(marked, original)
    assert original[0].contradicted_by is None
    assert original[1].contradicted_by is None
    assert original[0].score == 0.9


# === 8. Export/import circular parents round-trip ==========================


def test_transplant_circular_parents_round_trip() -> None:
    state = state_with([make_item("m1", parents=["m2"]), make_item("m2", parents=["m1"])])
    slice = export_slice(state, create_intent_state(), create_task_state(), memory_ids=["m1"], include_parents=True)
    result = import_slice(create_graph_state(), create_intent_state(), create_task_state(), slice)
    assert len(result.mem_state.items) == 2
    assert result.mem_state.items["m1"].parents == ["m2"]
    assert result.mem_state.items["m2"].parents == ["m1"]
