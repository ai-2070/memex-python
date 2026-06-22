"""Port of tests/edge-cases.test.ts."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from memex import (
    Edge,
    GraphState,
    MemoryItem,
    ScoredItem,
    _time,
    apply_command,
    apply_diversity,
    apply_many,
    bulk_adjust_scores,
    create_memory_item,
    decay_importance,
    filter_contradictions,
    get_alias_group,
    get_items,
    get_related_items,
    get_scored_items,
    get_stale_items,
    get_support_set,
    get_support_tree,
    mark_alias,
    mark_contradiction,
    replay_from_envelopes,
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


BASE = dict(scope="t", kind="observation", content={}, author="t", source_kind="observed")


# === Reducer edge cases ====================================================


def test_update_null_authority_sets_null() -> None:
    # null overwrites the field (only `undefined`/absent is stripped).
    state = state_with([make_item("m1", authority=0.9)])
    nxt, _ = apply_command(state, {"type": "memory.update", "item_id": "m1", "partial": {"authority": None}, "author": "test"})
    assert nxt.items["m1"].authority is None


def test_update_absent_key_does_not_overwrite() -> None:
    state = state_with([make_item("m1", authority=0.9, importance=0.7)])
    nxt, _ = apply_command(state, {"type": "memory.update", "item_id": "m1", "partial": {}, "author": "test"})
    assert nxt.items["m1"].importance == 0.7


def test_update_empty_partial_still_emits_event() -> None:
    state = state_with([make_item("m1", authority=0.9)])
    nxt, events = apply_command(state, {"type": "memory.update", "item_id": "m1", "partial": {}, "author": "test"})
    assert nxt.items["m1"].authority == 0.9
    assert len(events) == 1


def test_update_all_three_scores() -> None:
    state = state_with([make_item("m1", authority=0.5)])
    nxt, _ = apply_command(state, {"type": "memory.update", "item_id": "m1", "partial": {"authority": 0.1, "conviction": 0.2, "importance": 0.3}, "author": "test"})
    item = nxt.items["m1"]
    assert item.authority == 0.1 and item.conviction == 0.2 and item.importance == 0.3


# === Score validation boundaries ===========================================


def test_accepts_zero() -> None:
    create_memory_item(**BASE, authority=0)


def test_accepts_one() -> None:
    create_memory_item(**BASE, authority=1)


def test_rejects_just_below_zero() -> None:
    with pytest.raises(ValidationError):
        create_memory_item(**BASE, authority=-0.00001)


def test_rejects_just_above_one() -> None:
    with pytest.raises(ValidationError):
        create_memory_item(**BASE, authority=1.00001)


# === Query edge cases ======================================================


def test_resolve_path_deeply_nested() -> None:
    state = state_with([make_item("m1", meta={"a": {"b": {"c": {"d": "deep"}}}})])
    result = get_items(state, {"meta": {"a.b.c.d": "deep"}})
    assert len(result) == 1 and result[0].id == "m1"


def test_related_items_no_self_for_self_edge() -> None:
    state = state_with([make_item("m1")], [Edge(edge_id="e1", from_="m1", to="m1", kind="ABOUT", author="test", source_kind="observed", authority=1, active=True)])
    assert len(get_related_items(state, "m1")) == 0


def test_range_min_eq_max_exact_match() -> None:
    state = state_with([make_item("m1", authority=0.5), make_item("m2", authority=0.6)])
    result = get_items(state, {"range": {"authority": {"min": 0.5, "max": 0.5}}})
    assert len(result) == 1 and result[0].id == "m1"


def test_scored_min_score_exact_threshold() -> None:
    state = state_with([make_item("m1", authority=0.5)])
    result = get_scored_items(state, {"authority": 1}, {"min_score": 0.5})
    assert len(result) == 1 and result[0].score == 0.5


# === Integrity edge cases ==================================================


def test_stale_partial() -> None:
    state = state_with([make_item("m2"), make_item("m3", parents=["m1", "m2"])])
    stale = get_stale_items(state)
    assert len(stale) == 1 and stale[0].missing_parents == ["m1"]


def test_stale_multiple_missing() -> None:
    state = state_with([make_item("m3", parents=["m1", "m2"])])
    stale = get_stale_items(state)
    assert sorted(stale[0].missing_parents) == ["m1", "m2"]


def test_alias_group_cycle() -> None:
    state = state_with([make_item("m1"), make_item("m2"), make_item("m3")])
    state = mark_alias(state, "m1", "m2", "test").state
    state = mark_alias(state, "m2", "m3", "test").state
    state = mark_alias(state, "m3", "m1", "test").state
    group = get_alias_group(state, "m1")
    assert sorted(i.id for i in group) == ["m1", "m2", "m3"]


# === Retrieval edge cases ==================================================


def test_support_tree_cycle() -> None:
    state = state_with([make_item("m1", parents=["m2"]), make_item("m2", parents=["m1"])])
    tree = get_support_tree(state, "m1")
    assert tree is not None
    assert tree.item.id == "m1"
    assert len(tree.parents) == 1
    assert tree.parents[0].item.id == "m2"
    assert len(tree.parents[0].parents) == 1
    assert len(tree.parents[0].parents[0].parents) == 0


def test_support_set_cycle() -> None:
    state = state_with([make_item("m1", parents=["m2"]), make_item("m2", parents=["m1"])])
    s = get_support_set(state, "m1")
    assert sorted(i.id for i in s) == ["m1", "m2"]


def test_support_set_partial_chain() -> None:
    state = state_with([make_item("m1"), make_item("m3", parents=["m2"]), make_item("m4", parents=["m3", "m1"])])
    s = get_support_set(state, "m4")
    assert sorted(i.id for i in s) == ["m1", "m3", "m4"]


def test_filter_contradictions_neither_in_scored() -> None:
    state = state_with([make_item("m1"), make_item("m2"), make_item("m3")])
    marked, _ = mark_contradiction(state, "m1", "m2", "test")
    scored = [ScoredItem(item=marked.items["m3"], score=0.8)]
    filtered = filter_contradictions(marked, scored)
    assert len(filtered) == 1 and filtered[0].item.id == "m3"


def test_apply_diversity_empty() -> None:
    assert apply_diversity([], {"author_penalty": 0.5}) == []


# === Bulk edge cases =======================================================


def test_apply_many_empty_partial_no_events() -> None:
    state = state_with([make_item("m1"), make_item("m2")])
    nxt, events = apply_many(state, {}, lambda item: {}, "test")
    assert len(events) == 0
    assert nxt is state


def test_bulk_adjust_only_authority() -> None:
    state = state_with([make_item("m1", authority=0.5, conviction=0.8, importance=0.6)])
    nxt, _ = bulk_adjust_scores(state, {}, {"authority": 0.1}, "test")
    assert nxt.items["m1"].authority == pytest.approx(0.6)
    assert nxt.items["m1"].conviction == 0.8
    assert nxt.items["m1"].importance == 0.6


# === Created filter ========================================================


def test_created_before() -> None:
    item = create_memory_item(scope="test", kind="observation", content={"v": 1}, author="test", source_kind="observed", authority=0.5)
    result = get_items(state_with([item]), {"created": {"before": _time.now_ms() + 1000}})
    assert len(result) == 1


def test_created_after() -> None:
    item = create_memory_item(**BASE, authority=0.5)
    result = get_items(state_with([item]), {"created": {"after": _time.now_ms() - 10000}})
    assert len(result) == 1


def test_created_outside_range() -> None:
    item = create_memory_item(**BASE, authority=0.5)
    result = get_items(state_with([item]), {"created": {"after": _time.now_ms() + 60000}})
    assert len(result) == 0


def test_created_combined_with_filters() -> None:
    item = create_memory_item(scope="a", kind="observation", content={}, author="test", source_kind="observed", authority=0.5)
    state = state_with([item])
    past = _time.now_ms() - 10000
    assert len(get_items(state, {"scope": "a", "created": {"after": past}})) == 1
    assert len(get_items(state, {"scope": "b", "created": {"after": past}})) == 0


# === decayImportance =======================================================


def test_decay_importance_old_items() -> None:
    item = create_memory_item(**BASE, authority=0.5, importance=0.8)
    nxt, events = decay_importance(state_with([item]), -1000, 0.5, "system:decay")
    assert nxt.items[item.id].importance == pytest.approx(0.4)
    assert len(events) == 1


def test_decay_importance_skips_zero() -> None:
    item = create_memory_item(**BASE, authority=0.5, importance=0)
    _, events = decay_importance(state_with([item]), 0, 0.5, "system:decay")
    assert len(events) == 0


def test_decay_importance_skips_undefined() -> None:
    item = create_memory_item(**BASE, authority=0.5)
    _, events = decay_importance(state_with([item]), 0, 0.5, "system:decay")
    assert len(events) == 0


def test_decay_importance_skips_recent() -> None:
    item = create_memory_item(**BASE, authority=0.5, importance=0.9)
    nxt, events = decay_importance(state_with([item]), 999999999, 0.5, "system:decay")
    assert len(events) == 0
    assert nxt.items[item.id].importance == 0.9


# === Decay filter on getItems ==============================================


def fake_id_at_age(days_ago: float) -> str:
    ms = _time.now_ms() - int(days_ago * 86400000)
    h = format(ms, "x").rjust(12, "0")
    return f"{h[0:8]}-{h[8:12]}-7000-8000-000000000000"


def test_decay_filter_exponential() -> None:
    state = state_with([make_item(fake_id_at_age(0), authority=0.9), make_item(fake_id_at_age(1), authority=0.9), make_item(fake_id_at_age(3), authority=0.9)])
    result = get_items(state, {"decay": {"config": {"rate": 0.5, "interval": "day", "type": "exponential"}, "min": 0.4}})
    assert len(result) == 2


def test_decay_filter_aggressive() -> None:
    state = state_with([make_item(fake_id_at_age(2), authority=0.9), make_item(fake_id_at_age(5), authority=0.9)])
    result = get_items(state, {"decay": {"config": {"rate": 0.9, "interval": "day", "type": "exponential"}, "min": 0.5}})
    assert len(result) == 0


def test_decay_filter_gentle() -> None:
    state = state_with([make_item(fake_id_at_age(0), authority=0.5), make_item(fake_id_at_age(0.5), authority=0.5)])
    result = get_items(state, {"decay": {"config": {"rate": 0.1, "interval": "day", "type": "exponential"}, "min": 0.5}})
    assert len(result) == 2


def test_decay_filter_linear() -> None:
    state = state_with([make_item(fake_id_at_age(0), authority=0.9), make_item(fake_id_at_age(2), authority=0.9), make_item(fake_id_at_age(5), authority=0.9)])
    result = get_items(state, {"decay": {"config": {"rate": 0.3, "interval": "day", "type": "linear"}, "min": 0.1}})
    assert len(result) == 2


def test_decay_filter_step() -> None:
    state = state_with([make_item(fake_id_at_age(0.5), authority=0.9), make_item(fake_id_at_age(1.5), authority=0.9), make_item(fake_id_at_age(2.5), authority=0.9)])
    result = get_items(state, {"decay": {"config": {"rate": 0.5, "interval": "day", "type": "step"}, "min": 0.3}})
    assert len(result) == 2


def test_decay_filter_combined() -> None:
    recent1 = create_memory_item(scope="a", kind="observation", content={}, author="test", source_kind="observed", authority=0.9)
    old = make_item(fake_id_at_age(5), authority=0.9, scope="a")
    recent2 = create_memory_item(scope="b", kind="observation", content={}, author="test", source_kind="observed", authority=0.9)
    state = state_with([recent1, old, recent2])
    result = get_items(state, {"scope": "a", "decay": {"config": {"rate": 0.5, "interval": "day", "type": "exponential"}, "min": 0.1}})
    assert len(result) == 1 and result[0].id == recent1.id


def test_decay_filter_hourly() -> None:
    def hour_id(hours_ago: int) -> str:
        ms = _time.now_ms() - hours_ago * 3600000
        h = format(ms, "x").rjust(12, "0")
        return f"{h[0:8]}-{h[8:12]}-7000-8000-000000000000"

    state = state_with([make_item(hour_id(1), authority=0.9), make_item(hour_id(10), authority=0.9)])
    result = get_items(state, {"decay": {"config": {"rate": 0.2, "interval": "hour", "type": "exponential"}, "min": 0.5}})
    assert len(result) == 1


# === Replay edge cases =====================================================


def test_replay_duplicate_timestamps_stable() -> None:
    envelopes = [
        {"id": "ev1", "namespace": "memory", "type": "memory.create", "ts": "2026-01-01T00:00:00.000Z", "payload": {"type": "memory.create", "item": make_item("m1")}},
        {"id": "ev2", "namespace": "memory", "type": "memory.create", "ts": "2026-01-01T00:00:00.000Z", "payload": {"type": "memory.create", "item": make_item("m2")}},
    ]
    state, _, _ = replay_from_envelopes(envelopes)
    assert len(state.items) == 2
    assert "m1" in state.items and "m2" in state.items
