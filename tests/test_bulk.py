"""Port of tests/bulk.test.ts."""

from __future__ import annotations

from typing import Any

import pytest

from memex import GraphState, MemoryItem, apply_many, bulk_adjust_scores


def base_item(id: str, **overrides: Any) -> MemoryItem:
    base: dict[str, Any] = {
        "id": id, "scope": "test", "kind": "observation", "content": {},
        "author": "user:laz", "source_kind": "observed", "authority": 0.5,
    }
    base.update(overrides)
    return MemoryItem(**base)


def build_state(items: list[MemoryItem]) -> GraphState:
    return GraphState(items={i.id: i for i in items}, edges={})


# --- applyMany -------------------------------------------------------------


def test_apply_many_updates() -> None:
    state = build_state([base_item("m1", authority=0.5), base_item("m2", authority=0.6)])
    nxt, events = apply_many(state, {}, lambda item: {"authority": 0.9}, "system:eval")
    assert nxt.items["m1"].authority == 0.9
    assert nxt.items["m2"].authority == 0.9
    assert len(events) == 2
    assert all(e.type == "memory.updated" for e in events)


def test_apply_many_retracts_on_none() -> None:
    state = build_state([base_item("m1", authority=0.1), base_item("m2", authority=0.8)])
    nxt, events = apply_many(state, {}, lambda item: None if item.authority < 0.5 else {}, "system:cleanup")
    assert "m1" not in nxt.items
    assert "m2" in nxt.items
    assert len(events) == 1
    assert events[0].type == "memory.retracted"


def test_apply_many_skips_on_empty() -> None:
    state = build_state([base_item("m1"), base_item("m2")])
    nxt, events = apply_many(state, {}, lambda item: {}, "system:noop")
    assert nxt is state
    assert len(events) == 0


def test_apply_many_item_dependent_transform() -> None:
    state = build_state([base_item("m1", authority=0.8), base_item("m2", authority=0.4)])
    nxt, _ = apply_many(state, {}, lambda item: {"authority": item.authority * 0.5}, "system:decay")
    assert nxt.items["m1"].authority == pytest.approx(0.4)
    assert nxt.items["m2"].authority == pytest.approx(0.2)


def test_apply_many_conditional_retract_and_boost() -> None:
    state = build_state([
        base_item("m1", authority=0.1), base_item("m2", authority=0.7), base_item("m3", authority=0.3),
    ])
    nxt, events = apply_many(
        state, {}, lambda item: None if item.authority < 0.3 else {"authority": 1.0}, "system:evaluator"
    )
    assert "m1" not in nxt.items
    assert nxt.items["m2"].authority == 1.0
    assert nxt.items["m3"].authority == 1.0
    assert len(events) == 3


def test_apply_many_skips_already_retracted() -> None:
    state = build_state([base_item("m1", authority=0.1), base_item("m2", authority=0.1)])
    nxt, events = apply_many(state, {}, lambda item: None, "system:cleanup")
    assert len(nxt.items) == 0
    assert len(events) == 2


def test_apply_many_shallow_merges_meta() -> None:
    state = build_state([base_item("m1", meta={"agent_id": "agent:x", "session_id": "s1"})])
    nxt, _ = apply_many(state, {}, lambda item: {"meta": {"hot": True}}, "system:tagger")
    meta = nxt.items["m1"].meta
    assert meta is not None
    assert meta["hot"] is True
    assert meta["agent_id"] == "agent:x"
    assert meta["session_id"] == "s1"


def test_apply_many_filter_before_transform() -> None:
    state = build_state([base_item("m1", scope="a"), base_item("m2", scope="b")])
    nxt, _ = apply_many(state, {"scope": "a"}, lambda item: {"authority": 1.0}, "system:eval")
    assert nxt.items["m1"].authority == 1.0
    assert nxt.items["m2"].authority == 0.5


def test_apply_many_query_options_sort_limit() -> None:
    state = build_state([
        base_item("m1", authority=0.3), base_item("m2", authority=0.9), base_item("m3", authority=0.6),
    ])
    nxt, events = apply_many(
        state, {}, lambda item: {"meta": {"top": True}}, "system:tagger", None,
        {"sort": {"field": "authority", "order": "desc"}, "limit": 2},
    )
    assert nxt.items["m2"].meta["top"] is True
    assert nxt.items["m3"].meta["top"] is True
    assert nxt.items["m1"].meta is None
    assert len(events) == 2


def test_apply_many_does_not_mutate_original() -> None:
    state = build_state([base_item("m1", authority=0.5)])
    apply_many(state, {}, lambda item: {"authority": 1.0}, "test")
    assert state.items["m1"].authority == 0.5


# --- bulkAdjustScores ------------------------------------------------------


def test_bulk_adjust_authority() -> None:
    state = build_state([
        base_item("m1", authority=0.5), base_item("m2", authority=0.6), base_item("m3", authority=0.7),
    ])
    nxt, events = bulk_adjust_scores(
        state, {"range": {"authority": {"min": 0.5}}}, {"authority": -0.2}, "system:tuner", "decay"
    )
    assert nxt.items["m1"].authority == pytest.approx(0.3)
    assert nxt.items["m2"].authority == pytest.approx(0.4)
    assert nxt.items["m3"].authority == pytest.approx(0.5)
    assert len(events) == 3


def test_bulk_adjust_clamps_to_zero() -> None:
    state = build_state([base_item("m1", authority=0.1)])
    nxt, _ = bulk_adjust_scores(state, {}, {"authority": -0.5}, "system:tuner")
    assert nxt.items["m1"].authority == 0


def test_bulk_adjust_clamps_to_one() -> None:
    state = build_state([base_item("m1", authority=0.9)])
    nxt, _ = bulk_adjust_scores(state, {}, {"authority": 0.5}, "system:tuner")
    assert nxt.items["m1"].authority == 1


def test_bulk_adjust_undefined_importance_as_zero() -> None:
    state = build_state([base_item("m1")])
    nxt, _ = bulk_adjust_scores(state, {}, {"importance": 0.7}, "system:tuner")
    assert nxt.items["m1"].importance == 0.7


def test_bulk_adjust_no_matches() -> None:
    state = build_state([base_item("m1", scope="other")])
    nxt, events = bulk_adjust_scores(state, {"scope": "nonexistent"}, {"authority": 0.1}, "system:tuner")
    assert nxt is state
    assert len(events) == 0


def test_bulk_adjust_does_not_mutate_original() -> None:
    state = build_state([base_item("m1", authority=0.5)])
    bulk_adjust_scores(state, {}, {"authority": 0.3}, "system:tuner")
    assert state.items["m1"].authority == 0.5
