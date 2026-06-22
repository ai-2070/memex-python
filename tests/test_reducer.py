"""Port of tests/reducer.test.ts."""

from __future__ import annotations

import pytest

from memex import (
    DuplicateEdgeError,
    DuplicateMemoryError,
    EdgeNotFoundError,
    MemoryCreate,
    MemoryNotFoundError,
    apply_command,
    create_graph_state,
)
from support import make_edge, make_item, state_with

# --- memory.create ---------------------------------------------------------


def test_create_item_in_empty_state() -> None:
    item = make_item()
    state, events = apply_command(create_graph_state(), {"type": "memory.create", "item": item})
    assert state.items["m1"] == item
    assert len(events) == 1
    ev = events[0]
    assert ev.namespace == "memory"
    assert ev.type == "memory.created"
    assert ev.item == item
    assert ev.cause_type == "memory.create"


def test_create_returns_new_state_object() -> None:
    original = create_graph_state()
    state, _ = apply_command(original, MemoryCreate(item=make_item()))
    assert state is not original


def test_create_does_not_mutate_original() -> None:
    original = create_graph_state()
    apply_command(original, {"type": "memory.create", "item": make_item()})
    assert len(original.items) == 0


def test_create_duplicate_raises() -> None:
    state = state_with([make_item()])
    with pytest.raises(DuplicateMemoryError):
        apply_command(state, {"type": "memory.create", "item": make_item()})


# --- memory.update ---------------------------------------------------------


def test_update_authority() -> None:
    state = state_with([make_item()])
    nxt, events = apply_command(
        state,
        {"type": "memory.update", "item_id": "m1", "partial": {"authority": 0.5}, "author": "system:tuner"},
    )
    assert nxt.items["m1"].authority == 0.5
    assert events[0].type == "memory.updated"


def test_update_shallow_merges_content() -> None:
    state = state_with([make_item()])
    nxt, _ = apply_command(
        state,
        {"type": "memory.update", "item_id": "m1", "partial": {"content": {"newKey": "added"}}, "author": "test"},
    )
    assert nxt.items["m1"].content == {"key": "value", "nested": 1, "newKey": "added"}


def test_update_overwrites_existing_content_keys() -> None:
    state = state_with([make_item()])
    nxt, _ = apply_command(
        state,
        {"type": "memory.update", "item_id": "m1", "partial": {"content": {"key": "updated"}}, "author": "test"},
    )
    assert nxt.items["m1"].content["key"] == "updated"
    assert nxt.items["m1"].content["nested"] == 1


def test_update_ignores_id_in_partial() -> None:
    state = state_with([make_item()])
    nxt, _ = apply_command(
        state,
        {"type": "memory.update", "item_id": "m1", "partial": {"id": "sneaky-new-id"}, "author": "test"},
    )
    assert nxt.items["m1"].id == "m1"
    assert "sneaky-new-id" not in nxt.items


def test_update_shallow_merges_meta() -> None:
    state = state_with([make_item(meta={"agent_id": "agent:x", "session_id": "s1"})])
    nxt, _ = apply_command(
        state,
        {"type": "memory.update", "item_id": "m1", "partial": {"meta": {"tagged": True}}, "author": "test"},
    )
    meta = nxt.items["m1"].meta
    assert meta is not None
    assert meta["tagged"] is True
    assert meta["agent_id"] == "agent:x"
    assert meta["session_id"] == "s1"


def test_update_missing_raises() -> None:
    with pytest.raises(MemoryNotFoundError):
        apply_command(
            create_graph_state(),
            {"type": "memory.update", "item_id": "nope", "partial": {"authority": 0.1}, "author": "test"},
        )


def test_update_emits_fully_merged_item() -> None:
    state = state_with([make_item()])
    _, events = apply_command(
        state,
        {"type": "memory.update", "item_id": "m1", "partial": {"authority": 0.3, "importance": 0.7}, "author": "test"},
    )
    assert events[0].item.authority == 0.3
    assert events[0].item.importance == 0.7


def test_update_does_not_mutate_original() -> None:
    state = state_with([make_item()])
    apply_command(
        state,
        {"type": "memory.update", "item_id": "m1", "partial": {"authority": 0.1}, "author": "test"},
    )
    assert state.items["m1"].authority == 0.9


# --- memory.retract --------------------------------------------------------


def test_retract_removes_item() -> None:
    state = state_with([make_item()])
    nxt, events = apply_command(state, {"type": "memory.retract", "item_id": "m1", "author": "user:laz"})
    assert "m1" not in nxt.items
    assert events[0].type == "memory.retracted"


def test_retract_missing_raises() -> None:
    with pytest.raises(MemoryNotFoundError):
        apply_command(create_graph_state(), {"type": "memory.retract", "item_id": "nope", "author": "test"})


def test_retract_does_not_mutate_original() -> None:
    state = state_with([make_item()])
    apply_command(state, {"type": "memory.retract", "item_id": "m1", "author": "test"})
    assert "m1" in state.items


# --- memory.retract: orphan edge cleanup -----------------------------------


def test_retract_removes_edges_where_item_is_from() -> None:
    state = state_with(
        [make_item(id="m1"), make_item(id="m2")],
        [make_edge(edge_id="e1", from_="m1", to="m2")],
    )
    nxt, _ = apply_command(state, {"type": "memory.retract", "item_id": "m1", "author": "test"})
    assert len(nxt.edges) == 0


def test_retract_removes_edges_where_item_is_to() -> None:
    state = state_with(
        [make_item(id="m1"), make_item(id="m2")],
        [make_edge(edge_id="e1", from_="m2", to="m1")],
    )
    nxt, _ = apply_command(state, {"type": "memory.retract", "item_id": "m1", "author": "test"})
    assert len(nxt.edges) == 0


def test_retract_emits_edge_retracted_events() -> None:
    state = state_with(
        [make_item(id="m1"), make_item(id="m2")],
        [make_edge(edge_id="e1", from_="m1", to="m2")],
    )
    _, events = apply_command(state, {"type": "memory.retract", "item_id": "m1", "author": "test"})
    assert len(events) == 2
    assert events[0].type == "memory.retracted"
    assert events[1].type == "edge.retracted"
    assert events[1].edge.edge_id == "e1"


def test_retract_keeps_unrelated_edges() -> None:
    state = state_with(
        [make_item(id="m1"), make_item(id="m2"), make_item(id="m3")],
        [make_edge(edge_id="e1", from_="m2", to="m3")],
    )
    nxt, _ = apply_command(state, {"type": "memory.retract", "item_id": "m1", "author": "test"})
    assert len(nxt.edges) == 1
    assert "e1" in nxt.edges


# --- memory.update: None / absent content & meta (D5 semantics) ------------


def test_update_absent_content_key_is_preserved() -> None:
    # JS `undefined` ≡ key absent in Python; absent keys are never deleted (D5).
    state = state_with([make_item()])
    nxt, _ = apply_command(
        state,
        {"type": "memory.update", "item_id": "m1", "partial": {"content": {}}, "author": "test"},
    )
    assert nxt.items["m1"].content["key"] == "value"


def test_update_none_content_value_is_stored() -> None:
    # JS `null` ≡ Python None and is KEPT, not treated as a delete (D5).
    state = state_with([make_item()])
    nxt, _ = apply_command(
        state,
        {"type": "memory.update", "item_id": "m1", "partial": {"content": {"key": None}}, "author": "test"},
    )
    assert nxt.items["m1"].content["key"] is None


def test_update_absent_meta_key_is_preserved() -> None:
    state = state_with([make_item(meta={"agent_id": "agent:x", "session_id": "s1"})])
    nxt, _ = apply_command(
        state,
        {"type": "memory.update", "item_id": "m1", "partial": {"meta": {}}, "author": "test"},
    )
    assert nxt.items["m1"].meta["agent_id"] == "agent:x"


# --- edge.create -----------------------------------------------------------


def test_edge_create() -> None:
    edge = make_edge()
    state, events = apply_command(create_graph_state(), {"type": "edge.create", "edge": edge})
    assert state.edges["e1"] == edge
    ev = events[0]
    assert ev.namespace == "memory"
    assert ev.type == "edge.created"
    assert ev.edge == edge
    assert ev.cause_type == "edge.create"


def test_edge_create_duplicate_raises() -> None:
    state = state_with([], [make_edge()])
    with pytest.raises(DuplicateEdgeError):
        apply_command(state, {"type": "edge.create", "edge": make_edge()})


# --- edge.update -----------------------------------------------------------


def test_edge_update_weight() -> None:
    state = state_with([], [make_edge()])
    nxt, events = apply_command(
        state,
        {"type": "edge.update", "edge_id": "e1", "partial": {"weight": 0.5}, "author": "test"},
    )
    assert nxt.edges["e1"].weight == 0.5
    assert events[0].type == "edge.updated"


def test_edge_update_missing_raises() -> None:
    with pytest.raises(EdgeNotFoundError):
        apply_command(
            create_graph_state(),
            {"type": "edge.update", "edge_id": "nope", "partial": {"weight": 0.5}, "author": "test"},
        )


# --- edge.retract ----------------------------------------------------------


def test_edge_retract() -> None:
    state = state_with([], [make_edge()])
    nxt, events = apply_command(state, {"type": "edge.retract", "edge_id": "e1", "author": "test"})
    assert "e1" not in nxt.edges
    assert events[0].type == "edge.retracted"


def test_edge_retract_missing_raises() -> None:
    with pytest.raises(EdgeNotFoundError):
        apply_command(create_graph_state(), {"type": "edge.retract", "edge_id": "nope", "author": "test"})


# --- sequential operations -------------------------------------------------


def test_create_update_retract_sequence() -> None:
    all_events = []
    state = create_graph_state()
    res = apply_command(state, {"type": "memory.create", "item": make_item()})
    state = res.state
    all_events.extend(res.events)
    res = apply_command(state, {"type": "memory.update", "item_id": "m1", "partial": {"authority": 0.5}, "author": "test"})
    state = res.state
    all_events.extend(res.events)
    res = apply_command(state, {"type": "memory.retract", "item_id": "m1", "author": "test"})
    state = res.state
    all_events.extend(res.events)
    assert len(state.items) == 0
    assert len(all_events) == 3


def test_retract_one_of_two_removes_orphan_edge() -> None:
    state = create_graph_state()
    state = apply_command(state, {"type": "memory.create", "item": make_item(id="m1")}).state
    state = apply_command(state, {"type": "memory.create", "item": make_item(id="m2", scope="other")}).state
    state = apply_command(state, {"type": "edge.create", "edge": make_edge(edge_id="e1", from_="m1", to="m2")}).state
    state = apply_command(state, {"type": "memory.retract", "item_id": "m1", "author": "test"}).state
    assert len(state.items) == 1
    assert "m2" in state.items
    assert len(state.edges) == 0
