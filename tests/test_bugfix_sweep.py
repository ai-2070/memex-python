"""Port of tests/bugfix-sweep.test.ts."""

from __future__ import annotations

from typing import Any

import pytest

from memex import (
    DuplicateMemoryError,
    GraphState,
    InvalidTimestampError,
    MemoryItem,
    MemoryNotFoundError,
    apply_command,
    cascade_retract,
    create_edge,
    create_event_envelope,
    create_graph_state,
    create_intent_state,
    create_memory_item,
    create_task_state,
    extract_timestamp,
    get_aliases,
    get_items_by_budget,
    get_support_set,
    import_slice,
    mark_alias,
    mark_contradiction,
    merge_item,
    replay_commands,
    replay_from_envelopes,
    resolve_contradiction,
    smart_retrieve,
)


def mk_item(id: str, **overrides: Any) -> MemoryItem:
    base: dict[str, Any] = {
        "id": id, "scope": "test", "kind": "observation", "content": {},
        "author": "agent:a", "source_kind": "observed", "authority": 0.5,
    }
    base.update(overrides)
    return create_memory_item(**base)


def env(env_id: str, ts: str, item: MemoryItem) -> dict[str, Any]:
    return {"id": env_id, "namespace": "memory", "type": "memory.create", "ts": ts,
            "payload": {"type": "memory.create", "item": item}}


# === zero-cost items =======================================================


def test_smart_retrieve_zero_cost() -> None:
    state = create_graph_state()
    for i in range(3):
        item = mk_item(f"0190{str(i).rjust(4, '0')}-0000-7000-8000-000000000000", importance=0.5)
        state = apply_command(state, {"type": "memory.create", "item": item}).state
    out = smart_retrieve(state, budget=10, cost_fn=lambda i: 0, weights={"importance": 1})
    assert len(out) == 3


def test_smart_retrieve_rejects_negative() -> None:
    state = create_graph_state()
    state = apply_command(state, {"type": "memory.create", "item": mk_item("01900000-0000-7000-8000-000000000001")}).state
    with pytest.raises(ValueError):
        smart_retrieve(state, budget=10, cost_fn=lambda i: -1, weights={"importance": 1})


def test_budget_zero_cost() -> None:
    state = create_graph_state()
    state = apply_command(state, {"type": "memory.create", "item": mk_item("01900000-0000-7000-8000-000000000002", importance=0.9)}).state
    out = get_items_by_budget(state, budget=5, cost_fn=lambda i: 0, weights={"importance": 1})
    assert len(out) == 1


def test_smart_retrieve_zero_cost_after_exhaustion() -> None:
    state = create_graph_state()
    ids = [f"01900000-0000-7000-8000-00000000f00{i}" for i in range(1, 6)]
    for i, id_ in enumerate(ids):
        state = apply_command(state, {"type": "memory.create", "item": mk_item(id_, importance=0.9 if i < 2 else 0.5)}).state
    expensive = {ids[0], ids[1]}
    out = smart_retrieve(state, budget=10, cost_fn=lambda item: 5 if item.id in expensive else 0, weights={"importance": 1})
    assert len(out) == 5


def test_budget_zero_cost_after_exhaustion() -> None:
    state = create_graph_state()
    ids = [f"01900000-0000-7000-8000-00000000b00{i}" for i in range(1, 4)]
    for i, id_ in enumerate(ids):
        state = apply_command(state, {"type": "memory.create", "item": mk_item(id_, authority=0.9 if i == 0 else 0.5)}).state
    out = get_items_by_budget(state, budget=5, cost_fn=lambda item: 5 if item.id == ids[0] else 0, weights={"authority": 1})
    assert len(out) == 3


# === transplant shallowEqual nested arrays =================================


def test_shallow_equal_nested_arrays_skip() -> None:
    id_ = "01900000-0000-7000-8000-0000000000aa"
    state = apply_command(create_graph_state(), {"type": "memory.create", "item": mk_item(id_, content={"tags": [["x"], ["y", "z"]]})}).state
    same = mk_item(id_, content={"tags": [["x"], ["y", "z"]]})
    result = import_slice(state, create_intent_state(), create_task_state(), {"memories": [same], "intents": [], "tasks": [], "edges": []}, shallow_compare_existing=True, re_id_on_difference=True)
    assert same.id in result.report.skipped.memories
    assert len(result.report.created.memories) == 0


def test_shallow_equal_nested_arrays_conflict() -> None:
    id_ = "01900000-0000-7000-8000-0000000000ab"
    state = apply_command(create_graph_state(), {"type": "memory.create", "item": mk_item(id_, content={"tags": [["x"]]})}).state
    diff = mk_item(id_, content={"tags": [["y"]]})
    result = import_slice(state, create_intent_state(), create_task_state(), {"memories": [diff], "intents": [], "tasks": [], "edges": []}, shallow_compare_existing=True)
    assert diff.id in result.report.conflicts.memories


# === replay strict ISO sorting & validation ================================


def test_replay_orders_by_instant_across_timezones() -> None:
    id1 = "01900000-0000-7000-8000-000000000101"
    id2 = "01900000-0000-7000-8000-000000000102"
    env_early = env("e1", "2024-01-01T09:00:00Z", mk_item(id1))
    env_late = env("e2", "2024-01-01T10:00:00+02:00", mk_item(id2))  # 08:00Z
    result = replay_from_envelopes([env_early, env_late])
    assert result.events[0].item.id == id2
    assert result.events[1].item.id == id1


def test_replay_collects_unparsable_ts() -> None:
    e = env("e", "not-a-date", mk_item("01900000-0000-7000-8000-000000000103"))
    e2 = env("f", "2024-01-01T00:00:00Z", mk_item("01900000-0000-7000-8000-0000000001aa"))
    result = replay_from_envelopes([e, e2])
    assert len(result.state.items) == 1
    assert len(result.skipped) == 1
    assert result.skipped[0].envelope is e
    assert type(result.skipped[0].error).__name__ == "InvalidTimestampError"


@pytest.mark.parametrize("bad", [
    "Jan 1, 2024",
    "2024/01/01 10:00:00",
    "2024-01-01 10:00:00Z",
    "2024-01-01T10:00:00",
    "2024-01-01T10:00:00+0200",
])
def test_replay_skips_non_iso(bad: str) -> None:
    e = env("e", bad, mk_item("01900000-0000-7000-8000-000000000104"))
    result = replay_from_envelopes([e])
    assert len(result.skipped) == 1
    assert type(result.skipped[0].error).__name__ == "InvalidTimestampError"


@pytest.mark.parametrize("bad", [
    "2024-01-01T00:00:00.0001Z",
    "2024-01-01T00:00:00.000001Z",
    "2024-01-01T00:00:00.000000001Z",
])
def test_replay_skips_sub_millisecond(bad: str) -> None:
    e = env("e", bad, mk_item("01900000-0000-7000-8000-000000000110"))
    assert len(replay_from_envelopes([e]).skipped) == 1


@pytest.mark.parametrize("bad", [
    "2024-02-30T00:00:00Z", "2024-02-31T00:00:00Z", "2023-02-29T00:00:00Z",
    "2024-13-01T00:00:00Z", "2024-00-01T00:00:00Z", "2024-04-31T00:00:00Z",
    "2024-01-32T00:00:00Z", "2024-01-00T00:00:00Z", "2024-01-01T24:00:00Z",
    "2024-01-01T00:60:00Z", "2024-01-01T00:00:61Z",
])
def test_replay_skips_impossible_dates(bad: str) -> None:
    e = env("e", bad, mk_item("01900000-0000-7000-8000-000000000111"))
    assert len(replay_from_envelopes([e]).skipped) == 1


def test_replay_handles_years_0000_0099() -> None:
    old_env = env("a", "0050-01-01T00:00:00Z", mk_item("01900000-0000-7000-8000-000000000120"))
    modern_env = env("b", "1950-01-01T00:00:00Z", mk_item("01900000-0000-7000-8000-000000000121"))
    result = replay_from_envelopes([modern_env, old_env])
    assert len(result.skipped) == 0
    assert result.events[0].item.id == "01900000-0000-7000-8000-000000000120"


def test_replay_accepts_feb29_leap() -> None:
    e = env("e", "2024-02-29T12:00:00.500Z", mk_item("01900000-0000-7000-8000-000000000112"))
    replay_from_envelopes([e])  # does not raise


def test_replay_accepts_z_and_offset() -> None:
    ez = env("a", "2024-01-01T00:00:00.000Z", mk_item("01900000-0000-7000-8000-000000000105"))
    eo = env("b", "2024-01-01T02:00:00+02:00", mk_item("01900000-0000-7000-8000-000000000106"))
    replay_from_envelopes([ez, eo])  # does not raise


# === mergeItem does not rewrite created_at =================================


def test_merge_preserves_created_at() -> None:
    item = mk_item("01900000-0000-7000-8000-000000000201", created_at=1_700_000_000_000)
    merged = merge_item(item, {"authority": 0.9, "created_at": 1})
    assert merged.created_at == 1_700_000_000_000
    assert merged.authority == 0.9


def test_memory_update_cannot_rewrite_created_at() -> None:
    item = mk_item("01900000-0000-7000-8000-000000000202", created_at=1_700_000_000_000)
    state = apply_command(create_graph_state(), {"type": "memory.create", "item": item}).state
    res = apply_command(state, {"type": "memory.update", "item_id": item.id, "partial": {"created_at": 42}, "author": "tester"})
    assert res.state.items[item.id].created_at == 1_700_000_000_000


# === markAlias / markContradiction self-reference ==========================


def test_mark_alias_self_noop() -> None:
    id_ = "01900000-0000-7000-8000-000000000301"
    state = apply_command(create_graph_state(), {"type": "memory.create", "item": mk_item(id_)}).state
    result = mark_alias(state, id_, id_, "tester")
    assert len(result.events) == 0
    assert result.state is state


def test_mark_contradiction_self_records_edge() -> None:
    id_ = "01900000-0000-7000-8000-000000000302"
    state = apply_command(create_graph_state(), {"type": "memory.create", "item": mk_item(id_)}).state
    result = mark_contradiction(state, id_, id_, "tester")
    assert len(result.events) == 1
    contradicts = [e for e in result.state.edges.values() if e.kind == "CONTRADICTS"]
    assert len(contradicts) == 1
    assert contradicts[0].from_ == id_ and contradicts[0].to == id_


def test_mark_alias_self_no_pollution() -> None:
    id_ = "01900000-0000-7000-8000-000000000303"
    state = apply_command(create_graph_state(), {"type": "memory.create", "item": mk_item(id_)}).state
    state = mark_alias(state, id_, id_, "tester").state
    assert get_aliases(state, id_) == []


# === soft-failure semantics ================================================


def test_create_edge_self_ref() -> None:
    edge = create_edge(from_="m1", to="m1", kind="CONTRADICTS", author="agent:detector", source_kind="derived_deterministic", authority=1)
    assert edge.from_ == "m1"
    assert edge.to == "m1"
    assert isinstance(edge.edge_id, str)
    assert edge.active is True


def test_resolve_contradiction_noop_no_edge() -> None:
    a = "01900000-0000-7000-8000-000000000401"
    b = "01900000-0000-7000-8000-000000000402"
    state = create_graph_state()
    state = apply_command(state, {"type": "memory.create", "item": mk_item(a, authority=0.9)}).state
    state = apply_command(state, {"type": "memory.create", "item": mk_item(b, authority=0.7)}).state
    result = resolve_contradiction(state, a, b, "agent:resolver")
    assert len(result.events) == 0
    assert result.state.items[a].authority == 0.9
    assert result.state.items[b].authority == 0.7
    assert [e for e in result.state.edges.values() if e.kind == "SUPERSEDES"] == []


def test_resolve_contradiction_duplicate_calls() -> None:
    a = "01900000-0000-7000-8000-000000000403"
    b = "01900000-0000-7000-8000-000000000404"
    state = create_graph_state()
    state = apply_command(state, {"type": "memory.create", "item": mk_item(a, authority=0.9)}).state
    state = apply_command(state, {"type": "memory.create", "item": mk_item(b, authority=0.7)}).state
    state = mark_contradiction(state, a, b, "detector").state
    r1 = resolve_contradiction(state, a, b, "agent:resolver")
    assert len(r1.events) > 0
    r2 = resolve_contradiction(r1.state, a, b, "agent:resolver")
    assert len(r2.events) == 0
    assert r2.state is r1.state


# === bulk replay soft ======================================================


def test_replay_mix_good_bad_envelopes() -> None:
    good1 = mk_item("01900000-0000-7000-8000-000000000701")
    good2 = mk_item("01900000-0000-7000-8000-000000000702")
    envs = [
        env("a", "2024-01-01T00:00:00Z", good1),
        env("b", "garbage", mk_item("x")),
        env("c", "2024-01-02T00:00:00Z", good2),
    ]
    result = replay_from_envelopes(envs)
    assert len(result.state.items) == 2
    assert good1.id in result.state.items and good2.id in result.state.items
    assert len(result.skipped) == 1
    assert isinstance(result.skipped[0].error, InvalidTimestampError)


def test_replay_records_apply_failures() -> None:
    item = mk_item("01900000-0000-7000-8000-000000000703")
    envs = [
        env("a", "2024-01-01T00:00:00Z", item),
        env("b", "2024-01-02T00:00:00Z", item),  # duplicate id
        env("c", "2024-01-03T00:00:00Z", mk_item("01900000-0000-7000-8000-000000000704")),
    ]
    result = replay_from_envelopes(envs)
    assert len(result.state.items) == 2
    assert len(result.skipped) == 1
    assert isinstance(result.skipped[0].error, DuplicateMemoryError)


def test_replay_commands_collects_failures() -> None:
    item1 = mk_item("01900000-0000-7000-8000-000000000705")
    item2 = mk_item("01900000-0000-7000-8000-000000000706")
    commands = [
        {"type": "memory.create", "item": item1},
        {"type": "memory.update", "item_id": "missing", "partial": {"authority": 0.1}, "author": "tester"},
        {"type": "memory.create", "item": item2},
    ]
    result = replay_commands(commands)
    assert len(result.state.items) == 2
    assert len(result.skipped) == 1
    assert result.skipped[0].index == 1
    assert isinstance(result.skipped[0].error, MemoryNotFoundError)


def test_replay_empty_inputs() -> None:
    assert replay_from_envelopes([]).skipped == []
    assert replay_commands([]).skipped == []


# === extractTimestamp ======================================================


def test_extract_rejects_malformed_16_char() -> None:
    with pytest.raises(InvalidTimestampError):
        extract_timestamp("abcdefghijkl7mno")


def test_extract_rejects_non_hex() -> None:
    with pytest.raises(InvalidTimestampError):
        extract_timestamp("zzzzzzzz-zzzz-7zzz-8zzz-zzzzzzzzzzzz")


def test_extract_rejects_wrong_version() -> None:
    with pytest.raises(InvalidTimestampError):
        extract_timestamp("00000000-0000-4000-8000-000000000000")


def test_extract_accepts_valid_uuidv7() -> None:
    assert extract_timestamp("018bbd1b-3000-7000-8000-000000000000") == 1_699_684_757_504


def test_extract_error_is_exception_subclass() -> None:
    caught: Exception | None = None
    try:
        extract_timestamp("not-a-uuid")
    except Exception as e:  # noqa: BLE001
        caught = e
    assert isinstance(caught, Exception)
    assert isinstance(caught, InvalidTimestampError)
    assert type(caught).__name__ == "InvalidTimestampError"


# === cascadeRetract topological order ======================================


def test_cascade_shared_grandchild() -> None:
    a, b, c, d = (f"01900000-0000-7000-8000-00000000040{n}" for n in (1, 2, 3, 4))
    state = create_graph_state()
    for item in [mk_item(a), mk_item(b, parents=[a]), mk_item(c, parents=[a]), mk_item(d, parents=[b, c])]:
        state = apply_command(state, {"type": "memory.create", "item": item}).state
    res = cascade_retract(state, a, "tester")
    idx_d, idx_b, idx_c, idx_a = (res.retracted.index(x) for x in (d, b, c, a))
    assert idx_d < idx_b
    assert idx_d < idx_c
    assert idx_a == len(res.retracted) - 1
    assert len(res.state.items) == 0


def test_cascade_cyclic_terminates() -> None:
    x = "01900000-0000-7000-8000-000000000501"
    y = "01900000-0000-7000-8000-000000000502"
    state = GraphState(items={x: mk_item(x, parents=[y]), y: mk_item(y, parents=[x])}, edges={})
    res = cascade_retract(state, x, "tester")
    assert len(res.state.items) == 0


def test_cascade_cycle_back_to_root() -> None:
    a = "01900000-0000-7000-8000-000000000510"
    b = "01900000-0000-7000-8000-000000000511"
    state = GraphState(items={a: mk_item(a, parents=[b]), b: mk_item(b, parents=[a])}, edges={})
    res = cascade_retract(state, a, "tester")
    assert res.retracted == [b, a]
    assert len(res.state.items) == 0


def test_cascade_deep_chain_no_stack_overflow() -> None:
    n = 5000
    ids: list[str] = []
    items: dict[str, MemoryItem] = {}
    for i in range(n):
        id_ = f"chain-{str(i).rjust(6, '0')}"
        ids.append(id_)
        items[id_] = mk_item(id_, created_at=1_700_000_000_000 + i, parents=None if i == 0 else [ids[i - 1]])
    state = GraphState(items=items, edges={})
    res = cascade_retract(state, ids[0], "tester")
    assert len(res.retracted) == n
    assert res.retracted[-1] == ids[0]
    assert res.retracted[0] == ids[n - 1]
    assert len(res.state.items) == 0


# === createEventEnvelope round-trips =======================================


def test_event_envelope_round_trips_through_replay() -> None:
    id_ = "01900000-0000-7000-8000-000000000601"
    envelope = create_event_envelope("memory.create", {"type": "memory.create", "item": mk_item(id_)})
    result = replay_from_envelopes([envelope])
    assert id_ in result.state.items


# === getSupportSet with non-UUIDv7 ids =====================================


def test_support_set_non_uuid_ids() -> None:
    a = mk_item("custom-id-root", created_at=1_700_000_000_000)
    b = mk_item("custom-id-child", parents=["custom-id-root"], created_at=1_700_000_000_001)
    state = create_graph_state()
    state = apply_command(state, {"type": "memory.create", "item": a}).state
    state = apply_command(state, {"type": "memory.create", "item": b}).state
    support = get_support_set(state, b.id)
    assert sorted(i.id for i in support) == ["custom-id-child", "custom-id-root"]
