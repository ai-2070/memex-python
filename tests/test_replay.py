"""Port of tests/replay.test.ts."""

from __future__ import annotations

from memex import MemoryNotFoundError, replay_commands, replay_from_envelopes

ITEM1 = {"id": "m1", "scope": "test", "kind": "observation", "content": {"key": "v1"},
         "author": "user:laz", "source_kind": "observed", "authority": 0.9}
ITEM2 = {"id": "m2", "scope": "test", "kind": "assertion", "content": {"key": "v2"},
         "author": "user:laz", "source_kind": "user_explicit", "authority": 0.8}
EDGE1 = {"edge_id": "e1", "from": "m1", "to": "m2", "kind": "SUPPORTS",
         "author": "system:rule", "source_kind": "derived_deterministic", "authority": 0.7, "active": True}


# --- replayCommands --------------------------------------------------------


def test_replay_empty() -> None:
    state, events, _ = replay_commands([])
    assert len(state.items) == 0
    assert len(events) == 0


def test_replay_create_update_retract() -> None:
    commands = [
        {"type": "memory.create", "item": ITEM1},
        {"type": "memory.update", "item_id": "m1", "partial": {"authority": 0.5}, "author": "system:tuner"},
        {"type": "memory.retract", "item_id": "m1", "author": "user:laz"},
    ]
    state, events, _ = replay_commands(commands)
    assert len(state.items) == 0
    assert len(events) == 3
    assert events[0].type == "memory.created"
    assert events[1].type == "memory.updated"
    assert events[2].type == "memory.retracted"


def test_replay_create_create_edge() -> None:
    commands = [
        {"type": "memory.create", "item": ITEM1},
        {"type": "memory.create", "item": ITEM2},
        {"type": "edge.create", "edge": EDGE1},
    ]
    state, _, _ = replay_commands(commands)
    assert len(state.items) == 2
    assert len(state.edges) == 1


def test_replay_collects_failures() -> None:
    commands = [
        {"type": "memory.create", "item": ITEM1},
        {"type": "memory.update", "item_id": "nonexistent", "partial": {"authority": 0.1}, "author": "test"},
        {"type": "memory.create", "item": ITEM2},
    ]
    state, _, skipped = replay_commands(commands)
    assert len(state.items) == 2
    assert len(skipped) == 1
    assert skipped[0].index == 1
    assert isinstance(skipped[0].error, MemoryNotFoundError)


# --- replayFromEnvelopes ---------------------------------------------------


def test_replay_envelopes_sorts_by_timestamp() -> None:
    envelopes = [
        {"id": "ev2", "namespace": "memory", "type": "memory.create",
         "ts": "2026-04-10T19:30:00.000Z", "payload": {"type": "memory.create", "item": ITEM2}},
        {"id": "ev1", "namespace": "memory", "type": "memory.create",
         "ts": "2026-04-10T19:20:00.000Z", "payload": {"type": "memory.create", "item": ITEM1}},
        {"id": "ev3", "namespace": "memory", "type": "edge.create",
         "ts": "2026-04-10T19:40:00.000Z", "payload": {"type": "edge.create", "edge": EDGE1}},
    ]
    state, events, _ = replay_from_envelopes(envelopes)
    assert len(state.items) == 2
    assert len(state.edges) == 1
    assert events[0].item is not None and events[0].item.id == "m1"


def test_replay_envelopes_does_not_mutate_input() -> None:
    envelopes = [
        {"id": "ev2", "namespace": "memory", "type": "memory.create",
         "ts": "2026-04-10T19:30:00.000Z", "payload": {"type": "memory.create", "item": ITEM2}},
        {"id": "ev1", "namespace": "memory", "type": "memory.create",
         "ts": "2026-04-10T19:20:00.000Z", "payload": {"type": "memory.create", "item": ITEM1}},
    ]
    first_id = envelopes[0]["id"]
    replay_from_envelopes(envelopes)
    assert envelopes[0]["id"] == first_id
