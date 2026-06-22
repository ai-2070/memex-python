"""Port of tests/serialization.test.ts."""

from __future__ import annotations

import json

from memex import (
    Edge,
    MemoryItem,
    apply_command,
    create_graph_state,
    from_json,
    parse,
    stringify,
    to_json,
)


def make_item(id: str) -> MemoryItem:
    return MemoryItem(
        id=id, scope="test", kind="observation", content={"key": "value"},
        author="user:laz", source_kind="observed", authority=0.9,
        conviction=0.8, importance=0.7, meta={"agent_id": "agent:x"},
    )


def make_edge(id: str) -> Edge:
    return Edge(
        edge_id=id, from_="m1", to="m2", kind="SUPPORTS", author="system:rule",
        source_kind="derived_deterministic", authority=0.8, active=True, weight=0.5,
    )


def build_state():
    state = create_graph_state()
    state = apply_command(state, {"type": "memory.create", "item": make_item("m1")}).state
    state = apply_command(state, {"type": "memory.create", "item": make_item("m2")}).state
    state = apply_command(state, {"type": "edge.create", "edge": make_edge("e1")}).state
    return state


# --- toJSON / fromJSON -----------------------------------------------------


def test_round_trips_graph_state() -> None:
    restored = from_json(to_json(build_state()))
    assert len(restored.items) == 2
    assert len(restored.edges) == 1
    assert restored.items["m1"].content == {"key": "value"}
    assert restored.items["m1"].meta["agent_id"] == "agent:x"
    assert restored.edges["e1"].kind == "SUPPORTS"


def test_serialized_format_arrays() -> None:
    j = to_json(build_state())
    assert isinstance(j["items"], list)
    assert isinstance(j["edges"], list)
    assert len(j["items"]) == 2
    assert len(j["edges"]) == 1
    assert j["items"][0][0] == "m1"


def test_empty_state_round_trips() -> None:
    restored = from_json(to_json(create_graph_state()))
    assert len(restored.items) == 0
    assert len(restored.edges) == 0


def test_edge_from_alias_in_serialized_form() -> None:
    j = to_json(build_state())
    edge_dict = j["edges"][0][1]
    assert edge_dict["from"] == "m1"
    assert "from_" not in edge_dict


# --- stringify / parse -----------------------------------------------------


def test_stringify_round_trip() -> None:
    restored = parse(stringify(build_state()))
    assert len(restored.items) == 2
    assert len(restored.edges) == 1
    assert restored.items["m2"].authority == 0.9


def test_stringify_produces_valid_json() -> None:
    json.loads(stringify(build_state()))  # does not raise


def test_pretty_mode_is_longer() -> None:
    compact = stringify(build_state())
    pretty = stringify(build_state(), True)
    assert len(pretty) > len(compact)
    assert "\n" in pretty


def test_preserves_all_fields() -> None:
    restored = parse(stringify(build_state()))
    item = restored.items["m1"]
    assert item.scope == "test"
    assert item.kind == "observation"
    assert item.conviction == 0.8
    assert item.importance == 0.7
    assert item.meta["agent_id"] == "agent:x"
    edge = restored.edges["e1"]
    assert edge.weight == 0.5
    assert edge.active is True
