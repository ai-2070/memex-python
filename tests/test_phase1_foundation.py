"""Phase 1 smoke tests: models, factories, uuid, commands, graph state."""

from __future__ import annotations

import dataclasses

import pytest
from pydantic import ValidationError

import memex
from memex import (
    Edge,
    MemoryCommandAdapter,
    MemoryCreate,
    MemoryItem,
    clone_graph_state,
    create_edge,
    create_graph_state,
    create_memory_item,
    safe_extract_timestamp,
    uuid7,
)


def test_uuid7_roundtrips_timestamp() -> None:
    uid = uuid7(1_700_000_000_000)
    assert safe_extract_timestamp(uid) == 1_700_000_000_000


def test_uuid7_default_is_recent() -> None:
    ts = safe_extract_timestamp(uuid7())
    assert ts is not None and ts > 1_600_000_000_000


def test_safe_extract_timestamp_rejects_non_uuid7() -> None:
    assert safe_extract_timestamp("not-a-uuid") is None
    # a v4 uuid is well-formed but not version 7
    assert safe_extract_timestamp("00000000-0000-4000-8000-000000000000") is None


def test_create_memory_item_assigns_id_and_created_at() -> None:
    item = create_memory_item(
        scope="user:laz/general",
        kind="observation",
        content={"key": "login_count", "value": 42},
        author="agent:monitor",
        source_kind="observed",
        authority=0.9,
        importance=0.7,
    )
    assert item.id
    assert item.created_at == safe_extract_timestamp(item.id)
    assert item.authority == 0.9
    assert isinstance(item, MemoryItem)
    assert item.model_config["frozen"] is True


def test_score_out_of_range_raises_validation_error() -> None:
    # D2: validation is always on at construction.
    with pytest.raises(ValidationError):
        create_memory_item(
            scope="s",
            kind="observation",
            content={},
            author="a",
            source_kind="observed",
            authority=5,
        )


def test_model_construct_bypasses_validation() -> None:
    # Documented escape hatch for deliberately-invalid tolerance fixtures.
    item = MemoryItem.model_construct(
        id="x", scope="s", kind="observation", content={}, author="a",
        source_kind="observed", authority=5,
    )
    assert item.authority == 5


def test_frozen_item_is_immutable() -> None:
    item = create_memory_item(
        scope="s", kind="observation", content={}, author="a",
        source_kind="observed", authority=0.5,
    )
    with pytest.raises(ValidationError):
        item.authority = 0.6  # type: ignore[misc]


def test_edge_from_alias_roundtrip() -> None:
    edge = create_edge(
        from_="a", to="b", kind="DERIVED_FROM", author="agent:r",
        source_kind="agent_inferred", authority=0.8,
    )
    assert edge.from_ == "a"
    assert edge.active is True
    dumped = edge.model_dump(by_alias=True, exclude_none=True)
    assert dumped["from"] == "a"
    assert "from_" not in dumped
    # round-trips back through the alias
    assert Edge.model_validate(dumped).from_ == "a"


def test_command_discriminated_union_from_dict() -> None:
    item = create_memory_item(
        scope="s", kind="observation", content={}, author="a",
        source_kind="observed", authority=0.5,
    )
    cmd = MemoryCommandAdapter.validate_python({"type": "memory.create", "item": item})
    assert isinstance(cmd, MemoryCreate)
    assert cmd.item.id == item.id


def test_command_direct_construction_defaults_type() -> None:
    item = create_memory_item(
        scope="s", kind="observation", content={}, author="a",
        source_kind="observed", authority=0.5,
    )
    cmd = MemoryCreate(item=item)
    assert cmd.type == "memory.create"


def test_graph_state_clone_is_independent() -> None:
    state = create_graph_state()
    assert state.items == {} and state.edges == {}
    cloned = clone_graph_state(state)
    cloned.items["x"] = create_memory_item(
        scope="s", kind="observation", content={}, author="a",
        source_kind="observed", authority=0.5,
    )
    assert "x" not in state.items  # original untouched


def test_graph_state_is_frozen_dataclass() -> None:
    state = create_graph_state()
    with pytest.raises(dataclasses.FrozenInstanceError):
        state.items = {}  # type: ignore[misc]


def test_public_surface_exports() -> None:
    assert "create_memory_item" in memex.__all__
    assert "MemoryCommand" in memex.__all__
