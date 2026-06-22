"""Port of tests/types.test.ts (runtime-meaningful parts; TS type-level checks
become construction + isinstance checks)."""

from __future__ import annotations

from memex import (
    Edge,
    EventEnvelope,
    MemoryCommandAdapter,
    MemoryFilter,
    MemoryItem,
    MemoryLifecycleEvent,
)
from memex.commands import EdgeCreate, MemoryCreate, MemoryUpdate

# --- MemoryItem ------------------------------------------------------------


def test_memory_item_valid_literal() -> None:
    item = MemoryItem(
        id="01HV5W1YF3F8R9H1M6V3X6X8A0", scope="user:laz/general", kind="assertion",
        content={"key": "theme", "value": "dark"}, author="user:laz",
        source_kind="user_explicit", authority=0.99,
    )
    assert item.id == "01HV5W1YF3F8R9H1M6V3X6X8A0"


def test_memory_item_optional_fields() -> None:
    item = MemoryItem(
        id="m1", scope="test", kind="observation", content={}, author="test",
        source_kind="observed", authority=0.5, conviction=0.8, importance=0.3,
        meta={"agent_id": "agent:x", "session_id": "s1", "custom": True},
    )
    assert item.conviction == 0.8
    assert item.meta is not None and item.meta["custom"] is True


def test_memory_item_arbitrary_kind_and_source() -> None:
    item = MemoryItem(
        id="m2", scope="test", kind="custom_kind", content={}, author="test",
        source_kind="custom_source", authority=0.5,
    )
    assert item.kind == "custom_kind"
    assert item.source_kind == "custom_source"


# --- Edge ------------------------------------------------------------------


def test_edge_valid_literal() -> None:
    edge = Edge(
        edge_id="e1", from_="m1", to="m2", kind="DERIVED_FROM", author="system:rule_x",
        source_kind="derived_deterministic", authority=0.9, active=True,
    )
    assert edge.kind == "DERIVED_FROM"


def test_edge_optional_weight_and_meta() -> None:
    edge = Edge(
        edge_id="e2", from_="m1", to="m3", kind="SUPPORTS", weight=0.7,
        author="agent:reasoner", source_kind="agent_inferred", authority=0.6,
        active=True, meta={"reason": "correlation"},
    )
    assert edge.weight == 0.7


# --- MemoryCommand discriminated union -------------------------------------


def test_command_narrows_memory_create() -> None:
    cmd = MemoryCommandAdapter.validate_python({
        "type": "memory.create",
        "item": {
            "id": "m1", "scope": "test", "kind": "observation", "content": {},
            "author": "test", "source_kind": "observed", "authority": 1,
        },
    })
    assert isinstance(cmd, MemoryCreate)
    assert cmd.item.id == "m1"


def test_command_narrows_memory_update() -> None:
    cmd = MemoryCommandAdapter.validate_python({
        "type": "memory.update", "item_id": "m1", "partial": {"authority": 0.5}, "author": "system:tuner",
    })
    assert isinstance(cmd, MemoryUpdate)
    assert cmd.item_id == "m1"
    assert cmd.partial == {"authority": 0.5}


def test_command_narrows_edge_create() -> None:
    cmd = MemoryCommandAdapter.validate_python({
        "type": "edge.create",
        "edge": {
            "edge_id": "e1", "from": "m1", "to": "m2", "kind": "SUPPORTS",
            "author": "test", "source_kind": "observed", "authority": 1, "active": True,
        },
    })
    assert isinstance(cmd, EdgeCreate)
    assert cmd.edge.from_ == "m1"


# --- EventEnvelope ---------------------------------------------------------


def test_event_envelope_generic_payload() -> None:
    item = MemoryItem(
        id="m1", scope="test", kind="observation", content={}, author="test",
        source_kind="observed", authority=1,
    )
    env: EventEnvelope = EventEnvelope(
        id="ev1", namespace="memory", type="state.memory",
        ts="2026-04-10T19:30:00.010Z", payload={"item": item},
    )
    assert env.payload["item"].id == "m1"
    assert env.namespace == "memory"


def test_event_envelope_optional_trace_id() -> None:
    env: EventEnvelope = EventEnvelope(
        id="ev2", namespace="memory", type="test",
        ts="2026-01-01T00:00:00Z", trace_id="trace-123", payload=None,
    )
    assert env.trace_id == "trace-123"


# --- MemoryLifecycleEvent --------------------------------------------------


def test_lifecycle_event() -> None:
    item = MemoryItem(
        id="m1", scope="test", kind="observation", content={}, author="test",
        source_kind="observed", authority=1,
    )
    event = MemoryLifecycleEvent(type="memory.created", item=item, cause_type="memory.create")
    assert event.namespace == "memory"
    assert event.type == "memory.created"


# --- MemoryFilter ----------------------------------------------------------


def test_filter_empty() -> None:
    f = MemoryFilter()
    assert f.model_dump(exclude_none=True) == {}


def test_filter_all_fields() -> None:
    f = MemoryFilter.model_validate({
        "scope": "user:laz/general",
        "author": "user:laz",
        "kind": "observation",
        "source_kind": "observed",
        "range": {"authority": {"min": 0.3, "max": 0.9}, "conviction": {"min": 0.2}, "importance": {"max": 0.7}},
        "not": {"or": [{"kind": "simulation"}, {"kind": "hypothesis"}]},
        "meta": {"agent_id": "agent:foo"},
        "or": [{"kind": "trait"}],
    })
    assert f.scope == "user:laz/general"
    assert f.not_ is not None
    assert f.or_ is not None and f.or_[0].kind == "trait"
