"""Port of tests/helpers.test.ts (factories).

RangeError -> pydantic.ValidationError per D7.
"""

from __future__ import annotations

import re

import pytest
from pydantic import ValidationError

from memex import create_edge, create_event_envelope, create_memory_item

UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")

BASE_ITEM = dict(
    scope="test",
    kind="observation",
    content={"key": "value"},
    author="user:laz",
    source_kind="observed",
    authority=0.9,
)

BASE_EDGE = dict(
    from_="m1",
    to="m2",
    kind="SUPPORTS",
    author="system:rule",
    source_kind="derived_deterministic",
    authority=0.9,
)


# --- create_memory_item ----------------------------------------------------


def test_generates_valid_uuidv7_id() -> None:
    item = create_memory_item(**BASE_ITEM)
    assert UUID_RE.match(item.id)


def test_preserves_caller_supplied_id() -> None:
    item = create_memory_item(**BASE_ITEM, id="custom-id")
    assert item.id == "custom-id"


def test_authority_too_high_raises() -> None:
    with pytest.raises(ValidationError):
        create_memory_item(**{**BASE_ITEM, "authority": 1.5})


def test_authority_too_low_raises() -> None:
    with pytest.raises(ValidationError):
        create_memory_item(**{**BASE_ITEM, "authority": -0.1})


def test_conviction_out_of_range_raises() -> None:
    with pytest.raises(ValidationError):
        create_memory_item(**BASE_ITEM, conviction=2)


def test_importance_out_of_range_raises() -> None:
    with pytest.raises(ValidationError):
        create_memory_item(**BASE_ITEM, importance=-1)


def test_accepts_undefined_optional_scores() -> None:
    item = create_memory_item(**BASE_ITEM)
    assert item.conviction is None
    assert item.importance is None


def test_preserves_all_input_fields() -> None:
    item = create_memory_item(**BASE_ITEM, conviction=0.8, importance=0.5, meta={"agent_id": "agent:x"})
    assert item.conviction == 0.8
    assert item.importance == 0.5
    assert item.meta is not None and item.meta["agent_id"] == "agent:x"


def test_created_at_derived_from_uuidv7() -> None:
    item = create_memory_item(**BASE_ITEM)
    from memex import safe_extract_timestamp

    assert item.created_at == safe_extract_timestamp(item.id)


# --- create_edge -----------------------------------------------------------


def test_edge_generates_uuidv7_and_defaults_active() -> None:
    edge = create_edge(**BASE_EDGE)
    assert UUID_RE.match(edge.edge_id)
    assert edge.active is True


def test_edge_preserves_caller_id_and_active() -> None:
    edge = create_edge(**BASE_EDGE, edge_id="e-custom", active=False)
    assert edge.edge_id == "e-custom"
    assert edge.active is False


def test_edge_authority_out_of_range_raises() -> None:
    with pytest.raises(ValidationError):
        create_edge(**{**BASE_EDGE, "authority": 1.01})


def test_edge_weight_too_high_raises() -> None:
    with pytest.raises(ValidationError):
        create_edge(**BASE_EDGE, weight=1.5)


def test_edge_weight_too_low_raises() -> None:
    with pytest.raises(ValidationError):
        create_edge(**BASE_EDGE, weight=-0.1)


def test_edge_accepts_undefined_weight() -> None:
    edge = create_edge(**BASE_EDGE)
    assert edge.weight is None


def test_edge_accepts_valid_weight() -> None:
    edge = create_edge(**BASE_EDGE, weight=0.5)
    assert edge.weight == 0.5


def test_edge_allows_self_reference() -> None:
    edge = create_edge(**{**BASE_EDGE, "from_": "m1", "to": "m1"})
    assert edge.from_ == "m1"
    assert edge.to == "m1"
    assert edge.active is True


# --- create_event_envelope -------------------------------------------------


def test_envelope_namespace_ts_and_id() -> None:
    env = create_event_envelope("memory.create", {"test": True})
    assert UUID_RE.match(env.id)
    assert env.namespace == "memory"
    assert env.type == "memory.create"
    assert ISO_RE.match(env.ts)
    assert env.payload == {"test": True}


def test_envelope_defaults_namespace_memory() -> None:
    env = create_event_envelope("memory.create", {})
    assert env.namespace == "memory"


def test_envelope_custom_namespace() -> None:
    env = create_event_envelope("intent.create", {"id": "i1"}, namespace="intent")
    assert env.namespace == "intent"


def test_envelope_custom_namespace_with_trace_id() -> None:
    env = create_event_envelope("task.start", None, trace_id="t-1", namespace="task")
    assert env.namespace == "task"
    assert env.trace_id == "t-1"


def test_envelope_includes_trace_id() -> None:
    env = create_event_envelope("test", None, trace_id="t-123")
    assert env.trace_id == "t-123"


def test_envelope_omits_trace_id() -> None:
    env = create_event_envelope("test", None)
    assert env.trace_id is None
