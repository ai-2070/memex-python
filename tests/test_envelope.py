"""Port of tests/envelope.test.ts."""

from __future__ import annotations

import re

from memex import (
    Edge,
    MemoryItem,
    MemoryLifecycleEvent,
    wrap_edge_state_event,
    wrap_lifecycle_event,
    wrap_state_event,
)

UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")

ITEM = MemoryItem(id="m1", scope="test", kind="observation", content={}, author="test", source_kind="observed", authority=1)
EDGE = Edge(edge_id="e1", from_="m1", to="m2", kind="SUPPORTS", author="test", source_kind="observed", authority=1, active=True)


# --- wrapLifecycleEvent ----------------------------------------------------


def test_wrap_lifecycle_event() -> None:
    event = MemoryLifecycleEvent(type="memory.created", item=ITEM, cause_type="memory.create")
    env = wrap_lifecycle_event(event, "cmd-123")
    assert UUID_RE.match(env.id)
    assert env.namespace == "memory"
    assert env.type == "memory.created"
    assert ISO_RE.match(env.ts)
    assert env.payload["cause_id"] == "cmd-123"
    assert env.payload["item"] == ITEM


def test_wrap_lifecycle_event_trace_id() -> None:
    event = MemoryLifecycleEvent(type="memory.updated", item=ITEM)
    env = wrap_lifecycle_event(event, "cmd-1", "trace-abc")
    assert env.trace_id == "trace-abc"


def test_wrap_lifecycle_event_omits_trace_id() -> None:
    event = MemoryLifecycleEvent(type="memory.retracted", item=ITEM)
    env = wrap_lifecycle_event(event, "cmd-1")
    assert env.trace_id is None


# --- wrapStateEvent --------------------------------------------------------


def test_wrap_state_event() -> None:
    env = wrap_state_event(ITEM, "cmd-456")
    assert env.type == "state.memory"
    assert env.payload["item"] == ITEM
    assert env.payload["cause_id"] == "cmd-456"


def test_wrap_state_event_trace_id() -> None:
    env = wrap_state_event(ITEM, "cmd-1", "trace-xyz")
    assert env.trace_id == "trace-xyz"


# --- wrapEdgeStateEvent ----------------------------------------------------


def test_wrap_edge_state_event() -> None:
    env = wrap_edge_state_event(EDGE, "cmd-789")
    assert env.type == "state.edge"
    assert env.payload["edge"] == EDGE
    assert env.payload["cause_id"] == "cmd-789"
