"""Helpers that wrap lifecycle events / state snapshots into event envelopes."""

from __future__ import annotations

from typing import Any

from . import _time
from ._uuid import uuid7
from .models import Edge, EventEnvelope, MemoryItem, MemoryLifecycleEvent

__all__ = ["wrap_lifecycle_event", "wrap_state_event", "wrap_edge_state_event"]


def _event_fields(event: MemoryLifecycleEvent) -> dict[str, Any]:
    # Mirror the TS object spread: only keys that are set (item/edge/cause_type
    # may be absent). Nested item/edge are kept as models until serialized.
    fields: dict[str, Any] = {"namespace": event.namespace, "type": event.type}
    if event.item is not None:
        fields["item"] = event.item
    if event.edge is not None:
        fields["edge"] = event.edge
    if event.cause_type is not None:
        fields["cause_type"] = event.cause_type
    return fields


def wrap_lifecycle_event(
    event: MemoryLifecycleEvent,
    cause_id: str,
    trace_id: str | None = None,
) -> EventEnvelope[dict[str, Any]]:
    return EventEnvelope(
        id=uuid7(),
        namespace="memory",
        type=event.type,
        ts=_time.now_iso(),
        trace_id=trace_id,
        payload={**_event_fields(event), "cause_id": cause_id},
    )


def wrap_state_event(
    item: MemoryItem,
    cause_id: str,
    trace_id: str | None = None,
) -> EventEnvelope[dict[str, Any]]:
    return EventEnvelope(
        id=uuid7(),
        namespace="memory",
        type="state.memory",
        ts=_time.now_iso(),
        trace_id=trace_id,
        payload={"item": item, "cause_id": cause_id},
    )


def wrap_edge_state_event(
    edge: Edge,
    cause_id: str,
    trace_id: str | None = None,
) -> EventEnvelope[dict[str, Any]]:
    return EventEnvelope(
        id=uuid7(),
        namespace="memory",
        type="state.edge",
        ts=_time.now_iso(),
        trace_id=trace_id,
        payload={"edge": edge, "cause_id": cause_id},
    )
