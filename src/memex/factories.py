"""Factories that mint ids/timestamps, mirroring ``helpers.ts``.

``create_memory_item`` / ``create_edge`` validate score bounds at construction
(via the Pydantic ``Field`` constraints on the models). ``created_at`` is
derived from an explicit value, else the UUIDv7 timestamp, else the wall clock.
"""

from __future__ import annotations

from typing import Any

from . import _time
from ._uuid import safe_extract_timestamp, uuid7
from .models import Edge, EventEnvelope, MemoryItem

__all__ = ["create_memory_item", "create_edge", "create_event_envelope"]


def create_memory_item(
    *,
    scope: str,
    kind: str,
    content: dict[str, Any],
    author: str,
    source_kind: str,
    authority: float,
    id: str | None = None,
    parents: list[str] | None = None,
    conviction: float | None = None,
    importance: float | None = None,
    created_at: int | None = None,
    intent_id: str | None = None,
    task_id: str | None = None,
    meta: dict[str, Any] | None = None,
) -> MemoryItem:
    item_id = id if id is not None else uuid7()
    ts = created_at if created_at is not None else (safe_extract_timestamp(item_id) or _time.now_ms())
    return MemoryItem(
        id=item_id,
        scope=scope,
        kind=kind,
        content=content,
        author=author,
        source_kind=source_kind,
        parents=parents,
        authority=authority,
        conviction=conviction,
        importance=importance,
        created_at=ts,
        intent_id=intent_id,
        task_id=task_id,
        meta=meta,
    )


def create_edge(
    *,
    from_: str,
    to: str,
    kind: str,
    author: str,
    source_kind: str,
    authority: float,
    edge_id: str | None = None,
    active: bool | None = None,
    weight: float | None = None,
    meta: dict[str, Any] | None = None,
) -> Edge:
    return Edge(
        edge_id=edge_id if edge_id is not None else uuid7(),
        from_=from_,
        to=to,
        kind=kind,
        weight=weight,
        author=author,
        source_kind=source_kind,
        authority=authority,
        active=active if active is not None else True,
        meta=meta,
    )


def create_event_envelope(
    type: str,
    payload: Any,
    *,
    trace_id: str | None = None,
    namespace: str = "memory",
) -> EventEnvelope[Any]:
    return EventEnvelope(
        id=uuid7(),
        namespace=namespace,
        type=type,
        ts=_time.now_iso(),
        trace_id=trace_id,
        payload=payload,
    )
