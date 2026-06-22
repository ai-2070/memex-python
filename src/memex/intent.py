"""Intent graph — active goals with a status machine.

``active ⇄ paused → completed / cancelled``. Invalid transitions raise
``InvalidIntentTransitionError``. Same command → reducer → events pattern as the
memory graph, with its own reducer (``apply_intent_command``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from ._uuid import uuid7
from .errors import MemexError

__all__ = [
    "IntentStatus",
    "Intent",
    "IntentState",
    "IntentCommand",
    "IntentLifecycleEvent",
    "IntentFilter",
    "IntentResult",
    "IntentNotFoundError",
    "DuplicateIntentError",
    "InvalidIntentTransitionError",
    "create_intent_state",
    "create_intent",
    "apply_intent_command",
    "get_intents",
    "get_intent_by_id",
    "get_child_intents",
]

IntentStatus = Literal["active", "paused", "completed", "cancelled"]


class Intent(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    parent_id: str | None = None
    label: str
    description: str | None = None

    priority: float = Field(ge=0, le=1)
    owner: str
    status: IntentStatus

    context: dict[str, Any] | None = None
    root_memory_ids: list[str] | None = None

    meta: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class IntentState:
    intents: dict[str, Intent] = field(default_factory=dict)


def create_intent_state() -> IntentState:
    return IntentState(intents={})


def create_intent(
    *,
    label: str,
    priority: float,
    owner: str,
    id: str | None = None,
    parent_id: str | None = None,
    description: str | None = None,
    status: IntentStatus | None = None,
    context: dict[str, Any] | None = None,
    root_memory_ids: list[str] | None = None,
    meta: dict[str, Any] | None = None,
) -> Intent:
    return Intent(
        id=id if id is not None else uuid7(),
        parent_id=parent_id,
        label=label,
        description=description,
        priority=priority,
        owner=owner,
        status=status if status is not None else "active",
        context=context,
        root_memory_ids=root_memory_ids,
        meta=meta,
    )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


class IntentCreate(BaseModel):
    type: Literal["intent.create"] = "intent.create"
    intent: Intent


class IntentUpdate(BaseModel):
    type: Literal["intent.update"] = "intent.update"
    intent_id: str
    partial: dict[str, Any]
    author: str
    reason: str | None = None


class IntentComplete(BaseModel):
    type: Literal["intent.complete"] = "intent.complete"
    intent_id: str
    author: str
    reason: str | None = None


class IntentCancel(BaseModel):
    type: Literal["intent.cancel"] = "intent.cancel"
    intent_id: str
    author: str
    reason: str | None = None


class IntentPause(BaseModel):
    type: Literal["intent.pause"] = "intent.pause"
    intent_id: str
    author: str
    reason: str | None = None


class IntentResume(BaseModel):
    type: Literal["intent.resume"] = "intent.resume"
    intent_id: str
    author: str
    reason: str | None = None


IntentCommand = Annotated[
    IntentCreate | IntentUpdate | IntentComplete | IntentCancel | IntentPause | IntentResume,
    Field(discriminator="type"),
]

_INTENT_COMMAND_ADAPTER: TypeAdapter[IntentCommand] = TypeAdapter(IntentCommand)


IntentEventType = Literal[
    "intent.created",
    "intent.updated",
    "intent.completed",
    "intent.cancelled",
    "intent.paused",
    "intent.resumed",
]


class IntentLifecycleEvent(BaseModel):
    namespace: Literal["intent"] = "intent"
    type: IntentEventType
    intent: Intent
    cause_type: str


class IntentResult(NamedTuple):
    state: IntentState
    events: list[IntentLifecycleEvent]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class IntentNotFoundError(MemexError):
    def __init__(self, intent_id: str) -> None:
        super().__init__(f"Intent not found: {intent_id}")
        self.intent_id = intent_id


class DuplicateIntentError(MemexError):
    def __init__(self, intent_id: str) -> None:
        super().__init__(f"Intent already exists: {intent_id}")
        self.intent_id = intent_id


class InvalidIntentTransitionError(MemexError):
    def __init__(self, intent_id: str, from_status: str, to_status: str) -> None:
        super().__init__(f"Invalid intent transition: {intent_id} from {from_status} to {to_status}")
        self.intent_id = intent_id
        self.from_status = from_status
        self.to_status = to_status


# ---------------------------------------------------------------------------
# Reducer
# ---------------------------------------------------------------------------


def _set_status(
    state: IntentState,
    intent_id: str,
    target: IntentStatus,
    valid_from: tuple[IntentStatus, ...],
    cause_type: str,
    event_type: IntentEventType,
) -> IntentResult:
    existing = state.intents.get(intent_id)
    if existing is None:
        raise IntentNotFoundError(intent_id)
    if existing.status not in valid_from:
        raise InvalidIntentTransitionError(intent_id, existing.status, target)
    updated = existing.model_copy(update={"status": target})
    intents = {**state.intents, intent_id: updated}
    return IntentResult(
        IntentState(intents),
        [IntentLifecycleEvent(type=event_type, intent=updated, cause_type=cause_type)],
    )


def apply_intent_command(state: IntentState, cmd: IntentCommand | dict[str, Any]) -> IntentResult:
    command = cmd if isinstance(cmd, BaseModel) else _INTENT_COMMAND_ADAPTER.validate_python(cmd)

    match command:
        case IntentCreate(intent=intent):
            if intent.id in state.intents:
                raise DuplicateIntentError(intent.id)
            intents = {**state.intents, intent.id: intent}
            return IntentResult(
                IntentState(intents),
                [IntentLifecycleEvent(type="intent.created", intent=intent, cause_type="intent.create")],
            )

        case IntentUpdate(intent_id=intent_id, partial=partial):
            existing = state.intents.get(intent_id)
            if existing is None:
                raise IntentNotFoundError(intent_id)
            update = {k: v for k, v in partial.items() if k not in ("id", "status")}
            updated = existing.model_copy(update=update)
            intents = {**state.intents, intent_id: updated}
            return IntentResult(
                IntentState(intents),
                [IntentLifecycleEvent(type="intent.updated", intent=updated, cause_type="intent.update")],
            )

        case IntentComplete(intent_id=intent_id):
            return _set_status(state, intent_id, "completed", ("active", "paused"), "intent.complete", "intent.completed")

        case IntentCancel(intent_id=intent_id):
            return _set_status(state, intent_id, "cancelled", ("active", "paused"), "intent.cancel", "intent.cancelled")

        case IntentPause(intent_id=intent_id):
            return _set_status(state, intent_id, "paused", ("active",), "intent.pause", "intent.paused")

        case IntentResume(intent_id=intent_id):
            return _set_status(state, intent_id, "active", ("paused",), "intent.resume", "intent.resumed")

        case _:  # pragma: no cover
            raise TypeError(f"Unknown intent command: {command!r}")


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


class IntentFilter(BaseModel):
    owner: str | None = None
    status: IntentStatus | None = None
    statuses: list[IntentStatus] | None = None
    min_priority: float | None = None
    has_memory_id: str | None = None
    parent_id: str | None = None
    is_root: bool | None = None


def _coerce_intent_filter(f: IntentFilter | dict[str, Any] | None) -> IntentFilter | None:
    if f is None or isinstance(f, IntentFilter):
        return f
    return IntentFilter.model_validate(f)


def get_intents(state: IntentState, filter: IntentFilter | dict[str, Any] | None = None) -> list[Intent]:
    f = _coerce_intent_filter(filter)
    if f is None:
        return list(state.intents.values())

    results: list[Intent] = []
    for intent in state.intents.values():
        if f.owner is not None and intent.owner != f.owner:
            continue
        if f.status is not None and intent.status != f.status:
            continue
        if f.statuses is not None and intent.status not in f.statuses:
            continue
        if f.min_priority is not None and intent.priority < f.min_priority:
            continue
        if f.has_memory_id is not None:
            if not intent.root_memory_ids or f.has_memory_id not in intent.root_memory_ids:
                continue
        if f.parent_id is not None and intent.parent_id != f.parent_id:
            continue
        if f.is_root is not None:
            has_parent = intent.parent_id is not None
            if f.is_root and has_parent:
                continue
            if not f.is_root and not has_parent:
                continue
        results.append(intent)
    return results


def get_intent_by_id(state: IntentState, id: str) -> Intent | None:
    return state.intents.get(id)


def get_child_intents(state: IntentState, parent_id: str) -> list[Intent]:
    return get_intents(state, {"parent_id": parent_id})
