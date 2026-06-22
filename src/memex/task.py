"""Task graph — units of work tied to intents, with a status machine.

``pending → running → completed`` (``running → failed → running`` retry,
``cancel`` from any non-terminal state). Invalid transitions raise
``InvalidTaskTransitionError``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from ._uuid import uuid7
from .errors import MemexError

__all__ = [
    "TaskStatus",
    "Task",
    "TaskState",
    "TaskCommand",
    "TaskLifecycleEvent",
    "TaskFilter",
    "TaskResult",
    "TaskNotFoundError",
    "DuplicateTaskError",
    "InvalidTaskTransitionError",
    "create_task_state",
    "create_task",
    "apply_task_command",
    "get_tasks",
    "get_task_by_id",
    "get_tasks_by_intent",
    "get_child_tasks",
]

TaskStatus = Literal["pending", "running", "completed", "failed", "cancelled"]


class Task(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    intent_id: str
    parent_id: str | None = None

    action: str
    label: str | None = None

    status: TaskStatus
    priority: float = Field(ge=0, le=1)

    context: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    error: str | None = None

    input_memory_ids: list[str] | None = None
    output_memory_ids: list[str] | None = None

    agent_id: str | None = None
    attempt: int | None = None

    meta: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class TaskState:
    tasks: dict[str, Task] = field(default_factory=dict)


def create_task_state() -> TaskState:
    return TaskState(tasks={})


def create_task(
    *,
    intent_id: str,
    action: str,
    priority: float,
    id: str | None = None,
    parent_id: str | None = None,
    label: str | None = None,
    status: TaskStatus | None = None,
    context: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
    error: str | None = None,
    input_memory_ids: list[str] | None = None,
    output_memory_ids: list[str] | None = None,
    agent_id: str | None = None,
    attempt: int | None = None,
    meta: dict[str, Any] | None = None,
) -> Task:
    return Task(
        id=id if id is not None else uuid7(),
        intent_id=intent_id,
        parent_id=parent_id,
        action=action,
        label=label,
        status=status if status is not None else "pending",
        priority=priority,
        context=context,
        result=result,
        error=error,
        input_memory_ids=input_memory_ids,
        output_memory_ids=output_memory_ids,
        agent_id=agent_id,
        attempt=attempt if attempt is not None else 0,
        meta=meta,
    )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


class TaskCreate(BaseModel):
    type: Literal["task.create"] = "task.create"
    task: Task


class TaskUpdate(BaseModel):
    type: Literal["task.update"] = "task.update"
    task_id: str
    partial: dict[str, Any]
    author: str


class TaskStart(BaseModel):
    type: Literal["task.start"] = "task.start"
    task_id: str
    agent_id: str | None = None


class TaskComplete(BaseModel):
    type: Literal["task.complete"] = "task.complete"
    task_id: str
    result: dict[str, Any] | None = None
    output_memory_ids: list[str] | None = None


class TaskFail(BaseModel):
    type: Literal["task.fail"] = "task.fail"
    task_id: str
    error: str
    retryable: bool | None = None


class TaskCancel(BaseModel):
    type: Literal["task.cancel"] = "task.cancel"
    task_id: str
    reason: str | None = None


TaskCommand = Annotated[
    TaskCreate | TaskUpdate | TaskStart | TaskComplete | TaskFail | TaskCancel,
    Field(discriminator="type"),
]

_TASK_COMMAND_ADAPTER: TypeAdapter[TaskCommand] = TypeAdapter(TaskCommand)


TaskEventType = Literal[
    "task.created",
    "task.updated",
    "task.started",
    "task.completed",
    "task.failed",
    "task.cancelled",
]


class TaskLifecycleEvent(BaseModel):
    namespace: Literal["task"] = "task"
    type: TaskEventType
    task: Task
    cause_type: str


class TaskResult(NamedTuple):
    state: TaskState
    events: list[TaskLifecycleEvent]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TaskNotFoundError(MemexError):
    def __init__(self, task_id: str) -> None:
        super().__init__(f"Task not found: {task_id}")
        self.task_id = task_id


class DuplicateTaskError(MemexError):
    def __init__(self, task_id: str) -> None:
        super().__init__(f"Task already exists: {task_id}")
        self.task_id = task_id


class InvalidTaskTransitionError(MemexError):
    def __init__(self, task_id: str, from_status: str, to_status: str) -> None:
        super().__init__(f"Invalid task transition: {task_id} from {from_status} to {to_status}")
        self.task_id = task_id
        self.from_status = from_status
        self.to_status = to_status


# ---------------------------------------------------------------------------
# Reducer
# ---------------------------------------------------------------------------


def _event(task: Task, event_type: TaskEventType, cause_type: str) -> TaskLifecycleEvent:
    return TaskLifecycleEvent(type=event_type, task=task, cause_type=cause_type)


def apply_task_command(state: TaskState, cmd: TaskCommand | dict[str, Any]) -> TaskResult:
    command = cmd if isinstance(cmd, BaseModel) else _TASK_COMMAND_ADAPTER.validate_python(cmd)

    match command:
        case TaskCreate(task=task):
            if task.id in state.tasks:
                raise DuplicateTaskError(task.id)
            tasks = {**state.tasks, task.id: task}
            return TaskResult(TaskState(tasks), [_event(task, "task.created", "task.create")])

        case TaskUpdate(task_id=task_id, partial=partial):
            existing = state.tasks.get(task_id)
            if existing is None:
                raise TaskNotFoundError(task_id)
            update = {k: v for k, v in partial.items() if k not in ("id", "status")}
            updated = existing.model_copy(update=update)
            tasks = {**state.tasks, task_id: updated}
            return TaskResult(TaskState(tasks), [_event(updated, "task.updated", "task.update")])

        case TaskStart(task_id=task_id, agent_id=agent_id):
            existing = state.tasks.get(task_id)
            if existing is None:
                raise TaskNotFoundError(task_id)
            if existing.status not in ("pending", "failed"):
                raise InvalidTaskTransitionError(task_id, existing.status, "running")
            updated = existing.model_copy(update={
                "status": "running",
                "agent_id": agent_id if agent_id is not None else existing.agent_id,
                "attempt": (existing.attempt or 0) + 1,
            })
            tasks = {**state.tasks, task_id: updated}
            return TaskResult(TaskState(tasks), [_event(updated, "task.started", "task.start")])

        case TaskComplete(task_id=task_id, result=result, output_memory_ids=output_memory_ids):
            existing = state.tasks.get(task_id)
            if existing is None:
                raise TaskNotFoundError(task_id)
            if existing.status != "running":
                raise InvalidTaskTransitionError(task_id, existing.status, "completed")
            updated = existing.model_copy(update={
                "status": "completed",
                "result": result if result is not None else existing.result,
                "output_memory_ids": output_memory_ids if output_memory_ids is not None else existing.output_memory_ids,
            })
            tasks = {**state.tasks, task_id: updated}
            return TaskResult(TaskState(tasks), [_event(updated, "task.completed", "task.complete")])

        case TaskFail(task_id=task_id, error=error):
            existing = state.tasks.get(task_id)
            if existing is None:
                raise TaskNotFoundError(task_id)
            if existing.status != "running":
                raise InvalidTaskTransitionError(task_id, existing.status, "failed")
            updated = existing.model_copy(update={"status": "failed", "error": error})
            tasks = {**state.tasks, task_id: updated}
            return TaskResult(TaskState(tasks), [_event(updated, "task.failed", "task.fail")])

        case TaskCancel(task_id=task_id):
            existing = state.tasks.get(task_id)
            if existing is None:
                raise TaskNotFoundError(task_id)
            if existing.status in ("completed", "cancelled"):
                raise InvalidTaskTransitionError(task_id, existing.status, "cancelled")
            updated = existing.model_copy(update={"status": "cancelled"})
            tasks = {**state.tasks, task_id: updated}
            return TaskResult(TaskState(tasks), [_event(updated, "task.cancelled", "task.cancel")])

        case _:  # pragma: no cover
            raise TypeError(f"Unknown task command: {command!r}")


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


class TaskFilter(BaseModel):
    intent_id: str | None = None
    action: str | None = None
    status: TaskStatus | None = None
    statuses: list[TaskStatus] | None = None
    agent_id: str | None = None
    min_priority: float | None = None
    has_input_memory_id: str | None = None
    has_output_memory_id: str | None = None
    parent_id: str | None = None
    is_root: bool | None = None


def _coerce_task_filter(f: TaskFilter | dict[str, Any] | None) -> TaskFilter | None:
    if f is None or isinstance(f, TaskFilter):
        return f
    return TaskFilter.model_validate(f)


def get_tasks(state: TaskState, filter: TaskFilter | dict[str, Any] | None = None) -> list[Task]:
    f = _coerce_task_filter(filter)
    if f is None:
        return list(state.tasks.values())

    results: list[Task] = []
    for task in state.tasks.values():
        if f.intent_id is not None and task.intent_id != f.intent_id:
            continue
        if f.action is not None and task.action != f.action:
            continue
        if f.status is not None and task.status != f.status:
            continue
        if f.statuses is not None and task.status not in f.statuses:
            continue
        if f.agent_id is not None and task.agent_id != f.agent_id:
            continue
        if f.min_priority is not None and task.priority < f.min_priority:
            continue
        if f.has_input_memory_id is not None:
            if not task.input_memory_ids or f.has_input_memory_id not in task.input_memory_ids:
                continue
        if f.has_output_memory_id is not None:
            if not task.output_memory_ids or f.has_output_memory_id not in task.output_memory_ids:
                continue
        if f.parent_id is not None and task.parent_id != f.parent_id:
            continue
        if f.is_root is not None:
            has_parent = task.parent_id is not None
            if f.is_root and has_parent:
                continue
            if not f.is_root and not has_parent:
                continue
        results.append(task)
    return results


def get_task_by_id(state: TaskState, id: str) -> Task | None:
    return state.tasks.get(id)


def get_tasks_by_intent(state: TaskState, intent_id: str) -> list[Task]:
    return get_tasks(state, {"intent_id": intent_id})


def get_child_tasks(state: TaskState, parent_id: str) -> list[Task]:
    return get_tasks(state, {"parent_id": parent_id})
