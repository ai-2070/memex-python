"""Port of tests/task.test.ts."""

from __future__ import annotations

from typing import Any

import pytest

from memex import (
    DuplicateTaskError,
    InvalidTaskTransitionError,
    Task,
    TaskNotFoundError,
    apply_task_command,
    create_task,
    create_task_state,
    get_task_by_id,
    get_tasks,
    get_tasks_by_intent,
)


def make_task(**overrides: Any) -> Task:
    base: dict[str, Any] = {
        "id": "t1", "intent_id": "i1", "action": "search_linkedin",
        "status": "pending", "priority": 0.7, "attempt": 0,
    }
    base.update(overrides)
    return Task(**base)


def created_state(task: Task | None = None):
    return apply_task_command(create_task_state(), {"type": "task.create", "task": task or make_task()}).state


# --- task.create -----------------------------------------------------------


def test_create_task() -> None:
    task = make_task()
    state, events = apply_task_command(create_task_state(), {"type": "task.create", "task": task})
    assert state.tasks["t1"] == task
    assert len(events) == 1
    assert events[0].type == "task.created"
    assert events[0].namespace == "task"


def test_create_duplicate_raises() -> None:
    state = created_state()
    with pytest.raises(DuplicateTaskError):
        apply_task_command(state, {"type": "task.create", "task": make_task()})


def test_create_does_not_mutate_original() -> None:
    state = create_task_state()
    apply_task_command(state, {"type": "task.create", "task": make_task()})
    assert len(state.tasks) == 0


# --- task.update -----------------------------------------------------------


def test_update_priority() -> None:
    state = created_state()
    nxt, _ = apply_task_command(state, {"type": "task.update", "task_id": "t1", "partial": {"priority": 0.2}, "author": "test"})
    assert nxt.tasks["t1"].priority == 0.2


def test_update_missing_raises() -> None:
    with pytest.raises(TaskNotFoundError):
        apply_task_command(create_task_state(), {"type": "task.update", "task_id": "nope", "partial": {}, "author": "test"})


# --- task lifecycle --------------------------------------------------------


def test_pending_to_running() -> None:
    state = created_state()
    nxt, events = apply_task_command(state, {"type": "task.start", "task_id": "t1", "agent_id": "agent:worker"})
    assert nxt.tasks["t1"].status == "running"
    assert nxt.tasks["t1"].agent_id == "agent:worker"
    assert nxt.tasks["t1"].attempt == 1
    assert events[0].type == "task.started"


def test_running_to_completed() -> None:
    state = created_state()
    state = apply_task_command(state, {"type": "task.start", "task_id": "t1"}).state
    nxt, events = apply_task_command(state, {"type": "task.complete", "task_id": "t1", "result": {"found": True}, "output_memory_ids": ["m5"]})
    assert nxt.tasks["t1"].status == "completed"
    assert nxt.tasks["t1"].result == {"found": True}
    assert nxt.tasks["t1"].output_memory_ids == ["m5"]
    assert events[0].type == "task.completed"


def test_running_to_failed() -> None:
    state = created_state()
    state = apply_task_command(state, {"type": "task.start", "task_id": "t1"}).state
    nxt, events = apply_task_command(state, {"type": "task.fail", "task_id": "t1", "error": "timeout"})
    assert nxt.tasks["t1"].status == "failed"
    assert nxt.tasks["t1"].error == "timeout"
    assert events[0].type == "task.failed"


def test_failed_to_running_retry() -> None:
    state = created_state()
    state = apply_task_command(state, {"type": "task.start", "task_id": "t1"}).state
    state = apply_task_command(state, {"type": "task.fail", "task_id": "t1", "error": "timeout"}).state
    nxt, _ = apply_task_command(state, {"type": "task.start", "task_id": "t1"})
    assert nxt.tasks["t1"].status == "running"
    assert nxt.tasks["t1"].attempt == 2


def test_pending_to_cancelled() -> None:
    state = created_state()
    nxt, events = apply_task_command(state, {"type": "task.cancel", "task_id": "t1", "reason": "no longer needed"})
    assert nxt.tasks["t1"].status == "cancelled"
    assert events[0].type == "task.cancelled"


def test_running_to_cancelled() -> None:
    state = created_state()
    state = apply_task_command(state, {"type": "task.start", "task_id": "t1"}).state
    nxt, _ = apply_task_command(state, {"type": "task.cancel", "task_id": "t1"})
    assert nxt.tasks["t1"].status == "cancelled"


def test_completed_to_start_raises() -> None:
    state = created_state()
    state = apply_task_command(state, {"type": "task.start", "task_id": "t1"}).state
    state = apply_task_command(state, {"type": "task.complete", "task_id": "t1"}).state
    with pytest.raises(InvalidTaskTransitionError):
        apply_task_command(state, {"type": "task.start", "task_id": "t1"})


def test_completed_to_cancel_raises() -> None:
    state = created_state()
    state = apply_task_command(state, {"type": "task.start", "task_id": "t1"}).state
    state = apply_task_command(state, {"type": "task.complete", "task_id": "t1"}).state
    with pytest.raises(InvalidTaskTransitionError):
        apply_task_command(state, {"type": "task.cancel", "task_id": "t1"})


def test_pending_to_complete_raises() -> None:
    state = created_state()
    with pytest.raises(InvalidTaskTransitionError):
        apply_task_command(state, {"type": "task.complete", "task_id": "t1"})


# --- createTask factory ----------------------------------------------------


def test_create_task_factory() -> None:
    task = create_task(intent_id="i1", action="search", priority=0.5)
    assert task.id
    assert task.status == "pending"
    assert task.attempt == 0


# --- task queries ----------------------------------------------------------


def query_state():
    state = create_task_state()
    state = apply_task_command(state, {"type": "task.create", "task": make_task(id="t1", intent_id="i1", action="search", status="pending", priority=0.9, agent_id="agent:a", input_memory_ids=["m1"])}).state
    state = apply_task_command(state, {"type": "task.create", "task": make_task(id="t2", intent_id="i1", action="summarize", status="running", priority=0.5, agent_id="agent:b")}).state
    state = apply_task_command(state, {"type": "task.create", "task": make_task(id="t3", intent_id="i2", action="search", status="completed", priority=0.3, output_memory_ids=["m5"])}).state
    return state


def test_get_tasks_all() -> None:
    assert len(get_tasks(query_state())) == 3


def test_get_tasks_by_intent_id() -> None:
    assert len(get_tasks(query_state(), {"intent_id": "i1"})) == 2


def test_get_tasks_by_action() -> None:
    assert len(get_tasks(query_state(), {"action": "search"})) == 2


def test_get_tasks_by_status() -> None:
    result = get_tasks(query_state(), {"status": "running"})
    assert len(result) == 1 and result[0].id == "t2"


def test_get_tasks_by_statuses() -> None:
    assert len(get_tasks(query_state(), {"statuses": ["pending", "running"]})) == 2


def test_get_tasks_by_agent_id() -> None:
    result = get_tasks(query_state(), {"agent_id": "agent:a"})
    assert len(result) == 1 and result[0].id == "t1"


def test_get_tasks_by_min_priority() -> None:
    assert len(get_tasks(query_state(), {"min_priority": 0.5})) == 2


def test_get_tasks_by_has_input_memory_id() -> None:
    result = get_tasks(query_state(), {"has_input_memory_id": "m1"})
    assert len(result) == 1 and result[0].id == "t1"


def test_get_tasks_by_has_output_memory_id() -> None:
    result = get_tasks(query_state(), {"has_output_memory_id": "m5"})
    assert len(result) == 1 and result[0].id == "t3"


def test_get_task_by_id() -> None:
    state = query_state()
    assert get_task_by_id(state, "t2").action == "summarize"
    assert get_task_by_id(state, "nope") is None


def test_get_tasks_by_intent() -> None:
    assert len(get_tasks_by_intent(query_state(), "i1")) == 2
