"""Port of tests/intent.test.ts."""

from __future__ import annotations

from typing import Any

import pytest

from memex import (
    DuplicateIntentError,
    Intent,
    IntentNotFoundError,
    InvalidIntentTransitionError,
    apply_intent_command,
    create_intent,
    create_intent_state,
    get_intent_by_id,
    get_intents,
)


def make_intent(**overrides: Any) -> Intent:
    base: dict[str, Any] = {
        "id": "i1", "label": "find_kati", "priority": 0.8, "owner": "user:laz", "status": "active",
    }
    base.update(overrides)
    return Intent(**base)


def created_state(intent: Intent | None = None):
    return apply_intent_command(
        create_intent_state(), {"type": "intent.create", "intent": intent or make_intent()}
    ).state


# --- intent.create ---------------------------------------------------------


def test_create_intent() -> None:
    intent = make_intent()
    state, events = apply_intent_command(create_intent_state(), {"type": "intent.create", "intent": intent})
    assert state.intents["i1"] == intent
    assert len(events) == 1
    assert events[0].type == "intent.created"
    assert events[0].namespace == "intent"


def test_create_duplicate_raises() -> None:
    state = created_state()
    with pytest.raises(DuplicateIntentError):
        apply_intent_command(state, {"type": "intent.create", "intent": make_intent()})


def test_create_does_not_mutate_original() -> None:
    state = create_intent_state()
    apply_intent_command(state, {"type": "intent.create", "intent": make_intent()})
    assert len(state.intents) == 0


# --- intent.update ---------------------------------------------------------


def test_update_priority() -> None:
    state = created_state()
    nxt, events = apply_intent_command(
        state, {"type": "intent.update", "intent_id": "i1", "partial": {"priority": 0.5}, "author": "user:laz"}
    )
    assert nxt.intents["i1"].priority == 0.5
    assert events[0].type == "intent.updated"


def test_update_cannot_change_id() -> None:
    state = created_state()
    nxt, _ = apply_intent_command(
        state, {"type": "intent.update", "intent_id": "i1", "partial": {"id": "sneaky"}, "author": "test"}
    )
    assert nxt.intents["i1"].id == "i1"


def test_update_missing_raises() -> None:
    with pytest.raises(IntentNotFoundError):
        apply_intent_command(
            create_intent_state(),
            {"type": "intent.update", "intent_id": "nope", "partial": {"priority": 0.1}, "author": "test"},
        )


# --- status transitions ----------------------------------------------------


def test_active_to_paused() -> None:
    state = created_state(make_intent(status="active"))
    nxt, events = apply_intent_command(state, {"type": "intent.pause", "intent_id": "i1", "author": "user:laz"})
    assert nxt.intents["i1"].status == "paused"
    assert events[0].type == "intent.paused"


def test_paused_to_active_resume() -> None:
    state = created_state(make_intent(status="active"))
    state = apply_intent_command(state, {"type": "intent.pause", "intent_id": "i1", "author": "test"}).state
    nxt, _ = apply_intent_command(state, {"type": "intent.resume", "intent_id": "i1", "author": "test"})
    assert nxt.intents["i1"].status == "active"


def test_active_to_completed() -> None:
    state = created_state(make_intent(status="active"))
    nxt, events = apply_intent_command(state, {"type": "intent.complete", "intent_id": "i1", "author": "test"})
    assert nxt.intents["i1"].status == "completed"
    assert events[0].type == "intent.completed"


def test_active_to_cancelled() -> None:
    state = created_state(make_intent(status="active"))
    nxt, _ = apply_intent_command(state, {"type": "intent.cancel", "intent_id": "i1", "author": "test"})
    assert nxt.intents["i1"].status == "cancelled"


def test_paused_to_completed() -> None:
    state = created_state(make_intent(status="active"))
    state = apply_intent_command(state, {"type": "intent.pause", "intent_id": "i1", "author": "test"}).state
    nxt, _ = apply_intent_command(state, {"type": "intent.complete", "intent_id": "i1", "author": "test"})
    assert nxt.intents["i1"].status == "completed"


def test_completed_to_pause_raises() -> None:
    state = created_state(make_intent(status="active"))
    state = apply_intent_command(state, {"type": "intent.complete", "intent_id": "i1", "author": "test"}).state
    with pytest.raises(InvalidIntentTransitionError):
        apply_intent_command(state, {"type": "intent.pause", "intent_id": "i1", "author": "test"})


def test_cancelled_to_resume_raises() -> None:
    state = created_state(make_intent(status="active"))
    state = apply_intent_command(state, {"type": "intent.cancel", "intent_id": "i1", "author": "test"}).state
    with pytest.raises(InvalidIntentTransitionError):
        apply_intent_command(state, {"type": "intent.resume", "intent_id": "i1", "author": "test"})


# --- createIntent factory --------------------------------------------------


def test_create_intent_factory() -> None:
    intent = create_intent(label="test", priority=0.5, owner="user:laz")
    assert intent.id
    assert intent.status == "active"


# --- getIntents ------------------------------------------------------------


def query_state():
    state = create_intent_state()
    state = apply_intent_command(state, {"type": "intent.create", "intent": make_intent(id="i1", owner="user:laz", status="active", priority=0.9, root_memory_ids=["m1"])}).state
    state = apply_intent_command(state, {"type": "intent.create", "intent": make_intent(id="i2", owner="agent:reasoner", status="paused", priority=0.3)}).state
    state = apply_intent_command(state, {"type": "intent.create", "intent": make_intent(id="i3", owner="user:laz", status="completed", priority=0.7)}).state
    return state


def test_get_intents_all() -> None:
    assert len(get_intents(query_state())) == 3


def test_get_intents_by_owner() -> None:
    assert len(get_intents(query_state(), {"owner": "user:laz"})) == 2


def test_get_intents_by_status() -> None:
    result = get_intents(query_state(), {"status": "active"})
    assert len(result) == 1 and result[0].id == "i1"


def test_get_intents_by_statuses() -> None:
    assert len(get_intents(query_state(), {"statuses": ["active", "paused"]})) == 2


def test_get_intents_by_min_priority() -> None:
    assert len(get_intents(query_state(), {"min_priority": 0.5})) == 2


def test_get_intents_by_has_memory_id() -> None:
    result = get_intents(query_state(), {"has_memory_id": "m1"})
    assert len(result) == 1 and result[0].id == "i1"


def test_get_intent_by_id() -> None:
    state = query_state()
    assert get_intent_by_id(state, "i2").owner == "agent:reasoner"
    assert get_intent_by_id(state, "nope") is None
