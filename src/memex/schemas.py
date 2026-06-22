"""Validation entry points — the parity shim for ``@ai2070/memex/schemas``.

In Pydantic the models *are* the schema, so this module re-exports them and
offers ``validate_*`` helpers (backed by ``TypeAdapter`` for the command unions).
Use these to validate untrusted external input before folding it in:

    from memex.schemas import validate_command
    cmd = validate_command(raw)          # raises pydantic.ValidationError on bad shape
    state = apply_command(state, cmd).state
"""

from __future__ import annotations

from typing import Any

from pydantic import TypeAdapter

from .commands import MemoryCommand, MemoryCommandAdapter
from .intent import Intent, IntentCommand
from .models import Edge, MemoryItem
from .task import Task, TaskCommand

__all__ = [
    "validate_command",
    "validate_intent_command",
    "validate_task_command",
    "validate_memory_item",
    "validate_edge",
    # schema aliases (the Pydantic model IS the schema)
    "MemoryItemSchema",
    "EdgeSchema",
    "IntentSchema",
    "TaskSchema",
    "MemoryCommandAdapter",
    "IntentCommandAdapter",
    "TaskCommandAdapter",
]

# The models are the schemas.
MemoryItemSchema = MemoryItem
EdgeSchema = Edge
IntentSchema = Intent
TaskSchema = Task

IntentCommandAdapter: TypeAdapter[IntentCommand] = TypeAdapter(IntentCommand)
TaskCommandAdapter: TypeAdapter[TaskCommand] = TypeAdapter(TaskCommand)


def validate_command(raw: Any) -> MemoryCommand:
    """Validate a raw mapping into a typed memory command."""
    return MemoryCommandAdapter.validate_python(raw)


def validate_intent_command(raw: Any) -> IntentCommand:
    return IntentCommandAdapter.validate_python(raw)


def validate_task_command(raw: Any) -> TaskCommand:
    return TaskCommandAdapter.validate_python(raw)


def validate_memory_item(raw: Any) -> MemoryItem:
    return MemoryItem.model_validate(raw)


def validate_edge(raw: Any) -> Edge:
    return Edge.model_validate(raw)
