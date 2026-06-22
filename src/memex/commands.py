"""Memory commands as a Pydantic discriminated union.

This single module replaces the entire ``schemas.ts`` / Zod layer: in Pydantic
the command models *are* the schema. ``apply_command`` accepts either a command
model instance or a plain dict (validated through ``MemoryCommandAdapter``),
which keeps the TS object-literal call style portable.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, TypeAdapter

from .models import Edge, MemoryItem

__all__ = [
    "MemoryCreate",
    "MemoryUpdate",
    "MemoryRetract",
    "EdgeCreate",
    "EdgeUpdate",
    "EdgeRetract",
    "MemoryCommand",
    "MemoryCommandAdapter",
]


class MemoryCreate(BaseModel):
    type: Literal["memory.create"] = "memory.create"
    item: MemoryItem


class MemoryUpdate(BaseModel):
    type: Literal["memory.update"] = "memory.update"
    item_id: str
    partial: dict[str, Any]
    author: str
    reason: str | None = None
    basis: dict[str, Any] | None = None


class MemoryRetract(BaseModel):
    type: Literal["memory.retract"] = "memory.retract"
    item_id: str
    author: str
    reason: str | None = None


class EdgeCreate(BaseModel):
    type: Literal["edge.create"] = "edge.create"
    edge: Edge


class EdgeUpdate(BaseModel):
    type: Literal["edge.update"] = "edge.update"
    edge_id: str
    partial: dict[str, Any]
    author: str
    reason: str | None = None


class EdgeRetract(BaseModel):
    type: Literal["edge.retract"] = "edge.retract"
    edge_id: str
    author: str
    reason: str | None = None


MemoryCommand = Annotated[
    MemoryCreate | MemoryUpdate | MemoryRetract | EdgeCreate | EdgeUpdate | EdgeRetract,
    Field(discriminator="type"),
]

MemoryCommandAdapter: TypeAdapter[MemoryCommand] = TypeAdapter(MemoryCommand)
