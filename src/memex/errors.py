"""Domain errors for the memory graph.

Score-bound violations surface as ``pydantic.ValidationError`` (see D2/D7 in
PLAN.md); ``cost_fn`` contract violations surface as ``ValueError``. The typed
exceptions below mirror the named error classes thrown by the reducer layer in
the TypeScript library. Intent/Task graphs add their own typed errors in their
respective modules.
"""

from __future__ import annotations

__all__ = [
    "MemexError",
    "MemoryNotFoundError",
    "EdgeNotFoundError",
    "DuplicateMemoryError",
    "DuplicateEdgeError",
    "InvalidTimestampError",
]


class MemexError(Exception):
    """Base class for all memex domain errors."""


class MemoryNotFoundError(MemexError):
    def __init__(self, item_id: str) -> None:
        super().__init__(f"Memory item not found: {item_id}")
        self.item_id = item_id


class EdgeNotFoundError(MemexError):
    def __init__(self, edge_id: str) -> None:
        super().__init__(f"Edge not found: {edge_id}")
        self.edge_id = edge_id


class DuplicateMemoryError(MemexError):
    def __init__(self, item_id: str) -> None:
        super().__init__(f"Memory item already exists: {item_id}")
        self.item_id = item_id


class DuplicateEdgeError(MemexError):
    def __init__(self, edge_id: str) -> None:
        super().__init__(f"Edge already exists: {edge_id}")
        self.edge_id = edge_id


class InvalidTimestampError(MemexError):
    """Raised when a timestamp cannot be extracted or parsed from input."""
