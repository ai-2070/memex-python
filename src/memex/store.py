"""``MemexStore`` — an optional stateful, OO facade over the functional core.

The functional API (``apply_command`` / ``get_items`` / ...) is the backbone and
stays pure. ``MemexStore`` holds the three graph states, rebinds them on each
mutation, and returns the emitted lifecycle events — convenient for agents and
daemons that don't want to thread ``state =`` through every call.
"""

from __future__ import annotations

from typing import Any

from . import bulk, integrity, query, retrieval, serialization, stats
from .factories import create_edge, create_memory_item
from .graph import GraphState, create_graph_state
from .integrity import Contradiction, StaleItem
from .intent import (
    Intent,
    IntentLifecycleEvent,
    IntentState,
    apply_intent_command,
    create_intent,
    create_intent_state,
    get_intents,
)
from .models import Edge, MemoryItem, MemoryLifecycleEvent, ScoredItem
from .reducer import apply_command
from .retrieval import SupportNode
from .stats import GraphStats
from .task import (
    Task,
    TaskLifecycleEvent,
    TaskState,
    apply_task_command,
    create_task,
    create_task_state,
    get_tasks,
)
from .transplant import ImportReport, MemexExport, export_slice, import_slice

__all__ = ["MemexStore"]


class MemexStore:
    """A mutable container over the Memory / Intent / Task graphs."""

    def __init__(
        self,
        mem: GraphState | None = None,
        intents: IntentState | None = None,
        tasks: TaskState | None = None,
    ) -> None:
        self.mem = mem if mem is not None else create_graph_state()
        self.intents = intents if intents is not None else create_intent_state()
        self.tasks = tasks if tasks is not None else create_task_state()

    # -- memory mutations ---------------------------------------------------

    def apply(self, cmd: Any) -> list[MemoryLifecycleEvent]:
        result = apply_command(self.mem, cmd)
        self.mem = result.state
        return result.events

    def create(self, **kwargs: Any) -> MemoryItem:
        item = create_memory_item(**kwargs)
        self.apply({"type": "memory.create", "item": item})
        return item

    def add(self, item: MemoryItem) -> MemoryItem:
        self.apply({"type": "memory.create", "item": item})
        return item

    def update(self, item_id: str, partial: dict[str, Any], author: str, reason: str | None = None) -> list[MemoryLifecycleEvent]:
        return self.apply({"type": "memory.update", "item_id": item_id, "partial": partial, "author": author, "reason": reason})

    def retract(self, item_id: str, author: str, reason: str | None = None) -> list[MemoryLifecycleEvent]:
        return self.apply({"type": "memory.retract", "item_id": item_id, "author": author, "reason": reason})

    def add_edge(self, **kwargs: Any) -> Edge:
        edge = create_edge(**kwargs)
        self.apply({"type": "edge.create", "edge": edge})
        return edge

    # -- memory queries -----------------------------------------------------

    def items(self, filter: Any = None, options: Any = None) -> list[MemoryItem]:
        return query.get_items(self.mem, filter, options)

    def item(self, id: str) -> MemoryItem | None:
        return query.get_item_by_id(self.mem, id)

    def scored(self, weights: Any, options: Any = None) -> list[ScoredItem]:
        return query.get_scored_items(self.mem, weights, options)

    def edges(self, filter: Any = None) -> list[Edge]:
        return query.get_edges(self.mem, filter)

    def parents(self, item_id: str) -> list[MemoryItem]:
        return query.get_parents(self.mem, item_id)

    def children(self, item_id: str) -> list[MemoryItem]:
        return query.get_children(self.mem, item_id)

    def related(self, item_id: str, direction: str = "both") -> list[MemoryItem]:
        return query.get_related_items(self.mem, item_id, direction)

    def smart_retrieve(self, **kwargs: Any) -> list[ScoredItem]:
        return retrieval.smart_retrieve(self.mem, **kwargs)

    def support_tree(self, item_id: str) -> SupportNode | None:
        return retrieval.get_support_tree(self.mem, item_id)

    def support_set(self, item_id: str) -> list[MemoryItem]:
        return retrieval.get_support_set(self.mem, item_id)

    def stats(self) -> GraphStats:
        return stats.get_stats(self.mem)

    # -- integrity ----------------------------------------------------------

    def mark_contradiction(self, a: str, b: str, author: str, meta: dict[str, Any] | None = None) -> list[MemoryLifecycleEvent]:
        result = integrity.mark_contradiction(self.mem, a, b, author, meta)
        self.mem = result.state
        return result.events

    def resolve_contradiction(self, winner: str, loser: str, author: str, reason: str | None = None) -> list[MemoryLifecycleEvent]:
        result = integrity.resolve_contradiction(self.mem, winner, loser, author, reason)
        self.mem = result.state
        return result.events

    def mark_alias(self, a: str, b: str, author: str, meta: dict[str, Any] | None = None) -> list[MemoryLifecycleEvent]:
        result = integrity.mark_alias(self.mem, a, b, author, meta)
        self.mem = result.state
        return result.events

    def cascade_retract(self, item_id: str, author: str, reason: str | None = None) -> list[str]:
        result = integrity.cascade_retract(self.mem, item_id, author, reason)
        self.mem = result.state
        return result.retracted

    def contradictions(self) -> list[Contradiction]:
        return integrity.get_contradictions(self.mem)

    def stale_items(self) -> list[StaleItem]:
        return integrity.get_stale_items(self.mem)

    def aliases(self, item_id: str) -> list[MemoryItem]:
        return integrity.get_aliases(self.mem, item_id)

    def alias_group(self, item_id: str) -> list[MemoryItem]:
        return integrity.get_alias_group(self.mem, item_id)

    # -- bulk ---------------------------------------------------------------

    def apply_many(self, filter: Any, transform: Any, author: str, reason: str | None = None, options: Any = None) -> list[MemoryLifecycleEvent]:
        result = bulk.apply_many(self.mem, filter, transform, author, reason, options)
        self.mem = result.state
        return result.events

    def bulk_adjust_scores(self, criteria: Any, delta: Any, author: str, reason: str | None = None) -> list[MemoryLifecycleEvent]:
        result = bulk.bulk_adjust_scores(self.mem, criteria, delta, author, reason)
        self.mem = result.state
        return result.events

    def decay_importance(self, older_than_ms: int, factor: float, author: str, reason: str | None = None) -> list[MemoryLifecycleEvent]:
        result = bulk.decay_importance(self.mem, older_than_ms, factor, author, reason)
        self.mem = result.state
        return result.events

    # -- intent graph -------------------------------------------------------

    def apply_intent(self, cmd: Any) -> list[IntentLifecycleEvent]:
        result = apply_intent_command(self.intents, cmd)
        self.intents = result.state
        return result.events

    def create_intent(self, **kwargs: Any) -> Intent:
        intent = create_intent(**kwargs)
        self.apply_intent({"type": "intent.create", "intent": intent})
        return intent

    def get_intents(self, filter: Any = None) -> list[Intent]:
        return get_intents(self.intents, filter)

    # -- task graph ---------------------------------------------------------

    def apply_task(self, cmd: Any) -> list[TaskLifecycleEvent]:
        result = apply_task_command(self.tasks, cmd)
        self.tasks = result.state
        return result.events

    def create_task(self, **kwargs: Any) -> Task:
        task = create_task(**kwargs)
        self.apply_task({"type": "task.create", "task": task})
        return task

    def get_tasks(self, filter: Any = None) -> list[Task]:
        return get_tasks(self.tasks, filter)

    # -- transplant ---------------------------------------------------------

    def export_slice(self, **kwargs: Any) -> MemexExport:
        return export_slice(self.mem, self.intents, self.tasks, **kwargs)

    def import_slice(self, slice: MemexExport | dict[str, Any], **kwargs: Any) -> ImportReport:
        result = import_slice(self.mem, self.intents, self.tasks, slice, **kwargs)
        self.mem = result.mem_state
        self.intents = result.intent_state
        self.tasks = result.task_state
        return result.report

    # -- serialization (memory graph) --------------------------------------

    def to_json(self) -> serialization.SerializedGraphState:
        return serialization.to_json(self.mem)

    def dumps(self, pretty: bool = False) -> str:
        return serialization.stringify(self.mem, pretty)

    @classmethod
    def loads(cls, json_str: str) -> MemexStore:
        return cls(mem=serialization.parse(json_str))
