"""Transplant — export a slice of the three graphs and import it elsewhere.

The import side optionally re-ids colliding entities (``re_id_on_difference``),
and a per-graph re-id *pre-pass* populates the id maps before anything is
processed so cross-references (``parents`` / ``parent_id`` / ``intent_id`` /
memory id lists) remap correctly regardless of slice ordering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NamedTuple

from pydantic import BaseModel

from ._uuid import uuid7
from .errors import InvalidTimestampError
from .graph import GraphState
from .intent import Intent, IntentState, apply_intent_command
from .models import Edge, MemoryItem
from .query import extract_timestamp, get_children, get_edges
from .reducer import apply_command
from .task import Task, TaskState, apply_task_command

__all__ = [
    "ExportOptions",
    "MemexExport",
    "ImportOptions",
    "ImportBucket",
    "ImportReport",
    "ImportResult",
    "export_slice",
    "import_slice",
]


# ---------------------------------------------------------------------------
# Re-id helpers
# ---------------------------------------------------------------------------


def _uuid_from_ms(ms: int) -> str:
    """Build a UUIDv7-shaped id from a given ms timestamp + random suffix."""
    hex_ms = format(ms, "x").rjust(12, "0")
    rand = uuid7().replace("-", "")
    return "-".join([
        hex_ms[0:8],
        hex_ms[8:12],
        "7" + rand[13:16],
        rand[16:20],
        rand[20:32],
    ])


def _re_id_for(original_id: str, existing_ids: set[str], created_at: int | None = None) -> str:
    """Generate a fresh id 1ms after the original, incrementing on collision."""
    if created_at is None:
        try:
            created_at = extract_timestamp(original_id)
        except InvalidTimestampError as err:
            raise ValueError(
                f'Cannot re-id "{original_id}": provide created_at or use a UUIDv7 id'
            ) from err
    ms = created_at + 1
    new_id = _uuid_from_ms(ms)
    while new_id in existing_ids:
        ms += 1
        new_id = _uuid_from_ms(ms)
    return new_id


def _rewrite_id(id_: str, id_map: dict[str, str]) -> str:
    return id_map.get(id_, id_)


def _rewrite_ids(ids: list[str] | None, id_map: dict[str, str]) -> list[str] | None:
    if ids is None:
        return None
    return [id_map.get(i, i) for i in ids]


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


class ExportOptions(BaseModel):
    memory_ids: list[str] | None = None
    intent_ids: list[str] | None = None
    task_ids: list[str] | None = None
    include_parents: bool = False
    include_children: bool = False
    include_aliases: bool = False
    include_related_tasks: bool = False
    include_related_intents: bool = False


class MemexExport(BaseModel):
    memories: list[MemoryItem] = []
    edges: list[Edge] = []
    intents: list[Intent] = []
    tasks: list[Task] = []


def export_slice(
    mem_state: GraphState,
    intent_state: IntentState,
    task_state: TaskState,
    *,
    memory_ids: list[str] | None = None,
    intent_ids: list[str] | None = None,
    task_ids: list[str] | None = None,
    include_parents: bool = False,
    include_children: bool = False,
    include_aliases: bool = False,
    include_related_tasks: bool = False,
    include_related_intents: bool = False,
) -> MemexExport:
    memory_id_set: set[str] = set(memory_ids or [])
    intent_id_set: set[str] = set(intent_ids or [])
    task_id_set: set[str] = set(task_ids or [])
    edge_id_set: set[str] = set()

    # walk parents up-graph
    if include_parents:
        queue = list(memory_id_set)
        while queue:
            id_ = queue.pop()
            item = mem_state.items.get(id_)
            if item and item.parents:
                for pid in item.parents:
                    if pid not in memory_id_set:
                        memory_id_set.add(pid)
                        queue.append(pid)

    # walk children down-graph
    if include_children:
        queue = list(memory_id_set)
        while queue:
            id_ = queue.pop()
            for child in get_children(mem_state, id_):
                if child.id not in memory_id_set:
                    memory_id_set.add(child.id)
                    queue.append(child.id)

    # walk aliases (both directions)
    if include_aliases:
        queue = list(memory_id_set)
        visited: set[str] = set()
        while queue:
            id_ = queue.pop()
            if id_ in visited:
                continue
            visited.add(id_)
            for edge in get_edges(mem_state, {"from": id_, "kind": "ALIAS", "active_only": True}):
                edge_id_set.add(edge.edge_id)
                if edge.to not in memory_id_set:
                    memory_id_set.add(edge.to)
                    queue.append(edge.to)
            for edge in get_edges(mem_state, {"to": id_, "kind": "ALIAS", "active_only": True}):
                edge_id_set.add(edge.edge_id)
                if edge.from_ not in memory_id_set:
                    memory_id_set.add(edge.from_)
                    queue.append(edge.from_)

    # collect edges between included memories
    for edge in mem_state.edges.values():
        if edge.from_ in memory_id_set and edge.to in memory_id_set:
            edge_id_set.add(edge.edge_id)

    # walk related intents
    if include_related_intents:
        for intent in intent_state.intents.values():
            if intent.root_memory_ids:
                for mid in intent.root_memory_ids:
                    if mid in memory_id_set:
                        intent_id_set.add(intent.id)
                        break
        for mid in memory_id_set:
            item = mem_state.items.get(mid)
            if item and item.intent_id:
                intent_id_set.add(item.intent_id)

    # walk related tasks
    if include_related_tasks:
        for task in task_state.tasks.values():
            if task.intent_id in intent_id_set:
                task_id_set.add(task.id)
                continue
            input_match = task.input_memory_ids and any(mid in memory_id_set for mid in task.input_memory_ids)
            output_match = task.output_memory_ids and any(mid in memory_id_set for mid in task.output_memory_ids)
            if input_match or output_match:
                task_id_set.add(task.id)
        for mid in memory_id_set:
            item = mem_state.items.get(mid)
            if item and item.task_id:
                task_id_set.add(item.task_id)

    memories = [mem_state.items[i] for i in memory_id_set if i in mem_state.items]
    edges = [mem_state.edges[i] for i in edge_id_set if i in mem_state.edges]
    intents = [intent_state.intents[i] for i in intent_id_set if i in intent_state.intents]
    tasks = [task_state.tasks[i] for i in task_id_set if i in task_state.tasks]

    return MemexExport(memories=memories, edges=edges, intents=intents, tasks=tasks)


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------


class ImportOptions(BaseModel):
    skip_existing_ids: bool = True
    shallow_compare_existing: bool = False
    re_id_on_difference: bool = False


@dataclass
class ImportBucket:
    memories: list[str] = field(default_factory=list)
    intents: list[str] = field(default_factory=list)
    tasks: list[str] = field(default_factory=list)
    edges: list[str] = field(default_factory=list)


@dataclass
class ImportReport:
    created: ImportBucket = field(default_factory=ImportBucket)
    updated: ImportBucket = field(default_factory=ImportBucket)
    skipped: ImportBucket = field(default_factory=ImportBucket)
    conflicts: ImportBucket = field(default_factory=ImportBucket)


class ImportResult(NamedTuple):
    mem_state: GraphState
    intent_state: IntentState
    task_state: TaskState
    report: ImportReport


def _shallow_equal(a: BaseModel, b: BaseModel) -> bool:
    """Deep structural equality (order-independent), matching the TS helper."""
    return a.model_dump() == b.model_dump()


def import_slice(
    mem_state: GraphState,
    intent_state: IntentState,
    task_state: TaskState,
    slice: MemexExport | dict[str, Any],
    *,
    skip_existing_ids: bool = True,
    shallow_compare_existing: bool = False,
    re_id_on_difference: bool = False,
) -> ImportResult:
    sl = slice if isinstance(slice, MemexExport) else MemexExport.model_validate(slice)

    skip_existing = skip_existing_ids
    shallow_compare = shallow_compare_existing
    do_re_id = re_id_on_difference

    report = ImportReport()

    mem_id_map: dict[str, str] = {}
    intent_id_map: dict[str, str] = {}
    task_id_map: dict[str, str] = {}

    all_mem_ids = set(mem_state.items.keys())
    all_intent_ids = set(intent_state.intents.keys())
    all_task_ids = set(task_state.tasks.keys())

    current_mem = mem_state
    current_intent = intent_state
    current_task = task_state

    # --- memory re-id pre-pass ---
    if skip_existing and shallow_compare and do_re_id:
        for item in sl.memories:
            existing = mem_state.items.get(item.id)
            if existing is not None and not _shallow_equal(existing, item):
                new_id = _re_id_for(item.id, all_mem_ids, item.created_at)
                all_mem_ids.add(new_id)
                mem_id_map[item.id] = new_id

    # --- import memories ---
    for item in sl.memories:
        existing = current_mem.items.get(item.id)
        if existing is not None:
            if skip_existing:
                if shallow_compare and not _shallow_equal(existing, item):
                    if do_re_id:
                        new_id = mem_id_map[item.id]
                        remapped = item.model_copy(update={
                            "id": new_id,
                            "parents": _rewrite_ids(item.parents, mem_id_map),
                        })
                        current_mem = apply_command(current_mem, {"type": "memory.create", "item": remapped}).state
                        report.created.memories.append(new_id)
                    else:
                        report.conflicts.memories.append(item.id)
                else:
                    report.skipped.memories.append(item.id)
                continue
            # skip_existing False — update existing item
            partial = item.model_dump(exclude_none=True)
            partial.pop("id", None)
            rewritten = _rewrite_ids(item.parents, mem_id_map)
            if rewritten is not None:
                partial["parents"] = rewritten
            current_mem = apply_command(
                current_mem,
                {"type": "memory.update", "item_id": item.id, "partial": partial, "author": item.author},
            ).state
            report.updated.memories.append(item.id)
            continue
        # no collision — create
        remapped = item.model_copy(update={"parents": _rewrite_ids(item.parents, mem_id_map)})
        current_mem = apply_command(current_mem, {"type": "memory.create", "item": remapped}).state
        report.created.memories.append(item.id)

    all_edge_ids = set(current_mem.edges.keys())

    # --- import edges ---
    for edge in sl.edges:
        existing_edge = current_mem.edges.get(edge.edge_id)
        if existing_edge is not None:
            if skip_existing:
                if shallow_compare and not _shallow_equal(existing_edge, edge):
                    if do_re_id:
                        new_id = _re_id_for(edge.edge_id, all_edge_ids)
                        all_edge_ids.add(new_id)
                        remapped_edge = edge.model_copy(update={
                            "edge_id": new_id,
                            "from_": _rewrite_id(edge.from_, mem_id_map),
                            "to": _rewrite_id(edge.to, mem_id_map),
                        })
                        current_mem = apply_command(current_mem, {"type": "edge.create", "edge": remapped_edge}).state
                        report.created.edges.append(new_id)
                    else:
                        report.conflicts.edges.append(edge.edge_id)
                else:
                    report.skipped.edges.append(edge.edge_id)
                continue
            partial = edge.model_dump(exclude_none=True)
            for k in ("edge_id", "from", "from_", "to"):
                partial.pop(k, None)
            current_mem = apply_command(
                current_mem,
                {"type": "edge.update", "edge_id": edge.edge_id, "partial": partial, "author": edge.author},
            ).state
            report.updated.edges.append(edge.edge_id)
            continue
        remapped_edge = edge.model_copy(update={
            "from_": _rewrite_id(edge.from_, mem_id_map),
            "to": _rewrite_id(edge.to, mem_id_map),
        })
        current_mem = apply_command(current_mem, {"type": "edge.create", "edge": remapped_edge}).state
        report.created.edges.append(edge.edge_id)

    # --- intent re-id pre-pass ---
    if skip_existing and shallow_compare and do_re_id:
        for intent in sl.intents:
            existing_intent = current_intent.intents.get(intent.id)
            if existing_intent is not None and not _shallow_equal(existing_intent, intent):
                new_id = _re_id_for(intent.id, all_intent_ids)
                all_intent_ids.add(new_id)
                intent_id_map[intent.id] = new_id

    # --- import intents ---
    for intent in sl.intents:
        existing_intent = current_intent.intents.get(intent.id)
        if existing_intent is not None:
            if skip_existing:
                if shallow_compare and not _shallow_equal(existing_intent, intent):
                    if do_re_id:
                        new_id = intent_id_map[intent.id]
                        remapped_intent = intent.model_copy(update={
                            "id": new_id,
                            "parent_id": _rewrite_id(intent.parent_id, intent_id_map) if intent.parent_id else None,
                            "root_memory_ids": _rewrite_ids(intent.root_memory_ids, mem_id_map),
                        })
                        current_intent = apply_intent_command(current_intent, {"type": "intent.create", "intent": remapped_intent}).state
                        report.created.intents.append(new_id)
                    else:
                        report.conflicts.intents.append(intent.id)
                else:
                    report.skipped.intents.append(intent.id)
                continue
            partial = intent.model_dump(exclude_none=True)
            for k in ("id", "status"):
                partial.pop(k, None)
            partial["parent_id"] = _rewrite_id(intent.parent_id, intent_id_map) if intent.parent_id else None
            partial["root_memory_ids"] = _rewrite_ids(intent.root_memory_ids, mem_id_map)
            current_intent = apply_intent_command(
                current_intent,
                {"type": "intent.update", "intent_id": intent.id, "partial": partial, "author": intent.owner},
            ).state
            report.updated.intents.append(intent.id)
            continue
        remapped_intent = intent.model_copy(update={
            "parent_id": _rewrite_id(intent.parent_id, intent_id_map) if intent.parent_id else None,
            "root_memory_ids": _rewrite_ids(intent.root_memory_ids, mem_id_map),
        })
        current_intent = apply_intent_command(current_intent, {"type": "intent.create", "intent": remapped_intent}).state
        report.created.intents.append(intent.id)

    # --- task re-id pre-pass ---
    if skip_existing and shallow_compare and do_re_id:
        for task in sl.tasks:
            existing_task = current_task.tasks.get(task.id)
            if existing_task is not None and not _shallow_equal(existing_task, task):
                new_id = _re_id_for(task.id, all_task_ids)
                all_task_ids.add(new_id)
                task_id_map[task.id] = new_id

    # --- import tasks ---
    for task in sl.tasks:
        existing_task = current_task.tasks.get(task.id)
        if existing_task is not None:
            if skip_existing:
                if shallow_compare and not _shallow_equal(existing_task, task):
                    if do_re_id:
                        new_id = task_id_map[task.id]
                        remapped_task = task.model_copy(update={
                            "id": new_id,
                            "intent_id": _rewrite_id(task.intent_id, intent_id_map),
                            "parent_id": _rewrite_id(task.parent_id, task_id_map) if task.parent_id else None,
                            "input_memory_ids": _rewrite_ids(task.input_memory_ids, mem_id_map),
                            "output_memory_ids": _rewrite_ids(task.output_memory_ids, mem_id_map),
                        })
                        current_task = apply_task_command(current_task, {"type": "task.create", "task": remapped_task}).state
                        report.created.tasks.append(new_id)
                    else:
                        report.conflicts.tasks.append(task.id)
                else:
                    report.skipped.tasks.append(task.id)
                continue
            partial = task.model_dump(exclude_none=True)
            for k in ("id", "status"):
                partial.pop(k, None)
            partial["intent_id"] = _rewrite_id(task.intent_id, intent_id_map)
            partial["parent_id"] = _rewrite_id(task.parent_id, task_id_map) if task.parent_id else None
            partial["input_memory_ids"] = _rewrite_ids(task.input_memory_ids, mem_id_map)
            partial["output_memory_ids"] = _rewrite_ids(task.output_memory_ids, mem_id_map)
            current_task = apply_task_command(
                current_task,
                {"type": "task.update", "task_id": task.id, "partial": partial, "author": task.agent_id or "system:import"},
            ).state
            report.updated.tasks.append(task.id)
            continue
        remapped_task = task.model_copy(update={
            "intent_id": _rewrite_id(task.intent_id, intent_id_map),
            "parent_id": _rewrite_id(task.parent_id, task_id_map) if task.parent_id else None,
            "input_memory_ids": _rewrite_ids(task.input_memory_ids, mem_id_map),
            "output_memory_ids": _rewrite_ids(task.output_memory_ids, mem_id_map),
        })
        current_task = apply_task_command(current_task, {"type": "task.create", "task": remapped_task}).state
        report.created.tasks.append(task.id)

    # --- second pass: remap intent_id / task_id on imported memories ---
    if intent_id_map or task_id_map:
        imported_mem_ids = [*report.created.memories, *report.updated.memories]
        for mem_id in imported_mem_ids:
            stored_item = current_mem.items.get(mem_id)
            if stored_item is None:
                continue
            new_intent_id = intent_id_map.get(stored_item.intent_id) if stored_item.intent_id else None
            new_task_id = task_id_map.get(stored_item.task_id) if stored_item.task_id else None
            if new_intent_id or new_task_id:
                partial = {}
                if new_intent_id:
                    partial["intent_id"] = new_intent_id
                if new_task_id:
                    partial["task_id"] = new_task_id
                current_mem = apply_command(
                    current_mem,
                    {"type": "memory.update", "item_id": mem_id, "partial": partial, "author": "system:import"},
                ).state

    return ImportResult(current_mem, current_intent, current_task, report)
