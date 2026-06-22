"""Provenance walks, contradiction-aware packing, diversity, and smart retrieval."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import cmp_to_key
from typing import Any

from pydantic import BaseModel

from .graph import GraphState
from .models import Edge, MemoryFilter, MemoryItem, ScoredItem, ScoreWeights
from .query import get_edges, get_scored_items

__all__ = [
    "SupportNode",
    "DiversityOptions",
    "get_support_tree",
    "get_support_set",
    "filter_contradictions",
    "surface_contradictions",
    "apply_diversity",
    "smart_retrieve",
]


# ---------------------------------------------------------------------------
# 1. Support tree — provenance walk
# ---------------------------------------------------------------------------


@dataclass
class SupportNode:
    item: MemoryItem
    parents: list[SupportNode]


def get_support_tree(state: GraphState, item_id: str) -> SupportNode | None:
    """Build the full provenance tree for an item, deduplicating on cycles."""
    if item_id not in state.items:
        return None

    visited: set[str] = set()

    def walk(node_id: str) -> SupportNode | None:
        current = state.items.get(node_id)
        if current is None:
            return None
        if node_id in visited:
            return SupportNode(item=current, parents=[])
        visited.add(node_id)
        parent_nodes: list[SupportNode] = []
        if current.parents:
            for pid in current.parents:
                node = walk(pid)
                if node is not None:
                    parent_nodes.append(node)
        return SupportNode(item=current, parents=parent_nodes)

    return walk(item_id)


def get_support_set(state: GraphState, item_id: str) -> list[MemoryItem]:
    """Flatten the provenance chain into the set of items that justify a claim."""
    if item_id not in state.items:
        return []

    visited: set[str] = set()
    result: list[MemoryItem] = []

    def walk(node_id: str) -> None:
        if node_id in visited:
            return
        visited.add(node_id)
        current = state.items.get(node_id)
        if current is None:
            return
        result.append(current)
        if current.parents:
            for pid in current.parents:
                walk(pid)

    walk(item_id)
    return result


# ---------------------------------------------------------------------------
# 2. Contradiction-aware packing
# ---------------------------------------------------------------------------


def _superseded_ids(state: GraphState) -> set[str]:
    superseded: set[str] = set()
    for edge in state.edges.values():
        if edge.kind == "SUPERSEDES" and edge.active:
            superseded.add(edge.to)
    return superseded


def filter_contradictions(state: GraphState, scored: list[ScoredItem]) -> list[ScoredItem]:
    """Collapse contradictions: drop superseded items and the lower-scoring side
    of each unresolved CONTRADICTS pair (deterministic tie-breaks)."""
    superseded = _superseded_ids(state)
    filtered = [s for s in scored if s.item.id not in superseded]

    contradict_edges = get_edges(state, {"kind": "CONTRADICTS", "active_only": True})
    if contradict_edges:
        score_map = {entry.item.id: entry.score for entry in filtered}

        def _cmp(a: Edge, b: Edge) -> int:
            # Highest max-score pair first, then highest min-score, then edge_id.
            max_a = max(score_map.get(a.from_, -1), score_map.get(a.to, -1))
            max_b = max(score_map.get(b.from_, -1), score_map.get(b.to, -1))
            if max_a != max_b:
                return -1 if max_a > max_b else 1
            min_a = min(score_map.get(a.from_, -1), score_map.get(a.to, -1))
            min_b = min(score_map.get(b.from_, -1), score_map.get(b.to, -1))
            if min_a != min_b:
                return -1 if min_a > min_b else 1
            return -1 if a.edge_id < b.edge_id else 1

        contradict_edges = sorted(contradict_edges, key=cmp_to_key(_cmp))

        excluded: set[str] = set()
        for edge in contradict_edges:
            if edge.from_ in excluded or edge.to in excluded:
                continue
            score_a = score_map.get(edge.from_, -1)
            score_b = score_map.get(edge.to, -1)
            if score_a >= 0 and score_b >= 0:
                if score_a != score_b:
                    excluded.add(edge.to if score_a > score_b else edge.from_)
                else:
                    excluded.add(edge.to if edge.from_ < edge.to else edge.from_)

        if excluded:
            filtered = [s for s in filtered if s.item.id not in excluded]

    return filtered


def surface_contradictions(state: GraphState, scored: list[ScoredItem]) -> list[ScoredItem]:
    """Keep both sides of each contradiction, annotated via ``contradicted_by``.
    Superseded items are still removed."""
    superseded = _superseded_ids(state)
    result = [
        ScoredItem(
            item=s.item,
            score=s.score,
            contradicted_by=list(s.contradicted_by) if s.contradicted_by else None,
        )
        for s in scored
        if s.item.id not in superseded
    ]

    contradict_edges = get_edges(state, {"kind": "CONTRADICTS", "active_only": True})
    if not contradict_edges:
        return result

    item_map = {entry.item.id: entry for entry in result}

    # Dedup by item id — multiple/bidirectional CONTRADICTS edges may connect the
    # same pair, and a self-edge makes a is b.
    for edge in contradict_edges:
        a = item_map.get(edge.from_)
        b = item_map.get(edge.to)
        if a is None or b is None:
            continue
        if a is b:
            continue  # ignore self-contradictions
        if a.contradicted_by is None:
            a.contradicted_by = []
        if not any(i.id == b.item.id for i in a.contradicted_by):
            a.contradicted_by.append(b.item)
        if b.contradicted_by is None:
            b.contradicted_by = []
        if not any(i.id == a.item.id for i in b.contradicted_by):
            b.contradicted_by.append(a.item)

    return result


# ---------------------------------------------------------------------------
# 3. Diversity scoring
# ---------------------------------------------------------------------------


class DiversityOptions(BaseModel):
    author_penalty: float | None = None
    parent_penalty: float | None = None
    source_penalty: float | None = None


def apply_diversity(
    scored: list[ScoredItem],
    options: DiversityOptions | dict[str, Any],
) -> list[ScoredItem]:
    """Re-rank scored items with per-duplicate penalties (author/parent/source)."""
    opts = options if isinstance(options, DiversityOptions) else DiversityOptions.model_validate(options)

    author_counts: dict[str, int] | None = {} if opts.author_penalty else None
    parent_counts: dict[str, int] | None = {} if opts.parent_penalty else None
    source_counts: dict[str, int] | None = {} if opts.source_penalty else None

    diversified: list[ScoredItem] = []
    for entry in scored:
        penalty = 0.0
        if author_counts is not None:
            count = author_counts.get(entry.item.author, 0)
            penalty += count * opts.author_penalty  # type: ignore[operator]
            author_counts[entry.item.author] = count + 1
        if parent_counts is not None and entry.item.parents:
            for pid in entry.item.parents:
                count = parent_counts.get(pid, 0)
                penalty += count * opts.parent_penalty  # type: ignore[operator]
                parent_counts[pid] = count + 1
        if source_counts is not None:
            count = source_counts.get(entry.item.source_kind, 0)
            penalty += count * opts.source_penalty  # type: ignore[operator]
            source_counts[entry.item.source_kind] = count + 1
        diversified.append(
            ScoredItem(item=entry.item, score=max(0.0, entry.score - penalty), contradicted_by=entry.contradicted_by)
        )

    diversified.sort(key=lambda s: s.score, reverse=True)
    return diversified


# ---------------------------------------------------------------------------
# 4. Combined smart retrieval
# ---------------------------------------------------------------------------


def smart_retrieve(
    state: GraphState,
    *,
    budget: float,
    cost_fn: Callable[[MemoryItem], float],
    weights: ScoreWeights | dict[str, Any],
    filter: MemoryFilter | dict[str, Any] | None = None,
    contradictions: str | None = None,
    diversity: DiversityOptions | dict[str, Any] | None = None,
) -> list[ScoredItem]:
    """Score -> contradiction policy -> diversity -> greedy budget pack."""
    scored = get_scored_items(state, weights, {"pre": filter})

    if contradictions == "filter":
        scored = filter_contradictions(state, scored)
    elif contradictions == "surface":
        scored = surface_contradictions(state, scored)

    if diversity is not None:
        scored = apply_diversity(scored, diversity)

    results: list[ScoredItem] = []
    remaining = budget
    for entry in scored:
        cost = cost_fn(entry.item)
        if cost < 0 or not math.isfinite(cost):
            raise ValueError(f"cost_fn must return a finite non-negative number, got {cost}")
        if cost <= remaining:
            results.append(entry)
            remaining -= cost
    return results
