"""Micro-benchmarks for the graph hot paths.

Standalone (not collected by pytest — ``testpaths`` is ``tests``). Run with::

    python benchmarks/perf.py

Each benchmark contrasts the shipped, index-driven implementation with an inline
"naive" reference mirroring the pre-optimization code, so the speedup — and the
shift from quadratic to roughly linear scaling — is visible directly.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from functools import cmp_to_key
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from memex import (  # noqa: E402
    GraphState,
    MemoryItem,
    apply_command,
    cascade_retract,
    create_graph_state,
    create_memory_item,
    get_alias_group,
    get_dependents,
    get_items,
)
from memex.models import Edge  # noqa: E402
from memex.query import get_children, get_edges, get_sort_value  # noqa: E402


def _time(fn: Callable[[], Any], repeat: int = 1) -> float:
    """Return best-of-``repeat`` wall time in milliseconds."""
    best = float("inf")
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best * 1000


def _bench(title: str, sizes: list[int], variants: dict[str, Callable[[int], float]]) -> None:
    print(f"\n{title}")
    cols = "".join(f"{name:>18}" for name in variants)
    print(f"{'n':>8}{cols}{'speedup':>12}")
    for n in sizes:
        times = {name: fn(n) for name, fn in variants.items()}
        row = "".join(f"{times[name]:>15.2f}ms" for name in variants)
        vals = list(times.values())
        speedup = (vals[0] / vals[1]) if len(vals) == 2 and vals[1] else float("nan")
        tail = f"{speedup:>11.1f}x" if speedup == speedup else " " * 12  # noqa: PLR0124
        print(f"{n:>8}{row}{tail}")


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _chain_state(n: int) -> tuple[GraphState, list[str]]:
    """Linear provenance chain of n items (item k is the parent of item k+1)."""
    state = create_graph_state()
    ids: list[str] = []
    prev: str | None = None
    for k in range(n):
        item = create_memory_item(
            scope="s", kind="observation", content={"k": k},
            author="a", source_kind="observed", authority=0.5,
            parents=[prev] if prev else None,
        )
        state = apply_command(state, {"type": "memory.create", "item": item}).state
        ids.append(item.id)
        prev = item.id
    return state, ids


def _scored_state(n: int) -> GraphState:
    import random

    rng = random.Random(1)
    state = create_graph_state()
    for k in range(n):
        item = create_memory_item(
            scope="s", kind="observation", content={"k": k},
            author="a", source_kind="observed",
            authority=round(rng.random(), 2), importance=round(rng.random(), 2),
        )
        state = apply_command(state, {"type": "memory.create", "item": item}).state
    return state


def _alias_chain_state(n: int) -> tuple[GraphState, str]:
    items = {}
    edges = {}
    ids = [f"a{k:06d}" for k in range(n)]
    for k, id_ in enumerate(ids):
        items[id_] = MemoryItem(
            id=id_, scope="s", kind="observation", content={},
            author="a", source_kind="observed", authority=0.5,
        )
        if k > 0:
            eid = f"e{k:06d}"
            edges[eid] = Edge(
                edge_id=eid, from_=ids[k - 1], to=id_, kind="ALIAS",
                author="a", source_kind="derived_deterministic", authority=1.0, active=True,
            )
    return GraphState(items=items, edges=edges), ids[0]


# ---------------------------------------------------------------------------
# Naive references (pre-optimization behavior)
# ---------------------------------------------------------------------------


def _naive_cascade(state: GraphState, item_id: str) -> None:
    visited = {item_id}
    order: list[str] = []
    stack = [(c.id, "enter") for c in get_children(state, item_id)]
    while stack:
        fid, phase = stack.pop()
        if phase == "exit":
            order.append(fid)
            continue
        if fid in visited:
            continue
        visited.add(fid)
        stack.append((fid, "exit"))
        for c in get_children(state, fid):
            if c.id not in visited:
                stack.append((c.id, "enter"))
    current = state
    for dep in [*order, item_id]:
        if dep not in current.items:
            continue
        current = apply_command(current, {"type": "memory.retract", "item_id": dep, "author": "x"}).state


def _naive_multi_sort(items: list[MemoryItem], sorts: list[dict[str, str]]) -> None:
    def _cmp(a: MemoryItem, b: MemoryItem) -> int:
        for s in sorts:
            va = get_sort_value(a, s["field"])
            vb = get_sort_value(b, s["field"])
            if va < vb:
                return -1 if s["order"] == "asc" else 1
            if va > vb:
                return 1 if s["order"] == "asc" else -1
        return 0

    sorted(items, key=cmp_to_key(_cmp))


def _naive_alias_group(state: GraphState, item_id: str) -> None:
    visited: set[str] = set()
    queue = [item_id]
    while queue:
        node_id = queue.pop()
        if node_id in visited:
            continue
        visited.add(node_id)
        for edge in get_edges(state, {"from": node_id, "kind": "ALIAS", "active_only": True}):
            queue.append(edge.to)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # cascade_retract over a deep chain: naive is O(depth*(N+E)); batched is ~O(N+E).
    cascade_states = {n: _chain_state(n) for n in (200, 400, 800, 1600)}
    _bench(
        "cascade_retract - full chain retraction",
        [200, 400, 800, 1600],
        {
            "naive": lambda n: _time(lambda: _naive_cascade(cascade_states[n][0], cascade_states[n][1][0])),
            "optimized": lambda n: _time(lambda: cascade_retract(cascade_states[n][0], cascade_states[n][1][0], "x")),
        },
    )

    # multi-sort: cmp_to_key vs tuple key.
    sort_states = {n: _scored_state(n) for n in (1000, 2000, 4000, 8000)}
    spec = [{"field": "authority", "order": "desc"}, {"field": "importance", "order": "asc"}]
    sort_items = {n: list(sort_states[n].items.values()) for n in sort_states}
    _bench(
        "get_items multi-sort (authority desc, importance asc)",
        [1000, 2000, 4000, 8000],
        {
            "naive": lambda n: _time(lambda: _naive_multi_sort(sort_items[n], spec), repeat=5),
            "optimized": lambda n: _time(lambda: get_items(sort_states[n], None, {"sort": spec}), repeat=5),
        },
    )

    # get_alias_group over an alias chain: naive rescans all edges per node.
    alias_states = {n: _alias_chain_state(n) for n in (200, 400, 800, 1600)}
    _bench(
        "get_alias_group - alias chain traversal",
        [200, 400, 800, 1600],
        {
            "naive": lambda n: _time(lambda: _naive_alias_group(*alias_states[n])),
            "optimized": lambda n: _time(lambda: get_alias_group(*alias_states[n])),
        },
    )

    # get_dependents(transitive) over the chain: naive rescans all items per node.
    _bench(
        "get_dependents(transitive) - chain",
        [200, 400, 800, 1600],
        {
            "naive": lambda n: _time(lambda: _naive_dependents(cascade_states[n][0], cascade_states[n][1][0])),
            "optimized": lambda n: _time(lambda: get_dependents(cascade_states[n][0], cascade_states[n][1][0], True)),
        },
    )


def _naive_dependents(state: GraphState, item_id: str) -> None:
    visited: set[str] = set()
    queue = list(get_children(state, item_id))
    while queue:
        item = queue.pop()
        if item.id in visited:
            continue
        visited.add(item.id)
        queue.extend(get_children(state, item.id))


if __name__ == "__main__":
    main()
