"""Port of tests/retrieval.test.ts."""

from __future__ import annotations

from typing import Any

import pytest

from memex import (
    GraphState,
    InvalidTimestampError,
    MemoryItem,
    ScoredItem,
    _time,
    apply_diversity,
    create_memory_item,
    extract_timestamp,
    filter_contradictions,
    get_items,
    get_scored_items,
    get_support_set,
    get_support_tree,
    mark_contradiction,
    resolve_contradiction,
    smart_retrieve,
    surface_contradictions,
)


def make_item(id: str, **overrides: Any) -> MemoryItem:
    base: dict[str, Any] = {
        "id": id, "scope": "test", "kind": "observation", "content": {},
        "author": "agent:a", "source_kind": "observed", "authority": 0.5,
    }
    base.update(overrides)
    return MemoryItem(**base)


def state_with(items: list[MemoryItem]) -> GraphState:
    return GraphState(items={i.id: i for i in items}, edges={})


def to_scored(items: list[MemoryItem], scores: list[float]) -> list[ScoredItem]:
    return [ScoredItem(item=item, score=scores[i]) for i, item in enumerate(items)]


def fake_uuid7(ms: int) -> str:
    h = format(ms, "x").rjust(12, "0")
    return f"{h[0:8]}-{h[8:12]}-7000-8000-000000000000"


# === 1. Support tree & support set =========================================


def test_support_tree_none_for_nonexistent() -> None:
    assert get_support_tree(state_with([]), "nope") is None


def test_support_tree_leaf() -> None:
    tree = get_support_tree(state_with([make_item("m1")]), "m1")
    assert tree is not None
    assert tree.item.id == "m1"
    assert len(tree.parents) == 0


def test_support_tree_chain() -> None:
    state = state_with([
        make_item("m1"),
        make_item("m2", parents=["m1"]),
        make_item("m3", parents=["m2"]),
    ])
    tree = get_support_tree(state, "m3")
    assert tree is not None
    assert tree.item.id == "m3"
    assert len(tree.parents) == 1
    assert tree.parents[0].item.id == "m2"
    assert len(tree.parents[0].parents) == 1
    assert tree.parents[0].parents[0].item.id == "m1"
    assert len(tree.parents[0].parents[0].parents) == 0


def test_support_tree_diamond() -> None:
    state = state_with([
        make_item("m1"),
        make_item("m2", parents=["m1"]),
        make_item("m3", parents=["m1"]),
        make_item("m4", parents=["m2", "m3"]),
    ])
    tree = get_support_tree(state, "m4")
    assert tree is not None
    assert len(tree.parents) == 2
    all_ids: set[str] = set()

    def collect(node: Any) -> None:
        all_ids.add(node.item.id)
        for p in node.parents:
            collect(p)

    collect(tree)
    assert all_ids == {"m1", "m2", "m3", "m4"}


def test_support_tree_missing_parents() -> None:
    state = state_with([make_item("m2", parents=["m1"])])
    tree = get_support_tree(state, "m2")
    assert tree is not None
    assert tree.item.id == "m2"
    assert len(tree.parents) == 0


def test_support_set_nonexistent() -> None:
    assert len(get_support_set(state_with([]), "nope")) == 0


def test_support_set_root() -> None:
    s = get_support_set(state_with([make_item("m1")]), "m1")
    assert len(s) == 1 and s[0].id == "m1"


def test_support_set_full_chain_deduped() -> None:
    state = state_with([
        make_item("m1"),
        make_item("m2", parents=["m1"]),
        make_item("m3", parents=["m1"]),
        make_item("m4", parents=["m2", "m3"]),
    ])
    s = get_support_set(state, "m4")
    assert sorted(i.id for i in s) == ["m1", "m2", "m3", "m4"]


# === 2. Contradiction-aware packing ========================================


def test_filter_removes_superseded() -> None:
    state = state_with([make_item("m1", authority=0.9), make_item("m2", authority=0.3)])
    marked, _ = mark_contradiction(state, "m1", "m2", "system:detector")
    resolved, _ = resolve_contradiction(marked, "m1", "m2", "system:resolver")
    scored = to_scored([resolved.items["m1"], resolved.items["m2"]], [0.9, 0.03])
    filtered = filter_contradictions(resolved, scored)
    assert len(filtered) == 1 and filtered[0].item.id == "m1"


def test_filter_keeps_higher_side_of_unresolved() -> None:
    state = state_with([make_item("m1", authority=0.9), make_item("m2", authority=0.7)])
    marked, _ = mark_contradiction(state, "m1", "m2", "system:detector")
    scored = to_scored([marked.items["m1"], marked.items["m2"]], [0.9, 0.7])
    filtered = filter_contradictions(marked, scored)
    assert len(filtered) == 1 and filtered[0].item.id == "m1"


def test_filter_passes_through_no_contradictions() -> None:
    state = state_with([make_item("m1"), make_item("m2")])
    scored = to_scored([state.items["m1"], state.items["m2"]], [0.9, 0.7])
    assert len(filter_contradictions(state, scored)) == 2


def test_surface_keeps_both_and_flags() -> None:
    state = state_with([make_item("m1", authority=0.9), make_item("m2", authority=0.7)])
    marked, _ = mark_contradiction(state, "m1", "m2", "system:detector")
    scored = to_scored([marked.items["m1"], marked.items["m2"]], [0.9, 0.7])
    result = surface_contradictions(marked, scored)
    assert len(result) == 2
    m1 = next(s for s in result if s.item.id == "m1")
    m2 = next(s for s in result if s.item.id == "m2")
    assert m1.contradicted_by is not None and len(m1.contradicted_by) == 1
    assert m1.contradicted_by[0].id == "m2"
    assert m2.contradicted_by is not None and len(m2.contradicted_by) == 1
    assert m2.contradicted_by[0].id == "m1"


def test_surface_still_removes_superseded() -> None:
    state = state_with([make_item("m1", authority=0.9), make_item("m2", authority=0.3)])
    marked, _ = mark_contradiction(state, "m1", "m2", "system:detector")
    resolved, _ = resolve_contradiction(marked, "m1", "m2", "system:resolver")
    scored = to_scored([resolved.items["m1"], resolved.items["m2"]], [0.9, 0.03])
    result = surface_contradictions(resolved, scored)
    assert len(result) == 1
    assert result[0].item.id == "m1"
    assert result[0].contradicted_by is None


def test_surface_no_contradictions_no_flags() -> None:
    state = state_with([make_item("m1"), make_item("m2")])
    scored = to_scored([state.items["m1"], state.items["m2"]], [0.9, 0.7])
    result = surface_contradictions(state, scored)
    assert len(result) == 2
    assert result[0].contradicted_by is None
    assert result[1].contradicted_by is None


def test_smart_retrieve_surface_pipeline() -> None:
    state = state_with([
        make_item("m1", authority=0.9),
        make_item("m2", authority=0.7),
        make_item("m3", authority=0.5),
    ])
    marked, _ = mark_contradiction(state, "m1", "m2", "system:detector")
    result = smart_retrieve(marked, budget=100, cost_fn=lambda i: 1, weights={"authority": 1}, contradictions="surface")
    assert len(result) == 3
    m1 = next(s for s in result if s.item.id == "m1")
    m2 = next(s for s in result if s.item.id == "m2")
    m3 = next(s for s in result if s.item.id == "m3")
    assert m1.contradicted_by is not None and len(m1.contradicted_by) == 1
    assert m2.contradicted_by is not None and len(m2.contradicted_by) == 1
    assert m3.contradicted_by is None


# === 3. Diversity scoring ==================================================


def test_diversity_penalizes_duplicate_authors() -> None:
    items = [
        make_item("m1", author="agent:a"),
        make_item("m2", author="agent:a"),
        make_item("m3", author="agent:b"),
    ]
    diversified = apply_diversity(to_scored(items, [0.9, 0.8, 0.7]), {"author_penalty": 0.3})
    assert diversified[0].item.id == "m1"
    assert diversified[1].item.id == "m3"
    assert diversified[2].item.id == "m2"
    assert diversified[2].score == pytest.approx(0.5)


def test_diversity_penalizes_shared_parents() -> None:
    items = [
        make_item("m2", parents=["m1"]),
        make_item("m3", parents=["m1"]),
        make_item("m4", parents=["m5"]),
    ]
    diversified = apply_diversity(to_scored(items, [0.9, 0.8, 0.7]), {"parent_penalty": 0.4})
    assert diversified[0].item.id == "m2"
    assert diversified[1].item.id == "m4"
    assert diversified[2].item.id == "m3"


def test_diversity_penalizes_duplicate_source() -> None:
    items = [
        make_item("m1", source_kind="observed"),
        make_item("m2", source_kind="observed"),
        make_item("m3", source_kind="agent_inferred"),
    ]
    diversified = apply_diversity(to_scored(items, [0.9, 0.85, 0.7]), {"source_penalty": 0.2})
    assert diversified[0].item.id == "m1"
    assert diversified[1].item.id == "m3"
    assert diversified[2].item.id == "m2"


def test_diversity_combines_penalties() -> None:
    items = [
        make_item("m1", author="agent:a", source_kind="observed"),
        make_item("m2", author="agent:a", source_kind="observed"),
    ]
    diversified = apply_diversity(to_scored(items, [0.9, 0.9]), {"author_penalty": 0.1, "source_penalty": 0.1})
    assert diversified[0].score == pytest.approx(0.9)
    assert diversified[1].score == pytest.approx(0.7)


def test_diversity_clamps_to_zero() -> None:
    items = [make_item("m1", author="agent:a"), make_item("m2", author="agent:a")]
    diversified = apply_diversity(to_scored(items, [0.5, 0.1]), {"author_penalty": 0.5})
    assert diversified[1].score == 0


# === 4. extractTimestamp & recency sort ====================================


def test_extract_timestamp_from_uuidv7() -> None:
    item = create_memory_item(scope="test", kind="observation", content={}, author="test", source_kind="observed", authority=1)
    ts = extract_timestamp(item.id)
    assert abs(ts - _time.now_ms()) < 1000


def test_extract_timestamp_non_uuidv7_raises() -> None:
    with pytest.raises(InvalidTimestampError, match="not a valid UUIDv7"):
        extract_timestamp("not-a-uuid")


def test_extract_timestamp_custom_id_raises() -> None:
    with pytest.raises(InvalidTimestampError, match="not a valid UUIDv7"):
        extract_timestamp("custom-id-12345")


def test_extract_timestamp_preserves_ordering() -> None:
    item1 = create_memory_item(scope="test", kind="observation", content={}, author="test", source_kind="observed", authority=1)
    item2 = create_memory_item(scope="test", kind="observation", content={}, author="test", source_kind="observed", authority=1)
    assert extract_timestamp(item2.id) >= extract_timestamp(item1.id)


def test_recency_sort_descending() -> None:
    older = create_memory_item(scope="test", kind="observation", content={"order": 1}, author="test", source_kind="observed", authority=0.5)
    newer = create_memory_item(scope="test", kind="observation", content={"order": 2}, author="test", source_kind="observed", authority=0.5)
    state = state_with([older, newer])
    result = get_items(state, {}, {"sort": {"field": "recency", "order": "desc"}})
    assert extract_timestamp(result[0].id) >= extract_timestamp(result[1].id)


# === 5. Decay scoring ======================================================


def test_decay_exponential_recent_item() -> None:
    item = create_memory_item(scope="test", kind="observation", content={}, author="test", source_kind="observed", authority=1.0, importance=1.0)
    state = state_with([item])
    no_decay = get_scored_items(state, {"authority": 0.5, "importance": 0.5})
    assert no_decay[0].score == pytest.approx(1.0)
    with_decay = get_scored_items(state, {"authority": 0.5, "importance": 0.5, "decay": {"rate": 0.1, "interval": "day", "type": "exponential"}})
    assert with_decay[0].score == pytest.approx(1.0, abs=0.05)


def test_decay_exponential_formula() -> None:
    two_days_ago = _time.now_ms() - 2 * 86_400_000
    item = MemoryItem(id=fake_uuid7(two_days_ago), scope="test", kind="observation", content={}, author="test", source_kind="observed", authority=1.0)
    state = state_with([item])
    result = get_scored_items(state, {"authority": 1.0, "decay": {"rate": 0.5, "interval": "day", "type": "exponential"}})
    assert result[0].score == pytest.approx(0.25, abs=0.05)


def test_decay_linear_reaches_zero() -> None:
    five_days_ago = _time.now_ms() - 5 * 86_400_000
    item = MemoryItem(id=fake_uuid7(five_days_ago), scope="test", kind="observation", content={}, author="test", source_kind="observed", authority=1.0)
    state = state_with([item])
    result = get_scored_items(state, {"authority": 1.0, "decay": {"rate": 0.3, "interval": "day", "type": "linear"}})
    assert result[0].score == 0


def test_decay_step_boundaries() -> None:
    one_and_half_days = _time.now_ms() - int(1.5 * 86_400_000)
    item = MemoryItem(id=fake_uuid7(one_and_half_days), scope="test", kind="observation", content={}, author="test", source_kind="observed", authority=1.0)
    state = state_with([item])
    result = get_scored_items(state, {"authority": 1.0, "decay": {"rate": 0.5, "interval": "day", "type": "step"}})
    assert result[0].score == pytest.approx(0.5, abs=0.05)


def test_decay_hourly() -> None:
    three_hours_ago = _time.now_ms() - 3 * 3_600_000
    item = MemoryItem(id=fake_uuid7(three_hours_ago), scope="test", kind="observation", content={}, author="test", source_kind="observed", authority=1.0)
    state = state_with([item])
    result = get_scored_items(state, {"authority": 1.0, "decay": {"rate": 0.2, "interval": "hour", "type": "exponential"}})
    assert result[0].score == pytest.approx(0.512, abs=0.05)


def test_decay_no_config() -> None:
    item = make_item("m1", authority=0.8)
    result = get_scored_items(state_with([item]), {"authority": 1.0})
    assert result[0].score == pytest.approx(0.8)


# === 6. Smart retrieval ====================================================


def test_smart_retrieve_basic_budget() -> None:
    state = state_with([
        make_item("m1", authority=0.9),
        make_item("m2", authority=0.5),
        make_item("m3", authority=0.3),
    ])
    result = smart_retrieve(state, budget=20, cost_fn=lambda i: 10, weights={"authority": 1})
    assert len(result) == 2
    assert result[0].item.id == "m1"
    assert result[1].item.id == "m2"


def test_smart_retrieve_filter_contradictions() -> None:
    state = state_with([
        make_item("m1", authority=0.9),
        make_item("m2", authority=0.7),
        make_item("m3", authority=0.5),
    ])
    marked, _ = mark_contradiction(state, "m1", "m2", "system:detector")
    resolved, _ = resolve_contradiction(marked, "m1", "m2", "system:resolver")
    result = smart_retrieve(resolved, budget=100, cost_fn=lambda i: 1, weights={"authority": 1}, contradictions="filter")
    ids = [r.item.id for r in result]
    assert "m1" in ids
    assert "m2" not in ids
    assert "m3" in ids


def test_smart_retrieve_diversity() -> None:
    state = state_with([
        make_item("m1", author="agent:a", authority=0.9),
        make_item("m2", author="agent:a", authority=0.85),
        make_item("m3", author="agent:b", authority=0.8),
    ])
    result = smart_retrieve(state, budget=20, cost_fn=lambda i: 10, weights={"authority": 1}, diversity={"author_penalty": 0.5})
    assert len(result) == 2
    assert result[0].item.id == "m1"
    assert result[1].item.id == "m3"


def test_smart_retrieve_zero_cost() -> None:
    state = state_with([make_item("m1", authority=0.9)])
    result = smart_retrieve(state, budget=100, cost_fn=lambda i: 0, weights={"authority": 1})
    assert len(result) == 1


def test_smart_retrieve_negative_cost_raises() -> None:
    state = state_with([make_item("m1", authority=0.9)])
    with pytest.raises(ValueError):
        smart_retrieve(state, budget=100, cost_fn=lambda i: -5, weights={"authority": 1})


def test_smart_retrieve_nan_cost_raises() -> None:
    state = state_with([make_item("m1", authority=0.9)])
    with pytest.raises(ValueError):
        smart_retrieve(state, budget=100, cost_fn=lambda i: float("nan"), weights={"authority": 1})


def test_smart_retrieve_full_pipeline() -> None:
    state = state_with([
        make_item("m1", scope="a", author="agent:x", authority=0.9),
        make_item("m2", scope="a", author="agent:x", authority=0.85),
        make_item("m3", scope="a", author="agent:y", authority=0.8),
        make_item("m4", scope="b", author="agent:z", authority=0.95),
    ])
    result = smart_retrieve(
        state, budget=20, cost_fn=lambda i: 10, weights={"authority": 1},
        filter={"scope": "a"}, diversity={"author_penalty": 0.3},
    )
    assert len(result) == 2
    assert result[0].item.id == "m1"
    assert result[1].item.id == "m3"
