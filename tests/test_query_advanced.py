"""Port of tests/query-advanced.test.ts."""

from __future__ import annotations

from typing import Any

from memex import GraphState, MemoryItem, get_items


def make_item(id: str, **overrides: Any) -> MemoryItem:
    base: dict[str, Any] = {
        "id": id,
        "scope": "project:cyberdeck",
        "kind": "observation",
        "content": {},
        "author": "agent:a",
        "source_kind": "observed",
        "authority": 0.5,
    }
    base.update(overrides)
    return MemoryItem(**base)


def state_with(items: list[MemoryItem]) -> GraphState:
    return GraphState(items={i.id: i for i in items}, edges={})


def ids(result: list[MemoryItem]) -> list[str]:
    return sorted(i.id for i in result)


# --- ids filter ------------------------------------------------------------

IDS_STATE = state_with([make_item("m1"), make_item("m2"), make_item("m3"), make_item("m4")])


def test_ids_matching() -> None:
    assert ids(get_items(IDS_STATE, {"ids": ["m1", "m3"]})) == ["m1", "m3"]


def test_ids_ignores_nonexistent() -> None:
    result = get_items(IDS_STATE, {"ids": ["m1", "nonexistent"]})
    assert len(result) == 1 and result[0].id == "m1"


def test_ids_empty_returns_nothing() -> None:
    assert len(get_items(IDS_STATE, {"ids": []})) == 0


def test_ids_combined_with_range() -> None:
    state2 = state_with([
        make_item("m1", authority=0.9),
        make_item("m2", authority=0.3),
        make_item("m3", authority=0.8),
    ])
    result = get_items(state2, {"ids": ["m1", "m2", "m3"], "range": {"authority": {"min": 0.5}}})
    assert ids(result) == ["m1", "m3"]


# --- scope_prefix ----------------------------------------------------------

SCOPE_STATE = state_with([
    make_item("m1", scope="project:cyberdeck"),
    make_item("m2", scope="project:memex"),
    make_item("m3", scope="user:laz/general"),
    make_item("m4", scope="user:laz/settings"),
])


def test_scope_prefix_project() -> None:
    assert ids(get_items(SCOPE_STATE, {"scope_prefix": "project:"})) == ["m1", "m2"]


def test_scope_prefix_user() -> None:
    assert ids(get_items(SCOPE_STATE, {"scope_prefix": "user:laz/"})) == ["m3", "m4"]


def test_scope_prefix_no_match() -> None:
    assert len(get_items(SCOPE_STATE, {"scope_prefix": "system:"})) == 0


def test_scope_prefix_combined() -> None:
    result = get_items(SCOPE_STATE, {"scope_prefix": "project:", "not": {"scope": "project:memex"}})
    assert len(result) == 1 and result[0].id == "m1"


# --- parents (advanced) ----------------------------------------------------

PARENT_STATE = state_with([
    make_item("m1"),
    make_item("m2"),
    make_item("m3", parents=["m1"]),
    make_item("m4", parents=["m1", "m2"]),
    make_item("m5", parents=["m2"]),
])


def test_parents_includes() -> None:
    assert ids(get_items(PARENT_STATE, {"parents": {"includes": "m1"}})) == ["m3", "m4"]


def test_parents_includes_any() -> None:
    assert ids(get_items(PARENT_STATE, {"parents": {"includes_any": ["m1", "m2"]}})) == ["m3", "m4", "m5"]


def test_parents_includes_all() -> None:
    result = get_items(PARENT_STATE, {"parents": {"includes_all": ["m1", "m2"]}})
    assert len(result) == 1 and result[0].id == "m4"


def test_parents_count_min() -> None:
    result = get_items(PARENT_STATE, {"parents": {"count": {"min": 2}}})
    assert len(result) == 1 and result[0].id == "m4"


def test_parents_count_max() -> None:
    assert ids(get_items(PARENT_STATE, {"parents": {"count": {"max": 0}}})) == ["m1", "m2"]


def test_parents_count_range() -> None:
    assert ids(get_items(PARENT_STATE, {"parents": {"count": {"min": 1, "max": 1}}})) == ["m3", "m5"]


def test_parents_includes_with_count() -> None:
    result = get_items(PARENT_STATE, {"parents": {"includes": "m1", "count": {"min": 2}}})
    assert len(result) == 1 and result[0].id == "m4"


def test_parents_has_parent_sugar() -> None:
    assert ids(get_items(PARENT_STATE, {"has_parent": "m2"})) == ["m4", "m5"]


def test_parents_is_root_sugar() -> None:
    assert ids(get_items(PARENT_STATE, {"is_root": True})) == ["m1", "m2"]


# --- multi-sort ------------------------------------------------------------

NOW = 1_700_000_000_000
SORT_STATE = state_with([
    make_item("m1", authority=0.5, importance=0.9, created_at=NOW - 30),
    make_item("m2", authority=0.5, importance=0.3, created_at=NOW - 20),
    make_item("m3", authority=0.9, importance=0.1, created_at=NOW - 10),
    make_item("m4", authority=0.5, importance=0.6, created_at=NOW),
])


def test_single_sort_backwards_compat() -> None:
    result = get_items(SORT_STATE, {}, {"sort": {"field": "authority", "order": "desc"}})
    assert result[0].id == "m3"


def test_multi_sort_authority_then_importance_desc() -> None:
    result = get_items(
        SORT_STATE, {},
        {"sort": [{"field": "authority", "order": "desc"}, {"field": "importance", "order": "desc"}]},
    )
    assert [i.id for i in result] == ["m3", "m1", "m4", "m2"]


def test_multi_sort_importance_asc_then_authority_desc() -> None:
    result = get_items(
        SORT_STATE, {},
        {"sort": [{"field": "importance", "order": "asc"}, {"field": "authority", "order": "desc"}]},
    )
    assert [i.id for i in result] == ["m3", "m2", "m4", "m1"]


def test_multi_sort_recency_tiebreaker() -> None:
    result = get_items(
        SORT_STATE, {},
        {"sort": [{"field": "authority", "order": "desc"}, {"field": "recency", "order": "desc"}]},
    )
    assert result[0].id == "m3"
