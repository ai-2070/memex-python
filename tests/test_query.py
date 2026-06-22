"""Port of tests/query.test.ts."""

from __future__ import annotations

import pytest

from memex import (
    Edge,
    GraphState,
    MemoryItem,
    get_children,
    get_edge_by_id,
    get_edges,
    get_item_by_id,
    get_items,
    get_parents,
    get_related_items,
)

ITEMS = [
    MemoryItem(
        id="m1", scope="user:laz/general", kind="assertion",
        content={"key": "theme", "value": "dark"}, author="user:laz",
        source_kind="user_explicit", authority=0.95, importance=0.8, conviction=0.9,
        meta={"agent_id": "agent:x", "tags": {"primary": "preference", "env": "prod"}},
    ),
    MemoryItem(
        id="m2", scope="user:laz/general", kind="observation",
        content={"key": "login_count", "value": 42}, author="agent:reasoner",
        source_kind="observed", authority=0.7, importance=0.5, conviction=0.6,
        meta={"agent_id": "agent:y", "tags": {"primary": "metric", "env": "prod"}},
    ),
    MemoryItem(
        id="m3", scope="project:cyberdeck", kind="derivation",
        content={"key": "active_user", "value": True}, author="system:rule",
        source_kind="derived_deterministic", parents=["m1", "m2"], authority=0.6,
    ),
    MemoryItem(
        id="m4", scope="user:laz/general", kind="hypothesis",
        content={"key": "will_churn", "value": False}, author="agent:reasoner",
        source_kind="agent_inferred", parents=["m2"], authority=0.3, importance=0.9, conviction=0.4,
        meta={"agent_id": "agent:x", "tags": {"primary": "prediction", "env": "staging"}},
    ),
    MemoryItem(
        id="m5", scope="project:cyberdeck", kind="simulation",
        content={"key": "scenario", "value": "outage"}, author="agent:reasoner",
        source_kind="simulated", authority=0.2, importance=0.4,
    ),
]

EDGES = [
    Edge(edge_id="e1", from_="m1", to="m2", kind="SUPPORTS", author="system:rule",
         source_kind="derived_deterministic", authority=0.8, active=True),
    Edge(edge_id="e2", from_="m3", to="m4", kind="DERIVED_FROM", author="system:rule",
         source_kind="derived_deterministic", authority=0.9, active=True, weight=0.7),
    Edge(edge_id="e3", from_="m2", to="m3", kind="ABOUT", author="agent:reasoner",
         source_kind="agent_inferred", authority=0.5, active=False),
]


def build_state() -> GraphState:
    return GraphState(items={i.id: i for i in ITEMS}, edges={e.edge_id: e for e in EDGES})


def ids(result: list[MemoryItem]) -> list[str]:
    return sorted(i.id for i in result)


STATE = build_state()


# --- getItems basic filters ------------------------------------------------


def test_all_items_no_filter() -> None:
    assert len(get_items(STATE)) == 5


def test_filter_by_scope() -> None:
    result = get_items(STATE, {"scope": "project:cyberdeck"})
    assert len(result) == 2
    assert all(i.scope == "project:cyberdeck" for i in result)


def test_filter_by_kind() -> None:
    result = get_items(STATE, {"kind": "observation"})
    assert len(result) == 1 and result[0].id == "m2"


def test_filter_by_source_kind() -> None:
    result = get_items(STATE, {"source_kind": "observed"})
    assert len(result) == 1 and result[0].id == "m2"


def test_filter_by_author() -> None:
    assert len(get_items(STATE, {"author": "agent:reasoner"})) == 3


def test_not_with_or() -> None:
    result = get_items(STATE, {"not": {"or": [{"kind": "hypothesis"}, {"kind": "simulation"}]}})
    assert len(result) == 3
    assert all(i.kind not in ("hypothesis", "simulation") for i in result)


def test_not_single_kind() -> None:
    result = get_items(STATE, {"not": {"kind": "simulation"}})
    assert len(result) == 4
    assert all(i.kind != "simulation" for i in result)


def test_not_non_kind_field() -> None:
    result = get_items(STATE, {"not": {"author": "agent:reasoner"}})
    assert ids(result) == ["m1", "m3"]


def test_not_combined_with_other_filters() -> None:
    result = get_items(STATE, {"scope": "user:laz/general", "not": {"range": {"authority": {"max": 0.5}}}})
    assert ids(result) == ["m1", "m2"]


def test_filter_by_meta() -> None:
    result = get_items(STATE, {"meta": {"agent_id": "agent:x"}})
    assert ids(result) == ["m1", "m4"]


def test_and_logic() -> None:
    result = get_items(STATE, {"scope": "user:laz/general", "range": {"authority": {"min": 0.5}}})
    assert ids(result) == ["m1", "m2"]


# --- range filters ---------------------------------------------------------


def test_authority_min() -> None:
    result = get_items(STATE, {"range": {"authority": {"min": 0.6}}})
    assert len(result) == 3 and all(i.authority >= 0.6 for i in result)


def test_authority_max() -> None:
    assert ids(get_items(STATE, {"range": {"authority": {"max": 0.3}}})) == ["m4", "m5"]


def test_authority_range() -> None:
    assert ids(get_items(STATE, {"range": {"authority": {"min": 0.3, "max": 0.7}}})) == ["m2", "m3", "m4"]


def test_importance_min_excludes_undefined() -> None:
    assert ids(get_items(STATE, {"range": {"importance": {"min": 0.7}}})) == ["m1", "m4"]


def test_conviction_range() -> None:
    result = get_items(STATE, {"range": {"conviction": {"min": 0.5, "max": 0.8}}})
    assert len(result) == 1 and result[0].id == "m2"


def test_multiple_ranges() -> None:
    result = get_items(STATE, {"range": {"authority": {"min": 0.5}, "importance": {"min": 0.5}}})
    assert ids(result) == ["m1", "m2"]


# --- nested meta -----------------------------------------------------------


def test_nested_meta_dot_path() -> None:
    result = get_items(STATE, {"meta": {"tags.primary": "preference"}})
    assert len(result) == 1 and result[0].id == "m1"


def test_multiple_nested_meta_paths() -> None:
    result = get_items(STATE, {"meta": {"tags.env": "prod", "agent_id": "agent:x"}})
    assert len(result) == 1 and result[0].id == "m1"


def test_nonexistent_nested_meta() -> None:
    assert len(get_items(STATE, {"meta": {"tags.nonexistent": "foo"}})) == 0


def test_handles_items_with_no_meta() -> None:
    result = get_items(STATE, {"meta": {"tags.primary": "metric"}})
    assert len(result) == 1 and result[0].id == "m2"


# --- meta_has --------------------------------------------------------------


def test_meta_has_field_exists() -> None:
    assert ids(get_items(STATE, {"meta_has": ["tags"]})) == ["m1", "m2", "m4"]


def test_meta_has_dot_path() -> None:
    assert len(get_items(STATE, {"meta_has": ["tags.primary"]})) == 3


def test_meta_has_excludes_items_without_meta() -> None:
    assert len(get_items(STATE, {"meta_has": ["agent_id"]})) == 3


def test_meta_has_requires_all_paths() -> None:
    assert ids(get_items(STATE, {"meta_has": ["agent_id", "tags.env"]})) == ["m1", "m2", "m4"]


def test_meta_has_nonexistent() -> None:
    assert len(get_items(STATE, {"meta_has": ["nonexistent.deep.path"]})) == 0


def test_meta_has_combined_with_not() -> None:
    result = get_items(STATE, {"meta_has": ["agent_id"], "not": {"meta": {"agent_id": "agent:x"}}})
    assert len(result) == 1 and result[0].id == "m2"


# --- OR queries ------------------------------------------------------------


def test_or_matches_any() -> None:
    assert ids(get_items(STATE, {"or": [{"kind": "observation"}, {"kind": "assertion"}]})) == ["m1", "m2"]


def test_or_and_combined_with_top_level() -> None:
    result = get_items(STATE, {"scope": "user:laz/general", "or": [{"kind": "observation"}, {"kind": "hypothesis"}]})
    assert ids(result) == ["m2", "m4"]


def test_or_no_match() -> None:
    assert len(get_items(STATE, {"or": [{"kind": "policy"}, {"kind": "trait"}]})) == 0


def test_nested_or_recursive() -> None:
    result = get_items(STATE, {"or": [{"kind": "simulation"}, {"kind": "derivation", "scope": "project:cyberdeck"}]})
    assert ids(result) == ["m3", "m5"]


def test_or_with_meta_dot_path() -> None:
    result = get_items(
        STATE,
        {"or": [{"meta": {"tags.primary": "preference"}}, {"meta": {"tags.primary": "prediction"}}]},
    )
    assert ids(result) == ["m1", "m4"]


def test_empty_or_matches_all() -> None:
    assert len(get_items(STATE, {"or": []})) == 5


# --- sorting & pagination --------------------------------------------------


def test_sort_authority_asc() -> None:
    result = get_items(STATE, {}, {"sort": {"field": "authority", "order": "asc"}})
    authorities = [i.authority for i in result]
    assert authorities == sorted(authorities)


def test_sort_authority_desc() -> None:
    result = get_items(STATE, {}, {"sort": {"field": "authority", "order": "desc"}})
    authorities = [i.authority for i in result]
    assert authorities == sorted(authorities, reverse=True)


def test_sort_importance_desc_undefined_as_zero() -> None:
    result = get_items(STATE, {}, {"sort": {"field": "importance", "order": "desc"}})
    assert result[0].id == "m4"
    assert result[1].id == "m1"


def test_sort_conviction_asc() -> None:
    result = get_items(STATE, {}, {"sort": {"field": "conviction", "order": "asc"}})
    convictions = [i.conviction or 0 for i in result]
    assert convictions == sorted(convictions)


def test_limit() -> None:
    result = get_items(STATE, {}, {"sort": {"field": "authority", "order": "desc"}, "limit": 2})
    assert len(result) == 2
    assert result[0].authority == 0.95
    assert result[1].authority == 0.7


def test_offset() -> None:
    result = get_items(STATE, {}, {"sort": {"field": "authority", "order": "desc"}, "offset": 3})
    assert len(result) == 2


def test_offset_and_limit() -> None:
    result = get_items(STATE, {}, {"sort": {"field": "authority", "order": "desc"}, "offset": 1, "limit": 2})
    assert len(result) == 2
    assert result[0].authority == 0.7
    assert result[1].authority == 0.6


def test_filter_with_sort_and_limit() -> None:
    result = get_items(STATE, {"scope": "user:laz/general"}, {"sort": {"field": "authority", "order": "desc"}, "limit": 2})
    assert len(result) == 2
    assert result[0].id == "m1"
    assert result[1].id == "m2"


def test_unknown_sort_field_raises() -> None:
    with pytest.raises(ValueError, match="Unknown sort field.*bogus"):
        get_items(STATE, None, {"sort": {"field": "bogus", "order": "asc"}})


# --- getEdges --------------------------------------------------------------


def test_edges_active_only_default() -> None:
    result = get_edges(STATE)
    assert len(result) == 2 and all(e.active for e in result)


def test_edges_active_only_false() -> None:
    assert len(get_edges(STATE, {"active_only": False})) == 3


def test_edges_filter_by_from() -> None:
    result = get_edges(STATE, {"from": "m1"})
    assert len(result) == 1 and result[0].edge_id == "e1"


def test_edges_filter_by_kind() -> None:
    result = get_edges(STATE, {"kind": "DERIVED_FROM"})
    assert len(result) == 1 and result[0].edge_id == "e2"


def test_edges_filter_by_min_weight() -> None:
    result = get_edges(STATE, {"min_weight": 0.5})
    assert len(result) == 1 and result[0].edge_id == "e2"


# --- getItemById / getEdgeById ---------------------------------------------


def test_get_item_by_id() -> None:
    assert get_item_by_id(STATE, "m1").id == "m1"
    assert get_item_by_id(STATE, "nope") is None


def test_get_edge_by_id() -> None:
    assert get_edge_by_id(STATE, "e1").edge_id == "e1"
    assert get_edge_by_id(STATE, "nope") is None


# --- getRelatedItems -------------------------------------------------------


def test_related_both_directions() -> None:
    result = get_related_items(STATE, "m2")
    assert len(result) == 1 and result[0].id == "m1"


def test_related_from_only() -> None:
    result = get_related_items(STATE, "m1", "from")
    assert len(result) == 1 and result[0].id == "m2"


def test_related_to_only() -> None:
    result = get_related_items(STATE, "m2", "to")
    assert len(result) == 1 and result[0].id == "m1"


def test_related_unconnected() -> None:
    assert len(get_related_items(STATE, "m5")) == 0


# --- parents & children ----------------------------------------------------


def test_get_parents_multiple() -> None:
    assert ids(get_parents(STATE, "m3")) == ["m1", "m2"]


def test_get_parents_single() -> None:
    result = get_parents(STATE, "m4")
    assert len(result) == 1 and result[0].id == "m2"


def test_get_parents_root() -> None:
    assert len(get_parents(STATE, "m1")) == 0
    assert len(get_parents(STATE, "m2")) == 0


def test_get_parents_nonexistent() -> None:
    assert len(get_parents(STATE, "nope")) == 0


def test_get_children_multiple() -> None:
    assert ids(get_children(STATE, "m2")) == ["m3", "m4"]


def test_get_children_single() -> None:
    result = get_children(STATE, "m1")
    assert len(result) == 1 and result[0].id == "m3"


def test_get_children_leaf() -> None:
    assert len(get_children(STATE, "m5")) == 0


# --- parent filters --------------------------------------------------------


def test_has_parent() -> None:
    assert ids(get_items(STATE, {"has_parent": "m2"})) == ["m3", "m4"]


def test_has_parent_no_match() -> None:
    assert len(get_items(STATE, {"has_parent": "m5"})) == 0


def test_is_root_true() -> None:
    assert ids(get_items(STATE, {"is_root": True})) == ["m1", "m2", "m5"]


def test_is_root_false() -> None:
    assert ids(get_items(STATE, {"is_root": False})) == ["m3", "m4"]


def test_has_parent_combined() -> None:
    result = get_items(STATE, {"has_parent": "m2", "kind": "hypothesis"})
    assert len(result) == 1 and result[0].id == "m4"


def test_not_has_parent() -> None:
    result = get_items(STATE, {"is_root": False, "not": {"has_parent": "m1"}})
    assert len(result) == 1 and result[0].id == "m4"
