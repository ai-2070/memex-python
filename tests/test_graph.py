"""Port of tests/graph.test.ts (Maps -> dicts)."""

from __future__ import annotations

from memex import MemoryItem, clone_graph_state, create_graph_state


def test_create_graph_state_returns_empty_dicts() -> None:
    state = create_graph_state()
    assert isinstance(state.items, dict)
    assert isinstance(state.edges, dict)
    assert len(state.items) == 0
    assert len(state.edges) == 0


def test_clone_returns_new_dict_references() -> None:
    original = create_graph_state()
    cloned = clone_graph_state(original)
    assert cloned is not original
    assert cloned.items is not original.items
    assert cloned.edges is not original.edges


def test_clone_mutation_does_not_affect_original() -> None:
    original = create_graph_state()
    item = MemoryItem(
        id="m1", scope="test", kind="observation", content={},
        author="test", source_kind="observed", authority=1,
    )
    original.items["m1"] = item
    cloned = clone_graph_state(original)
    del cloned.items["m1"]
    assert "m1" in original.items
    assert "m1" not in cloned.items
