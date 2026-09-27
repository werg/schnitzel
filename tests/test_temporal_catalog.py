import torch

from schnitz.key_index import PublishedKeyIndex
from schnitz.store import DiskStore, StoredRecord
from schnitz.temporal_catalog import CatalogStore, GrowingCatalogIndex


def _record(record_id, key, payload, *, space, generation, created_at):
    return StoredRecord(
        record_id, torch.tensor(key, dtype=torch.float32),
        torch.tensor(payload, dtype=torch.bfloat16), namespace="corpus",
        space=space, generation=generation, domain="research",
        created_at=created_at, source_id=record_id,
    )


def test_growing_catalog_merges_generations_and_preserves_payload_origin(tmp_path):
    parent_store = DiskStore(tmp_path / "parent.sqlite")
    parent_store.put_many([
        _record(record_id, key, payload, space=space, generation="g0", created_at=1)
        for record_id, key, payload in (
            ("parent-a", [1, 0], [10, 11]),
            ("parent-b", [0, 1], [20, 21]),
        ) for space in ("s0", "s1")
    ])
    parent = PublishedKeyIndex(
        parent_store, namespace="corpus", generation="g0",
        spaces=("s0", "s1"), expected_sources=2)
    authored_store = DiskStore(tmp_path / "authored.sqlite")
    index = GrowingCatalogIndex(
        parent, authored_store, authored_namespace="corpus",
        authored_generation="g1", capacity=4, catalog_generation="g0+g1")
    store = CatalogStore(parent_store, authored_store, index)

    records = [
        _record("authored-a", [0.99, 0.01], [30, 31], space=space,
                generation="g1", created_at=3)
        for space in ("s0", "s1")
    ]
    assert authored_store.commit_event(
        records, namespace="corpus", stream="tasks", generation="g1",
        position=0, event_id="trajectory-0", visibility_time=3,
        expected_spaces=("s0", "s1"),
        lineage={"authored-a": ("parent-a",)},
        record_metadata={"authored-a": {"call_id": "write-0"}})
    index.add_records(records)

    early = index.search_batch(
        torch.tensor([[1.0, 0.0]]), top_k=2, namespace="catalog", space="s0",
        generation="g0+g1", domains=("research",), query_times=(3,))[0]
    assert [selection.record_id for selection in early.selections] == ["parent-a", "parent-b"]
    later = index.search_batch(
        torch.tensor([[1.0, 0.0]]), top_k=2, namespace="catalog", space="s0",
        generation="g0+g1", domains=("research",), query_times=(4,))[0]
    assert [selection.record_id for selection in later.selections] == [
        "parent-a", "authored-a"]
    values = store.fetch(later)
    torch.testing.assert_close(values[0], torch.tensor([10, 11], dtype=torch.bfloat16))
    torch.testing.assert_close(values[1], torch.tensor([30, 31], dtype=torch.bfloat16))
    keys = index.keys_for_ids(
        "s0", ["authored-a", "parent-a"], domain="research", query_time=4)
    assert keys.shape == (2, 2)


def test_growing_catalog_reopens_committed_frontier(tmp_path):
    parent_store = DiskStore(tmp_path / "parent.sqlite")
    parent_store.put_many([
        _record("parent", [1, 0], [1, 2], space="s0",
                generation="g0", created_at=1)])
    parent = PublishedKeyIndex(
        parent_store, namespace="corpus", generation="g0",
        spaces=("s0",), expected_sources=1)
    authored = DiskStore(tmp_path / "authored.sqlite")
    records = [_record("new", [0, 1], [3, 4], space="s0",
                       generation="g1", created_at=2)]
    authored.commit_event(
        records, namespace="corpus", stream="tasks", generation="g1", position=0,
        event_id="event", visibility_time=2, expected_spaces=("s0",))
    reopened = GrowingCatalogIndex(
        parent, authored, authored_namespace="corpus", authored_generation="g1",
        capacity=2, catalog_generation="catalog")
    assert reopened.authored_count == 1
    assert reopened.origin("s0", "new") == "authored"
