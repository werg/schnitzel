from contextlib import contextmanager

import pytest
import torch

from schnitz.store import (DiskStore, StoredRecord, ReadPlan, Selection, AsyncRetriever, lookup_record)


def record(rid, key, **kwargs):
    return StoredRecord(rid, torch.tensor(key).float(), torch.randn(8).bfloat16(), **kwargs)


def test_disk_roundtrip_and_reopen(tmp_path):
    path = tmp_path / "bank.sqlite"
    store = DiskStore(path)
    r = record("a", [1, 0, 0])
    store.put(r)
    reopened = DiskStore(path)
    plan = reopened.search(torch.tensor([1., 0, 0]), top_k=1)
    assert plan.selections[0].record_id == "a"
    torch.testing.assert_close(reopened.fetch(plan)[0], r.payload)
    assert reopened.fetch(plan)[0].dtype == torch.bfloat16
    assert reopened.sizes()["records"] == 1


def test_filter_before_topk_and_causal_cutoff(tmp_path):
    store = DiskStore(tmp_path / "bank.sqlite")
    store.put(record("secret", [1, 0], domain="secret"))
    store.put(record("future", [1, 0], created_at=20))
    store.put(record("wrong_version", [1, 0], generation="v9"))
    store.put(record("right", [0.8, 0.2], created_at=2))
    plan = store.search(torch.tensor([1., 0]), top_k=1, query_time=10)
    assert [s.record_id for s in plan.selections] == ["right"]


def test_search_chunk_independence_and_ties(tmp_path):
    store = DiskStore(tmp_path / "bank.sqlite")
    for rid in ["c", "b", "a"]:
        store.put(record(rid, [1, 0]))
    one = store.search(torch.tensor([1., 0]), top_k=3, key_chunk_size=1)
    many = store.search(torch.tensor([1., 0]), top_k=3, key_chunk_size=10)
    assert one == many
    assert [s.record_id for s in one.selections] == ["a", "b", "c"]


def test_immutable_record_version(tmp_path):
    import sqlite3
    store = DiskStore(tmp_path / "bank.sqlite")
    store.put(record("a", [1, 0]))
    with pytest.raises(sqlite3.IntegrityError):
        store.put(record("a", [0, 1]))


def test_deletion_transitive_and_outstanding_plan_invalid(tmp_path):
    store = DiskStore(tmp_path / "bank.sqlite")
    for rid in ["a", "b"]:
        store.put(record(rid, [1, 0]))
    store.put(record("compact", [1, 0]), children=("a", "b"))
    store.put(record("compact2", [1, 0]), children=("compact",))
    plan = ReadPlan("default", "s0", "v0", "research", 100, (Selection("compact2", 1.0),))
    assert store.delete("default", "a") == {"a", "compact", "compact2"}
    with pytest.raises(KeyError):
        store.fetch(plan)
    with pytest.raises(ValueError):
        store.put(record("a", [1, 0]))
    assert store.sizes()["records"] == 1


def test_mixed_authorization_compaction_rejected(tmp_path):
    store = DiskStore(tmp_path / "bank.sqlite")
    store.put(record("a", [1, 0], domain="one"))
    with pytest.raises(PermissionError):
        store.put(record("compact", [1, 0], domain="two"), children=("a",))


def test_async_read(tmp_path):
    store = DiskStore(tmp_path / "bank.sqlite")
    store.put(record("a", [1, 0]))
    with AsyncRetriever(store) as retrieve:
        plan, values = retrieve.submit(torch.tensor([1., 0]), top_k=1).result(timeout=5)
    assert plan.selections[0].record_id == "a" and len(values) == 1


def test_lookup_version_mismatch(tmp_path):
    store = DiskStore(tmp_path / "bank.sqlite")
    store.put(record("a", [1, 0]))
    with pytest.raises(KeyError):
        lookup_record(store, "a", namespace="default", space="s0", generation="new")


def test_empty_search(tmp_path):
    store = DiskStore(tmp_path / "bank.sqlite")
    assert store.search(torch.randn(4)).selections == ()
    assert store.fetch(store.search(torch.randn(4), top_k=0)) == []


def test_fetch_many_batches_each_plan_and_preserves_selection_order(tmp_path):
    store = DiskStore(tmp_path / "bank.sqlite")
    store.put(record("a", [1, 0], created_at=1))
    store.put(record("b", [0, 1], created_at=1))
    plan = ReadPlan("default", "s0", "v0", "research", 2, (
        Selection("b", 2.0), Selection("a", 1.0), Selection("b", 0.5)))
    expected = [store.fetch(ReadPlan(
        plan.namespace, plan.space, plan.generation, plan.domain, plan.query_time,
        (selection,)))[0] for selection in plan.selections]
    actual = store.fetch_many((plan, ReadPlan(
        "default", "s0", "v0", "research", 2, ())))
    assert len(actual) == 2 and actual[1] == []
    for value, reference in zip(actual[0], expected, strict=True):
        torch.testing.assert_close(value, reference)


def test_fetch_many_uses_offline_record_index_when_present(tmp_path):
    store = DiskStore(tmp_path / "bank.sqlite")
    store.put(record("a", [1, 0], created_at=1))
    with store.connect() as db:
        db.execute("CREATE INDEX offline_record_scope ON records "
                   "(namespace,generation,record_id,space)")
    statements = []
    original = store.connect

    @contextmanager
    def traced():
        with original() as db:
            db.set_trace_callback(statements.append)
            yield db

    store.connect = traced
    store.fetch_many((ReadPlan("default", "s0", "v0", "research", 2,
                               (Selection("a", 1.0),)),))
    assert any("INDEXED BY offline_record_scope" in statement for statement in statements)


def test_temporal_event_commit_is_atomic_causal_and_idempotent(tmp_path):
    store = DiskStore(tmp_path / "events.sqlite")
    with store.connect() as db:
        assert db.execute("SELECT 1 FROM sqlite_master WHERE name='event_streams'").fetchone() is None
    records = [record("lesson", [1, 0], namespace="experience", space=space,
                      generation="g1", created_at=1, source_id="trajectory-0")
               for space in ("s0", "s1")]
    assert store.event_frontier(namespace="experience", stream="tasks", generation="g1") == {
        "next_position": 0, "visibility_time": -1}
    assert store.commit_event(
        records, namespace="experience", stream="tasks", generation="g1", position=0,
        event_id="task-0", visibility_time=1, expected_spaces=("s0", "s1"))
    assert store.search(torch.tensor([1., 0]), namespace="experience", space="s0",
                        generation="g1", query_time=1).selections == ()
    assert [item.record_id for item in store.search(
        torch.tensor([1., 0]), namespace="experience", space="s0",
        generation="g1", query_time=2).selections] == ["lesson"]
    assert not store.commit_event(
        records, namespace="experience", stream="tasks", generation="g1", position=0,
        event_id="task-0", visibility_time=1, expected_spaces=("s0", "s1"))
    assert store.event_frontier(namespace="experience", stream="tasks", generation="g1") == {
        "next_position": 1, "visibility_time": 1}


def test_temporal_event_persists_external_lineage_and_metadata_atomically(tmp_path):
    store = DiskStore(tmp_path / "events.sqlite")
    records = [record("lesson", [1, 0], namespace="experience", space=space,
                      generation="g1", created_at=4, source_id="trajectory-0")
               for space in ("s0", "s1")]
    kwargs = dict(
        namespace="experience", stream="tasks", generation="g1", position=0,
        event_id="task-0", visibility_time=4, expected_spaces=("s0", "s1"),
        lineage={"lesson": ("g0/source-a", "g0/source-b")},
        record_metadata={"lesson": {"call_id": "write-0", "parents": ["read-0"]}},
    )
    assert store.commit_event(records, **kwargs)
    summary = store.generation_summary(
        namespace="experience", generation="g1", spaces=("s0", "s1"))
    assert summary["logical_records"] == 1 and summary["views"] == 2
    assert len(summary["sha256"]) == 64
    assert not store.commit_event(records, **kwargs)
    with store.connect() as db:
        assert db.execute("SELECT parent_ref FROM event_lineage ORDER BY parent_ref").fetchall() == [
            ("g0/source-a",), ("g0/source-b",)]
        assert db.execute("SELECT metadata FROM event_record_metadata").fetchone()[0] == (
            '{"call_id":"write-0","parents":["read-0"]}')
    changed = dict(kwargs)
    changed["lineage"] = {"lesson": ("g0/source-a",)}
    with pytest.raises(ValueError, match="already committed differently"):
        store.commit_event(records, **changed)


def test_temporal_event_rejects_partial_views_without_advancing(tmp_path):
    store = DiskStore(tmp_path / "events.sqlite")
    partial = [record("lesson", [1, 0], namespace="experience", space="s0",
                      generation="g1", created_at=1)]
    with pytest.raises(ValueError, match="every expected space"):
        store.commit_event(
            partial, namespace="experience", stream="tasks", generation="g1", position=0,
            event_id="task-0", visibility_time=1, expected_spaces=("s0", "s1"))
    assert store.sizes()["records"] == 0
    assert store.event_frontier(namespace="experience", stream="tasks", generation="g1")[
        "next_position"] == 0
