import torch

from schnitz.storage_contract import KeySearchBackend, StoredReadBackend
from schnitz.store import DiskStore, ReadPlan, Selection, StoredRecord, lookup_record


class PublicReadProxy:
    """Exercise the remote-ready surface without exposing SQLite internals."""

    def __init__(self, store):
        self.store = store

    def search(self, query, **kwargs):
        return self.store.search(query, **kwargs)

    def fetch(self, plan):
        return self.store.fetch(plan)

    def fetch_many(self, plans):
        return self.store.fetch_many(plans)

    def iter_fetch(self, plan, chunk_size=32):
        return self.store.iter_fetch(plan, chunk_size)

    def lookup(self, record_id, **scope):
        return self.store.lookup(record_id, **scope)


def test_disk_store_satisfies_remote_read_contract(tmp_path):
    store = DiskStore(tmp_path / "bank.sqlite")
    store.put(StoredRecord(
        "a", torch.tensor([1.0, 0.0]), torch.tensor([3.0]),
        namespace="corpus", generation="g", created_at=1, source_id="source-a",
    ))
    proxy = PublicReadProxy(store)
    assert isinstance(store, StoredReadBackend)
    assert isinstance(store, KeySearchBackend)
    assert isinstance(proxy, StoredReadBackend)
    record = lookup_record(
        proxy, "a", namespace="corpus", space="s0", generation="g", query_time=2,
    )
    assert record.source_id == "source-a"
    plan = ReadPlan(
        "corpus", "s0", "g", "research", 2,
        (Selection("a", 1.0), Selection("a", 0.5)), bank_cursor=7,
    )
    values = proxy.fetch_many((plan,))[0]
    assert len(values) == 2
    torch.testing.assert_close(values[0], torch.tensor([3.0]))
    torch.testing.assert_close(values[1], torch.tensor([3.0]))

