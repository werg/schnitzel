import torch
import pytest

from schnitz.store import DiskStore, StoredRecord


def test_published_exact_keys_match_sqlite_selection_and_visibility(tmp_path):
    from schnitz.key_index import PublishedKeyIndex
    store = DiskStore(tmp_path / 'bank.sqlite')
    records = []
    for space in ('s0', 's1'):
        for record_id, key, domain, created_at in (
                ('a', [1., 0.], 'research', 1),
                ('b', [0., 1.], 'research', 1),
                ('c', [1., 0.], 'research', 3),
                ('d', [1., 0.], 'private', 1)):
            records.append(StoredRecord(record_id, torch.tensor(key), torch.ones(3),
                                        namespace='corpus', space=space,
                                        generation='g1', domain=domain,
                                        created_at=created_at, source_id=record_id))
    store.put_many(records)
    index = PublishedKeyIndex(store, namespace='corpus', generation='g1',
                              spaces=('s0', 's1'), expected_sources=4)
    q = torch.tensor([1., 0.])
    for space in ('s0', 's1'):
        for domain, when, excluded in (
                ('research', 2, frozenset()),
                ('research', 4, frozenset({'a'})),
                ('private', 2, frozenset())):
            kwargs = dict(namespace='corpus', generation='g1', space=space,
                          top_k=3, domain=domain, query_time=when,
                          exclude_ids=excluded)
            reference = store.search(q, **kwargs)
            fast = index.search(q, **kwargs)
            assert [x.record_id for x in fast.selections] == [x.record_id for x in reference.selections]
            for actual, expected in zip(fast.selections, reference.selections, strict=True):
                assert abs(actual.score - expected.score) < 1e-6
    assert index.key_bytes == 4 * 2 * 2 * 4
    batched = index.search_batch(
        torch.stack((q, torch.tensor([0., 1.]))), namespace='corpus', generation='g1',
        space='s0', top_k=3, domains=('research', 'private'), query_times=(2, 2),
        exclude_ids=(frozenset(), frozenset()),
    )
    assert [[item.record_id for item in plan.selections] for plan in batched] == [
        ['a', 'b'], ['d']]
    fetched = store.fetch_many(batched)
    assert [len(values) for values in fetched] == [2, 1]
    selected_before_delete = index.search(q, namespace='corpus', generation='g1',
                                          space='s0', domain='research', query_time=2,
                                          top_k=1)
    store.delete('corpus', selected_before_delete.selections[0].record_id)
    with pytest.raises(KeyError, match='Unavailable'):
        store.fetch(selected_before_delete)


def test_published_index_rejects_incomplete_space(tmp_path):
    from schnitz.key_index import PublishedKeyIndex
    store = DiskStore(tmp_path / 'bank.sqlite')
    store.put(StoredRecord('a', torch.ones(2), torch.ones(3),
                           namespace='corpus', space='s0', generation='g1'))
    try:
        PublishedKeyIndex(store, namespace='corpus', generation='g1',
                          spaces=('s0', 's1'), expected_sources=1)
    except ValueError as exc:
        assert 'complete' in str(exc)
    else:
        raise AssertionError('A missing space must not become a published index')
