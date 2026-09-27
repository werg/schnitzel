import pytest
import torch

from schnitz.offline_bank import ensure_offline_shard, publish_offline_generation
from schnitz.store import DiskStore, StoredRecord


def _records(source_ids, *, namespace='bank', generation='g1'):
    for source_id in source_ids:
        for space in ('s0', 's1'):
            yield StoredRecord(source_id, torch.ones(2), torch.ones(3),
                               namespace=namespace, space=space, generation=generation,
                               created_at=1, source_id=source_id)


def test_shards_commit_complete_records_and_resume_without_reencoding(tmp_path):
    store = DiskStore(tmp_path / 'bank.sqlite')
    kwargs = dict(identity={'writer': 'frozen-a'}, namespace='bank', generation='g1',
                  spaces=('s0', 's1'))
    assert ensure_offline_shard(store, lambda: _records(('a', 'b')),
                                shard_id='000', source_ids=('a', 'b'), **kwargs)
    with store.connect() as db:
        assert "offline_record_scope" in {row[1] for row in db.execute(
            "PRAGMA index_list('records')")}
    assert not ensure_offline_shard(store, lambda: (_ for _ in ()).throw(AssertionError('reencoded')),
                                    shard_id='000', source_ids=('a', 'b'), **kwargs)
    assert ensure_offline_shard(store, lambda: _records(('c',)),
                                shard_id='001', source_ids=('c',), **kwargs)
    manifest = publish_offline_generation(store, shard_ids=('000', '001'),
                                          source_count=3, **kwargs)
    assert manifest['records'] == 6 and manifest['sources'] == 3
    assert publish_offline_generation(store, shard_ids=('000', '001'),
                                      source_count=3, verify_only=True, **kwargs) == manifest
    with store.connect() as writer:
        writer.execute('BEGIN IMMEDIATE')
        assert publish_offline_generation(store, shard_ids=('000', '001'),
                                          source_count=3, verify_only=True, **kwargs) == manifest
    assert store.sizes()['records'] == 6
    with pytest.raises(ValueError, match='identity'):
        ensure_offline_shard(store, lambda: (), shard_id='000', source_ids=('a', 'b'),
                             **(kwargs | {'identity': {'writer': 'changed'}}))


def test_incomplete_shard_rolls_back_and_cannot_publish(tmp_path):
    store = DiskStore(tmp_path / 'bank.sqlite')
    kwargs = dict(identity={'writer': 'frozen-a'}, namespace='bank', generation='g1',
                  spaces=('s0', 's1'))
    with pytest.raises(ValueError, match='complete'):
        ensure_offline_shard(store, lambda: list(_records(('a',)))[:1],
                             shard_id='000', source_ids=('a',), **kwargs)
    assert store.sizes()['records'] == 0
    with pytest.raises(ValueError, match='shard'):
        publish_offline_generation(store, shard_ids=('000',), source_count=1, **kwargs)
