import torch
import random
import pytest
from safetensors.torch import save

from sdkb.agent import SDKBAgent
from sdkb.checkpoints import restore_checkpoint, save_checkpoint
from sdkb.key_index import PublishedKeyIndex
from sdkb.optimizers import make_optimizer
from sdkb.store import DiskStore, ReadPlan, Selection, StoredRecord
from sdkb.training_bank import TrainingBank


def test_training_overlay_updates_search_keys_and_stored_payloads(tmp_path):
    base = DiskStore(tmp_path / 'base.sqlite')
    base.put(StoredRecord('a', torch.tensor([1., 0.]), torch.tensor([1., 2.]),
        namespace='corpus', space='s0', generation='g', created_at=1))
    cache = DiskStore(tmp_path / 'cache.sqlite')
    index = PublishedKeyIndex(base, namespace='corpus', generation='g', spaces=('s0',),
                              expected_sources=1)
    bank = TrainingBank(base, cache, index)
    bank.update([StoredRecord('a', torch.tensor([0., 2.]), torch.tensor([7., 8.]),
                              space='s0')])
    plan = ReadPlan('corpus', 's0', 'g', 'research', 2, (Selection('a', 0.),))
    torch.testing.assert_close(bank.fetch_many((plan,))[0][0], torch.tensor([7., 8.]))
    torch.testing.assert_close(index.keys_for_ids('s0', ('a',), domain='research',
                                                  query_time=2)[0], torch.tensor([0., 1.]))
    reopened_index = PublishedKeyIndex(base, namespace='corpus', generation='g',
                                       spaces=('s0',), expected_sources=1)
    reopened = TrainingBank(base, cache, reopened_index)
    torch.testing.assert_close(reopened.fetch_many((plan,))[0][0], torch.tensor([7., 8.]))
    torch.testing.assert_close(reopened_index.keys_for_ids(
        's0', ('a',), domain='research', query_time=2)[0], torch.tensor([0., 1.]))


def test_training_overlay_batches_multiple_plans_and_spaces(tmp_path):
    base = DiskStore(tmp_path / 'base.sqlite')
    for space in ('s0', 's1'):
        for offset, record_id in enumerate(('a', 'b')):
            base.put(StoredRecord(record_id, torch.tensor([1., float(offset)]),
                torch.tensor([float(offset)]), namespace='corpus', space=space,
                generation='g', created_at=1))
    cache = DiskStore(tmp_path / 'cache.sqlite')
    index = PublishedKeyIndex(base, namespace='corpus', generation='g',
                              spaces=('s0', 's1'), expected_sources=2)
    bank = TrainingBank(base, cache, index)
    bank.update([
        StoredRecord('a', torch.tensor([1., 0.]), torch.tensor([7.]), space='s0'),
        StoredRecord('a', torch.tensor([1., 0.]), torch.tensor([70.]), space='s1'),
        StoredRecord('b', torch.tensor([1., 1.]), torch.tensor([80.]), space='s0'),
        StoredRecord('b', torch.tensor([1., 1.]), torch.tensor([8.]), space='s1'),
    ])
    plans = tuple(ReadPlan('corpus', space, 'g', 'research', 2,
                           tuple(Selection(record_id, 0.) for record_id in ('a', 'b')))
                  for space in ('s0', 's1'))
    rows = bank.fetch_many(plans)
    torch.testing.assert_close(rows[0][0], torch.tensor([7.]))
    torch.testing.assert_close(rows[0][1], torch.tensor([80.]))
    torch.testing.assert_close(rows[1][0], torch.tensor([70.]))
    torch.testing.assert_close(rows[1][1], torch.tensor([8.]))


def test_training_overlay_is_part_of_exact_checkpoint_resume(tiny_config, tmp_path):
    agent = SDKBAgent(tiny_config)
    optimizer = make_optimizer(agent)
    run = tmp_path / 'run'
    run.mkdir()
    base = DiskStore(tmp_path / 'base.sqlite')
    base.put(StoredRecord('a', torch.tensor([1.] * tiny_config.memory.key_dim),
        torch.ones(tiny_config.memory.payload_dims[0]), namespace='corpus', space='s0',
        generation='g', created_at=1))
    cache = DiskStore(run / 'training_cache.sqlite')
    index = PublishedKeyIndex(base, namespace='corpus', generation='g', spaces=('s0',),
                              expected_sources=1)
    bank = TrainingBank(base, cache, index)
    first = torch.full((tiny_config.memory.payload_dims[0],), 3.)
    bank.update([StoredRecord('a', torch.arange(tiny_config.memory.key_dim).float() + 1,
                              first, space='s0')])
    checkpoint = save_checkpoint(
        agent, optimizer, run, 1, random.Random(7), cache, 'data', keep=2)
    assert (checkpoint / 'bank-state.json').is_file()
    assert not (checkpoint / 'training_cache.sqlite').exists()
    bank.update([StoredRecord('a', torch.arange(tiny_config.memory.key_dim).float() + 2,
                              torch.full_like(first, 9.), space='s0')])
    assert bank.maintenance_ids(1) == ('a',)
    assert restore_checkpoint(agent, optimizer, run, random.Random(9), 'data') == 1
    restored_index = PublishedKeyIndex(base, namespace='corpus', generation='g',
                                       spaces=('s0',), expected_sources=1)
    restored = TrainingBank(base, cache, restored_index)
    plan = ReadPlan('corpus', 's0', 'g', 'research', 2, (Selection('a', 0.),))
    torch.testing.assert_close(restored.fetch_many((plan,))[0][0], first)
    assert cache.mutable_bank_state()['maintenance_position'] == 0


def test_maintenance_rotates_without_scanning_revision_ages(tmp_path):
    base = DiskStore(tmp_path / 'base.sqlite')
    for record_id in ('a', 'b', 'c'):
        base.put(StoredRecord(record_id, torch.tensor([1., 0.]),
                              torch.tensor([1.]), namespace='corpus', space='s0',
                              generation='g', created_at=1))
    cache = DiskStore(tmp_path / 'journal.sqlite')
    index = PublishedKeyIndex(base, namespace='corpus', generation='g',
                              spaces=('s0',), expected_sources=3)
    bank = TrainingBank(base, cache, index)
    bank.update([StoredRecord('a', torch.tensor([0., 1.]), torch.tensor([2.]),
                              namespace='corpus', space='s0', generation='g',
                              created_at=1)])
    assert bank.maintenance_ids(1) == ('a',)
    assert bank.maintenance_ids(1, exclude=('b',)) == ('c',)
    assert bank.maintenance_ids(1) == ('a',)


def test_mutable_revision_is_all_space_atomic_and_journaled(tmp_path):
    base = DiskStore(tmp_path / 'base.sqlite')
    for space in ('s0', 's1'):
        base.put(StoredRecord('a', torch.tensor([1., 0.]), torch.tensor([1.]),
            namespace='corpus', space=space, generation='g', created_at=1))
    cache = DiskStore(tmp_path / 'journal.sqlite')
    index = PublishedKeyIndex(base, namespace='corpus', generation='g',
                              spaces=('s0', 's1'), expected_sources=1)
    bank = TrainingBank(base, cache, index)
    with pytest.raises(ValueError, match='every configured space'):
        bank.update([StoredRecord('a', torch.tensor([0., 1.]), torch.tensor([2.]),
                                  space='s0')])
    records = [StoredRecord('a', torch.tensor([0., 1.]), torch.tensor([2. + i]),
                            space=space) for i, space in enumerate(('s0', 's1'))]
    assert bank.update(records, optimizer_step=7) == 2
    assert bank.cursor == 1
    pinned = index.search(torch.tensor([0., 1.]), top_k=1, namespace='corpus',
                          space='s0', generation='g', query_time=2)
    assert pinned.bank_cursor == 1
    records = [StoredRecord('a', torch.tensor([-1., 0.]), torch.tensor([4. + i]),
                            space=space) for i, space in enumerate(('s0', 's1'))]
    bank.update(records, optimizer_step=8)
    torch.testing.assert_close(bank.fetch_many((pinned,))[0][0], torch.tensor([2.]))
    with cache.connect() as db:
        assert db.execute('SELECT COUNT(*) FROM mutable_bank_revisions').fetchone()[0] == 4
        assert db.execute('SELECT optimizer_step FROM mutable_bank_commits '
                          'ORDER BY cursor').fetchall() == [(7,), (8,)]


def test_mutable_bank_adds_records_and_invalidates_compact_descendants(tmp_path):
    base = DiskStore(tmp_path / 'base.sqlite')
    for record_id in ('a', 'compact'):
        base.put(StoredRecord(record_id, torch.tensor([1., 0.]), torch.tensor([1.]),
            namespace='corpus', space='s0', generation='g', created_at=1),
            children=('a',) if record_id == 'compact' else ())
    cache = DiskStore(tmp_path / 'journal.sqlite')
    index = PublishedKeyIndex(base, namespace='corpus', generation='g', spaces=('s0',),
                              expected_sources=2)
    bank = TrainingBank(base, cache, index)
    bank.update([StoredRecord('a', torch.tensor([0., 1.]), torch.tensor([2.]), space='s0')])
    assert index.spaces['s0'].deleted[list(index.spaces['s0'].ids).index('compact')]
    assert bank.invalidated_records() == ({
        'record_id': 'compact', 'cause_id': 'a', 'invalidated_at': 1,
        'children': ('a',)},)
    plan = ReadPlan('corpus', 's0', 'g', 'research', 2, (Selection('compact', 0.),))
    with pytest.raises(KeyError, match='Invalidated'):
        bank.fetch_many((plan,))
    bank.update([StoredRecord('compact', torch.tensor([1., 1.]), torch.tensor([3.]),
                              space='s0')], children={'compact': ('a',)})
    assert not index.spaces['s0'].deleted[list(index.spaces['s0'].ids).index('compact')]
    assert bank.invalidated_records() == ()
    bank.update([StoredRecord('new', torch.tensor([-1., 0.]), torch.tensor([9.]),
        namespace='corpus', space='s0', generation='g', created_at=1, source_id='src')])
    found = index.search(torch.tensor([-1., 0.]), top_k=1, namespace='corpus',
                         space='s0', generation='g', query_time=2)
    assert found.selections[0].record_id == 'new'
    torch.testing.assert_close(bank.fetch_many((found,))[0][0], torch.tensor([9.]))


def test_compaction_rebuild_claims_are_leased_layered_and_cleared(tmp_path):
    base = DiskStore(tmp_path / 'base.sqlite')
    for record_id, children in (('a', ()), ('compact', ('a',)),
                                ('recursive', ('compact',))):
        base.put(StoredRecord(record_id, torch.tensor([1., 0.]), torch.tensor([1.]),
            namespace='corpus', space='s0', generation='g', created_at=1),
            children=children)
    journal = DiskStore(tmp_path / 'journal.sqlite')
    index = PublishedKeyIndex(base, namespace='corpus', generation='g', spaces=('s0',),
                              expected_sources=3)
    bank = TrainingBank(base, journal, index)
    bank.update([StoredRecord('a', torch.tensor([0., 1.]), torch.tensor([2.]), space='s0')])

    first = bank.claim_rebuilds('worker-a', limit=2, lease_seconds=30, now_ns=100)
    assert [job['record_id'] for job in first] == ['compact']
    assert bank.claim_rebuilds('worker-b', limit=2, lease_seconds=30, now_ns=101) == ()
    bank.release_rebuilds('worker-a', ('compact',))
    assert bank.claim_rebuilds('worker-b', limit=1, lease_seconds=30, now_ns=102)[0][
        'record_id'] == 'compact'
    bank.update([StoredRecord('compact', torch.tensor([1., 1.]), torch.tensor([3.]),
                              space='s0')], children={'compact': ('a',)})
    assert [job['record_id'] for job in bank.claim_rebuilds(
        'worker-a', limit=2, lease_seconds=30, now_ns=103)] == ['recursive']
    bank.update([StoredRecord('recursive', torch.tensor([-1., 1.]), torch.tensor([4.]),
                              space='s0')], children={'recursive': ('compact',)})
    assert bank.invalidated_records() == ()
    with journal.connect() as db:
        assert db.execute('SELECT COUNT(*) FROM mutable_bank_rebuild_leases').fetchone()[0] == 0


def test_mutable_event_frontier_and_revision_commit_are_atomic(tmp_path):
    base = DiskStore(tmp_path / 'base.sqlite')
    base.put(StoredRecord('a', torch.tensor([1., 0.]), torch.tensor([1.]),
        namespace='corpus', space='s0', generation='g', created_at=1,
        source_id='source-a'))
    journal = DiskStore(tmp_path / 'journal.sqlite')
    index = PublishedKeyIndex(base, namespace='corpus', generation='g', spaces=('s0',),
                              expected_sources=1)
    bank = TrainingBank(base, journal, index)
    initial_state = journal.mutable_bank_state()
    record = StoredRecord('new', torch.tensor([0., 1.]), torch.tensor([2.]),
        namespace='corpus', space='s0', generation='g', created_at=2,
        source_id='trajectory-1')
    event = {'stream': 'events', 'position': 0, 'event_id': 'trajectory-1',
             'visibility_time': 2, 'lineage': {'new': ('record:a',)},
             'record_metadata': {'new': {'kind': 'lesson'}}}
    assert bank.update([record], children={'new': ('a',)}, event=event) == 1
    assert journal.event_frontier(namespace='corpus', stream='events', generation='g') == {
        'next_position': 1, 'visibility_time': 2}
    assert bank.update([record], children={'new': ('a',)}, event=event) == 0
    assert bank.cursor == 1
    bad = StoredRecord('later', torch.tensor([-1., 0.]), torch.tensor([3.]),
        namespace='corpus', space='s0', generation='g', created_at=3,
        source_id='trajectory-2')
    changed = event | {'event_id': 'trajectory-2', 'visibility_time': 3,
                       'lineage': {'later': ()},
                       'record_metadata': {'later': {}}}
    with pytest.raises(ValueError, match='committed differently'):
        bank.update([bad], event=changed)
    assert bank.cursor == 1
    journal.restore_mutable_bank(initial_state)
    assert journal.event_frontier(namespace='corpus', stream='events', generation='g') == {
        'next_position': 0, 'visibility_time': -1}


def test_mutable_gc_respects_checkpoint_pins(tmp_path):
    base = DiskStore(tmp_path / 'base.sqlite')
    base.put(StoredRecord('a', torch.tensor([1., 0.]), torch.tensor([1.]),
        namespace='corpus', space='s0', generation='g', created_at=1))
    journal = DiskStore(tmp_path / 'journal.sqlite')
    index = PublishedKeyIndex(base, namespace='corpus', generation='g', spaces=('s0',),
                              expected_sources=1)
    bank = TrainingBank(base, journal, index)
    bank.update([StoredRecord('a', torch.tensor([0., 1.]), torch.tensor([2.]), space='s0')])
    journal.pin_mutable_bank('checkpoint-one', 1)
    bank.update([StoredRecord('a', torch.tensor([-1., 0.]), torch.tensor([3.]), space='s0')])
    with pytest.raises(ValueError, match='pinned'):
        journal.gc_mutable_bank(2)
    journal.unpin_mutable_bank('checkpoint-one')
    journal.gc_mutable_bank(2)
    with pytest.raises(ValueError, match='retained'):
        bank.rollback(1)


def test_legacy_overlay_migrates_to_revision_journal(tmp_path):
    base = DiskStore(tmp_path / 'base.sqlite')
    for space in ('s0', 's1'):
        base.put(StoredRecord('a', torch.tensor([1., 0.]), torch.tensor([1.]),
            namespace='corpus', space=space, generation='g', created_at=1,
            source_id='source-a'))
    cache = DiskStore(tmp_path / 'cache.sqlite')
    with cache.connect() as db:
        db.execute('''CREATE TABLE training_bank_overlay (
            space TEXT NOT NULL, record_id TEXT NOT NULL, key BLOB NOT NULL,
            key_dim INTEGER NOT NULL, payload BLOB NOT NULL,
            PRIMARY KEY(space,record_id))''')
        for offset, space in enumerate(('s0', 's1')):
            key = torch.tensor([0., 1.]).numpy().astype('<f4').tobytes()
            db.execute('INSERT INTO training_bank_overlay VALUES (?,?,?,?,?)',
                       (space, 'a', key, 2,
                        save({'payload': torch.tensor([7. + offset])})))
    index = PublishedKeyIndex(base, namespace='corpus', generation='g',
                              spaces=('s0', 's1'), expected_sources=1)
    bank = TrainingBank(base, cache, index)
    assert bank.cursor == 1
    with cache.connect() as db:
        assert 'training_bank_overlay' not in bank._tables(db)
    plans = tuple(ReadPlan('corpus', space, 'g', 'research', 2,
                           (Selection('a', 0.),)) for space in ('s0', 's1'))
    rows = bank.fetch_many(plans)
    torch.testing.assert_close(rows[0][0], torch.tensor([7.]))
    torch.testing.assert_close(rows[1][0], torch.tensor([8.]))


def test_legacy_checkpoint_reuses_and_rolls_back_completed_migration(tiny_config, tmp_path):
    agent = SDKBAgent(tiny_config)
    optimizer = make_optimizer(agent)
    run = tmp_path / 'run'
    run.mkdir()
    base = DiskStore(tmp_path / 'base.sqlite')
    base.put(StoredRecord('a', torch.ones(tiny_config.memory.key_dim),
        torch.ones(tiny_config.memory.payload_dims[0]), namespace='corpus', space='s0',
        generation='g', created_at=1, source_id='source-a'))
    cache = DiskStore(run / 'training_cache.sqlite')
    with cache.connect() as db:
        db.execute('''CREATE TABLE training_bank_overlay (
            space TEXT NOT NULL, record_id TEXT NOT NULL, key BLOB NOT NULL,
            key_dim INTEGER NOT NULL, payload BLOB NOT NULL,
            PRIMARY KEY(space,record_id))''')
        key = torch.arange(tiny_config.memory.key_dim).float().add(1)
        db.execute('INSERT INTO training_bank_overlay VALUES (?,?,?,?,?)', (
            's0', 'a', key.numpy().astype('<f4').tobytes(), key.numel(),
            save({'payload': torch.full((tiny_config.memory.payload_dims[0],), 3.)})))
    save_checkpoint(agent, optimizer, run, 4, random.Random(1), cache, 'data')
    assert restore_checkpoint(agent, optimizer, run, random.Random(2), 'data') == 4
    index = PublishedKeyIndex(base, namespace='corpus', generation='g', spaces=('s0',),
                              expected_sources=1)
    bank = TrainingBank(base, cache, index)
    migrated_cursor = bank.cursor
    assert migrated_cursor == 1
    bank.update([StoredRecord('a', torch.arange(tiny_config.memory.key_dim).float() + 2,
                              torch.full((tiny_config.memory.payload_dims[0],), 9.),
                              space='s0')])
    assert bank.cursor == 2
    assert restore_checkpoint(agent, optimizer, run, random.Random(3), 'data') == 4
    assert cache.mutable_bank_state()['cursor'] == migrated_cursor


def test_payload_cache_serves_newest_revisions_under_snapshot_and_scope(tmp_path):
    base = DiskStore(tmp_path / 'base.sqlite')
    for record_id in ('a', 'b'):
        base.put(StoredRecord(record_id, torch.tensor([1., 0.]), torch.tensor([1., 2.]),
                              namespace='corpus', space='s0', generation='g', created_at=1))
    index = PublishedKeyIndex(base, namespace='corpus', generation='g', spaces=('s0',),
                              expected_sources=2)
    bank = TrainingBank(base, DiskStore(tmp_path / 'cache.sqlite'), index)
    bank.enable_payload_cache(1 << 20)
    plan = ReadPlan('corpus', 's0', 'g', 'research', 2, (Selection('a', 0.), Selection('b', 0.)))
    first = bank.fetch_many((plan,))[0]  # base payloads, now cached
    bank.update([StoredRecord('a', torch.tensor([0., 1.]), torch.tensor([7., 8.]), space='s0')])
    misses = bank.payload_cache_misses
    values = bank.fetch_many((plan,))[0]
    assert bank.payload_cache_misses == misses  # both served from memory
    torch.testing.assert_close(values[0], torch.tensor([7., 8.]))
    torch.testing.assert_close(values[1], first[1])
    # An older snapshot never sees the newer cached revision.
    old = ReadPlan('corpus', 's0', 'g', 'research', 2, (Selection('a', 0.),), bank_cursor=0)
    torch.testing.assert_close(bank.fetch_many((old,))[0][0], torch.tensor([1., 2.]))
    # Cached payloads still respect the record's authorization scope.
    with pytest.raises(KeyError):
        bank.fetch_many((ReadPlan('corpus', 's0', 'g', 'other', 2, (Selection('b', 0.),)),))
    with pytest.raises(KeyError):
        bank.fetch_many((ReadPlan('corpus', 's0', 'g', 'research', 1, (Selection('b', 0.),)),))
    bank.rollback(0)
    torch.testing.assert_close(bank.fetch_many((plan,))[0][0], torch.tensor([1., 2.]))
    bank.enable_payload_cache(8)  # evicts down to the byte cap
    assert bank._payload_bytes <= 8
