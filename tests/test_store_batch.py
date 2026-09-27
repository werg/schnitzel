import sqlite3
from contextlib import contextmanager

import pytest
import torch

from schnitz.store import DiskStore, StoredRecord
from schnitz.agent import SchnitzelAgent
from schnitz.data import make_multiuse_world
from schnitz.evaluation import build_shared_bank, stored_transfer_evaluation
from schnitz.trajectory_eval import build_teacher_bank


def record(name, **kwargs):
    return StoredRecord(name, torch.tensor([1., 2.]),
                        torch.tensor([3., 4.], dtype=torch.bfloat16), **kwargs)


def test_bulk_write_matches_individual_serialized_records(tmp_path):
    records = [record(str(i), namespace='world', domain='owner', created_at=i,
                      source_id=f'opaque-{i}', generation='frozen-2') for i in range(4)]
    single, bulk = DiskStore(tmp_path / 'single.sqlite'), DiskStore(tmp_path / 'bulk.sqlite')
    for r in records:
        single.put(r)
    bulk.put_many(iter(records))
    with single.connect() as a, bulk.connect() as b:
        assert a.execute('SELECT * FROM records ORDER BY record_id').fetchall() == b.execute(
            'SELECT * FROM records ORDER BY record_id').fetchall()


def test_bulk_failure_rolls_back_every_new_record(tmp_path):
    store = DiskStore(tmp_path / 'bank.sqlite')
    store.put(record('existing'))
    def records():
        yield record('new')
        # A concurrent reader sees no half-written bank, even though the
        # generator is consumed incrementally rather than materialized first.
        with store.connect() as db:
            assert db.execute('SELECT record_id FROM records').fetchall() == [('existing',)]
        yield record('existing')
    with pytest.raises(sqlite3.IntegrityError):
        store.put_many(records())
    with store.connect() as db:
        assert db.execute('SELECT record_id FROM records').fetchall() == [('existing',)]


def test_bulk_write_cannot_resurrect_deleted_identity(tmp_path):
    store = DiskStore(tmp_path / 'bank.sqlite')
    store.put(record('deleted'))
    store.delete('default', 'deleted')
    with pytest.raises(ValueError, match='resurrected'):
        store.put_many([record('new'), record('deleted', generation='v2')])
    assert store.sizes()['records'] == 0


def test_tombstone_check_and_insert_exclude_concurrent_deletion(tmp_path, monkeypatch):
    store = DiskStore(tmp_path / 'bank.sqlite')
    original_connect = store.connect
    outcomes = []
    def interleave(statement):
        if statement.startswith('INSERT INTO records'):
            # The authorization/deletion check has already happened. Attempt a
            # deletion on another connection immediately before the insert.
            peer = sqlite3.connect(store.path, timeout=0.01)
            try:
                with peer:
                    peer.execute("INSERT INTO tombstones VALUES ('default','victim')")
                outcomes.append('deletion slipped between check and insert')
            except sqlite3.OperationalError as exc:
                outcomes.append('blocked' if 'locked' in str(exc) else str(exc))
            finally:
                peer.close()
    @contextmanager
    def traced_connect():
        with original_connect() as db:
            db.set_trace_callback(interleave)
            yield db
    monkeypatch.setattr(store, 'connect', traced_connect)
    store.put(record('victim'))
    assert outcomes == ['blocked']


def test_deletion_snapshot_excludes_new_compacted_parents(tmp_path, monkeypatch):
    store = DiskStore(tmp_path / 'bank.sqlite')
    store.put(record('victim'))
    original_connect = store.connect
    outcomes = []
    def interleave(statement):
        if statement.startswith('UPDATE records') and not outcomes:
            peer = sqlite3.connect(store.path, timeout=0.01)
            try:
                with peer:
                    peer.execute("""INSERT INTO records SELECT namespace,'late-parent',space,
                        generation,domain,created_at,key,key_dim,payload,source_id,deleted
                        FROM records WHERE record_id='victim'""")
                    peer.execute("INSERT INTO lineage VALUES ('default','late-parent','victim')")
                outcomes.append('parent escaped the deletion snapshot')
            except sqlite3.OperationalError as exc:
                outcomes.append('blocked' if 'locked' in str(exc) else str(exc))
            finally:
                peer.close()
    @contextmanager
    def traced_connect():
        with original_connect() as db:
            db.set_trace_callback(interleave)
            yield db
    monkeypatch.setattr(store, 'connect', traced_connect)
    store.delete('default', 'victim')
    assert outcomes == ['blocked']
    assert store.sizes()['records'] == 0


@pytest.mark.parametrize('teacher', [False, True])
def test_offline_batch_matches_reference_bank_and_reads(tmp_path, tiny_config, teacher):
    class IndividualStore(DiskStore):
        def put_many(self, records):
            for r in records:
                self.put(r)
    agent = SchnitzelAgent(tiny_config).eval()
    episodes = make_multiuse_world(0, bindings=2)
    reference = IndividualStore(tmp_path / 'reference.sqlite')
    bulk = DiskStore(tmp_path / 'bulk.sqlite')
    builder = build_teacher_bank if teacher else build_shared_bank
    assert builder(agent, reference, episodes) == builder(agent, bulk, episodes)
    with reference.connect() as a, bulk.connect() as b:
        assert a.execute('SELECT * FROM records ORDER BY namespace,record_id,space').fetchall() == b.execute(
            'SELECT * FROM records ORDER BY namespace,record_id,space').fetchall()
    if not teacher:
        assert stored_transfer_evaluation(agent, reference, episodes)['rows'] == stored_transfer_evaluation(
            agent, bulk, episodes)['rows']
