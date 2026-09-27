import pytest
import torch

from schnitz.compaction_maintenance import rebuild_invalidated
from schnitz.key_index import PublishedKeyIndex
from schnitz.store import DiskStore, StoredRecord
from schnitz.training_bank import TrainingBank


def make_bank(tmp_path):
    base = DiskStore(tmp_path / 'base.sqlite')
    base.put(StoredRecord('a', torch.tensor([1., 0.]), torch.tensor([1.]),
        namespace='corpus', space='s0', generation='g', created_at=1))
    base.put(StoredRecord('compact', torch.tensor([1., 0.]), torch.tensor([1.]),
        namespace='corpus', space='s0', generation='g', created_at=1), children=('a',))
    journal = DiskStore(tmp_path / 'journal.sqlite')
    index = PublishedKeyIndex(base, namespace='corpus', generation='g', spaces=('s0',),
                              expected_sources=2)
    bank = TrainingBank(base, journal, index)
    bank.update([StoredRecord('a', torch.tensor([0., 1.]), torch.tensor([2.]), space='s0')])
    return bank, journal


def test_rebuild_worker_publishes_claimed_parent(tmp_path):
    bank, _ = make_bank(tmp_path)

    def build(job):
        assert job['children'] == ('a',)
        return [StoredRecord(job['record_id'], torch.tensor([1., 1.]),
                             torch.tensor([3.]), space='s0')]

    assert rebuild_invalidated(bank, 'worker', build) == {'claimed': 1, 'rebuilt': 1}
    assert bank.invalidated_records() == ()


def test_rebuild_worker_releases_failed_claim(tmp_path):
    bank, journal = make_bank(tmp_path)

    with pytest.raises(RuntimeError, match='compactor failed'):
        rebuild_invalidated(bank, 'worker', lambda _job: (_ for _ in ()).throw(
            RuntimeError('compactor failed')))
    with journal.connect() as db:
        assert db.execute('SELECT COUNT(*) FROM mutable_bank_rebuild_leases').fetchone()[0] == 0
    assert bank.claim_rebuilds('retry', limit=1)[0]['record_id'] == 'compact'


def test_rebuild_worker_rejects_output_when_child_changes_after_claim(tmp_path):
    bank, journal = make_bank(tmp_path)

    def build(_job):
        bank.update([StoredRecord('a', torch.tensor([-1., 0.]),
                                  torch.tensor([9.]), space='s0')])
        return [StoredRecord('compact', torch.tensor([1., 1.]),
                             torch.tensor([3.]), space='s0')]

    with pytest.raises(RuntimeError, match='changed while its compaction was built'):
        rebuild_invalidated(bank, 'worker', build)
    assert bank.invalidated_records()[0]['record_id'] == 'compact'
    with journal.connect() as db:
        assert db.execute('SELECT COUNT(*) FROM mutable_bank_rebuild_leases').fetchone()[0] == 0


def test_rebuild_publication_requires_the_live_lease_owner(tmp_path):
    bank, _ = make_bank(tmp_path)
    job = bank.claim_rebuilds('worker-a', limit=1)[0]
    record = StoredRecord('compact', torch.tensor([1., 1.]),
                          torch.tensor([3.]), space='s0')
    with pytest.raises(RuntimeError, match='lease is stale or belongs to another worker'):
        bank.update(
            [record], children={'compact': ('a',)},
            expected_child_cursors={'compact': job['child_cursors']},
            rebuild_claims={'compact': ('worker-b', job['invalidated_at'])},
        )
    assert bank.invalidated_records()[0]['record_id'] == 'compact'


def test_exact_bank_restore_clears_outstanding_rebuild_leases(tmp_path):
    bank, journal = make_bank(tmp_path)
    state = journal.mutable_bank_state()
    assert bank.claim_rebuilds('worker-a', limit=1)
    journal.restore_mutable_bank(state)
    with journal.connect() as db:
        assert db.execute('SELECT COUNT(*) FROM mutable_bank_rebuild_leases').fetchone()[0] == 0
