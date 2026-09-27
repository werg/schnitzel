from dataclasses import replace

import pytest
import torch

from schnitz.agent import SchnitzelAgent
from schnitz.data import make_episode
from schnitz.evaluation import build_shared_bank
from schnitz.store import DiskStore, StoredRecord


def test_committed_bank_resumes_without_writer_and_ignores_code_views(tiny_config, tmp_path, monkeypatch):
    agent = SchnitzelAgent(tiny_config).eval()
    store = DiskStore(tmp_path / 'bank.sqlite')
    episodes = [make_episode(0)]
    first = build_shared_bank(agent, store, episodes, writer_identity='checkpoint-sha')
    assert first['writer_calls'] > 0
    store.put(StoredRecord('code', torch.ones(2), torch.ones(2), namespace='global',
                           space='compact-test', generation='frozen-v1'))
    monkeypatch.setattr(agent, 'produce', lambda *_: pytest.fail('Resume re-encoded sources'))
    resumed = build_shared_bank(agent, store, episodes, writer_identity='checkpoint-sha')
    assert resumed['writer_calls'] == 0
    assert resumed['unique_sources'] == first['unique_sources']


def test_interrupted_bank_and_manifest_roll_back_together(tiny_config, tmp_path, monkeypatch):
    agent = SchnitzelAgent(tiny_config).eval()
    store = DiskStore(tmp_path / 'bank.sqlite')
    produce, calls = agent.produce, 0
    def interrupted(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('interruption')
        return produce(*args)
    monkeypatch.setattr(agent, 'produce', interrupted)
    with pytest.raises(RuntimeError, match='interruption'):
        build_shared_bank(agent, store, [make_episode(0)], writer_identity='a')
    with store.connect() as db:
        assert db.execute('SELECT COUNT(*) FROM records').fetchone()[0] == 0
    monkeypatch.setattr(agent, 'produce', produce)
    assert build_shared_bank(agent, store, [make_episode(0)], writer_identity='a')['writer_calls'] > 0


@pytest.mark.parametrize('change', ['writer', 'source', 'payload', 'deleted', 'extra', 'precision'])
def test_resume_rejects_changed_identity_or_raw_records(tiny_config, tmp_path, change):
    agent = SchnitzelAgent(tiny_config).eval()
    store = DiskStore(tmp_path / 'bank.sqlite')
    episodes = [make_episode(0)]
    build_shared_bank(agent, store, episodes, writer_identity='a')
    identity = 'a'
    if change == 'writer':
        identity = 'b'
    elif change == 'source':
        episodes = [replace(episodes[0], supports=tuple(replace(s, created_at=s.created_at + 1)
                                                       for s in episodes[0].supports))]
    elif change == 'precision':
        agent.config.memory.storage_dtype = 'float32'
    else:
        with store.connect() as db:
            if change == 'extra':
                db.execute("UPDATE records SET record_id=record_id || '-changed'")
            else:
                db.execute('UPDATE records SET ' + ('payload=?' if change == 'payload' else 'deleted=?'),
                           (b'changed' if change == 'payload' else 1,))
    with pytest.raises(ValueError, match='identity|contents'):
        build_shared_bank(agent, store, episodes, writer_identity=identity)


def test_legacy_bank_is_not_silently_adopted(tiny_config, tmp_path):
    agent = SchnitzelAgent(tiny_config).eval()
    store = DiskStore(tmp_path / 'bank.sqlite')
    episodes = [make_episode(0)]
    build_shared_bank(agent, store, episodes)
    with pytest.raises(ValueError, match='manifest'):
        build_shared_bank(agent, store, episodes, writer_identity='a')
