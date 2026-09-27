import pytest
import torch

from schnitz.cluster_store import ClusterBank, state_fingerprint
from schnitz.readers import SetReader
from schnitz.store import DiskStore, StoredRecord, ReadPlan, Selection


def setup_bank(tmp_path, domain='research'):
    store = DiskStore(tmp_path / 'bank.sqlite')
    for i in range(4):
        store.put(StoredRecord(str(i), torch.ones(4), torch.full((8,), float(i), dtype=torch.bfloat16),
                               domain=domain, created_at=i+1))
    bank = ClusterBank(store, view='c1', reader_hash='reader1')
    plan = ReadPlan('default', 's0', 'v0', domain, 10, (Selection('0',0.),Selection('1',0.)))
    parent = bank.put(plan, torch.full((1,8), .5, dtype=torch.bfloat16), torch.tensor([2.]))
    return store, bank, plan, parent


def test_persistent_roundtrip_and_exact_partial_fallback(tmp_path):
    store, bank, plan, parent = setup_bank(tmp_path)
    reread = ClusterBank(DiskStore(store.path), view='c1', reader_hash='reader1')
    full = reread.fetch(plan)
    assert full.used_clusters == [parent] and full.raw_fallback_ids == []
    assert full.values.shape == (1,1,8) and full.weights.item() == 2
    partial = ReadPlan('default','s0','v0','research',10,(Selection('0',0.),Selection('3',0.)))
    result = reread.fetch(partial)
    assert result.used_clusters == [] and result.raw_fallback_ids == ['0','3']
    torch.testing.assert_close(result.values[0,1], torch.full((8,),3.,dtype=torch.bfloat16))


def test_cluster_deletion_identity_and_authorization(tmp_path):
    store, bank, plan, parent = setup_bank(tmp_path)
    with pytest.raises(ValueError, match='Stale'):
        ClusterBank(store, view='c1', reader_hash='different').fetch(plan)
    denied = ReadPlan('default','s0','v0','other',10,plan.selections)
    with pytest.raises(PermissionError):
        bank.fetch(denied)
    assert parent in store.delete('default','0')
    with pytest.raises(KeyError):
        bank.fetch(plan)
    # A surviving partial child remains readable from the raw path.
    assert bank.fetch(ReadPlan('default','s0','v0','research',10,(Selection('1',0.),))).raw_fallback_ids == ['1']


def test_disjoint_view_and_multiplicity_checks(tmp_path):
    _, bank, plan, _ = setup_bank(tmp_path)
    with pytest.raises(ValueError, match='multiplicity'):
        bank.put(plan, torch.ones(1,8), torch.ones(1))
    with pytest.raises(ValueError, match='disjoint'):
        bank.put(plan, torch.ones(1,8), torch.tensor([2.]))


def test_state_fingerprint_detects_reader_change():
    reader = SetReader(8,4,8,width=8,slots=2,rounds=2)
    first = state_fingerprint(reader)
    with torch.no_grad():
        reader.initial_slots.add_(.01)
    assert first != state_fingerprint(reader)


@pytest.mark.parametrize('timing', ['prefix', 'loop_boundary'])
def test_stored_codes_need_neither_writer_nor_compactor_at_read(tmp_path, tiny_config, monkeypatch, timing):
    from schnitz.agent import SchnitzelAgent
    from schnitz.data import make_boolean_world
    from schnitz.evaluation import build_shared_bank, build_persistent_codes, stored_transfer_evaluation
    tiny_config.memory.compaction = 'synthetic'
    tiny_config.memory.compact_records = 1
    if timing == 'loop_boundary':
        tiny_config.model.tiny_layers = 4
        tiny_config.model.recurrence_mode = 'middle_block'
        tiny_config.model.recurrent_start, tiny_config.model.recurrent_end = 1, 3
        tiny_config.model.loops, tiny_config.model.writer_loops = 3, 1
        tiny_config.memory.read_timing = timing
        tiny_config.memory.read_steps, tiny_config.memory.read_top_k = 2, 1
        tiny_config.memory.compaction = 'none'
    tiny_config.validate()
    agent = SchnitzelAgent(tiny_config).eval()
    episodes = make_boolean_world(0, operations=('a','b','xor'))
    store = DiskStore(tmp_path / 'bank.sqlite')
    build_shared_bank(agent, store, episodes)
    if timing == 'prefix':
        codes, manifest = build_persistent_codes(agent, store, episodes)
    else:
        # Offline compaction is separate from the recurrent training config.
        from schnitz.compaction import SyntheticCompactor
        compactor = SyntheticCompactor(24, 24, 1).eval()
        codes = ClusterBank(store, view='offline', reader_hash=state_fingerprint(agent.reader))
        plan = ReadPlan('global', 's0', 'frozen-v1', 'research', episodes[2].query_time,
                        tuple(Selection(rid, 0.) for rid in episodes[2].required_ids))
        raw = torch.stack(store.fetch(plan)).float()[None]
        with torch.no_grad():
            values, weights = compactor(raw, raw.new_ones(raw.shape[:2]))
        codes.put(plan, values[0].bfloat16(), weights[0])
        manifest = codes.sizes()
    assert manifest['codes'] == 1
    def forbidden(*_args, **_kwargs):
        raise AssertionError('Inference regenerated stored representations')
    monkeypatch.setattr(agent, 'produce', forbidden)
    monkeypatch.setattr(agent.compactor if timing == 'prefix' else compactor, 'forward', forbidden)
    updates = []
    result = stored_transfer_evaluation(agent, DiskStore(store.path), episodes, cluster_bank=codes,
                                        progress=updates.append)
    assert updates == [{'completed_queries': 3, 'total_queries': 3, 'scored_rows': 12}]
    persistent = [r for r in result['rows'] if r['condition'] == 'persistent']
    assert len(persistent) == 3
    assert persistent[0]['payload_accounting'][0]['raw_fallback_ids']
    assert persistent[2]['payload_accounting'][-1]['clusters']
    if timing == 'loop_boundary':
        # The first boundary has a partial cluster and must read the raw child.
        # Only the complete cumulative selection at the second can use its code.
        assert persistent[2]['payload_accounting'][0]['raw_fallback_ids']
        assert not persistent[2]['payload_accounting'][0]['clusters']
        assert not persistent[2]['payload_accounting'][1]['raw_fallback_ids']
        from schnitz.sessions import read_session
        captured = []
        handle = agent.reader.register_forward_pre_hook(
            lambda _module, args: captured.append((args[0].clone(), args[2].clone())))
        arguments = dict(namespace='global', generation='frozen-v1',
                         query_time=episodes[2].query_time, oracle_ids=episodes[2].required_ids,
                         cluster_bank=codes)
        try:
            session = read_session(agent, store, agent.prompt_ids(episodes[2].query),
                                   ablate_values=True, **arguments)
        finally:
            handle.remove()
        assert len(captured) == len(session.plans) == 2
        assert all(torch.count_nonzero(values) == 0 for values, _ in captured)
        assert [weights.sum().item() for _, weights in captured] == [1., 2.]
        assert [values.shape[1] for values, _ in captured] == [1, 1]
        # Stored-code use does not relax visibility or derivative invalidation.
        with pytest.raises(KeyError, match='unauthorized'):
            read_session(agent, store, agent.prompt_ids('query'), domain='denied', **arguments)
        store.delete('global', episodes[2].required_ids[0])
        with pytest.raises(KeyError):
            read_session(agent, store, agent.prompt_ids('query'), **arguments)


def test_failed_code_publication_leaves_no_marker_or_lineage(tmp_path):
    import sqlite3
    store, _, plan, _ = setup_bank(tmp_path)
    bank = ClusterBank(store, view='interrupted', reader_hash='reader1')
    with store.connect() as db:
        records = db.execute('SELECT * FROM records').fetchall()
        lineage = db.execute('SELECT * FROM lineage').fetchall()
        db.execute("""CREATE TRIGGER fail_members BEFORE INSERT ON cluster_members
                    WHEN NEW.view='interrupted' BEGIN SELECT RAISE(ABORT,'interrupted'); END""")
    with pytest.raises(sqlite3.IntegrityError, match='interrupted'):
        bank.put(plan, torch.ones(1, 8), torch.tensor([2.]))
    with store.connect() as db:
        assert db.execute('SELECT * FROM records').fetchall() == records
        assert db.execute('SELECT * FROM lineage').fetchall() == lineage
        assert not db.execute("SELECT * FROM cluster_codes WHERE view='interrupted'").fetchall()
        db.execute('DROP TRIGGER fail_members')
    parent = bank.put(plan, torch.ones(1, 8), torch.tensor([2.]))
    assert bank.fetch(plan).used_clusters == [parent]
