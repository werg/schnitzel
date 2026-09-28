import json

import numpy as np
import pytest
import torch

import schnitz.kb_store as kbs
from schnitz.kb_store import KnowledgeBase, NewItem, Provenance, SpaceSpec, search_kbs

SPACES = {'A': SpaceSpec(8, 4, 1.0), 'B': SpaceSpec(12, 6, 0.5)}


def item(space='A', n=3, time=0, mass=1.0, sources=('r0',), id=None, producer='codec'):
    spec = SPACES[space]
    return NewItem(torch.randn(n, spec.width), torch.randn(spec.key_width),
                   Provenance(tuple(sources), producer, 5), mass, time, id)


def make(tmp_path, name='kb', dataset='ds'):
    return KnowledgeBase.create(tmp_path / name, name=name, dataset=dataset, spaces=SPACES)


def test_roundtrip_and_reopen(tmp_path):
    kb = make(tmp_path)
    new = [item(n=n, sources=(f'r{n}',), time=n) for n in (1, 3, 5)]
    ids = kb.append('A', new)
    b = kb.append('B', [item('B', id='b-item')])
    assert b == ['b-item'] and kb.cursor == 2
    kb.close()
    kb = KnowledgeBase(tmp_path / 'kb')
    for got, want in zip(kb.read('A', ids), new):
        assert got.values.dtype == torch.bfloat16 and got.version == 1 and got.current
        torch.testing.assert_close(got.values, want.values.to(torch.bfloat16))
        torch.testing.assert_close(got.key, want.key)
        assert got.provenance == Provenance(want.provenance.sources, 'codec', 5, 'ds')
        assert got.time == want.time and got.mass == 1.0 and got.lineage == ()
    assert kb.read('B', ['b-item'])[0].values.shape == (3, 12)
    with pytest.raises(KeyError):
        kb.read('A', ['b-item'])          # an item lives in one space
    with pytest.raises(PermissionError):
        kb.append('A', [item()])          # opened read-only
    assert kb.stats()['A'] == {'items': 3, 'positions': 9, 'current': 3, 'live': False}


def test_validation_and_dataset_boundary(tmp_path):
    kb = make(tmp_path)
    with pytest.raises(ValueError):
        kb.append('A', [NewItem(torch.randn(2, 9), torch.randn(4), Provenance((), 'codec'))])
    with pytest.raises(ValueError):
        kb.append('A', [item(producer='oracle')])
    foreign = item()
    foreign.provenance = Provenance(('r',), 'codec', 0, 'other-dataset')
    with pytest.raises(PermissionError):
        kb.append('A', [foreign])
    kb.append('A', [item(id='x')])
    with pytest.raises(ValueError):
        kb.append('B', [item('B', id='x')])  # ids are unique within a KB
    with pytest.raises(RuntimeError):
        KnowledgeBase(tmp_path / 'kb', writable=True)  # one writer


def test_supersede_and_rewrite_lineage(tmp_path):
    kb = make(tmp_path)
    ids = kb.append('A', [item(sources=(f's{i}',), mass=m, time=t)
                          for i, (m, t) in enumerate([(1., 2), (2., 7), (0.5, 3), (1., 1)])])
    before = kb.cursor
    new = item(time=0)
    new.id = ids[0]
    assert kb.supersede('A', [new]) == [(ids[0], 2)]
    cur = kb.read('A', [ids[0]])[0]
    old = kb.read('A', [ids[0]], versions=[1])[0]
    assert cur.version == 2 and cur.lineage == ((ids[0], 1),) and cur.time == 2  # never earlier
    assert not old.current and kb.read('A', [ids[0]], cursor=before)[0].current
    torch.testing.assert_close(cur.values, new.values.to(torch.bfloat16))

    with pytest.raises(ValueError, match='mass'):
        kb.rewrite('A', ids[1:3], [item(mass=1.0)])
    outs = kb.rewrite('A', ids[1:3], [item(sources=(), mass=1.5, producer='rewrite'),
                                      item(sources=(), mass=1.0, producer='rewrite')])
    got = kb.read('A', outs)
    assert all(g.lineage == ((ids[1], 1), (ids[2], 1)) for g in got)
    assert all(g.provenance.sources == ('s1', 's2') and g.time == 7 for g in got)
    for gone in ids[1:3]:
        with pytest.raises(KeyError):
            kb.read('A', [gone])
        assert not kb.read('A', [gone], versions=[1])[0].current
    with pytest.raises(KeyError):
        kb.rewrite('A', [ids[1]], [item(mass=2.0)])   # superseded inputs cannot be rewritten
    assert kb.stats()['A']['current'] == 4 and kb.stats()['A']['items'] == 7


def brute(keys, valid, queries, k, metric):
    if metric == 'cosine':
        keys = keys / keys.norm(dim=1, keepdim=True)
        queries = queries / queries.norm(dim=1, keepdim=True)
    scores = queries @ keys.T
    scores[:, ~valid] = float('-inf')
    top = scores.topk(min(k, int(valid.sum())), 1)
    return top.values, top.indices


@pytest.mark.parametrize('metric', ['cosine', 'dot'])
def test_exact_search_matches_brute_force(tmp_path, metric):
    kb = make(tmp_path)
    items = [item(n=int(torch.randint(1, 5, ())), time=int(t))
             for t in torch.randint(0, 10, (60,))]
    ids = kb.append('A', items[:40]) + kb.append('A', items[40:])
    keys = torch.stack([i.key for i in items])
    for j in range(0, 60, 7):   # supersede some with new keys, rewrite two others away
        new = item(time=items[j].time)
        new.id = ids[j]
        kb.supersede('A', [new])
        keys[j] = new.key
    kb.rewrite('A', [ids[1], ids[2]], [item(mass=2.0, producer='rewrite', time=4)])
    new_key = kb.read('A', [kb._row_ids['A'][-1]])[0].key
    queries = torch.randn(5, 4)
    valid = torch.ones(60, dtype=torch.bool)
    valid[[1, 2]] = False
    all_keys, all_valid = torch.cat((keys, new_key[None])), torch.cat((valid, torch.tensor([True])))
    all_ids = ids + [kb._row_ids['A'][-1]]
    for chunk in (3, 16, 10_000):
        hits = kb.search('A', queries, 8, metric=metric, chunk_rows=chunk)
        values, index = brute(all_keys, all_valid, queries, 8, metric)
        for b in range(5):
            assert hits.ids[b] == [all_ids[i] for i in index[b]]
            torch.testing.assert_close(hits.scores[b], values[b])
    # causal filter: a query at time 3 sees only items available by then
    times = torch.tensor([i.time for i in items] + [4])
    hits = kb.search('A', queries, 100, metric=metric, query_time=[3, 3, 3, 3, 9])
    for b, qt in enumerate([3, 3, 3, 3, 9]):
        mask = all_valid & (times <= qt)
        values, index = brute(all_keys, mask, queries, 100, metric)
        assert hits.ids[b] == [all_ids[i] for i in index[b]]
    hits = kb.search('A', queries[:1], 2, return_items=True)
    assert [i.id for i in hits.items[0]] == hits.ids[0] and all(i.current for i in hits.items[0])


def test_reader_snapshot_isolated_from_writer(tmp_path):
    kb = make(tmp_path)
    ids = kb.append('A', [item() for _ in range(4)])
    reader = KnowledgeBase(tmp_path / 'kb')
    q = torch.randn(1, 4)
    before = reader.search('A', q, 10)
    new = item()
    new.id = ids[0]
    kb.supersede('A', [new])
    kb.append('A', [item() for _ in range(3)])
    again = reader.search('A', q, 10)
    assert again.ids == before.ids and again.versions == before.versions  # pinned cursor
    reader.refresh()
    assert len(reader.search('A', q, 10).ids[0]) == 7 and reader.cursor == kb.cursor
    # the writer can also read a pinned older cursor
    assert kb.search('A', q, 10, cursor=1).ids == before.ids


def test_interrupted_commit_is_discarded(tmp_path, monkeypatch):
    kb = make(tmp_path)
    ids = kb.append('A', [item() for _ in range(3)])
    q = torch.randn(2, 4)
    reference = kb.search('A', q, 5)
    replace = item()
    replace.id = ids[1]

    def crash(path, value):
        raise OSError('power loss')
    monkeypatch.setattr(kbs, '_atomic_json', crash)
    with pytest.raises(OSError):   # rows written and a dead mark set, manifest not replaced
        kb.supersede('A', [replace])
    monkeypatch.undo()
    kb.close()
    (tmp_path / 'kb' / 'manifest.json.pending').write_text('{"torn": ')
    kb = KnowledgeBase(tmp_path / 'kb', writable=True)
    assert kb.cursor == 1 and kb.stats()['A']['items'] == 3
    assert not (tmp_path / 'kb' / 'manifest.json.pending').exists()
    assert (tmp_path / 'kb' / 'A' / 'keys.f32').stat().st_size == 3 * 4 * 4
    after = kb.search('A', q, 5)
    assert after.ids == reference.ids and kb.read('A', [ids[1]])[0].version == 1
    kb.supersede('A', [replace])    # resumes cleanly
    assert kb.read('A', [ids[1]])[0].version == 2


def test_scoping_and_cross_kb_authorization(tmp_path):
    one, two = make(tmp_path, 'one', 'ds1'), make(tmp_path, 'two', 'ds2')
    a = one.append('A', [item() for _ in range(5)])
    b = two.append('A', [item() for _ in range(5)])
    q = torch.randn(3, 4)
    assert set(sum(one.search('A', q, 10).ids, [])) <= set(a)   # one KB, one domain
    with pytest.raises(PermissionError):
        search_kbs([one, two], ['ds1'], 'A', q, 4)
    with pytest.raises(ValueError):
        search_kbs([one, one], ['ds1'], 'A', q, 4)
    hits = search_kbs([one, two], ['ds1', 'ds2'], 'A', q, 4, return_items=True)
    for row in range(3):
        pool = sorted([(s, i, 'ds1') for s, i in zip(one.search('A', q, 4).scores[row].tolist(),
                                                     one.search('A', q, 4).ids[row])] +
                      [(s, i, 'ds2') for s, i in zip(two.search('A', q, 4).scores[row].tolist(),
                                                     two.search('A', q, 4).ids[row])],
                      key=lambda x: -x[0])[:4]
        assert hits.ids[row] == [i for _, i, _ in pool]
        assert hits.datasets[row] == [d for _, _, d in pool]
        assert all((d == 'ds1') == (i in a) for i, d in zip(hits.ids[row], hits.datasets[row]))
    assert set(b).isdisjoint(a)


@pytest.mark.parametrize('weight_decay', [0.0, 0.1])
def test_live_adam_matches_torch_per_item(tmp_path, weight_decay):
    kb = make(tmp_path)
    items = [item(n=n) for n in (2, 4, 3, 1)]
    ids = kb.append('A', items[:2])
    kb.enable_live('A')
    ids += kb.append('A', items[2:])      # items appended in live mode join live state
    params = [torch.nn.Parameter(i.values.to(torch.bfloat16).float()) for i in items]
    make_opt = torch.optim.AdamW if weight_decay else torch.optim.Adam
    opts = [make_opt([p], lr=0.01, betas=(0.8, 0.95), eps=1e-6, weight_decay=weight_decay,
                     foreach=False) for p in params]
    schedule = [[0, 1, 2, 3], [1, 3], [3], [0, 1, 2, 3], [0, 2]]   # sparse, uneven steps
    for touched in schedule:
        grads = [torch.randn_like(params[j]) for j in touched]
        for j, g in zip(touched, grads):
            params[j].grad = g
            opts[j].step()
        kb.live_step('A', [ids[j] for j in touched], grads, lr=0.01, betas=(0.8, 0.95),
                     eps=1e-6, weight_decay=weight_decay)
    assert kb.live_updates == len(schedule)
    kb.close()
    kb = KnowledgeBase(tmp_path / 'kb', writable=True)   # persisted
    for j, got in enumerate(kb.read('A', ids, live=True)):
        torch.testing.assert_close(got.values, params[j].detach(), rtol=1e-6, atol=1e-7)
        state = opts[j].state[params[j]]
        row = kb._rows['A'][ids[j]][-1]
        off, n = kb._map('A', 'rows.i64')[row, [0, 1]]
        torch.testing.assert_close(torch.from_numpy(np.array(kb._map('A', 'live_m.f32')[off:off + n])),
                                   state['exp_avg'], rtol=1e-6, atol=1e-8)
        assert kb._map('A', 'live_step.i64')[row] == int(state['step'])
    # stored bf16 payloads are untouched by live updates
    torch.testing.assert_close(kb.read('A', [ids[0]])[0].values, items[0].values.to(torch.bfloat16))
    with pytest.raises(ValueError):
        kb.live_step('A', [ids[0], ids[0]], [torch.zeros(2, 8)] * 2, lr=0.1)
    with pytest.raises(ValueError):
        kb.live_step('B', [], [], lr=0.1)       # B is not live


def test_live_journal_replay_and_torn_journal(tmp_path, monkeypatch):
    kb = make(tmp_path)
    ids = kb.append('A', [item(n=2) for _ in range(3)])
    kb.enable_live('A')
    kb.live_step('A', ids[:1], [torch.ones(2, 8)], lr=0.1)
    applied = kb.read('A', ids, live=True)
    # a journal that is durable but not applied is replayed at the next writer open
    monkeypatch.setattr(KnowledgeBase, '_apply_journal', lambda self, path: None)
    kb.live_step('A', ids[1:], [torch.ones(2, 8), -torch.ones(2, 8)], lr=0.1)
    monkeypatch.undo()
    assert kb.live_updates == 1
    kb.close()
    kb = KnowledgeBase(tmp_path / 'kb', writable=True)
    assert kb.live_updates == 2 and not (tmp_path / 'kb' / 'live.journal').exists()
    got = kb.read('A', ids, live=True)
    torch.testing.assert_close(got[0].values, applied[0].values)
    torch.testing.assert_close(got[1].values, applied[1].values - 0.1)
    torch.testing.assert_close(got[2].values, applied[2].values + 0.1)
    # a pending (not renamed) journal is ignored
    kb.close()
    (tmp_path / 'kb' / 'live.journal.pending').write_bytes(b'garbage')
    kb = KnowledgeBase(tmp_path / 'kb', writable=True)
    assert kb.live_updates == 2
    torch.testing.assert_close(kb.read('A', ids, live=True)[1].values, got[1].values)


def test_export_live_is_frozen_snapshot(tmp_path):
    kb = make(tmp_path)
    ids = kb.append('A', [item(n=2, sources=(f's{i}',)) for i in range(3)])
    bid = kb.append('B', [item('B')])
    replaced = item(n=2)
    replaced.id = ids[2]
    kb.supersede('A', [replaced])
    kb.enable_live('A')
    kb.live_step('A', ids[:2], [torch.ones(2, 8)] * 2, lr=0.05)
    kb.set_live_keys('A', ids[:1], torch.ones(1, 4))
    live = kb.read('A', ids, live=True)
    out = kb.export_live(tmp_path / 'export')
    assert out.dataset == 'ds' and out.name == 'kb@live2' and out.stats()['A']['current'] == 3
    manifest = json.loads((tmp_path / 'export' / 'manifest.json').read_text())
    assert manifest['frozen'] and manifest['origin'] == {'name': 'kb', 'cursor': kb.cursor,
                                                         'live_updates': 2}
    got = out.read('A', ids)
    for g, want in zip(got, live):
        torch.testing.assert_close(g.values, want.values.to(torch.bfloat16))
        assert g.provenance.producer == 'live-update' and g.id == want.id
    assert [g.provenance.step for g in got] == [1, 1, 0] and got[2].version == 2
    assert got[2].lineage == ((ids[2], 1),)
    torch.testing.assert_close(got[0].key, torch.ones(4))
    assert out.read('B', bid)[0].provenance.producer == 'codec'
    with pytest.raises(PermissionError):
        KnowledgeBase(tmp_path / 'export', writable=True)
    kb.live_step('A', ids[:1], [torch.ones(2, 8)], lr=0.05)      # live training continues
    torch.testing.assert_close(out.read('A', ids[:1])[0].values, got[0].values)
    assert not (tmp_path / 'export.pending').exists()
