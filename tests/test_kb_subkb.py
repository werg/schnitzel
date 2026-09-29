"""KB-size curriculum: sub-KBs cut from a bank (``schnitz.kb.subkb``, ``l1 subkb``); CPU,
tiny fixtures, no downloads."""
from __future__ import annotations

import json

import pytest
import torch

from schnitz.kb import subkb as sk
from schnitz.kb import teacher_keys as tk
from schnitz.kb.read import source_index
from schnitz.kb.stages import l1
from schnitz.kb_store import KnowledgeBase, NewItem, Provenance
from test_kb_read import make_kb
from test_kb_teacher_keys import HashEmbedder, transcript

RECORDS = tuple((f'r{i}', 1) for i in range(1, 9)) + (('r9', 5),)


def bank(tmp_path, rows, *, records=RECORDS, extra_kb=None):
    """A leaf bank (``l1 build`` layout) over ``records`` with the transcripts ``rows``
    (and optionally a second KB ``secret``)."""
    src = tmp_path / 'corpus'
    src.mkdir()
    (src / 'sources.jsonl').write_text(''.join(
        json.dumps({'record_id': r, 'text': f'record {r} text', 'created_at': t}) + '\n'
        for r, t in records))
    d = tmp_path / 'transcripts'
    d.mkdir()
    (d / 'manifest.json').write_text(json.dumps({'input': str(src), 'kb': 'ds'}))
    (d / 'transcripts-train.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in rows))
    (d / 'transcripts-validation.jsonl').write_text('')
    banks = tmp_path / 'banks'
    banks.mkdir()
    make_kb(banks, 'ds', 'ds', records=records, live=False).close()
    kbs = {'ds': {'dir': 'ds', 'records': len(records)}}
    if extra_kb:
        make_kb(banks, 'secret', 'secret', records=extra_kb, live=False).close()
        kbs['secret'] = {'dir': 'secret', 'records': len(extra_kb)}
    (banks / 'banks.json').write_text(json.dumps({
        'command': 'build', 'transcripts': [str(d)], 'limit': None, 'eval_limit': None,
        'codec_step': 3, 'kbs': kbs}))
    torch.save({'dims': {}}, banks / 'stack.pt')
    torch.save({}, banks / 'key_heads_init.pt')
    return banks, d


ROWS = [transcript('e', (['r1'], ['r2']), neutral=['r3']),
        transcript('f', (['r5'],))]
ROWS[0]['messages'][3]['content']['slot']['alternatives'] = ['r1', 'r4']


def items_of(kb, space):
    return {i.id: i for i in kb.read(space, kb._row_ids[space])}


def test_subkbs_copy_stored_items_and_rewrite_only_the_kb(tmp_path):
    banks, d = bank(tmp_path, ROWS)
    out = sk.build(banks, tmp_path / 'sub', size=5, mode='random', group_size=1, seed=0,
                   log=lambda x: None)
    assert sorted(out['kbs']) == ['ds#t00000', 'ds#t00001']
    parent = KnowledgeBase(banks / 'ds')
    before = parent.stats()
    by_episode = {info['episodes'][0]: (name, info) for name, info in out['kbs'].items()}
    name, info = by_episode['e']
    # the slots' records and alternatives, then distractors up to the size; distractors
    # are never a slot's positive or neutral record and are visible at the query time
    assert info['core'] == 3 and info['records'] == 5 and info['distractors'] == 2
    child = KnowledgeBase(tmp_path / 'sub' / info['dir'])
    assert child.dataset == name and sk.parent_kb(name) == 'ds'
    records = set(source_index(child, 'D'))
    assert {'r1', 'r2', 'r4'} <= records and 'r3' not in records and 'r9' not in records
    assert len(records - {'r1', 'r2', 'r4'}) == 2 and not (records & {'r3'})
    for s in child.spaces:                   # bit-identical copies, ids and provenance kept
        mine, theirs = items_of(child, s), items_of(parent, s)
        assert mine and set(mine) <= set(theirs)
        for item_id, item in mine.items():
            ref = theirs[item_id]
            assert torch.equal(item.values.view(torch.int16), ref.values.view(torch.int16))
            assert torch.equal(item.key, ref.key)
            assert (item.mass, item.time, item.version) == (ref.mass, ref.time, ref.version)
            assert item.provenance.sources == ref.provenance.sources
            assert (item.provenance.producer, item.provenance.step) == \
                (ref.provenance.producer, ref.provenance.step)
    assert parent.stats() == before
    # transcripts: only the KB names change
    rows = [json.loads(line) for line in
            (tmp_path / 'sub' / 'transcripts' / d.name / 'transcripts-train.jsonl').open()]
    got = {r['episode_id']: r for r in rows}
    assert got['e']['kb'] == name and got['e']['provenance']['parent_kb'] == 'ds'
    slot = got['e']['messages'][3]['content']['slot']
    assert slot['kb'] == name and slot['alternatives'] == ['r1', 'r4'] and slot['neutral'] == ['r3']
    source = dict(ROWS[0], kb=name)
    assert got['e']['messages'][1] == source['messages'][1]
    manifest = json.loads((tmp_path / 'sub' / 'banks.json').read_text())
    assert manifest['transcripts'] == [str(tmp_path / 'sub' / 'transcripts' / d.name)]
    assert (tmp_path / 'sub' / 'stack.pt').exists()
    assert (tmp_path / 'sub' / 'key_heads_init.pt').exists()


def test_per_batch_groups_and_neutral_option(tmp_path):
    banks, _ = bank(tmp_path, ROWS)
    out = sk.build(banks, tmp_path / 'sub', size=3, group_size=2, with_neutral=True,
                   log=lambda x: None)
    (name, info), = out['kbs'].items()
    assert sorted(info['episodes']) == ['e', 'f'] and info['distractors'] == 0
    child = KnowledgeBase(tmp_path / 'sub' / info['dir'])
    assert set(source_index(child, 'A')) == {'r1', 'r2', 'r3', 'r4', 'r5'}   # over size: kept
    assert out['counts'] == {'groups_over_size': 1}


def test_teacher_and_mixed_distractors_are_near_misses(tmp_path):
    banks, d = bank(tmp_path, ROWS)
    records = {r: {'text': f'record {r} text', 'kb': 'ds', 'created_at': t} for r, t in RECORDS}
    tk.build(tmp_path / 'teacher', HashEmbedder(), [d], {'train': None}, records, top=16)
    cache = tk.TeacherKeys(tmp_path / 'teacher')
    out = sk.build(banks, tmp_path / 'sub', size=5, mode='teacher', teacher_dir=tmp_path / 'teacher',
                   log=lambda x: None)
    info = next(v for v in out['kbs'].values() if v['episodes'] == ['e'])
    named = {'r1', 'r2', 'r3', 'r4'}
    lists = [[r for r in cache.mined('e', j, 16) if r not in named and r != 'r9']
             for j in range(2)]
    want = list(dict.fromkeys(x for pair in zip(*lists) for x in pair))[:2]
    child = KnowledgeBase(tmp_path / 'sub' / info['dir'])
    assert set(source_index(child, 'D')) == {'r1', 'r2', 'r4', *want}
    assert info['distractor_sources'] == {'teacher': 2, 'random': 0}
    mixed = sk.build(banks, tmp_path / 'mixed', size=7, mode='mixed',
                     teacher_dir=tmp_path / 'teacher', log=lambda x: None)
    info = next(v for v in mixed['kbs'].values() if v['episodes'] == ['e'])
    assert info['distractor_sources'] == {'teacher': 2, 'random': 2}
    with pytest.raises(ValueError):
        sk.build(banks, tmp_path / 'none', size=5, mode='teacher', log=lambda x: None)
    # the teacher accepts a sub-KB of its site's KB, never another KB
    assert cache.mined('e', 0, 3, kb='ds#t00000') == cache.mined('e', 0, 3, kb='ds')
    with pytest.raises(PermissionError):
        cache.mined('e', 0, 3, kb='secret#t00000')


def test_subkbs_never_mix_kbs_and_refuse_derived_items(tmp_path):
    rows = ROWS + [transcript('g', (['s1'],), kb='secret')]
    banks, _ = bank(tmp_path, rows, extra_kb=(('s1', 1), ('s2', 1), ('s3', 1)))
    out = sk.build(banks, tmp_path / 'sub', size=4, log=lambda x: None)
    for name, info in out['kbs'].items():
        child = KnowledgeBase(tmp_path / 'sub' / info['dir'])
        parent = KnowledgeBase(banks / sk.parent_kb(name))
        assert set(source_index(child, 'B')) <= set(source_index(parent, 'B'))
        assert info['parent'] == sk.parent_kb(name)
    secret = [v for k, v in out['kbs'].items() if sk.parent_kb(k) == 'secret']
    assert len(secret) == 1 and secret[0]['records'] == 3        # only the parent's records
    kb = KnowledgeBase(banks / 'ds', writable=True)
    ids = kb._row_ids['A'][:2]
    new = kb.rewrite('A', ids, [NewItem(torch.zeros(2, 256), torch.zeros(256),
                                        Provenance(('r1', 'r2'), 'rewrite'), 2.0, 1)])
    kb.close()
    with pytest.raises(ValueError):
        sk.copy_items(KnowledgeBase(banks / 'ds'), None, ['r1'],
                      sk.ParentIndex(items={'r1': {'A': new}}, time={'r1': 1}))
    with pytest.raises(ValueError):
        sk.sub_name('ds#t00000', 'train', 1)


def test_l1_cli_routes_subkb_and_hard_negatives(tmp_path):
    import argparse
    banks, d = bank(tmp_path, ROWS)
    parser = argparse.ArgumentParser()
    l1.add_args(parser)
    args = parser.parse_args(['subkb', '--banks', str(banks), '--output', str(tmp_path / 'sub'),
                              '--size', '4', '--distractors', 'random'])
    l1.run(args)
    assert json.loads((tmp_path / 'sub' / 'banks.json').read_text())['size'] == 4
    with pytest.raises(SystemExit):
        parser.parse_args(['build', '--output', 'x', '--distractors', 'nearest'])
    assert parser.parse_args(['build', '--output', 'x', '--distractors', '12']).distractors == 12
    hits = tmp_path / 'hits.json'
    from schnitz.kb import hard_negatives as hn
    hn.save(hits, [{'episode_id': 'e', 'call': 0, 'kb': 'ds', 'records': ['r1', 'r6', 'r7']}],
            fmt=hn.HITS_FORMAT)
    args = parser.parse_args(['hard-negatives', '--transcripts', str(d), '--hits', str(hits),
                              '--output', str(tmp_path / 'hn.json')])
    l1.run(args)
    got = hn.HardNegatives.load(tmp_path / 'hn.json')
    assert got.records('e', 0, 'ds') == ['r6', 'r7']       # the slot's own record dropped
