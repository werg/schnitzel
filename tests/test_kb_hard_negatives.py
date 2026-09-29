"""Mined hard negatives (``schnitz.kb.hard_negatives``, ``l1 train --hard-negatives`` and
``--dump-hits``); CPU, tiny models, no downloads."""
from __future__ import annotations

import random

import pytest
import torch

from schnitz.kb import hard_negatives as hn
from schnitz.kb.read import ItemCache
from schnitz.kb.stages import l1
from test_kb_read import (IDS, SPACES, _torch_conv, episode, make_kb, reader,  # noqa: F401
                          targets_of, tiny_lm, train_args)

RECORDS = tuple((f'r{i}', 1) for i in range(1, 7)) + (('r7', 5),)


def big_context(tmp_path, **kw):
    lm, _ = tiny_lm()
    kb = make_kb(tmp_path, records=RECORDS)
    return l1.Context(l1.Frozen(lm, 2), reader(**kw), {'ds': kb})


def row(eid, slots, kb='ds'):
    messages = [{'role': 'user', 'content': 'q'}]
    for slot in slots:
        messages += [{'role': 'assistant', 'content': '', 'tool_calls': [
            {'type': 'function', 'function': {'name': 'memory_search', 'arguments': {}}}]},
            {'role': 'tool', 'name': 'memory_search', 'content': {'slot': {'kb': kb, **slot}}}]
    return {'episode_id': eid, 'kb': kb, 'messages': messages}


class FakeTeacher:
    """``TeacherKeys.mined`` over a fixed ranking per site."""

    def __init__(self, ranking, kb='ds'):
        self.ranking, self.kb = ranking, kb

    def mined(self, episode_id, call, k, kb=None, skip=()):
        if (episode_id, call) not in self.ranking:
            raise KeyError((episode_id, call))
        if kb is not None and kb != self.kb:
            raise PermissionError(kb)
        return [r for r in self.ranking[episode_id, call] if r not in set(skip)][:k]


def test_mine_interleaves_hits_and_teacher_and_never_names_positives_or_neutral():
    rows = [row('e', [{'record_ids': ['r1'], 'alternatives': ['r1', 'r2'], 'neutral': ['r3']},
                      {'record_ids': ['r4']}])]
    teacher = FakeTeacher({('e', 0): ['r1', 'r3', 'r5', 'r6', 'r7'], ('e', 1): ['r4', 'r1']})
    hits = {('e', 0): ['r2', 'r6', 'r4'], ('e', 1): ['r5']}
    sites = hn.mine(rows, teacher=teacher, hits=hits, k=3)
    assert sites[0] == {'episode_id': 'e', 'call': 0, 'kb': 'ds', 'records': ['r6', 'r5', 'r4'],
                        'sources': {'hits': 2, 'teacher': 1}}
    # site 1's own positive (r4) is excluded; site 0's positive r1 is a negative there
    assert sites[1]['records'] == ['r5', 'r1']
    assert hn.mine(rows, k=3) == []                      # no source, no entry
    only_teacher = hn.mine(rows, teacher=teacher, k=8)
    assert only_teacher[0]['records'] == ['r5', 'r6', 'r7']


def test_file_round_trip_and_authorization(tmp_path):
    sites = [{'episode_id': 'e', 'call': 0, 'kb': 'ds', 'records': ['r5']}]
    hn.save(tmp_path / 'hn.json', sites, {'k': 1})
    got = hn.HardNegatives.load(tmp_path / 'hn.json')
    assert len(got) == 1 and got.records('e', 0, 'ds') == ['r5']
    assert got.records('e', 1, 'ds') == [] and got.records('f', 0, 'ds') == []
    assert got.records('e', 0, 'ds#t00003') == ['r5']      # a sub-KB of the site's KB
    with pytest.raises(PermissionError):
        got.records('e', 0, 'secret')
    hn.dump_hits(tmp_path / 'hits.json', {}, 3)
    with pytest.raises(ValueError):
        hn.HardNegatives.load(tmp_path / 'hits.json')
    assert hn.load_hits(tmp_path / 'hits.json') == {}


def test_items_map_records_of_the_own_kb_without_the_slots_positives(tmp_path):
    ctx = big_context(tmp_path)
    ep = episode(IDS, [3, 10], [6, 13], records=(['r1'], ['r2']))
    ep.slots[0]['neutral'] = ['r3']
    hard = hn.HardNegatives([{'episode_id': 'e', 'call': 0, 'kb': 'ds',
                              'records': ['r1', 'r3', 'r5', 'missing']},
                             {'episode_id': 'e', 'call': 1, 'kb': 'ds', 'records': ['r6']}])
    kb = ctx.kbs['ds']
    assert hard.items(ctx, ep, 0) == targets_of(kb, ['r5'])
    merged = hard.extend(ctx, ep, None)
    assert merged == targets_of(kb, ['r5', 'r6'])
    base = targets_of(kb, ['r2', 'r5'])
    got = hard.extend(ctx, ep, base)
    assert all(got[s] == list(dict.fromkeys(base[s] + targets_of(kb, ['r5', 'r6'])[s]))
               for s in SPACES)
    assert hn.HardNegatives([]).extend(ctx, ep, base) is base
    other = episode(IDS, [3, 10], [6, 13], kb='secret')
    with pytest.raises(PermissionError):
        hard.items(ctx, other, 0)


def test_collect_hits_ranks_wrong_scored_records(tmp_path):
    ctx = big_context(tmp_path)
    ep = episode(IDS, [3, 10], [6, 13], records=(['r1'], ['r2']))
    ep.slots[0]['alternatives'] = ['r1', 'r4']
    ep.slots[0]['neutral'] = ['r3']
    with torch.no_grad():
        _, _, reads, _ = l1.run_episode(ctx, ep, ItemCache(train=False))
    sink = {}
    hn.collect_hits(sink, ctx, ep, reads, top=4, step=7)
    assert set(sink) == {('e', 0), ('e', 1)}
    first = sink['e', 0]
    assert first['step'] == 7 and first['kb'] == 'ds' and len(first['records']) <= 4
    assert not {'r1', 'r4', 'r3'} & set(first['records'])
    assert 'r2' not in sink['e', 1]['records']
    assert 'r7' not in first['records']                   # later than the query: never scored
    scored = {s: [(ctx.origin['ds'][s][i][1][0], float(g))
                  for (_, i), g in zip(info.scored, info.scored_gates)]
              for s, info in reads[0].spaces.items()}
    best = max(scored['D'], key=lambda x: x[1])[0] if 'D' in scored else None
    if best is not None and best not in {'r1', 'r4', 'r3'}:
        assert best in first['records'][:len(SPACES)]
    hn.dump_hits(tmp_path / 'hits.json', sink, 7)
    assert hn.load_hits(tmp_path / 'hits.json')[('e', 0)] == first['records']


def _step(tmp_path, hard=None, **kw):
    ctx = big_context(tmp_path)
    if hard is not None:
        ctx.hard_negatives = hard
    a = episode(IDS, [3, 10], [6, 13], records=(['r1'], ['r2']))
    b = episode(IDS, [3, 10], [6, 13], records=(['r3'], ['r1']))
    b.episode_id = 'f'
    trainable = l1.set_phase(l1.parameter_sets(ctx.reader), l1.L1A_SET)
    opt = torch.optim.AdamW(trainable, lr=1e-3)
    out = l1.train_step(ctx, [a, b], opt, train_args(**kw), 0, trainable=trainable)
    return out, [p.detach().clone() for p in trainable], ctx, (a, b)


@pytest.mark.parametrize('retrieval_only', [True, False])
def test_without_hard_negatives_the_step_is_bit_identical(tmp_path, retrieval_only):
    base, params, _, _ = _step(tmp_path / 'a', retrieval_only=retrieval_only)
    # an empty file changes nothing, bit for bit
    empty, params_e, _, _ = _step(tmp_path / 'b', hn.HardNegatives([]),
                                  retrieval_only=retrieval_only)
    assert empty['aux'] == base['aux'] and empty.get('nll') == base.get('nll')
    assert all(torch.equal(x, y) for x, y in zip(params, params_e))
    # and the loss is the pre-flag formula: in-batch negatives per episode, nothing else
    ctx = big_context(tmp_path / 'c')
    a = episode(IDS, [3, 10], [6, 13], records=(['r1'], ['r2']))
    b = episode(IDS, [3, 10], [6, 13], records=(['r3'], ['r1']))
    b.episode_id = 'f'
    pool = l1.batch_negatives(ctx, [a, b], 16, random.Random(0))
    cache = ItemCache(train=True)
    want = 0.0
    for ep in (a, b):
        _, _, reads, _ = l1.run_episode(ctx, ep, cache, retrieval_only=retrieval_only,
                                        negatives=pool.get(ep.kb))
        want += torch.stack([r.aux for r in reads]).mean().item() / 2
    assert base['aux'] == want
    # with hard negatives the loss changes
    hard = hn.HardNegatives([{'episode_id': 'e', 'call': 0, 'kb': 'ds',
                              'records': ['r4', 'r5', 'r6']}])
    mined, _, _, _ = _step(tmp_path / 'd', hard, retrieval_only=retrieval_only)
    assert mined['aux'] != base['aux']


def test_dump_hits_collects_during_training(tmp_path):
    ctx = big_context(tmp_path)
    ctx.hit_sink = {}
    a = episode(IDS, [3, 10], [6, 13], records=(['r1'], ['r2']))
    trainable = l1.set_phase(l1.parameter_sets(ctx.reader), l1.L1A_SET)
    opt = torch.optim.AdamW(trainable, lr=1e-3)
    l1.train_step(ctx, [a], opt, train_args(retrieval_only=True, dump_hits_top=3), 5,
                  trainable=trainable)
    assert set(ctx.hit_sink) == {('e', 0), ('e', 1)}
    assert all(s['step'] == 5 and len(s['records']) <= 3 for s in ctx.hit_sink.values())
