"""Key-teacher distillation of the retrieval heads (``schnitz.kb.teacher_keys``,
``L1Reader.read(teacher=...)``, ``l1 --key-teacher``); CPU, tiny fixtures, no downloads."""
from __future__ import annotations

import hashlib
import json
import random

import pytest
import torch

from schnitz.kb import teacher_keys as tk
from schnitz.kb.losses import retrieval_loss, teacher_kl
from schnitz.kb.read import MINED_WEIGHT, ItemCache, TeacherSite
from schnitz.kb.stack import KEY_WIDTH
from schnitz.kb.stages import l1
from test_kb_read import (HIDDEN, IDS, SPACES, _torch_conv, context, episode, make_kb,  # noqa: F401
                          reader, targets_of, train_args)

OFFSET = {}
_at = 0
for _s in SPACES:
    OFFSET[_s] = _at
    _at += KEY_WIDTH[_s]
TOTAL = _at


def block(space, vector):
    """A per-space key placed in its own block of one teacher vector, so one teacher
    query serves every space and the dot product is the space's cosine."""
    out = torch.zeros(TOTAL)
    out[OFFSET[space]:OFFSET[space] + vector.shape[-1]] = vector.detach().float()
    return out


def mirror(r, kb, cache, state, tau=0.1, seen=None, positives=None):
    """A teacher whose logits are exactly the reader's scores at ``tau`` = 1/scale."""
    query = torch.cat([r.keys.query_key(s, state).detach() for s in SPACES])

    def keys(space, refs):
        if seen is not None:
            seen.setdefault(space, []).extend(refs)
        return [block(space, r.keys.item_key(space, cache.get(kb, space, [i])[0][0]))
                for _, i in refs]
    return TeacherSite(query, keys, tau, positives)


def random_teacher(seen=None, positives=None, tau=0.05, missing=()):
    gen = torch.Generator().manual_seed(7)
    query = torch.nn.functional.normalize(torch.randn(24, generator=gen), dim=0)
    table = {}

    def keys(space, refs):
        if seen is not None:
            seen.setdefault(space, []).extend(refs)
        out = []
        for ref in refs:
            if ref[1] in missing:
                out.append(None)
                continue
            if ref not in table:
                table[ref] = torch.nn.functional.normalize(torch.randn(24, generator=gen), dim=0)
            out.append(table[ref])
        return out
    return TeacherSite(query, keys, tau, positives)


def test_teacher_kl_is_zero_for_equal_distributions():
    scores = torch.randn(9)
    assert float(teacher_kl(scores, scores.clone())) == 0.0
    assert float(teacher_kl(scores, scores + 3.0)) == pytest.approx(0.0, abs=1e-6)
    assert float(teacher_kl(scores, torch.randn(9))) > 0


def test_kl_is_zero_when_our_scores_equal_the_teachers(tmp_path):
    kb = make_kb(tmp_path, records=tuple((f'r{i}', 1) for i in range(8)))
    r = reader(candidates={'A': 3, 'B': 3, 'C': 3, 'D': 3})
    state, cache = torch.randn(HIDDEN), ItemCache(train=True)
    targets, negs = targets_of(kb, ['r0']), targets_of(kb, ['r4', 'r6'])
    scale = float(r.keys.log_scale['A'].detach().exp())
    read = r.read(state, [kb], ['ds'], 3, cache, targets=targets, negatives=negs,
                  teacher=mirror(r, kb, cache, state, tau=1 / scale))
    assert float(read.teacher_kl) == pytest.approx(0.0, abs=1e-5)
    other = r.read(state, [kb], ['ds'], 3, cache, targets=targets, negatives=negs,
                   teacher=mirror(r, kb, cache, state, tau=0.05))
    assert float(other.teacher_kl) > 1e-3
    # the teacher changes neither the retrieval loss nor what is read
    plain = r.read(state, [kb], ['ds'], 3, ItemCache(train=True), targets=targets, negatives=negs)
    assert torch.equal(plain.aux, other.aux) and plain.teacher_kl is None
    assert all(plain.spaces[s].refs == other.spaces[s].refs for s in SPACES)


def test_teacher_kl_reaches_query_and_item_key_heads(tmp_path):
    kb = make_kb(tmp_path, records=tuple((f'r{i}', 1) for i in range(8)))
    r = reader(candidates={'A': 3, 'B': 3, 'C': 3, 'D': 3})
    cache = ItemCache(train=True)
    read = r.read(torch.randn(HIDDEN), [kb], ['ds'], 3, cache, targets=targets_of(kb, ['r0']),
                  negatives=targets_of(kb, ['r4', 'r6']), teacher=random_teacher())
    read.teacher_kl.backward()
    for s in SPACES:
        assert r.keys.query[s][1].weight.grad.abs().sum() > 0
        assert r.keys.item[s][1].weight.grad.abs().sum() > 0
        assert r.keys.log_scale[s].grad is not None
    assert all(p.grad is None for p in r.operators.parameters())


def test_teacher_list_is_the_retrieval_list_without_neutral_items(tmp_path):
    kb = make_kb(tmp_path, records=tuple((f'r{i}', 1) for i in range(8)))
    every = {'A': 8, 'B': 8, 'C': 8, 'D': 8}
    r = reader(candidates=every, keep=every)
    state, cache, seen = torch.randn(HIDDEN), ItemCache(train=True), {}
    targets, neutral = targets_of(kb, ['r0']), targets_of(kb, ['r1', 'r2', 'r3'])
    read = r.read(state, [kb], ['ds'], 3, cache, targets=targets, neutral=neutral,
                  teacher=random_teacher(seen))
    for s in SPACES:
        assert set(neutral[s]) <= set(read.spaces[s].scored)      # scored and read ...
        assert not set(seen[s]) & set(neutral[s])                 # ... but not distilled
        assert sorted(seen[s]) == sorted(p for p in read.spaces[s].scored
                                         if p not in set(neutral[s]))
    # items without a teacher embedding (written items) leave the KL, the rest stay
    r2 = r.read(state, [kb], ['ds'], 3, ItemCache(train=True), targets=targets,
                neutral=neutral, teacher=random_teacher(missing={i for _, i in targets['A']}))
    assert r2.teacher_kl is not None and not torch.equal(r2.teacher_kl, read.teacher_kl)


def test_mined_positives_join_the_loss_at_half_weight(tmp_path):
    kb = make_kb(tmp_path, records=tuple((f'r{i}', 1) for i in range(8)))
    r = reader(candidates={'A': 3, 'B': 3, 'C': 3, 'D': 3})
    state = torch.randn(HIDDEN)
    targets, neutral = targets_of(kb, ['r0']), targets_of(kb, ['r6'])
    mined = targets_of(kb, ['r5', 'r6', 'r0'])     # r6 neutral: dropped; r0 stays weight 1
    base = r.read(state, [kb], ['ds'], 3, ItemCache(train=True), targets=targets,
                  neutral=neutral)
    cache = ItemCache(train=True)
    read = r.read(state, [kb], ['ds'], 3, cache, targets=targets, neutral=neutral,
                  teacher=random_teacher(positives=mined))
    assert read.recall_at == base.recall_at            # recall over the slot's own positives
    want = []
    for s in SPACES:
        info = read.spaces[s]
        scored = [p for p in info.scored if p not in set(neutral[s])]
        extra = [p for p in targets[s] + targets_of(kb, ['r5'])[s] if p not in set(scored)]
        pool = scored + extra
        weight = torch.tensor([1.0 if p in set(targets[s]) else
                               MINED_WEIGHT if p in set(targets_of(kb, ['r5'])[s]) else 0.0
                               for p in pool])
        q = r.keys.query_key(s, state)
        keys = torch.stack([r.keys.item_key(s, cache.get(kb, s, [i])[0][0]) for _, i in pool])
        want.append(retrieval_loss(r.keys.scores(s, q[None], keys), weight[None])[0])
    torch.testing.assert_close(read.aux, torch.stack(want).mean())
    with pytest.raises(PermissionError):          # mined positives must be readable here
        other = make_kb(tmp_path, 'other', 'secret')
        r.read(state, [kb], ['ds'], 3, ItemCache(train=False), targets=targets,
               teacher=random_teacher(positives=targets_of(other, ['r1'])))


# -- the cache -------------------------------------------------------------------------
class HashEmbedder:
    """Deterministic unit vectors from the text (no model); records what it embedded."""
    name, instruction, query_tokens, record_tokens = 'hash', 'find', 0, 0

    def __init__(self):
        self.queries: list[str] = []

    def encode(self, texts, query=False):
        if query:
            self.queries += list(texts)
        out = []
        for t in texts:
            seed = int(hashlib.sha256(t.encode()).hexdigest()[:8], 16)
            out.append(torch.randn(16, generator=torch.Generator().manual_seed(seed)))
        return torch.nn.functional.normalize(torch.stack(out), dim=-1).half()


def transcript(eid, records, answer='SECRET ANSWER', kb='ds', qt=3, neutral=()):
    messages = [{'role': 'system', 'content': 'Use memory.'},
                {'role': 'user', 'content': f'Question {eid}?'}]
    for n, r in enumerate(records):
        slot = {'kb': kb, 'record_ids': list(r)}
        if neutral and n == 0:
            slot['neutral'] = list(neutral)
        messages += [{'role': 'assistant', 'content': '', 'tool_calls': [
            {'type': 'function', 'function': {'name': 'memory_search', 'arguments': {}}}]},
            {'role': 'tool', 'name': 'memory_search', 'content': {'slot': slot}}]
    messages.append({'role': 'assistant', 'content': answer})
    return {'episode_id': eid, 'kb': kb, 'messages': messages,
            'provenance': {'source_query_time': qt}}


def corpus(tmp_path, rows, records=(('r1', 1), ('r2', 1), ('r3', 1), ('r4', 5))):
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
    return d


def test_site_prefixes_are_causal():
    row = transcript('e', (['r1'], ['r2']))
    first, second = tk.site_prefixes(row)
    assert first.endswith('memory_search()') and 'Question e?' in first
    assert second.startswith(first) and second.count(tk.MEMORY_RESULT) == 1
    for text in (first, second):       # no answer, no slot record ids or contents
        assert 'SECRET' not in text and 'r1' not in text and 'r2' not in text


def test_cache_keys_by_episode_and_call_and_refuses_unknown_sites(tmp_path):
    rows = [transcript('e', (['r1'], ['r2'])), transcript('f', (['r3'],), qt=3)]
    d = corpus(tmp_path, rows)
    embedder = HashEmbedder()
    # the slots' records by default; a bank's records (here one later distractor) given
    assert tk.build(tmp_path / 'default', HashEmbedder(), [d], {'train': None})['records'] == 3
    records = {r: {'text': f'record {r} text', 'kb': 'ds', 'created_at': t}
               for r, t in (('r1', 1), ('r2', 1), ('r3', 1), ('r4', 5))}
    manifest = tk.build(tmp_path / 'teacher', embedder, [d], {'train': None}, records, top=8)
    assert manifest['sites'] == 3 and manifest['records'] == 4
    assert not any('SECRET' in q for q in embedder.queries)
    cache = tk.TeacherKeys(tmp_path / 'teacher')
    prefixes = tk.site_prefixes(rows[0])
    for j in range(2):
        torch.testing.assert_close(cache.query('e', j),
                                   embedder.encode([prefixes[j]], query=True)[0])
    assert not torch.equal(cache.query('e', 0), cache.query('e', 1))
    for site in (('e', 2), ('g', 0)):
        with pytest.raises(KeyError):
            cache.query(*site)
    assert cache.record('r9') is None
    # mined: the site's top records of its KB, never later than its query time
    top = cache.mined('e', 0, 8, kb='ds')
    assert sorted(top) == ['r1', 'r2', 'r3'] and 'r4' not in top
    assert cache.mined('e', 0, 2, kb='ds', skip={top[0]}) == top[1:3]
    with pytest.raises(PermissionError):
        cache.mined('e', 0, 2, kb='other')


def test_key_teacher_in_the_trainer(tmp_path):
    ctx, _ = context(tmp_path)
    d = corpus(tmp_path, [transcript('e', (['r1'], ['r2']), neutral=['r3'])])
    tk.build(tmp_path / 'teacher', HashEmbedder(), [d], {'train': None}, top=8)
    ctx.key_teacher = l1.KeyTeacher(tk.TeacherKeys(tmp_path / 'teacher'), 0.05, positives=2)
    ep = episode(IDS, [3, 10], [6, 13])
    ep.slots[0]['neutral'] = ['r3']
    site = ctx.key_teacher.site(ctx, ep, 0)
    kb = ctx.kbs['ds']
    for s in SPACES:     # bank items get their record's embedding, others none
        got = site.keys(s, targets_of(kb, ['r1', 'r2'])[s] + [('ds', 'missing')])
        torch.testing.assert_close(got[0], ctx.key_teacher.cache.record('r1'))
        assert got[2] is None
        assert not set(site.positives[s]) & set(targets_of(kb, ['r3'])[s])   # neutral
        assert len(site.positives[s]) == 2
    # K2: the KL is logged and trains the heads alongside the retrieval loss
    args = train_args(retrieval_only=True, retrieval_weight=1.0, key_teacher_weight=1.0)
    params = ctx.reader.trainable()
    opt = torch.optim.AdamW(params, lr=1e-3)
    before = ctx.reader.keys.query['A'][1].weight.detach().clone()
    out = l1.train_step(ctx, [ep], opt, args, 0, rng=random.Random(0))
    assert out['teacher_kl'] > 0
    assert not torch.equal(before, ctx.reader.keys.query['A'][1].weight)
    # the KL only: with retrieval weight 0 the heads still move
    ctx.reader.zero_grad(set_to_none=True)
    nll, _, reads, _ = l1.run_episode(ctx, ep, ItemCache(train=True), retrieval_only=True,
                                      teacher=True)
    kl = torch.stack([rd.teacher_kl for rd in reads]).mean()
    kl.backward()
    assert ctx.reader.keys.item['B'][1].weight.grad.abs().sum() > 0
    # evaluation reads never use the teacher
    _, _, plain, _ = l1.run_episode(ctx, ep, ItemCache(train=False), retrieval_only=True)
    assert all(rd.teacher_kl is None for rd in plain)
    # a training site missing from the cache is refused
    stranger = episode(IDS, [3, 10], [6, 13])
    stranger.episode_id = 'zzz'
    with pytest.raises(KeyError):
        l1.run_episode(ctx, stranger, ItemCache(train=True), retrieval_only=True, teacher=True)


# -- alignment to the teacher's principal subspace (--key-align-weight) ---------------------
def test_teacher_alignment_value():
    from schnitz.kb.read import teacher_alignment
    basis = torch.eye(24)[:, :4]                     # the first four teacher directions
    teacher = torch.zeros(2, 24)
    teacher[0, 0], teacher[1, 1], teacher[1, 9] = 1.0, 1.0, 5.0   # dim 9 is projected away
    keys = torch.zeros(2, 4)
    keys[0, 0] = 1.0                                 # aligned: 0
    keys[1, 2] = 1.0                                 # orthogonal: 1
    assert float(teacher_alignment(keys, teacher, basis)) == pytest.approx(0.5)
    assert float(teacher_alignment(keys[:1], teacher[:1], basis)) == 0.0


def test_basis_is_the_records_principal_directions(tmp_path):
    rows = [transcript('e', (['r1'], ['r2'])), transcript('f', (['r3'],), qt=3)]
    d = corpus(tmp_path, rows)
    tk.build(tmp_path / 'teacher', HashEmbedder(), [d], {'train': None}, top=8)
    cache = tk.TeacherKeys(tmp_path / 'teacher')
    basis = cache.basis(8)
    assert basis.shape == (16, 8)
    n = cache.records.shape[0]                       # n directions, the rest zero columns
    torch.testing.assert_close(basis[:, :n].t() @ basis[:, :n], torch.eye(n), atol=1e-5, rtol=0)
    assert torch.equal(basis[:, n:], torch.zeros(16, 8 - n))
    # the records lie in the span: projecting keeps their norm
    torch.testing.assert_close((cache.records.float() @ basis).norm(dim=-1),
                               torch.ones(n), atol=1e-3, rtol=0)


def test_alignment_trains_the_heads_and_is_logged(tmp_path):
    ctx, _ = context(tmp_path, query_pool=True)
    d = corpus(tmp_path, [transcript('e', (['r1'], ['r2']))])
    tk.build(tmp_path / 'teacher', HashEmbedder(), [d], {'train': None}, top=8)
    cache = tk.TeacherKeys(tmp_path / 'teacher')
    ep = episode(IDS, [3, 10], [6, 13])
    ctx.key_teacher = l1.KeyTeacher(cache, 0.05)     # align off: no term
    _, _, reads, _ = l1.run_episode(ctx, ep, ItemCache(train=True), retrieval_only=True,
                                    teacher=True)
    assert all(r.teacher_align is None for r in reads)
    ctx.key_teacher = l1.KeyTeacher(cache, 0.05, align=True)
    site = ctx.key_teacher.site(ctx, ep, 0)
    assert site.basis.shape == (16, KEY_WIDTH['A'])
    _, _, reads, _ = l1.run_episode(ctx, ep, ItemCache(train=True), retrieval_only=True,
                                    teacher=True)
    align = torch.stack([r.teacher_align for r in reads]).mean()
    assert 0 < float(align) < 4
    align.backward()
    for s in SPACES:
        assert ctx.reader.keys.query[s][1].weight.grad.abs().sum() > 0
        assert ctx.reader.keys.item[s][1].weight.grad.abs().sum() > 0
    assert ctx.reader.keys.pool.out.weight.grad.abs().sum() > 0
    # evaluation reads have no teacher, so no alignment
    _, _, plain, _ = l1.run_episode(ctx, ep, ItemCache(train=False), retrieval_only=True)
    assert all(r.teacher_align is None for r in plain)
    ctx.reader.zero_grad(set_to_none=True)
    args = train_args(retrieval_only=True, retrieval_weight=1.0, key_teacher_weight=1.0,
                      key_align_weight=10.0)
    opt = torch.optim.AdamW(ctx.reader.trainable(), lr=1e-3)
    out = l1.train_step(ctx, [ep], opt, args, 0, rng=random.Random(0))
    assert out['teacher_align'] > 0
    # the query module's gradient and output-projection norm are logged
    assert out['grad_norm']['query_module'] > 0 and out['query_module_out_norm'] == 0.0


def test_centered_alignment_removes_the_common_direction_attractor():
    """Teacher vectors sharing a large common direction: one constant key at that
    direction scores well against all of them uncentered, and not at all centered."""
    from schnitz.kb.read import teacher_alignment
    gen = torch.Generator().manual_seed(0)
    common = torch.nn.functional.normalize(torch.randn(24, generator=gen), dim=0)
    teacher = torch.nn.functional.normalize(
        10.0 * common + torch.randn(200, 24, generator=gen), dim=-1)
    basis = torch.eye(24)
    constant = common[None].expand(200, -1)
    plain = float(teacher_alignment(constant, teacher, basis))
    centered = float(teacher_alignment(constant, teacher, basis, teacher.mean(0)))
    assert plain < 0.2 and centered > 0.9


def test_key_teacher_center_passes_centers_and_a_centered_basis(tmp_path):
    ctx, _ = context(tmp_path)
    d = corpus(tmp_path, [transcript('e', (['r1'], ['r2']))])
    tk.build(tmp_path / 'teacher', HashEmbedder(), [d], {'train': None}, top=8)
    cache = tk.TeacherKeys(tmp_path / 'teacher')
    ep = episode(IDS, [3, 10], [6, 13])
    plain = l1.KeyTeacher(cache, 0.05, align=True).site(ctx, ep, 0)
    site = l1.KeyTeacher(cache, 0.05, align=True, center=True).site(ctx, ep, 0)
    assert plain.centers is None and site.centers is not None
    mq, mr = site.centers
    torch.testing.assert_close(mq, cache.queries.float().mean(0))
    torch.testing.assert_close(mr, cache.records.float().mean(0))
    assert not torch.equal(site.basis, plain.basis)
    ctx.key_teacher = l1.KeyTeacher(cache, 0.05, align=True, center=True)
    _, _, reads, _ = l1.run_episode(ctx, ep, ItemCache(train=True), retrieval_only=True,
                                    teacher=True)
    assert all(r.teacher_align is not None for r in reads)
