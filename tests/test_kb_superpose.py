"""The superposed KB (``schnitz.kb.superpose``): rows as combiner outputs over fields, the
read path over them, the write fit, and the query-conditioned R; CPU, tiny, no downloads."""
from __future__ import annotations

import math
import random

import numpy as np
import pytest
import torch
from torch import nn

from schnitz.kb import superpose as sp
from schnitz.kb.read import ItemCache, KeyOptimizer, L1Reader, ReadConfig, conditioned, current_ids
from schnitz.kb.stages import l1
from schnitz.kb_store import DEFAULT_SPACES, KnowledgeBase, NewItem, Provenance

SPACES = DEFAULT_SPACES
HIDDEN = 32
DIMS = {'state': 16, 'hidden': 12, 'layers': 2}


@pytest.fixture(autouse=True)
def _torch_conv(monkeypatch):
    """The pure-torch convolution path (the CUDA kernel refuses CPU tensors)."""
    import inspect
    from transformers.models.lfm2 import modeling_lfm2
    monkeypatch.setattr(modeling_lfm2, 'causal_conv1d_fn',
                        inspect.unwrap(modeling_lfm2.causal_conv1d_fn))


def leaf_kb(tmp_path, name='leaves', dataset='ds', n=24, times=None, live=True, seed=0):
    """A KB of ``n`` leaves per space (records r0..), clustered keys, lengths 1-4."""
    gen = torch.Generator().manual_seed(seed)
    kb = KnowledgeBase.create(tmp_path / name, name=name, dataset=dataset, spaces=SPACES)
    centres = {s: torch.randn(4, spec.key_width, generator=gen) for s, spec in SPACES.items()}
    for s, spec in SPACES.items():
        items = []
        for i in range(n):
            key = centres[s][i % 4] + 0.3 * torch.randn(spec.key_width, generator=gen)
            items.append(NewItem(torch.randn(1 + i % 4, spec.width, generator=gen),
                                 nn.functional.normalize(key, dim=-1),
                                 Provenance((f'r{i}',), 'codec'), 1.0,
                                 1 if times is None else times[i]))
        kb.append(s, items)
    if live:
        for s in SPACES:
            kb.enable_live(s)
    return kb


def config(**kw) -> sp.SuperposeConfig:
    base = dict(depth=2, field={'A': 4, 'B': 4, 'C': 6, 'D': 6}, overlap=2, cache_every=1,
                deep_grad=1.0)
    return sp.SuperposeConfig(**{**base, **kw})


def rows_for(kb, cfg, tmp_path, name='rows'):
    rows_kb, _ = sp.build_rows(kb, tmp_path / name, cfg)
    anchors = sp.rows_of(rows_kb)
    return rows_kb, anchors


def view_of(kb, cfg, tmp_path, name='rows', seed=0):
    rows_kb, anchors = rows_for(kb, cfg, tmp_path, name)
    rows_kb.close()
    torch.manual_seed(seed)
    ops = sp.WriteOps(cfg.depth, DIMS, cfg.per_level)
    view = sp.SuperposedKB(kb, ops, cfg, anchors)
    view.rebuild(0)
    return view, ops


def reader(**kw) -> L1Reader:
    kw = {'candidates': {'A': 3, 'B': 4, 'C': 6, 'D': 6}, **kw}
    torch.manual_seed(0)
    return L1Reader(ReadConfig(hidden=HIDDEN, span_width=HIDDEN, state=16, op_hidden=12,
                               layers=2, key_hidden=16, checkpointing=False, **kw))


# -- fields, rows, time, authorization -------------------------------------------------------
def test_rows_are_fewer_than_leaves_and_every_input_splits_its_mass(tmp_path):
    kb = leaf_kb(tmp_path, n=40)
    cfg = config()
    assert cfg.row_count('D', 40) == math.ceil(2 / 6 * 40) < 40
    assert sp.SuperposeConfig().row_count('D', 100) == 19      # c/f = 3/16 by default
    view, _ = view_of(kb, cfg, tmp_path)
    for s, g in view.graphs.items():
        assert len(g.row_ids) == cfg.row_count(s, 40) < len(g.ids)
        below = g.mass
        for level in g.levels:
            matrix = level.matrix(len(below)).toarray()
            np.testing.assert_allclose(matrix.sum(0), 1.0, rtol=1e-9)     # invariant 7
            np.testing.assert_allclose(level.mass, matrix @ below, rtol=1e-9)
            assert level.mass.sum() == pytest.approx(below.sum())          # mass conserved
            assert all(level.inputs)                                       # no empty field
            below = level.mass
        np.testing.assert_allclose(np.asarray(g.comp.sum(1)).ravel(), 1.0, rtol=1e-9)
        stats = g.stats()
        assert stats['levels'][0]['fields_per_input'] >= cfg.overlap


def test_time_is_the_latest_leaf_and_later_rows_are_never_read(tmp_path):
    times = [9 if i in (16, 20) else 1 for i in range(24)]
    kb = leaf_kb(tmp_path, n=24, times=times)
    view, _ = view_of(kb, config(), tmp_path)
    for s, g in view.graphs.items():
        level = g.levels[0]
        for j, ins in enumerate(level.inputs):
            assert level.time[j] == max(times[i] for i in ins)
        top = g.levels[-1]
        for j, ins in enumerate(top.inputs):
            assert top.time[j] == max(g.levels[0].time[i] for i in ins)
        late = {g.row_ids[j] for j in range(len(g.row_ids)) if g.top_time[j] > 3}
        assert late, 'the late leaves reach some rows'
        hits = view.search(s, torch.randn(SPACES[s].key_width), len(g.row_ids), 3)
        assert hits and not {i for _, i in hits} & late
    cache = sp.SuperposedCache({'ds': view}, train=False)
    r = reader(read_combine='r', learned_keys=True)
    read = r.read(torch.randn(HIDDEN), [kb], ['ds'], 3, cache)
    for s, info in read.spaces.items():
        assert all(view.top_time(s, i) <= 3 for _, i in info.scored)


def test_no_mixing_across_kbs_and_authorization(tmp_path):
    a = leaf_kb(tmp_path, 'a', 'ds', n=16, seed=1)
    b = leaf_kb(tmp_path, 'b', 'secret', n=16, seed=2)
    va, _ = view_of(a, config(), tmp_path, 'rows_a')
    vb, _ = view_of(b, config(), tmp_path, 'rows_b')
    for view, kb in ((va, a), (vb, b)):
        for s, g in view.graphs.items():
            leaves = set(current_ids(kb, s))
            assert all(set(shares) <= leaves for shares in g.composition().values())
    cache = sp.SuperposedCache({'ds': va, 'secret': vb}, train=False)
    r = reader(read_combine='r', learned_keys=True)
    read = r.read(torch.randn(HIDDEN), [a], ['ds'], 5, cache)
    rows_a = {i for g in va.graphs.values() for i in g.row_ids}
    assert all(d == 'ds' and i in rows_a for info in read.spaces.values() for d, i in info.scored)
    with pytest.raises(PermissionError):
        r.read(torch.randn(HIDDEN), [a, b], ['ds'], 5, cache)


# -- lazy evaluation, gradients, keys ------------------------------------------------------------
def test_lazy_rows_equal_the_full_stack_when_caches_are_fresh(tmp_path):
    kb = leaf_kb(tmp_path)
    view, ops = view_of(kb, config(deep_grad=0.3), tmp_path)
    for s, g in view.graphs.items():
        leaves = [(it.values.float(), nn.functional.normalize(it.key.float(), dim=-1),
                   torch.tensor(it.mass)) for it in kb.read(s, g.ids, live=True)]
        levels = sp.aggregate(g, ops, leaves)
        for j in range(len(g.row_ids)):
            value, key, mass = view.item(s, g.depth, j)
            torch.testing.assert_close(value, levels[-1][j][0], rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(key, levels[-1][j][1], rtol=1e-4, atol=1e-5)
            assert float(mass) == pytest.approx(float(levels[-1][j][2]), rel=1e-5)
            cache = ItemCache(train=True)

            def leaf(space, ids):
                return [(v, nn.functional.normalize(k, dim=-1), torch.tensor(1.0)) for (v, _), k
                        in zip(cache.get(kb, space, ids), cache.keys(kb, space, ids))]
            grad_v, grad_k, _ = view.grad_value(s, g.depth, j, leaf)   # sampled deep path
            # the recomputation is batched differently from the cached pass: equal up to
            # summation order
            torch.testing.assert_close(grad_v.detach(), value, rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(grad_k.detach(), key, rtol=1e-5, atol=1e-5)


def test_field_key_is_the_share_weighted_mean_without_correction():
    keys = [nn.functional.normalize(torch.randn(8), dim=-1) for _ in range(3)]
    gates = [0.5, 0.25, 1.0]
    head = nn.Linear(4, 8)
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    got = sp.field_key(head, keys, gates, torch.randn(3, 4))
    want = nn.functional.normalize(sum(g * k for g, k in zip(gates, keys)) / sum(gates), dim=-1)
    torch.testing.assert_close(got, want)
    torch.testing.assert_close(sp.mean_key(keys, gates), want)
    ops = sp.WriteOps(2, DIMS)
    assert all(float(p.abs().sum()) == 0 for p in ops.key_heads.parameters())


def test_gradients_reach_leaves_aggregators_and_key_heads(tmp_path):
    kb = leaf_kb(tmp_path)
    view, ops = view_of(kb, config(), tmp_path)
    cache = sp.SuperposedCache({'ds': view}, train=True)
    r = reader(read_combine='r', learned_keys=True)
    read = r.read(torch.randn(HIDDEN), [kb], ['ds'], 5, cache)
    (read.span.square().sum()).backward()
    stats = cache.backward()
    assert stats['rows'] > 0 and stats['drift'] < 1e-5
    assert any(v.grad is not None and v.grad.abs().sum() > 0 for v in cache.values.values())
    assert any(k.grad is not None and k.grad.abs().sum() > 0 for k in cache.key_leaves.values())
    for s in SPACES:
        for level in ('1', '2'):
            assert ops.levels[level][s].op.layers[0].out.weight.grad.abs().sum() > 0
        assert ops.key_heads['2'][s].weight.grad.abs().sum() > 0
    assert r.read_r.head[1].weight.grad.abs().sum() > 0
    assert r.keys.query['A'][1].weight.grad.abs().sum() > 0
    assert cache.apply(item_lr=0.01)['items'] > 0


def test_learned_keys_move_and_change_the_retrieval_order(tmp_path):
    kb = leaf_kb(tmp_path, n=12)
    kb.load_live()
    r = reader(read_combine='r', learned_keys=True, candidates={'A': 12, 'B': 12, 'C': 12,
                                                                  'D': 12})
    state = torch.randn(HIDDEN)
    index = {s: l1.source_index(kb, s) for s in SPACES}
    target = {s: [('ds', index[s]['r5'][0])] for s in SPACES}

    def rank(s):
        q = r.keys.query_key(s, state).detach()
        items = kb.read(s, current_ids(kb, s), live=True)
        scores = torch.stack([nn.functional.normalize(it.key, dim=-1) for it in items]) @ q
        order = [items[i].id for i in scores.argsort(descending=True).tolist()]
        return order.index(target[s][0][1])
    before = {s: rank(s) for s in SPACES}
    keys_before = {s: kb.read(s, [target[s][0][1]], live=True)[0].key.clone() for s in SPACES}
    opt = KeyOptimizer(lr=0.1)
    for _ in range(8):
        cache = ItemCache(train=True)
        read = r.read(state, [kb], ['ds'], 5, cache, targets=target)
        read.aux.backward()
        assert cache.apply(0.0, key_optimizer=opt)['keys'] > 0
        r.zero_grad(set_to_none=True)
    for s in SPACES:
        assert not torch.equal(kb.read(s, [target[s][0][1]], live=True)[0].key, keys_before[s])
        assert rank(s) <= before[s]
    assert any(rank(s) < before[s] for s in SPACES)
    assert r.rekey(kb) == 0                      # learned keys are not overwritten


# -- read path ablation and depth 0 --------------------------------------------------------------
def test_conditioned_recombiner_starts_as_the_stack_recombiner():
    r = reader()
    cond = conditioned(r.stack.recombiner, 4 * 256)
    items = [(s, torch.randn(2, SPACES[s].width), 0.7) for s in SPACES]
    a, _ = r.stack.recombiner(items, 5)
    b, _ = cond(items, 5, cond=torch.randn(1, 4 * 256))
    torch.testing.assert_close(a, b)


def test_depth_zero_is_the_plain_read_and_both_combines_run(tmp_path):
    kb = leaf_kb(tmp_path, n=8)
    ctx = l1.Context(l1.Frozen(_tiny_lm(), 2), reader(), {'ds': kb})
    assert type(ctx.new_cache(train=True)) is ItemCache        # no views: the plain path
    state = torch.randn(HIDDEN)
    for combine in ('s_s', 'r'):
        rd = reader(read_combine=combine)
        read = rd.read(state, [kb], ['ds'], 5, ItemCache(train=False))
        assert read.span.shape[0] == read.n > 0
        values = {s: [v for v, _ in ItemCache(train=False).get(kb, s, [i for _, i in info.refs])]
                  for s, info in read.spaces.items() if info.refs}
        scales = {s: read.spaces[s].scales for s in values}
        torch.testing.assert_close(rd.reread(state, values, scales), read.span)


# -- export, write fit, insert, consolidation ------------------------------------------------------
def test_export_keeps_lineage_shares_and_masses_down_to_the_sources(tmp_path):
    kb = leaf_kb(tmp_path, n=20)
    view, _ = view_of(kb, config(), tmp_path)
    out = view.export(tmp_path / 'export', step=3)
    assert out.frozen
    comp = out.source_composition()
    for s, g in view.graphs.items():
        assert current_ids(out, s) == g.row_ids
        want = g.composition()
        records = {i: it.provenance.sources[0] for i, it in zip(g.ids, kb.read(s, g.ids))}
        for row in g.row_ids:
            got = comp[row]
            expected = {}
            for leaf, share in want[row].items():
                expected[records[leaf]] = expected.get(records[leaf], 0.0) + share
            assert set(got) == set(expected)
            for r, v in expected.items():
                assert got[r] == pytest.approx(v, rel=1e-4)
        stored = out.read(s, g.row_ids)
        np.testing.assert_allclose([it.mass for it in stored], g.top_mass, rtol=1e-5)
        assert [it.time for it in stored] == g.top_time.tolist()
        np.testing.assert_allclose(sum(it.mass for it in stored), len(g.ids), rtol=1e-5)
    out.close()


def test_write_fit_loss_is_zero_when_the_stack_reproduces_its_targets(tmp_path):
    kb = leaf_kb(tmp_path, live=False)
    view, ops = view_of(kb, config(), tmp_path)
    for s, g in view.graphs.items():
        rows = g.row_ids[:3]
        pairs = [view.item(s, g.depth, g.top_index[i]) for i in rows]

        def leaf(space, ids, g=g):
            return view._sources(space, [g.index[i] for i in ids])
        loss, parts, loads = sp.fit_losses(view, s, rows, [v for v, _, _ in pairs],
                                           torch.stack([k for _, k, _ in pairs]), leaf)
        assert float(loss) == pytest.approx(0.0, abs=1e-5)
        assert loads.shape == (3,) and bool((loads > 0).all())
        other = [v + 0.5 * torch.randn_like(v) for v, _, _ in pairs]
        loss2, _, _ = sp.fit_losses(view, s, rows, other,
                                    torch.stack([k for _, k, _ in pairs]), leaf)
        assert float(loss2) > 0.01
        loss2.backward()
        assert ops.levels['1'][s].op.layers[0].out.weight.grad.abs().sum() > 0


def test_insert_places_a_new_leaf_and_recomputes_only_its_rows(tmp_path):
    kb = leaf_kb(tmp_path, n=20)
    view, _ = view_of(kb, config(), tmp_path)
    s = 'A'
    g = view.graphs[s]
    before = [view.item(s, g.depth, j)[0].clone() for j in range(len(g.row_ids))]
    new = NewItem(torch.randn(2, SPACES[s].width), g.keys[0].clone(),
                  Provenance(('write:e#0',), 'write'), 1.0, 1, id='w-new')
    kb.append(s, [new])
    kb.enable_live(s) if not kb.is_live(s) else None
    report = view.insert(s, ['w-new'])
    g = view.graphs[s]
    assert g.index['w-new'] == len(g.ids) - 1
    matrix = g.levels[0].matrix(len(g.ids)).toarray()
    np.testing.assert_allclose(matrix.sum(0), 1.0, rtol=1e-9)
    touched = report['rows'][g.depth]
    for j in range(len(g.row_ids)):
        now = view.item(s, g.depth, j)[0]
        if j in touched:
            continue
        torch.testing.assert_close(now, before[j])
    assert touched and any(not torch.equal(view.item(s, g.depth, j)[0], before[j])
                           for j in touched)


def test_consolidation_returns_the_rows_within_tolerance(tmp_path):
    kb = leaf_kb(tmp_path, n=16)
    kb.load_live()
    view, ops = view_of(kb, config(), tmp_path)
    refs = [('ds', 'A', i) for i in view.graphs['A'].row_ids[:3]]
    before = {ref: view.top_value(ref[1], ref[2]).clone() for ref in refs}
    with torch.no_grad():                       # an aggregator update
        for p in ops.levels['1']['A'].parameters():
            p.add_(0.05 * torch.randn_like(p))
    out = sp.consolidate({'ds': view}, refs, before, steps=40, item_lr=0.05)
    assert out['moved_rel'] > 0.01
    assert out['refit_rel'] < 0.5 * out['moved_rel']


# -- the trainer with rows from the stack ---------------------------------------------------------
def _tiny_lm():
    from transformers import Lfm2Config, Lfm2ForCausalLM
    from schnitz.span_protocol import ProtocolTokens, untie
    torch.manual_seed(0)
    cfg = Lfm2Config(vocab_size=64, hidden_size=HIDDEN, intermediate_size=64, num_hidden_layers=4,
                     num_attention_heads=4, num_key_value_heads=2,
                     layer_types=['conv', 'full_attention', 'conv', 'full_attention'],
                     block_auto_adjust_ff_dim=False, max_position_embeddings=256)
    lm = Lfm2ForCausalLM(cfg).eval()
    untie(lm)
    protocol = ProtocolTokens(lm.get_input_embeddings(), lm.lm_head, lambda f: 0.0)
    protocol.install(lm.get_input_embeddings(), lm.lm_head)
    for p in list(lm.parameters()) + list(protocol.parameters()):
        p.requires_grad_(False)
    return lm


def test_train_step_with_rows_from_the_stack_and_the_phases(tmp_path):
    from types import SimpleNamespace
    from schnitz.span_tokens import SPAN_TOKENS
    mem, mem_end = SPAN_TOKENS['mem'][1], SPAN_TOKENS['mem_end'][1]
    kb = leaf_kb(tmp_path, n=12)
    kb.load_live()
    view, ops = view_of(kb, config(deep_grad=0.5), tmp_path)
    r = reader(read_combine='r', learned_keys=True)
    ctx = l1.Context(l1.Frozen(_tiny_lm(), 2), r, {'ds': kb}, views={'ds': view})
    ctx.key_optimizer = KeyOptimizer(1e-2)
    ids = [1, 9, 10, 11, 12, 13, mem, mem_end, 14, 15, 16, 17, 18, mem, mem_end, 19, 20, 21, 22, 23]
    ep = l1.Episode('e', 'ds', torch.tensor(ids), torch.tensor(list(range(16, 20))), [3, 10],
                    [6, 13], [{'kb': 'ds', 'record_ids': ['r1']}, {'kb': 'ds', 'record_ids': ['r2']}],
                    3, {})
    args = SimpleNamespace(phase='l1a', retrieval_only=False, inbatch_negatives=4,
                           balance_weight=0.01, retrieval_weight=0.5, retrieval_anneal=0,
                           retrieval_floor=0.0, clip=1.0, item_lr=0.01, write_level_index=0,
                           consolidate_every=0, read_anchor=0.0, rows_from_stack='x')
    sets = l1.parameter_sets(r, None, ops)
    params = [p for ps in sets.values() for p in ps]
    opt = torch.optim.AdamW(l1.optimizer_groups(sets, SimpleNamespace(
        lr=1e-3, l1b_codec_lr=0.0, l1b_writer_lr=0.0, write_lr=None, **vars(args))), lr=1e-3)
    assert l1.phase_set('l1a', args) == l1.L1A_SET + l1.WRITE_SET
    assert l1.phase_set('r', args) == l1.READ_SET and l1.phase_set('w', args) == l1.WRITE_SET
    schedule = l1.parse_schedule('r:1,w:2', 'l1a')
    assert [l1.phase_at(schedule, i, read_warmup=2) for i in range(6)] == \
        ['r', 'r', 'r', 'w', 'w', 'r']

    def live():
        return [it.values.clone() for it in kb.read('A', current_ids(kb, 'A'), live=True)]
    op_weight = ops.levels['2']['A'].op.layers[0].out.weight
    start, start_op = live(), op_weight.detach().clone()
    out = l1.train_step(ctx, [ep], opt, args, 0, phase='r',
                        trainable=l1.set_phase(sets, l1.phase_set('r', args)))
    assert 'superpose' not in out and out['items'] == 0
    assert all(torch.equal(a, b) for a, b in zip(live(), start))
    assert torch.equal(op_weight, start_op)
    out = l1.train_step(ctx, [ep], opt, args, 1, phase='l1a',
                        trainable=l1.set_phase(sets, l1.phase_set('l1a', args)))
    assert out['superpose']['rows'] > 0 and out['superpose']['drift'] < 1e-5
    assert out['items'] > 0 and not torch.equal(op_weight, start_op)
    assert any(not torch.equal(a, b) for a, b in zip(live(), start))
    report = l1.superposition_metrics(ctx, {})
    assert report['ds']['A']['rows'] < report['ds']['A']['leaves']
    del params


def test_rows_command_writes_a_rows_banks_dir(tmp_path):
    import json
    import shutil
    from types import SimpleNamespace
    banks = tmp_path / 'banks'
    banks.mkdir()
    kb = leaf_kb(banks, 'ds', live=False, n=20)
    kb.close()
    (banks / 'banks.json').write_text(json.dumps({'kbs': {'ds': {'dir': 'ds'}},
                                                  'transcripts': [], 'level': 's0'}))
    for name in ('stack.pt', 'key_heads_init.pt'):
        torch.save({}, banks / name)
    args = SimpleNamespace(banks=banks, output=tmp_path / 'rows', field='A=4,B=4,C=5,D=5',
                           overlap=2, budget=None, temperature=None, deep_grad=None,
                           cache_every=None, graph_every=None, max_positives=None, seed=0)
    l1.rows(args)
    manifest = json.loads((tmp_path / 'rows' / 'banks.json').read_text())
    assert manifest['rows']['leaves'] == str(banks)
    rows = KnowledgeBase(tmp_path / 'rows' / 'ds')
    assert len(current_ids(rows, 'A')) == math.ceil(2 / 4 * 20)
    assert len(current_ids(rows, 'D')) == math.ceil(2 / 5 * 20)
    comp = rows.source_composition()
    assert all(abs(sum(v.values()) - 1) < 1e-6 for v in comp.values())
    rows.close()
    shutil.rmtree(tmp_path / 'rows')
    random.seed(0)


def test_full_deep_gradient_is_the_exact_gradient_with_fresh_caches(tmp_path):
    kb = leaf_kb(tmp_path, n=16)
    view, ops = view_of(kb, config(deep_grad=1.0), tmp_path)
    s = 'B'
    g = view.graphs[s]
    items = kb.read(s, g.ids, live=True)
    leaves = [it.values.float().clone().requires_grad_() for it in items]
    keys = [nn.functional.normalize(it.key.float(), dim=-1) for it in items]
    levels = sp.aggregate(g, ops, [(v, k, torch.tensor(1.0)) for v, k in zip(leaves, keys)])
    levels[-1][0][0].square().sum().backward()
    exact = {g.ids[i]: v.grad for i, v in enumerate(leaves) if v.grad is not None}
    cache = ItemCache(train=True)

    def leaf(space, ids):
        return [(v, nn.functional.normalize(k.detach(), dim=-1), torch.tensor(1.0)) for (v, _), k
                in zip(cache.get(kb, space, ids), cache.keys(kb, space, ids))]
    out, _, _ = view.grad_value(s, g.depth, 0, leaf)
    out.square().sum().backward()
    got = {i: v.grad for (_, _, i), v in cache.values.items() if v.grad is not None}
    assert set(got) == set(exact)
    for i in exact:
        torch.testing.assert_close(got[i], exact[i], rtol=1e-4, atol=1e-6)


def test_dynamic_shares_sum_to_one_and_a_zero_mass_input_changes_nothing():
    anchors = nn.functional.normalize(torch.randn(5, 8), dim=-1)
    key = nn.functional.normalize(torch.randn(8), dim=-1)
    p = sp.dynamic_shares(key, anchors, [0, 3, 4], torch.tensor(7.0))
    assert float(p.sum()) == pytest.approx(1.0) and bool((p > 0).all())
    from schnitz.kb.stack import SuperpositionOperator
    torch.manual_seed(0)
    op = SuperpositionOperator('A', 16, 12, 2)
    values = [torch.randn(2, 256), torch.randn(3, 256)]
    offs = [torch.randn(256), torch.randn(256)]
    one, _ = sp.apply_field(op, values[:1], torch.tensor([0.4]), offs[:1], anchors[0].repeat(32), 2)
    two, _ = sp.apply_field(op, values, torch.tensor([0.4, 0.0]), offs, anchors[0].repeat(32), 2)
    torch.testing.assert_close(one, two, rtol=1e-5, atol=1e-6)


def test_the_fit_moves_an_input_key_toward_the_row_its_content_helps():
    """Two rows share one input j (candidates of j: both rows); row 1's target is its
    output with j at full weight, row 2's its output without j, so the fit gradient on
    j's key raises j's share to row 1 and lowers it to row 2."""
    torch.manual_seed(0)
    width = SPACES['A'].width
    anchors = nn.functional.normalize(torch.stack([torch.randn(256), torch.randn(256)]), dim=-1)
    ops = sp.WriteOps(1, DIMS)
    level = sp.Level(inputs=[[0, 1], [0, 2]], shares=[[0.5, 1.0], [0.5, 1.0]],
                     mass=np.array([1.5, 1.5]), time=np.array([1, 1]), count=[2, 2],
                     candidates=[[0, 1], [0], [1]])
    keys = nn.functional.normalize(torch.stack([anchors[0] + anchors[1], anchors[0],
                                                anchors[1]]), dim=-1)
    graph = sp.SpaceGraph('ds', 'A', ['j', 'b1', 'b2'], [1.0, 1.0, 1.0], [1, 1, 1], [2, 2, 2],
                          keys, ['r1', 'r2'], anchors, [level])
    values = [torch.randn(2, width) for _ in range(3)]
    key_j = keys[0].clone().requires_grad_()
    mass = [torch.tensor(1.0)] * 3

    def rows(k, weight_j=None):
        leaves = [(values[0], k, mass[0] if weight_j is None else torch.tensor(weight_j)),
                  (values[1], keys[1], mass[1]), (values[2], keys[2], mass[2])]
        return sp.aggregate(graph, ops, leaves)[-1]
    with torch.no_grad():
        t1 = rows(keys[0], 1e3)[0][0]            # j dominating row 1
        t2 = rows(keys[0], 0.0)[1][0]            # j absent from row 2
    from schnitz.kb.producer import item_losses
    got = rows(key_j)
    loss = sum(item_losses({'A': got[i][0]}, {'A': t})['mse_A'] for i, t in ((0, t1), (1, t2)))
    loss.backward()
    assert key_j.grad is not None and key_j.grad.abs().sum() > 0
    tau = ops.tau(1, 'A').detach()
    before = sp.dynamic_shares(keys[0], anchors, [0, 1], tau)
    after = sp.dynamic_shares(keys[0] - 0.05 * key_j.grad / key_j.grad.norm(), anchors, [0, 1], tau)
    assert after[0] > before[0] and after[1] < before[1]


def test_field_ranges_parse_sample_and_set_the_level_one_fill(tmp_path):
    fields = sp.parse_fields('A=4:16,B=8')
    assert fields == {'A': (4, 16), 'B': 8}
    cfg = config(field={'A': (4, 16), 'B': 8, 'C': 6, 'D': 6}, depth=1)
    assert cfg.field_size('A') == pytest.approx(8.0)
    rng = random.Random(0)
    draws = [cfg.sample_field('A', rng) for _ in range(200)]
    assert 4 <= min(draws) < 6 and 12 < max(draws) <= 16
    kb = leaf_kb(tmp_path, n=40)
    view, _ = view_of(kb, cfg, tmp_path)
    fills = {}
    for f in (4, 16):
        view.refield({'A': f}, 0)
        g = view.graphs['A']
        fills[f] = g.stats()['levels'][0]['field_fill']
        np.testing.assert_allclose(g.levels[0].matrix(len(g.ids)).toarray().sum(0), 1.0)
    assert fills[16] > 2 * fills[4]


def test_stack_fit_learns_the_rows_and_reports_sweeps(tmp_path):
    from schnitz.kb.producer import L2Weights
    from schnitz.kb.stack import KeyHeads
    from schnitz.kb.stages import l2
    banks = tmp_path / 'banks'
    banks.mkdir()
    leaf_kb(banks, 'ds', live=False, n=24).close()
    manifest = {'kbs': {'ds': {'dir': 'ds'}}, 'transcripts': []}
    cfg = config(field={'A': (2, 6), 'B': 4, 'C': 6, 'D': 6}, depth=2)
    leaves = KnowledgeBase(banks / 'ds')
    rows_kb, _ = sp.build_rows(leaves, tmp_path / 'rows' / 'ds', cfg)
    rows_kb.close()
    leaves.close()
    torch.manual_seed(0)
    heads = KeyHeads(HIDDEN, 16)
    ops = sp.WriteOps(2, DIMS, temperature=cfg.temperature)
    fit = l2.StackFit(ops, heads, cfg, banks, manifest, 'cpu', 5)
    fit.load(l2.load_row_targets(tmp_path / 'rows', {'ds': 'ds'}))
    train, held = fit.split(False), fit.split(True)
    assert train and held
    params = list(ops.parameters()) + list(heads.item.parameters()) + \
        list(fit.corrections.parameters())
    opt = torch.optim.Adam(params, lr=3e-3)
    weights = L2Weights()
    rng = random.Random(0)
    losses = []
    for step in range(12):
        fit.refield({s: cfg.sample_field(s, rng) for s in SPACES}, step)
        picks = {s: rows[:6] for s, rows in train.items()}
        opt.zero_grad()
        loss, parts = fit.loss(picks, weights, balance=0.01)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
    assert any(float(p.detach().abs().sum()) > 0 for p in fit.corrections.parameters())
    report = fit.measure(held)
    assert any(k.startswith('cos_') for k in report) and any(k.startswith('keycos_') for k in report)
    cost = fit.cost({s: rows[:2] for s, rows in train.items()}, weights,
                    {'A': 6, 'B': 4, 'C': 6, 'D': 6}, 12)
    assert cost['step_s'] > 0
    half = fit.density(0.5, held, 12)
    assert half['rows'] <= report['rows']
    stats = fit.write_stats()['ds']['A']
    assert stats['share_entropy'] >= 0 and stats['row_load']['n'] > 0
    fit.close()


def test_batched_rows_equal_row_by_row_values_and_gradients(tmp_path):
    kb = leaf_kb(tmp_path)
    cfg = config(max_pairs=200)          # several chunks per pass
    view, ops = view_of(kb, cfg, tmp_path)
    with torch.no_grad():                 # zero-initialized paths must matter here
        for p in ops.parameters():
            p.add_(0.02 * torch.randn_like(p))
    for s, g in view.graphs.items():
        cache = ItemCache(train=True)

        def leaf(space, ids):
            return [(v, nn.functional.normalize(k, dim=-1), torch.tensor(1.0)) for (v, _), k
                    in zip(cache.get(kb, space, ids), cache.keys(kb, space, ids))]
        results = []
        for batched in (True, False):
            view.config.batched = batched
            view.clear()
            rows = list(range(len(g.row_ids)))
            got = view.grad_values(s, g.depth, rows, leaf, deep=1.0)
            loss = sum((v ** 2).sum() + (k * torch.arange(k.shape[0])).sum() + m
                       for v, k, m in got)
            params = list(ops.parameters())
            ids = g.ids
            leaves = [v for v, _ in cache.get(kb, s, ids)] + cache.keys(kb, s, ids)
            grads = torch.autograd.grad(loss, params + leaves, allow_unused=True)
            results.append(([t.detach() for triple in got for t in triple], grads))
        view.config.batched = True
        (va, ga), (vb, gb) = results
        for a, b in zip(va, vb):
            torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
        for a, b in zip(ga, gb):
            assert (a is None) == (b is None)
            if a is not None:       # up to summation order, relative to the tensor's scale
                assert float((a - b).abs().max()) <= 1e-4 * float(b.abs().max()) + 1e-5
