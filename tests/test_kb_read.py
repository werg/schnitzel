"""L1 read path (``schnitz.kb.read``) and trainer passes (``schnitz.kb.stages.l1``);
CPU, tiny models, no downloads."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from schnitz.kb.losses import retrieval_loss
from schnitz.kb.read import ItemCache, L1Reader, ReadConfig, current_ids, splice
from schnitz.kb import producer
from schnitz.kb.stages import l1
from schnitz.kb_store import DEFAULT_SPACES, KnowledgeBase, NewItem, Provenance
from schnitz.span_protocol import ProtocolTokens, untie
from schnitz.span_tokens import SPAN_TOKENS

SPACES = DEFAULT_SPACES
HIDDEN = 32
MEM, MEM_END = SPAN_TOKENS['mem'][1], SPAN_TOKENS['mem_end'][1]


def config(**kw):
    kw = {'candidates': {'A': 3, 'B': 4, 'C': 8, 'D': 8}, **kw}
    return ReadConfig(hidden=HIDDEN, span_width=HIDDEN, state=16, op_hidden=12, layers=2,
                      key_hidden=16, checkpointing=False, **kw)


def reader(**kw):
    torch.manual_seed(0)
    return L1Reader(config(**kw))


def make_kb(tmp_path, name='kb', dataset='ds', records=(('r1', 1), ('r2', 1), ('r3', 5)),
            live=True):
    kb = KnowledgeBase.create(tmp_path / name, name=name, dataset=dataset, spaces=SPACES)
    for k, (s, spec) in enumerate(SPACES.items()):
        items = []
        for i, (r, t) in enumerate(records):     # content depends on (space, record) only
            gen = torch.Generator().manual_seed(1000 * k + int(r[1:]))
            items.append(NewItem(torch.randn(2 + i, spec.width, generator=gen),
                                 torch.randn(spec.key_width, generator=gen),
                                 Provenance((r,), 'codec'), 1.0, t))
        kb.append(s, items)
    if live:
        for s in SPACES:
            kb.enable_live(s)
    return kb


def targets_of(kb, records):
    index = {s: l1.source_index(kb, s) for s in SPACES}
    return {s: [(kb.dataset, i) for r in records for i in index[s][r]] for s in SPACES}


def test_causal_filtering_by_time(tmp_path):
    kb = make_kb(tmp_path)
    late = targets_of(kb, ['r3'])
    r = reader()
    read = r.read(torch.randn(HIDDEN), [kb], ['ds'], 3, ItemCache(train=False))
    for s, info in read.spaces.items():
        assert len(info.scored) == 2 and not set(info.scored) & set(late[s])
    # a gold target that is not yet available is dropped, never read
    gold = r.read(torch.randn(HIDDEN), [kb], ['ds'], 3, ItemCache(train=False),
                  targets=targets_of(kb, ['r1', 'r3']), gold=True)
    assert all(len(info.refs) == 1 for info in gold.spaces.values())
    # nothing available yet: an empty read (no memory), not an error
    empty = r.read(torch.randn(HIDDEN), [kb], ['ds'], 0, ItemCache(train=False))
    assert empty.span.shape == (0, HIDDEN) and empty.n == 0


def test_future_items_do_not_change_the_read(tmp_path):
    """Gate-zero exactness end to end: items outside the causal window change nothing."""
    a = make_kb(tmp_path, 'a', records=(('r1', 1), ('r2', 1)))
    b = make_kb(tmp_path, 'b', records=(('r1', 1), ('r2', 1), ('r3', 5)))
    r = reader()
    state = torch.randn(HIDDEN)
    one = r.read(state, [a], ['ds'], 3, ItemCache(train=False))
    two = r.read(state, [b], ['ds'], 3, ItemCache(train=False))
    torch.testing.assert_close(one.span, two.span)


def test_unread_candidates_and_gate_zero_spaces_change_nothing(tmp_path):
    r = reader()
    reads = {'A': torch.randn(3, SPACES['A'].width), 'B': torch.randn(2, 512)}
    alone, _ = r.stack.recombiner([('A', reads['A'], 1.0)], 5)
    gated, _ = r.stack.recombiner([('A', reads['A'], 1.0), ('B', reads['B'], 0.0)], 5)
    torch.testing.assert_close(alone, gated, atol=1e-6, rtol=1e-6)
    # a sparse read: candidates beyond keep_s are scored but never enter the span
    kb = make_kb(tmp_path, records=tuple((f'r{i}', 1) for i in range(8)))
    cache = ItemCache(train=True)
    read = r.read(torch.randn(HIDDEN), [kb], ['ds'], 3, cache)
    read.span.square().sum().backward()
    for s, info in read.spaces.items():
        assert len(info.refs) == r.config.keep[s] < len(info.scored)
        unread = {i for _, i in info.scored} - {i for _, i in info.refs}
        for (_, space, item_id), v in cache.values.items():
            if space == s and item_id in unread:
                assert v.grad is None or v.grad.abs().sum() == 0


def test_authorization(tmp_path):
    kb = make_kb(tmp_path, 'kb', 'ds')
    other = make_kb(tmp_path, 'other', 'secret')
    r = reader()
    with pytest.raises(PermissionError):
        r.read(torch.randn(HIDDEN), [kb, other], ['ds'], 3, ItemCache(train=False))
    with pytest.raises(PermissionError):   # a target of an unauthorized KB, even in gold mode
        r.read(torch.randn(HIDDEN), [kb], ['ds'], 3, ItemCache(train=False),
               targets=targets_of(other, ['r1']), gold=True)
    both = r.read(torch.randn(HIDDEN), [kb, other], ['ds', 'secret'], 3, ItemCache(train=False))
    assert {d for info in both.spaces.values() for d, _ in info.scored} <= {'ds', 'secret'}


def test_gradients_reach_live_items_and_key_heads(tmp_path):
    kb = make_kb(tmp_path)
    r = reader()
    cache = ItemCache(train=True)
    before = {s: [i.values.clone() for i in kb.read(s, kb._row_ids[s], live=True)]
              for s in SPACES}
    read = r.read(torch.randn(HIDDEN), [kb], ['ds'], 3, cache, targets=targets_of(kb, ['r1']))
    assert read.aux is not None and read.span.shape[0] >= 1
    (read.span.square().sum() + read.aux).backward()
    for s in SPACES:
        assert r.keys.query[s][1].weight.grad.abs().sum() > 0
        assert r.keys.item[s][1].weight.grad.abs().sum() > 0
        assert r.operators[s].op.layers[0].out.weight.grad.abs().sum() > 0
        assert r.gate_offset[s].grad is not None
    assert r.stack.recombiner.head[1].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in r.stack.codecs.parameters())
    assert all(v.grad is not None and v.grad.abs().sum() > 0 for v in cache.values.values())
    assert cache.apply(item_lr=0.01) == {'items': 8}   # two visible items in each space
    for s in SPACES:
        after = kb.read(s, kb._row_ids[s], live=True)
        changed = [not torch.equal(a.values, b) for a, b in zip(after, before[s])]
        assert changed == [True, True, False]           # r3 is in the future
        stored = kb.read(s, kb._row_ids[s])             # the frozen bf16 payload is untouched
        torch.testing.assert_close(stored[0].values.float(), before[s][0])
    # the search keys are refreshed from the item-key heads
    assert r.rekey(kb) == 12
    for s in SPACES:
        for item in kb.read(s, kb._row_ids[s], live=True):
            torch.testing.assert_close(item.key, r.keys.item_key(s, item.values).detach(),
                                       atol=1e-6, rtol=1e-5)


def test_retrieval_loss_over_candidates_and_missed_targets(tmp_path):
    kb = make_kb(tmp_path, records=tuple((f'r{i}', 1) for i in range(8)))
    r = reader(candidates={'A': 3, 'B': 3, 'C': 3, 'D': 3})
    cache = ItemCache(train=True)
    state = torch.randn(HIDDEN)
    targets = targets_of(kb, ['r0', 'r5', 'r7'])
    read = r.read(state, [kb], ['ds'], 3, cache, targets=targets)
    want = []
    for s, info in read.spaces.items():
        assert info.recall == sum(t in set(info.scored) for t in targets[s]) / 3
        assert info.recall_read == sum(t in set(info.refs) for t in targets[s]) / 3
        pool = info.scored + [t for t in targets[s] if t not in set(info.scored)]
        q = r.keys.query_key(s, state)
        keys = torch.stack([r.keys.item_key(s, cache.get(kb, s, [i])[0][0]) for _, i in pool])
        positive = torch.tensor([p in set(targets[s]) for p in pool])
        want.append(retrieval_loss(r.keys.scores(s, q[None], keys), positive[None])[0])
    torch.testing.assert_close(read.aux, torch.stack(want).mean())
    read.aux.backward()
    for s in SPACES:     # every target gets a gradient, retrieved or not
        for ref in targets[s]:
            assert cache.values[('ds', s, ref[1])].grad.abs().sum() > 0


def test_alternatives_are_positives_never_negatives_and_gold_reads_targets(tmp_path):
    kb = make_kb(tmp_path, records=tuple((f'r{i}', 1) for i in range(8)))
    r = reader(candidates={'A': 3, 'B': 3, 'C': 3, 'D': 3})
    state = torch.randn(HIDDEN)
    targets, copies = targets_of(kb, ['r0']), targets_of(kb, ['r5', 'r7'])
    cache = ItemCache(train=True)
    # a copy offered as a negative is still a positive
    read = r.read(state, [kb], ['ds'], 3, cache, targets=targets, alternatives=copies,
                  negatives=copies)
    want = []
    for s, info in read.spaces.items():
        positives = targets[s] + copies[s]
        assert info.recall == float(any(p in set(info.scored) for p in positives))
        pool = info.scored + [p for p in positives if p not in set(info.scored)]
        q = r.keys.query_key(s, state)
        keys = torch.stack([r.keys.item_key(s, cache.get(kb, s, [i])[0][0]) for _, i in pool])
        positive = torch.tensor([p in set(positives) for p in pool])
        want.append(retrieval_loss(r.keys.scores(s, q[None], keys), positive[None])[0])
    torch.testing.assert_close(read.aux, torch.stack(want).mean())
    gold = r.read(state, [kb], ['ds'], 3, ItemCache(train=False), targets=targets,
                  alternatives=copies, gold=True)
    assert all(info.refs == targets[s] for s, info in gold.spaces.items())


def _loss_over(r, kb, cache, state, pool, positives):
    """The retrieval loss per space over ``pool`` (scored from current values)."""
    want = []
    for s in SPACES:
        q = r.keys.query_key(s, state)
        keys = torch.stack([r.keys.item_key(s, cache.get(kb, s, [i])[0][0]) for _, i in pool[s]])
        positive = torch.tensor([p in set(positives[s]) for p in pool[s]])
        want.append(retrieval_loss(r.keys.scores(s, q[None], keys), positive[None])[0])
    return torch.stack(want).mean()


def test_neutral_items_leave_the_retrieval_loss(tmp_path):
    """Neutral items are neither positives nor negatives: scored candidates and in-batch
    negatives that are neutral leave the loss's list; a neutral positive stays positive;
    they may still be read."""
    kb = make_kb(tmp_path, records=tuple((f'r{i}', 1) for i in range(8)))
    every = {'A': 8, 'B': 8, 'C': 8, 'D': 8}
    r = reader(candidates=every, keep=every)
    state = torch.randn(HIDDEN)
    targets, copies = targets_of(kb, ['r0']), targets_of(kb, ['r1'])
    neutral = targets_of(kb, ['r1', 'r2', 'r3'])      # r1 is also a positive: stays one
    cache = ItemCache(train=True)
    read = r.read(state, [kb], ['ds'], 3, cache, targets=targets, alternatives=copies,
                  neutral=neutral)
    pool, positives = {}, {}
    for s, info in read.spaces.items():
        assert set(neutral[s]) <= set(info.scored) and set(neutral[s]) <= set(info.refs)
        positives[s] = targets[s] + copies[s]
        pool[s] = [p for p in info.scored if p not in set(neutral[s]) or p in positives[s]]
        assert len(pool[s]) == 6
    torch.testing.assert_close(read.aux, _loss_over(r, kb, cache, state, pool, positives))
    read.aux.backward()
    for s in SPACES:       # a neutral item gets no retrieval gradient (the read is detached
        for _, i in targets_of(kb, ['r2', 'r3'])[s]:    # from aux here: no task loss)
            assert cache.values[('ds', s, i)].grad is None or \
                cache.values[('ds', s, i)].grad.abs().sum() == 0

    # a neutral item offered as an in-batch negative is dropped; the others stay
    one = {'A': 1, 'B': 1, 'C': 1, 'D': 1}
    r = reader(candidates=one, keep=one)
    cache = ItemCache(train=True)
    negs = targets_of(kb, ['r2', 'r3', 'r4', 'r5'])
    read = r.read(state, [kb], ['ds'], 3, cache, targets=targets, negatives=negs,
                  neutral=targets_of(kb, ['r2', 'r3']))
    only = r.read(state, [kb], ['ds'], 3, ItemCache(train=True), targets=targets,
                  negatives=targets_of(kb, ['r4', 'r5']), neutral=targets_of(kb, ['r2', 'r3']))
    assert torch.equal(read.aux, only.aux)
    with_neutral_negatives = r.read(state, [kb], ['ds'], 3, ItemCache(train=True),
                                    targets=targets, negatives=negs)
    assert not torch.equal(read.aux, with_neutral_negatives.aux)


def test_absent_neutral_leaves_the_loss_unchanged_bit_for_bit(tmp_path):
    kb = make_kb(tmp_path, records=tuple((f'r{i}', 1) for i in range(8)) + (('r9', 5),))
    r = reader(candidates={'A': 3, 'B': 3, 'C': 3, 'D': 3})
    state = torch.randn(HIDDEN)
    targets, copies = targets_of(kb, ['r0']), targets_of(kb, ['r5'])
    negs = targets_of(kb, ['r2', 'r3', 'r4'])

    def aux(**kw):
        return r.read(state, [kb], ['ds'], 3, ItemCache(train=True), targets=targets,
                      alternatives=copies, negatives=negs, **kw)
    base = aux()
    for neutral in (None, {}, {s: [] for s in SPACES},
                    targets_of(kb, ['r9'])):   # only later than the query: never scored
        got = aux(neutral=neutral)
        assert torch.equal(got.aux, base.aux)
        assert got.recall_at == base.recall_at
        for s in SPACES:
            assert got.spaces[s].refs == base.spaces[s].refs
            assert got.spaces[s].recall == base.spaces[s].recall


def test_neutral_items_are_authorized_like_the_other_named_items(tmp_path):
    kb = make_kb(tmp_path, 'kb', 'ds')
    other = make_kb(tmp_path, 'other', 'secret')
    with pytest.raises(PermissionError):
        reader().read(torch.randn(HIDDEN), [kb], ['ds'], 3, ItemCache(train=False),
                      targets=targets_of(kb, ['r1']), neutral=targets_of(other, ['r2']))


def test_read_count_is_mass_weighted_length_in_reps_capped(tmp_path):
    kb = make_kb(tmp_path, records=(('r1', 1), ('r2', 1)))   # items 2 and 3 positions long
    targets = targets_of(kb, ['r1', 'r2'])
    read = reader(max_reps=64).read(torch.randn(HIDDEN), [kb], ['ds'], 3,
                                    ItemCache(train=False), targets=targets, gold=True)
    # gates 1: reps per item m / r_s: A 2, 3; B 4, 6; C 8, 12; D 16, 24 -> mean 9.375
    assert read.n == 9
    capped = reader(max_reps=3).read(torch.randn(HIDDEN), [kb], ['ds'], 3,
                                     ItemCache(train=False), targets=targets, gold=True)
    assert capped.n == 3


def test_splice_places_span_between_mem_tokens():
    ids = torch.tensor([5, 6, MEM, MEM_END, 7, MEM, MEM_END, 8])
    embeds = torch.nn.functional.one_hot(ids, 64).float()
    spans = [torch.full((3, 64), 0.5), torch.full((2, 64), -0.5)]
    out, index = splice(embeds, [2, 5], spans)
    assert out.shape[0] == 8 + 5
    torch.testing.assert_close(out[index], embeds)
    for p, span in zip((2, 5), spans):
        start = int(index[p]) + 1
        torch.testing.assert_close(out[start:start + len(span)], span)
        assert int(index[p + 1]) == start + len(span)          # then <|/mem|>
    with pytest.raises(ValueError):
        splice(embeds, [2], spans)


# -- tiny decoder -----------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _torch_conv(monkeypatch):
    """The pure-torch convolution path (the CUDA kernel refuses CPU tensors)."""
    import inspect
    from transformers.models.lfm2 import modeling_lfm2
    monkeypatch.setattr(modeling_lfm2, 'causal_conv1d_fn',
                        inspect.unwrap(modeling_lfm2.causal_conv1d_fn))


def tiny_lm():
    from transformers import Lfm2Config, Lfm2ForCausalLM
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
    return lm, protocol


def episode(ids, calls, mems, kb='ds', records=(['r1'], ['r2']), qt=3):
    ids = torch.tensor(ids)
    targets = torch.tensor([t for t in range(1, len(ids)) if t > mems[-1] + 1])
    return l1.Episode('e', kb, ids, targets, calls, mems,
                      [{'kb': kb, 'record_ids': list(r)} for r in records], qt, {})


# two sites: call at 3, result at 6-7; call at 10, result at 13-14; answer after
IDS = [1, 9, 10, 11, 12, 13, MEM, MEM_END, 14, 15, 16, 17, 18, MEM, MEM_END, 19, 20, 21, 22, 23]


def context(tmp_path, **kw):
    lm, protocol = tiny_lm()
    kb = make_kb(tmp_path, records=(('r1', 1), ('r2', 1), ('r3', 1), ('r4', 5)))
    frozen = l1.Frozen(lm, 2)
    return l1.Context(frozen, reader(**kw), {'ds': kb}), protocol


def test_protocol_embeddings_frame_the_span(tmp_path):
    ctx, protocol = context(tmp_path)
    ep = episode(IDS, [3, 10], [6, 13])
    embeds = ctx.frozen.embed(ep.ids)
    torch.testing.assert_close(embeds[6], protocol.embedding('mem').float())
    torch.testing.assert_close(embeds[7], protocol.embedding('mem_end').float())
    span = torch.randn(4, HIDDEN)
    out, index = splice(embeds, [6], [span])
    torch.testing.assert_close(out[int(index[6]) + 1:int(index[7])], span)


def test_queries_use_only_the_causal_prefix(tmp_path):
    ctx, _ = context(tmp_path)
    ep = episode(IDS, [3, 10], [6, 13])
    changed = list(IDS)
    changed[11] = 30                  # after the second call: must not change either query
    changed[16:] = [40, 41, 42, 43]   # the answer (targets) must not change the reads
    with torch.no_grad():
        _, _, _, spans = l1.run_episode(ctx, ep, ItemCache(train=False))
        _, _, _, spans2 = l1.run_episode(ctx, episode(changed, [3, 10], [6, 13]),
                                         ItemCache(train=False))
        for a, b in zip(spans, spans2):
            torch.testing.assert_close(a, b)
        early = list(IDS)             # a token before the second call changes read 2 only
        early[9] = 31
        _, _, _, spans3 = l1.run_episode(ctx, episode(early, [3, 10], [6, 13]),
                                         ItemCache(train=False))
    torch.testing.assert_close(spans[0], spans3[0])
    assert not torch.equal(spans[1], spans3[1])


def test_later_query_depends_on_earlier_read(tmp_path):
    """The prefix pass of site 2 carries span 1, and its gradient reaches what produced
    span 1 (the full graph; nothing is detached)."""
    ctx, _ = context(tmp_path)
    ep = episode(IDS, [3, 10], [6, 13])
    _, _, reads, _ = l1.run_episode(ctx, ep, ItemCache(train=True))
    reads[1].aux.backward()           # only the second read's retrieval loss
    # it depends on R's weights only through span 1 -> the prefix pass -> query 2
    assert ctx.reader.stack.recombiner.head[1].weight.grad.abs().sum() > 0


def test_task_loss_trains_items_and_is_deterministic(tmp_path):
    ctx, _ = context(tmp_path)
    ep = episode(IDS, [3, 10], [6, 13])
    runs = []
    for _ in range(2):
        cache = ItemCache(train=True)
        nll, n, reads, spans = l1.run_episode(ctx, ep, cache)
        (nll / n).backward()
        runs.append((nll.detach(), [s.detach() for s in spans]))
        grads = [v.grad for v in cache.values.values() if v.grad is not None]
        assert grads and any(g.abs().sum() > 0 for g in grads)
        assert ctx.reader.keys.query['A'][1].weight.grad.abs().sum() > 0
        ctx.reader.zero_grad(set_to_none=True)
    torch.testing.assert_close(runs[0][0], runs[1][0], rtol=0, atol=0)
    for a, b in zip(runs[0][1], runs[1][1]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    with torch.no_grad():             # the empty-span control reads nothing
        empty = [torch.zeros(0, HIDDEN)] * 2
        _, n0, reads0, _ = l1.run_episode(ctx, ep, ItemCache(train=False), 'fixed', empty)
    assert reads0 == [] and n0 == n


def test_retrieval_only_trains_heads_not_the_read(tmp_path):
    ctx, _ = context(tmp_path)
    ep = episode(IDS, [3, 10], [6, 13])
    nll, _, reads, spans = l1.run_episode(ctx, ep, ItemCache(train=True), retrieval_only=True)
    assert nll is None and not any(s.requires_grad for s in spans)
    torch.stack([rd.aux for rd in reads]).mean().backward()
    assert ctx.reader.keys.query['A'][1].weight.grad.abs().sum() > 0
    assert ctx.reader.keys.item['A'][1].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in ctx.reader.operators.parameters())
    assert all(p.grad is None for p in ctx.reader.stack.parameters())


def test_gold_mode_reads_exactly_the_targets(tmp_path):
    ctx, _ = context(tmp_path)
    ep = episode(IDS, [3, 10], [6, 13], records=(['r1', 'r2'], ['r4']))
    with torch.no_grad():
        _, _, reads, spans = l1.run_episode(ctx, ep, ItemCache(train=False), 'gold')
    want = ctx.targets(ep, 0)
    for s in SPACES:
        assert reads[0].spaces[s].refs == want[s]
        assert torch.equal(reads[0].spaces[s].gates, torch.ones(2))
    assert spans[1].shape[0] == 0          # r4 is later than the query: nothing to read


def test_balance_loss_over_scored_candidates(tmp_path):
    ctx, _ = context(tmp_path)
    ep = episode(IDS, [3, 10], [6, 13])
    _, _, reads, _ = l1.run_episode(ctx, ep, ItemCache(train=True))
    usage = {}
    loss = l1._balance(ctx, reads, usage)
    assert loss is not None and loss.requires_grad
    assert set(usage) == {f'ds/{s}' for s in SPACES}
    assert all(bool(u.touched.any()) for u in usage.values())


def test_chat_drops_query_text_and_uses_current_schema():
    row = {'episode_id': 'e', 'kb': 'ds', 'messages': [
        {'role': 'user', 'content': 'q'},
        {'role': 'assistant', 'content': '', 'tool_calls': [
            {'type': 'function', 'function': {'name': 'memory_search',
                                              'arguments': {'query': 'secret header'}}}]},
        {'role': 'tool', 'name': 'memory_search',
         'content': {'slot': {'kb': 'ds', 'record_ids': ['r1', 'r2']}}},
        {'role': 'tool', 'name': 'memory_write', 'content': {'write_result': {'kb': 'ds'}}},
        {'role': 'assistant', 'content': 'a'}],
        'tools': [{'name': 'memory_search', 'parameters': {'properties': {'query': {}}}},
                  {'name': 'api', 'parameters': {}}]}
    messages, tools = l1.chat(row)
    assert messages[1]['tool_calls'][0]['function']['arguments'] == {}
    assert messages[2]['content'] == SPAN_TOKENS['mem'][0] + SPAN_TOKENS['mem_end'][0]
    assert len(messages) == 4             # the write acknowledgement is dropped
    kept, _ = l1.chat(row, keep_writes=True)
    assert len(kept) == 5 and isinstance(kept[3]['content'], str)
    assert [t['name'] for t in tools] == ['memory_search', 'memory_write', 'api']
    assert tools[0]['parameters']['properties'] == {}
    text_messages, _ = l1.chat(row, {'r1': 'one', 'r2': 'two'})
    assert text_messages[2]['content'] == 'one\n\ntwo'
    assert row['messages'][1]['tool_calls'][0]['function']['arguments'] == {'query': 'secret header'}


TOKENIZER = Path('/runs/hf-models/LFM2.5-350M')


@pytest.mark.skipif(not TOKENIZER.exists(), reason='needs the LFM2.5 tokenizer')
def test_layout_with_the_real_template():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    call = {'type': 'function', 'function': {'name': 'memory_search', 'arguments': {}}}
    row = {'episode_id': 'e', 'kb': 'ds', 'provenance': {'source_query_time': 4}, 'messages': [
        {'role': 'user', 'content': 'q'},
        {'role': 'assistant', 'content': '', 'tool_calls': [call, call]},
        {'role': 'tool', 'name': 'memory_search', 'content': {'slot': {'kb': 'ds', 'record_ids': ['a']}}},
        {'role': 'tool', 'name': 'memory_search', 'content': {'slot': {'kb': 'ds', 'record_ids': ['b']}}},
        {'role': 'assistant', 'content': '1842'}], 'tools': []}
    ep = l1.layout(row, tok)
    assert len(ep.calls) == 2 and ep.calls[0] < ep.calls[1] < ep.mems[0] < ep.mems[1]
    assert all(')' in tok.decode([int(ep.ids[c])]) for c in ep.calls)
    assert [int(ep.ids[m + 1]) for m in ep.mems] == [MEM_END, MEM_END]
    assert ep.query_time == 4
    assert tok.decode(ep.ids[ep.targets]).startswith('<|tool_call_start|>[memory_search()')
    assert '1842' in tok.decode(ep.ids[ep.targets])
    text = l1.layout(row, tok, {'a': 'alpha', 'b': 'beta'})
    assert text.targets.numel() == ep.targets.numel()


# -- L1 completion: K2 chaining, in-batch negatives, writes, L1b, phases ---------------------
BG, BG_END = SPAN_TOKENS['bg'][1], SPAN_TOKENS['bg_end'][1]
IDS_W = IDS + [BG, BG_END, 24]           # a write site at the end (its <|bg|> at 20)


def train_args(**kw):
    from types import SimpleNamespace
    base = dict(phase='l1a', retrieval_only=False, inbatch_negatives=16, balance_weight=0.01,
                retrieval_weight=0.5, retrieval_anneal=0, retrieval_floor=0.0, clip=1.0,
                item_lr=0.01, write_level_index=0)
    return SimpleNamespace(**{**base, **kw})


def writer_model(lm):
    """``Model`` on the tiny decoder with the real writer path (``write``, ``free_run``,
    span heads); the prefix cache is replaced by its exact equivalent, a full forward
    per example (no padding, so results do not depend on batch composition)."""
    import contextlib
    from types import SimpleNamespace
    from schnitz.bgkit_span import SpanWriter
    from schnitz.kb.decoder import Model
    torch.manual_seed(1)
    model = Model.__new__(Model)
    model.decoder = SimpleNamespace(embed_tokens=lm.get_input_embeddings(), base_lm=lm)
    model.writer = SpanWriter(HIDDEN, 1.0)
    model.device = torch.device('cpu')
    model.core = SimpleNamespace(autocast=contextlib.nullcontext)
    model.prompts = {'memory': (torch.tensor([1, 2]), torch.tensor([3]))}
    model.gate, model.adapter, model.gate_value, model.merged = None, None, 0.0, False
    model.protocol, model.protocol_losses, model.record_protocol = None, [], False
    model.reference, model.port = None, None
    model.text_ids = lambda text: torch.tensor([4 + ord(c) % 20 for c in text][:1024])

    def span_hidden(cache, spans):
        out = []
        for ex, span in zip(cache['examples'], spans):
            x = torch.cat([model.write_inputs(ex), span.float()])
            h = lm.model(inputs_embeds=x[None], use_cache=False).last_hidden_state[0]
            out.append(h[-span.shape[0]:])
        return out

    model.prefix = lambda examples: {'examples': examples}
    model.span_hidden = span_hidden
    return model


def test_rekey_through_batched_reads(tmp_path):
    kb = make_kb(tmp_path, records=tuple((f'r{i}', 1) for i in range(6)))
    kb.load_live()
    ids = kb._row_ids['B']
    kb.live_step('B', ids[:2], [torch.ones_like(x.values) for x in kb.read('B', ids[:2], live=True)],
                 lr=0.1)
    r = reader()
    assert r.rekey(kb) == 24
    for s in SPACES:     # batched keys equal the per-item key head
        for item in kb.read(s, kb._row_ids[s], live=True):
            torch.testing.assert_close(item.key, r.keys.item_key(s, item.values).detach(),
                                       atol=1e-6, rtol=1e-5)


def test_init_reader_loads_k2_key_heads(tmp_path):
    k2 = reader()
    with torch.no_grad():
        for p in k2.keys.parameters():
            p.add_(0.5)
        k2.gate_offset['A'].fill_(0.9)
    torch.save({'reader': k2.state_dict(), 'step': 10}, tmp_path / 'reader.pt')
    torch.manual_seed(5)
    fresh = L1Reader(config())
    operators = [p.clone() for p in fresh.operators.parameters()]
    loaded = l1.load_init_reader(fresh, tmp_path / 'reader.pt')
    assert loaded and all(k.startswith(('keys.', 'gate_offset.')) for k in loaded)
    for a, b in zip(fresh.keys.parameters(), k2.keys.parameters()):
        assert torch.equal(a, b)
    assert float(fresh.gate_offset['A']) == pytest.approx(0.9)
    assert all(torch.equal(a, b) for a, b in zip(fresh.operators.parameters(), operators))
    torch.save({'reader': {'operators.A.x': torch.zeros(1)}}, tmp_path / 'bad.pt')
    with pytest.raises(ValueError):
        l1.load_init_reader(fresh, tmp_path / 'bad.pt')


def test_inbatch_negatives_stay_in_their_kb(tmp_path):
    import random
    ctx, _ = context(tmp_path)
    other = make_kb(tmp_path, 'other', 'secret', records=(('r1', 1), ('r2', 1)))
    ctx = l1.Context(ctx.frozen, ctx.reader, {'ds': ctx.kbs['ds'], 'secret': other})
    a = episode(IDS, [3, 10], [6, 13], records=(['r1'], ['r2']))
    b = episode(IDS, [3, 10], [6, 13], records=(['r3'], ['r1']))
    c = episode(IDS, [3, 10], [6, 13], kb='secret', records=(['r1'], ['r2']))
    pool = l1.batch_negatives(ctx, [a, b, c], 16, random.Random(0))
    assert set(pool) == {'ds', 'secret'}
    assert all(d == 'ds' for refs in pool['ds'].values() for d, _ in refs)
    assert all(d == 'secret' for refs in pool['secret'].values() for d, _ in refs)
    kb = ctx.kbs['ds']
    with pytest.raises(PermissionError):      # the reader refuses another KB's negatives
        ctx.reader.read(torch.randn(HIDDEN), [kb], ['ds'], 3, ItemCache(train=False),
                        targets=targets_of(kb, ['r1']), negatives=pool['secret'])
    # negatives the search did not score enter the loss and get gradients; a negative
    # later than the query time never does
    one = {'A': 1, 'B': 1, 'C': 1, 'D': 1}
    r = reader(candidates=one, keep=one)
    state = torch.randn(HIDDEN)
    plain = r.read(state, [kb], ['ds'], 3, ItemCache(train=False), targets=targets_of(kb, ['r1']))
    cache = ItemCache(train=True)
    negs = targets_of(kb, ['r2', 'r3', 'r4'])
    read = r.read(state, [kb], ['ds'], 3, cache, targets=targets_of(kb, ['r1']), negatives=negs)
    assert not torch.equal(read.aux, plain.aux)
    read.aux.backward()
    late = targets_of(kb, ['r4'])
    for s in SPACES:
        for d, i in negs[s]:
            value = cache.values.get((d, s, i))
            if (d, i) in late[s]:
                assert value is None or value.grad is None
            elif (d, i) not in read.spaces[s].scored:
                assert value.grad.abs().sum() > 0


def test_context_neutral_and_batch_negatives_skip_a_slots_neutral(tmp_path):
    import random
    ctx, _ = context(tmp_path)
    a = episode(IDS, [3, 10], [6, 13], records=(['r1'], ['r2']))
    a.slots[0]['alternatives'] = ['r1', 'r3']
    a.slots[0]['neutral'] = ['r1', 'r2', 'r3', 'r4']   # positives never neutral
    b = episode(IDS, [3, 10], [6, 13], records=(['r3'], ['r1']))
    kb = ctx.kbs['ds']
    assert ctx.neutral(a, 0) == targets_of(kb, ['r2', 'r4'])
    assert ctx.neutral(a, 1) is None and ctx.neutral(b, 0) is None
    # the batch pool offers r2 (a's slot 1, b's slots) to a's slot 0, which drops it
    pool = l1.batch_negatives(ctx, [a, b], 16, random.Random(0))
    assert all(set(targets_of(kb, ['r2'])[s]) <= set(pool['ds'][s]) for s in SPACES)
    one = {'A': 1, 'B': 1, 'C': 1, 'D': 1}
    ctx.reader = reader(candidates=one, keep=one)
    state = torch.randn(HIDDEN)
    kw = dict(targets=ctx.targets(a, 0), alternatives=ctx.alternatives(a, 0))
    got = ctx.reader.read(state, [kb], ['ds'], 3, ItemCache(train=False),
                          negatives=pool['ds'], neutral=ctx.neutral(a, 0), **kw)
    kept = {s: [p for p in refs if p not in set(targets_of(kb, ['r2', 'r4'])[s])]
            for s, refs in pool['ds'].items()}
    want = ctx.reader.read(state, [kb], ['ds'], 3, ItemCache(train=False),
                           negatives=kept, neutral=ctx.neutral(a, 0), **kw)
    assert torch.equal(got.aux, want.aux)


def write_context(tmp_path, live=True):
    ctx, _ = context(tmp_path)
    other = make_kb(tmp_path, 'other', 'secret', records=(('r1', 1), ('r2', 1)))
    kbs = {'ds': ctx.kbs['ds'], 'secret': other}
    if live:
        for kb in kbs.values():
            kb.load_live()
    ctx = l1.Context(ctx.frozen, ctx.reader, kbs)
    model = writer_model(ctx.frozen.lm)
    writer = producer.Writer(model, ctx.reader.stack, ctx.frozen.embed, batch=2)
    return ctx, model, writer


def write_episode(name, qt=3, kb='ds', records=(['r1'], ['r2'])):
    ep = episode(IDS_W, [3, 10], [6, 13], kb=kb, records=records, qt=qt)
    ep.episode_id, ep.writes = name, [20]
    ep.write_sites = [{'teacher_text': f'what {name} learned about the task'}]
    return ep


def test_writes_enter_their_kb_for_later_steps_only(tmp_path):
    ctx, model, writer = write_context(tmp_path)
    log = producer.WriteLog(tmp_path / 'writes')
    args = train_args()
    opt = torch.optim.AdamW(ctx.reader.trainable(), lr=1e-3)
    e1, e2 = write_episode('e1'), episode(IDS, [3, 10], [6, 13])
    e2.episode_id = 'e2'
    sizes = {n: {s: len(kb._row_ids[s]) for s in SPACES} for n, kb in ctx.kbs.items()}
    first = l1.train_step(ctx, [e1, e2], opt, args, 0, writer=writer, log=log,
                          text_ids=model.text_ids)
    assert first['writes'] == 1
    assert all(first.get(f'written_{s}', 0) == 0 for s in SPACES)   # nothing in the same step
    kb = ctx.kbs['ds']
    for s in SPACES:
        assert len(kb._row_ids[s]) == sizes['ds'][s] + 1
        assert len(ctx.kbs['secret']._row_ids[s]) == sizes['secret'][s]   # never another KB
        item = kb.read(s, [l1.write_item_id('e1', 0, s)])[0]
        assert item.provenance.producer == 'write' and item.time == 3
        assert item.provenance.sources == ('write:e1#0',) and item.provenance.step == 1
    # a later step: another episode retrieves the write (D scores every visible item);
    # the writing episode never retrieves its own write; an earlier query time neither
    second = l1.train_step(ctx, [e2], opt, args, 1, writer=writer, log=log,
                           text_ids=model.text_ids)
    assert second['written_D'] == 1.0 and 'writes' not in second
    cache = ItemCache(train=False)
    with torch.no_grad():
        _, _, reads, _ = l1.run_episode(ctx, e1, cache)
        own = {(d, i) for rd in reads for info in rd.spaces.values() for d, i in info.scored}
        assert not own & {('ds', l1.write_item_id('e1', 0, s)) for s in SPACES}
        early = episode(IDS, [3, 10], [6, 13], qt=2)
        _, _, reads, _ = l1.run_episode(ctx, early, cache)
        seen = {i for rd in reads for info in rd.spaces.values() for _, i in info.scored}
        assert not seen & {l1.write_item_id('e1', 0, s) for s in SPACES}
    # a revisit supersedes the site's items (same ids, new version)
    l1.train_step(ctx, [e1], opt, args, 2, writer=writer, log=log, text_ids=model.text_ids)
    for s in SPACES:
        assert kb.read(s, [l1.write_item_id('e1', 0, s)])[0].version == 2
        assert len(current_ids(kb, s)) == sizes['ds'][s] + 1
    assert set(log.index) == {'write:e1#0'} and log.index['write:e1#0']['step'] == 3
    log.truncate(1)
    assert log.index['write:e1#0']['step'] == 1
    assert producer.WriteLog(tmp_path / 'writes').index['write:e1#0']['step'] == 1


def test_write_prefix_is_causal(tmp_path):
    ctx, model, writer = write_context(tmp_path, live=False)
    ep = write_episode('e1')
    with torch.no_grad():
        _, _, _, spans = l1.run_episode(ctx, ep, ItemCache(train=False))
    req, = l1.write_requests(ep, spans, model.text_ids, 0)
    assert req.prefix_ids.tolist() == IDS_W[:20] and req.mems == [6, 13]
    assert all(torch.equal(a, b) for a, b in zip(req.reads, spans))
    x = writer.inputs(req.prefix_ids, req.mems, req.reads)
    assert x.shape[0] == 20 + sum(s.shape[0] for s in spans)
    changed = list(IDS_W)
    changed[21:] = [BG_END, 40]                      # after the site: no effect on the span
    ep2 = write_episode('e1')
    ep2.ids = torch.tensor(changed)
    with torch.no_grad():
        _, _, _, spans2 = l1.run_episode(ctx, ep2, ItemCache(train=False))
    req2, = l1.write_requests(ep2, spans2, model.text_ids, 0)
    a, b = l1.generate_writes(writer, [req, req2])
    assert torch.equal(a, b) and a.shape[0] == req.count


def codec_bank(tmp_path, ctx, model, records):
    """A bank as ``l1 build`` makes it: the writer's spans of the records (span cache,
    bf16), the codecs' items per space, keys from the item-key heads."""
    from schnitz.kb.bank import SpanCache
    cache = SpanCache.build(tmp_path / 'spans' / 'ds', model,
                            {r: text for r, (text, _) in records.items()}, 's0', batch_size=2)
    kb = KnowledgeBase.create(tmp_path / 'bank', name='bank', dataset='ds', spaces=SPACES)
    stack = ctx.reader.stack
    per_space = {s: [] for s in SPACES}
    with torch.no_grad():
        for r, (_, t) in records.items():
            items = stack.encode(cache.get(r).float())
            for s in SPACES:
                values = items[s].float()
                per_space[s].append(NewItem(values, ctx.reader.keys.item_key(s, values),
                                            Provenance((r,), 'codec'), 1.0, t))
    for s, items in per_space.items():
        kb.append(s, items)
        kb.enable_live(s)
    kb.load_live()
    return kb, cache


RECORDS = {'r1': ('the first record text', 1), 'r2': ('another record, longer text', 1),
           'r3': ('third', 1)}


def l1_producers(ctx, writer, texts, caches, log, mode='free', batch=1):
    """The shared replay as ``l1.load_producers`` configures it for L1b."""
    return producer.Producers(writer, origin=ctx.origin, texts=texts, caches=caches, level=0,
                              log=log, mode=mode, batch=batch,
                              stored=lambda d, s, i: ctx.kbs[d].read(s, [i])[0].values)


def producer_setup(tmp_path, mode='free'):
    ctx, model, writer = write_context(tmp_path, live=False)
    kb, cache = codec_bank(tmp_path, ctx, model, RECORDS)
    ctx = l1.Context(ctx.frozen, ctx.reader, {'ds': kb})
    writer = producer.Writer(model, ctx.reader.stack, ctx.frozen.embed, batch=2)
    log = producer.WriteLog(None)
    producers = l1_producers(ctx, writer, {r: t for r, (t, _) in RECORDS.items()},
                             {'ds': cache}, log, mode, batch=2)
    return ctx, model, writer, producers, log


def test_l1b_recompute_equals_stored_items_and_reaches_producers(tmp_path):
    ctx, model, writer, producers, log = producer_setup(tmp_path)
    kb = ctx.kbs['ds']
    sets = l1.parameter_sets(ctx.reader, model)
    l1.set_phase(sets, ('writer', 'codecs', 'keys', 'operators', 'recombiner'))
    producers.begin()
    for s in SPACES:
        ids = current_ids(kb, s)
        got = producers.values(s, [('ds', i) for i in ids])
        for value, item in zip(got, kb.read(s, ids)):
            assert torch.equal(value, item.values.float())   # the stored bf16 payload exactly
    loss = sum((v.float() ** 2).mean() for leaves in producers.leaves.values()
               for v in leaves.values())
    loss.backward()
    stats = producers.backward()
    assert stats['match_exact'] == 1.0 and stats['match_rel'] == 0.0 and stats['drift'] == 0.0
    assert stats['backward_sources'] == len(RECORDS)
    assert model.writer.rep.net[1].weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.writer.ratio.parameters())
    for s in SPACES:
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in ctx.reader.stack.codecs[s].parameters())
    # written items: recomputed from the write log, they equal their stored payload too
    ep = write_episode('e9', records=(['r1'], ['r2']))
    with torch.no_grad():
        _, _, _, spans = l1.run_episode(ctx, ep, ItemCache(train=False))
    requests = l1.write_requests(ep, spans, model.text_ids, 0)
    requests += l1.write_requests(write_episode('e8'), spans, model.text_ids, 0)
    l1.commit_writes(ctx, writer, requests, l1.generate_writes(writer, requests), 1, log)
    log.flush(1)
    assert log.index['write:e9#0']['group'] == ['write:e9#0', 'write:e8#0']
    producers.begin()
    for s in SPACES:
        item_id = l1.write_item_id('e9', 0, s)
        value, = producers.values(s, [('ds', item_id)])
        assert torch.equal(value, kb.read(s, [item_id])[0].values.float())
    # replayed in the generation's batch composition (one unit for both writes)
    assert len(producers.unit_of[(('write', 'ds', 'write:e9#0'), 1)]) == 2
    # teacher-fed replay: one pass on the stored span, close to the stored item
    teacher = l1_producers(ctx, writer, producers.texts, producers.caches, log, 'teacher')
    teacher.begin()
    ids = current_ids(kb, 'D')
    for value, item in zip(teacher.values('D', [('ds', i) for i in ids]), kb.read('D', ids)):
        torch.testing.assert_close(value, item.values.float(), atol=0.05, rtol=0.05)


def test_l1b_read_uses_recomputed_items(tmp_path):
    ctx, model, writer, producers, _ = producer_setup(tmp_path)
    kb = ctx.kbs['ds']
    producers.begin()
    cache = ItemCache(train=True)
    read = ctx.reader.read(torch.randn(HIDDEN), [kb], ['ds'], 3, cache, producer=producers)
    assert all(info.recomputed == len(info.refs) for info in read.spaces.values())
    read.span.square().sum().backward()
    for s, info in read.spaces.items():   # the gradient goes to the recomputed leaves ...
        for d, i in info.refs:
            key = producers.source(d, s, i)
            assert producers.leaves[(key, 1)][s].grad.abs().sum() > 0
    l1.set_phase(l1.parameter_sets(ctx.reader, model), ('writer', 'codecs'))
    producers.backward()                  # ... and on into the producers
    assert model.writer.rep.net[1].weight.grad.abs().sum() > 0


def test_phase_schedule_and_l1b_step(tmp_path):
    schedule = l1.parse_schedule('a:2,b:1', 'l1a')
    assert [l1.phase_at(schedule, i) for i in range(7)] == \
        ['l1a', 'l1a', 'l1b', 'l1a', 'l1a', 'l1b', 'l1a']
    assert l1.parse_schedule(None, 'l1b') == [('l1b', 1)]
    with pytest.raises(ValueError):
        l1.parse_schedule('a:2,c:1', 'l1a')
    ctx, model, writer, producers, log = producer_setup(tmp_path)
    kb = ctx.kbs['ds']
    sets = l1.parameter_sets(ctx.reader, model)
    params = [p for ps in sets.values() for p in ps]
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=1e-3)
    ep = episode(IDS, [3, 10], [6, 13], records=(['r1'], ['r2']))
    args = train_args()
    rep = model.writer.rep.net[1].weight

    def live(s):
        return [x.values.clone() for x in kb.read(s, current_ids(kb, s), live=True)]
    before_rep, before_live = rep.detach().clone(), {s: live(s) for s in SPACES}
    trainable = l1.set_phase(sets, l1.L1A_SET)
    out = l1.train_step(ctx, [ep], opt, args, 0, phase='l1a', trainable=trainable)
    assert out['phase'] == 'l1a' and out['items'] > 0 and torch.equal(rep, before_rep)
    after_a = {s: live(s) for s in SPACES}
    assert any(not torch.equal(a, b) for s in SPACES for a, b in zip(after_a[s], before_live[s]))
    trainable = l1.set_phase(sets, ('writer', 'codecs', 'keys', 'operators', 'recombiner'))
    out = l1.train_step(ctx, [ep], opt, args, 1, phase='l1b', trainable=trainable,
                        producers=producers)
    assert out['phase'] == 'l1b' and out['items'] == 0 and out['l1b']['backward_sources'] > 0
    assert not torch.equal(rep, before_rep)                     # the writer trained
    assert out['l1b']['change_rel'] > 0 and out['l1b']['change_units'] > 0   # logged
    assert all(torch.equal(a, b) for s in SPACES for a, b in zip(live(s), after_a[s]))
    with pytest.raises(ValueError):
        l1.train_step(ctx, [ep], opt, args, 2, phase='l1b', trainable=trainable)


@pytest.mark.skipif(not TOKENIZER.exists(), reason='needs the LFM2.5 tokenizer')
def test_layout_with_write_sites_matches_the_site_prefix():
    from transformers import AutoTokenizer
    from schnitz.memory_transcripts import write_site_prefix
    from schnitz.span_tokens import MEMORY_TOOLS
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    search = {'type': 'function', 'function': {'name': 'memory_search', 'arguments': {}}}
    write = {'type': 'function', 'function': {'name': 'memory_write', 'arguments': {}}}
    slot = {'slot': {'kb': 'ds', 'record_ids': ['a']}}
    row = {'episode_id': 'e', 'kb': 'ds', 'provenance': {'source_query_time': 2}, 'messages': [
        {'role': 'user', 'content': 'q'},
        {'role': 'assistant', 'content': '', 'tool_calls': [search]},
        {'role': 'tool', 'name': 'memory_search', 'content': slot},
        {'role': 'assistant', 'content': 'SELECT 1'},
        {'role': 'assistant', 'content': '', 'tool_calls': [write],
         'write_span': {'kb': 'ds', 'write_site': 0}},
        {'role': 'tool', 'name': 'memory_write', 'content': {'write_result': {'kb': 'ds'}}}],
        'tools': list(MEMORY_TOOLS),       # v3 rows list the memory tools (as L1 renders)
        'write_sites': [{'message': 4, 'call': 0, 'teacher_text': 'secret'}]}
    ep = l1.layout(row, tok, writes=True)
    assert len(ep.writes) == 1 and int(ep.ids[ep.writes[0] + 1]) == BG_END
    assert ep.writes[0] not in ep.targets.tolist() and ep.writes[0] + 1 not in ep.targets.tolist()
    assert ep.ids[:ep.writes[0]].tolist() == write_site_prefix(tok, row, 0)
    assert 'secret' not in tok.decode(ep.ids)
    plain = l1.layout(row, tok)
    assert plain.writes == [] and BG not in plain.ids.tolist()
    assert '[memory_write()]' not in tok.decode(plain.ids)
    text = l1.layout(row, tok, {'a': 'alpha'}, writes=True)
    assert text.targets.numel() == ep.targets.numel()


def test_gates_carry_stored_mass_and_caller_weights(tmp_path):
    """Invariant 5: an item's gate is multiplied by its stored mass (and by the caller's
    per-item weight), so its share of the read mass and read_count follow."""
    kb = make_kb(tmp_path, records=(('r1', 1),))
    half = KnowledgeBase.create(tmp_path / 'half', name='half', dataset='ds', spaces=SPACES)
    for s in SPACES:
        item = kb.read(s, kb._row_ids[s])[0]
        half.append(s, [NewItem(item.values, item.key, item.provenance, 0.5, item.time)])
    r = reader()
    state = torch.randn(HIDDEN)
    one = r.read(state, [kb], ['ds'], 3, ItemCache(train=False), targets=targets_of(kb, ['r1']),
                 gold=True)
    masses = r.read(state, [half], ['ds'], 3, ItemCache(train=False),
                    targets=targets_of(half, ['r1']), gold=True)
    weighted = r.read(state, [kb], ['ds'], 3, ItemCache(train=False),
                      targets=targets_of(kb, ['r1']), gold=True,
                      weights={i: 0.5 for s in SPACES for i in kb._row_ids[s]})
    for s in SPACES:
        assert masses.spaces[s].mass == pytest.approx(0.5 * one.spaces[s].mass, rel=1e-6)
        assert weighted.spaces[s].mass == pytest.approx(masses.spaces[s].mass, rel=1e-6)
    torch.testing.assert_close(masses.span, weighted.span)


def test_search_exclude_masks_items_before_top_k(tmp_path):
    kb = make_kb(tmp_path, records=tuple((f'r{i}', 1) for i in range(6)))
    q = torch.randn(1, SPACES['A'].key_width)
    full = kb.search('A', q, 3)
    gone = set(full.ids[0][:2])
    hits = kb.search('A', q, 3, exclude=gone | {'not-an-item'})
    assert len(hits.ids[0]) == 3 and not gone & set(hits.ids[0])
    assert hits.ids[0] == [i for i in kb.search('A', q, 5).ids[0] if i not in gone][:3]


def test_context_rows_follow_appends(tmp_path):
    ctx, _ = context(tmp_path)
    kb = ctx.kbs['ds']
    before = dict(ctx.rows('ds', 'A'))
    item = kb.read('A', kb._row_ids['A'][:1])[0]
    new, = kb.append('A', [NewItem(item.values, item.key, Provenance(('r9',), 'codec'), 1.0, 1)])
    rows = ctx.rows('ds', 'A')
    assert rows[new] == len(before) and all(rows[k] == v for k, v in before.items())


def test_l1b_producers_have_their_own_learning_rates(tmp_path):
    from types import SimpleNamespace
    ctx, model, _ = write_context(tmp_path, live=False)
    sets = l1.parameter_sets(ctx.reader, model)
    args = SimpleNamespace(lr=3e-4, l1b_codec_lr=3e-5, l1b_writer_lr=3e-6)
    reader_group, codecs, writer = l1.optimizer_groups(sets, args)
    assert (reader_group['lr'], codecs['lr'], writer['lr']) == (3e-4, 3e-5, 3e-6)
    own = {id(p) for p in codecs['params'] + writer['params']}
    assert codecs['params'] == list(ctx.reader.stack.codecs.parameters())
    assert writer['params'] == list(model.writer.parameters())
    assert not own & {id(p) for p in reader_group['params']}
    assert len(reader_group['params']) + len(own) == sum(len(v) for v in sets.values())


def test_ste_round_is_exactly_the_serialized_value():
    x = (torch.randn(4096) * 3).requires_grad_()
    y = producer.ste_round(x)
    assert torch.equal(y, x.detach().bfloat16().float())        # bitwise, not up to an ulp
    y.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))



# -- content contrast, null prefix, reader decoder layers ------------------------------
@pytest.mark.parametrize('combine', ['s_s', 'r'])
def test_contrast_is_zero_for_the_identity_permutation(tmp_path, combine):
    """Reading an episode's own items at its own gates through the contrast arm gives the
    retrieved NLL exactly, so the term (margin 0) and its gradient vanish."""
    ctx, _ = context(tmp_path, read_combine=combine)
    ep = episode(IDS, [3, 10], [6, 13])
    cache = ItemCache(train=True)
    nll, _, reads, _ = l1.run_episode(ctx, ep, cache)
    term, stats = l1.contrast_term(ctx, ep, nll, reads, cache, train_args(contrast_margin=0.0))
    assert float(term) == 0.0 and stats['content_nats_train'] == 0.0
    term.backward()
    assert all(p.grad is None or not p.grad.any() for p in ctx.reader.parameters())


@pytest.mark.parametrize('combine', ['s_s', 'r'])
def test_contrast_arm_matches_counts_and_gates_and_trains_r_not_donor_items(tmp_path, combine):
    ctx, _ = context(tmp_path, read_combine=combine)
    ep = episode(IDS, [3, 10], [6, 13], records=(['r1'], ['r2']))
    other = episode([2, 8, 7, 11, 12, 13, MEM, MEM_END, 5, 4, 16, 17, 18, MEM, MEM_END, 3, 20,
                     21, 22, 23], [3, 10], [6, 13], records=(['r3'], ['r1']))
    donor_cache = ItemCache(train=True)
    _, _, donor_reads, _ = l1.run_episode(ctx, other, donor_cache)
    cache = ItemCache(train=True)
    shuffled, _, reads, _ = l1.run_episode(ctx, ep, cache, 'donor', donors=donor_reads)
    assert len(reads) == len(donor_reads)
    for mine, theirs in zip(reads, donor_reads):     # same items, gates and span length
        assert mine.n == theirs.n and mine.span.shape == theirs.span.shape
        for s in SPACES:
            assert mine.spaces[s].refs == theirs.spaces[s].refs
            torch.testing.assert_close(mine.spaces[s].gates, theirs.spaces[s].gates,
                                       rtol=0, atol=0)
    shuffled.backward()
    assert ctx.reader.recombiner.head[1].weight.grad.abs().sum() > 0      # R trains
    # the donor's item payloads are detached in the contrast arm
    assert all(v.grad is None for v in donor_cache.values.values())
    assert all(v.grad is None for v in cache.values.values())
    # the full term: gradients reach R through both arms (margin keeps it active)
    ctx.reader.zero_grad(set_to_none=True)
    cache = ItemCache(train=True)
    nll, _, _, _ = l1.run_episode(ctx, ep, cache)
    term, stats = l1.contrast_term(ctx, ep, nll, donor_reads, cache,
                                   train_args(contrast_margin=10.0))
    assert float(term) > 0 and stats['content_nats_train'] != 0.0
    term.backward()
    assert ctx.reader.recombiner.head[1].weight.grad.abs().sum() > 0
    assert any(v.grad is not None and v.grad.abs().sum() > 0 for v in cache.values.values())


def test_contrast_donors_are_a_derangement_within_each_kb():
    eps = [episode(IDS, [3, 10], [6, 13], kb=kb) for kb in ('a', 'b', 'a', 'a', 'c')]
    donors = l1.contrast_donors(eps)
    assert donors == [3, None, 0, 2, None]
    assert all(d is None or (d != i and eps[d].kb == eps[i].kb) for i, d in enumerate(donors))


def test_train_step_logs_the_contrast(tmp_path):
    ctx, _ = context(tmp_path, read_combine='r')
    a = episode(IDS, [3, 10], [6, 13], records=(['r1'], ['r2']))
    b = episode(IDS, [3, 10], [6, 13], records=(['r3'], ['r1']))
    b.episode_id = 'f'
    sets = l1.parameter_sets(ctx.reader)
    trainable = l1.set_phase(sets, l1.L1A_SET)
    opt = torch.optim.AdamW(trainable, lr=1e-3)
    out = l1.train_step(ctx, [a, b], opt, train_args(contrast_weight=1.0, contrast_margin=0.5),
                        0, trainable=trainable)
    assert out['contrast_episodes'] == 2 and 'content_nats_train' in out and 'contrast' in out
    out = l1.train_step(ctx, [a], opt, train_args(contrast_weight=1.0), 1, trainable=trainable)
    assert 'contrast' not in out            # no same-KB donor in the batch: no term


def test_gold_reads_read_the_slots_items_and_anneal(tmp_path):
    ctx, _ = context(tmp_path, read_combine='r')
    a = episode(IDS, [3, 10], [6, 13], records=(['r1'], ['r2']))
    sets = l1.parameter_sets(ctx.reader)
    trainable = l1.set_phase(sets, l1.L1A_SET)
    opt = torch.optim.AdamW(trainable, lr=1e-3)
    seen = []
    real = l1.run_episode

    def spy(ctx_, ep, cache, mode='retrieve', **kw):
        seen.append(mode)
        return real(ctx_, ep, cache, mode, **kw)

    l1.run_episode = spy
    try:
        out = l1.train_step(ctx, [a], opt, train_args(gold_reads=1.0), 0, trainable=trainable)
        assert seen == ['gold'] and out['gold_episodes'] == 1
        seen.clear()
        l1.train_step(ctx, [a], opt, train_args(gold_reads=1.0, gold_anneal=10), 10,
                      trainable=trainable)
        assert seen == ['retrieve']
    finally:
        l1.run_episode = real
    assert l1.gold_read_rate(train_args(gold_reads=0.8, gold_anneal=100), 50) == 0.4


def test_null_prefix_sits_in_front_of_every_span_and_trains(tmp_path):
    ctx, _ = context(tmp_path, read_combine='r')
    prefix = l1.NullPrefix(['ds'], 3, ctx.reader.stack)
    ctx = l1.Context(ctx.frozen, ctx.reader, ctx.kbs, prefix=prefix)
    ep = episode(IDS, [3, 10], [6, 13])
    empty = [torch.zeros(0, HIDDEN)] * 2
    nll, _, _, spans = l1.run_episode(ctx, ep, ItemCache(train=False), 'fixed', empty)
    assert all(s.shape[0] == 0 for s in spans)       # the returned spans are the reads'
    nll.backward()                                     # the no-memory arm carries it too
    assert prefix.z['ds'].grad.abs().sum() > 0
    assert [s.shape[0] for s in ctx.spliced(ep, empty)] == [3, 3]
    args = train_args(null_prefix=3, decoder_train_below=0)
    assert 'prefix' in l1.phase_set('l1a', args) and 'prefix' in l1.phase_set('r', args)
    assert 'prefix' not in l1.phase_set('l1b', args, ('keys',))
    sets = l1.parameter_sets(ctx.reader, prefix=prefix)
    assert sets['prefix'] == [prefix.z['ds']]


def test_decoder_train_below_zero_is_the_frozen_reader(tmp_path):
    ctx, _ = context(tmp_path)
    assert ctx.frozen.parent is None and len(ctx.frozen.own) == 0 and not ctx.frozen.trains
    sets = l1.parameter_sets(ctx.reader, decoder=ctx.frozen)
    assert sets['decoder'] == []
    args = train_args(decoder_train_below=0, lr=1e-3, l1b_codec_lr=1e-4, l1b_writer_lr=1e-5)
    assert 'decoder' not in l1.phase_set('l1a', args)
    assert l1.decoder_replay_kl(args) == 0.0
    sets.update(codecs=[], writer=[])
    assert len(l1.optimizer_groups(sets, args)) == 3


def _reader_decoder_context(tmp_path, below=2, checkpointing=False):
    lm, _ = tiny_lm()
    kb = make_kb(tmp_path, records=(('r1', 1), ('r2', 1), ('r3', 1), ('r4', 5)))
    frozen = l1.Frozen(lm, 3, train_below=below, checkpointing=checkpointing)
    for p in frozen.own.parameters():
        p.requires_grad_(True)
    return l1.Context(frozen, reader(), {'ds': kb}), lm


def test_reader_decoder_starts_as_its_parent(tmp_path):
    ctx, lm = _reader_decoder_context(tmp_path)
    x = ctx.frozen.embed(torch.tensor(IDS))[None]
    with torch.no_grad():
        mine = ctx.frozen.logits(ctx.frozen.final(x))
        parent = ctx.frozen.parent.logits(ctx.frozen.parent.final(x))
        mid, parent_mid = ctx.frozen.mid(x), ctx.frozen.parent.mid(x)
    torch.testing.assert_close(mine, parent, rtol=0, atol=0)
    torch.testing.assert_close(mid, parent_mid, rtol=0, atol=0)
    assert ctx.frozen.lm.model.layers is not lm.model.layers
    assert ctx.frozen.lm.model.layers[2] is lm.model.layers[2]           # shared, frozen
    assert ctx.frozen.lm.model.layers[0] is not lm.model.layers[0]       # own copy
    assert ctx.frozen.lm.lm_head is lm.lm_head
    kl = l1.parent_kl(ctx, episode(IDS, [3, 10], [6, 13]))
    assert abs(float(kl)) < 1e-6
    args = train_args(decoder_train_below=2)
    assert l1.decoder_replay_kl(args) == 0.1
    assert 'decoder' in l1.phase_set('l1a', args) and 'decoder' in l1.phase_set('r', args)


@pytest.mark.parametrize('checkpointing', [False, True])
def test_only_reader_layers_below_n_get_gradients(tmp_path, checkpointing):
    ctx, lm = _reader_decoder_context(tmp_path, checkpointing=checkpointing)
    ep = episode(IDS, [3, 10], [6, 13])
    nll, n, _, _ = l1.run_episode(ctx, ep, ItemCache(train=True))
    (nll / n).backward()
    own = list(ctx.frozen.own.parameters())
    assert own and all(p.grad is not None for p in own)
    assert any(p.grad.abs().sum() > 0 for p in ctx.frozen.own[0].parameters())
    assert all(p.grad is None for p in lm.parameters())       # parent: embeddings, layers
    assert all(p.grad is None for layer in ctx.frozen.lm.model.layers[2:]
               for p in layer.parameters())                   # the frozen top, LM head
    grads = [p.grad.clone() for p in own]
    if checkpointing:     # recomputation gives the same gradients as stored activations
        plain, _ = _reader_decoder_context(tmp_path / 'plain')
        nll2, n2, _, _ = l1.run_episode(plain, ep, ItemCache(train=True))
        (nll2 / n2).backward()
        for a, b in zip(grads, [p.grad for p in plain.frozen.own.parameters()]):
            torch.testing.assert_close(a, b)
    # the parent-preservation KL reaches the own layers only
    ctx.frozen.own.zero_grad(set_to_none=True)
    with torch.no_grad():
        for p in ctx.frozen.own.parameters():
            p.add_(0.01 * torch.randn_like(p))
    kl = l1.parent_kl(ctx, ep)
    assert float(kl) > 0
    kl.backward()
    assert all(p.grad is not None for p in ctx.frozen.own.parameters())
    assert all(p.grad is None for p in lm.parameters())
