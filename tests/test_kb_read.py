"""L1 read path (``schnitz.kb.read``) and trainer passes (``schnitz.kb.stages.l1``);
CPU, tiny models, no downloads."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from schnitz.kb.read import ItemCache, L1Reader, ReadConfig, splice
from schnitz.kb.stages import l1
from schnitz.kb_store import KnowledgeBase, NewItem, Provenance, SpaceSpec
from schnitz.span_protocol import ProtocolTokens, untie
from schnitz.span_tokens import SPAN_TOKENS


SPACES = {'A': SpaceSpec(8, 4, 1.0), 'B': SpaceSpec(12, 4, 0.5)}
HIDDEN = 32
MEM, MEM_END = SPAN_TOKENS['mem'][1], SPAN_TOKENS['mem_end'][1]


def config(**kw):
    kw = {'candidates': {'A': 3, 'B': 4}, **kw}
    return ReadConfig(spaces=SPACES, hidden=HIDDEN, span_width=HIDDEN, target_norm=1.0,
                      state=16, op_hidden=12, layers=2, **kw)


def reader(**kw):
    torch.manual_seed(0)
    return L1Reader(config(**kw))


def make_kb(tmp_path, name='kb', dataset='ds', records=(('r1', 1), ('r2', 1), ('r3', 5)),
            live=True):
    kb = KnowledgeBase.create(tmp_path / name, name=name, dataset=dataset, spaces=SPACES)
    for k, (s, spec_) in enumerate(SPACES.items()):
        items = []
        for i, (r, t) in enumerate(records):     # content depends on (space, record) only
            gen = torch.Generator().manual_seed(1000 * k + int(r[1:]))
            items.append(NewItem(torch.randn(2 + i, spec_.width, generator=gen),
                                 torch.randn(spec_.key_width, generator=gen),
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
        assert len(info.refs) == 2 and not set(info.refs) & set(late[s])
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


def test_gate_zero_space_is_removed_exactly():
    r = reader()
    reads = {'A': torch.randn(3, 8), 'B': torch.randn(2, 12)}
    alone, _ = r.recombiner([('A', reads['A'], 1.0)], 5)
    gated, _ = r.recombiner([('A', reads['A'], 1.0), ('B', reads['B'], 0.0)], 5)
    torch.testing.assert_close(alone, gated, atol=1e-6, rtol=1e-6)
    # scaling every gate together changes nothing but the mass
    items = [torch.randn(3, 8), torch.randn(2, 8)]
    cond = torch.randn(1, 4)
    x, m1 = r.operators['A']([('A', items[0], 0.2), ('A', items[1], 0.6)], 4, cond=cond)
    y, m2 = r.operators['A']([('A', items[0], 0.4), ('A', items[1], 1.2)], 4, cond=cond)
    torch.testing.assert_close(x, y, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(m2, 2 * m1)


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
    assert {d for info in both.spaces.values() for d, _ in info.refs} <= {'ds', 'secret'}


def test_gradients_reach_live_items_keys_and_key_heads(tmp_path):
    kb = make_kb(tmp_path)
    r = reader()
    cache = ItemCache(train=True)
    before_values = {s: [i.values.clone() for i in kb.read(s, kb._row_ids[s], live=True)]
                     for s in SPACES}
    before_keys = {s: [i.key.clone() for i in kb.read(s, kb._row_ids[s], live=True)]
                   for s in SPACES}
    read = r.read(torch.randn(HIDDEN), [kb], ['ds'], 3, cache, targets=targets_of(kb, ['r1']))
    assert read.aux is not None and read.span.shape[0] >= 1
    (read.span.square().sum() + read.aux).backward()
    assert all(r.keys.heads[s].weight.grad.abs().sum() > 0 for s in SPACES)
    assert all(r.operators[s].layers[0].out.weight.grad.abs().sum() > 0 for s in SPACES)
    assert r.recombiner.head[1].weight.grad.abs().sum() > 0
    assert all(v.grad is not None and v.grad.abs().sum() > 0 for v in cache.values.values())
    assert all(k.grad is not None and k.grad.abs().sum() > 0 for k in cache.keys.values())
    counts = cache.apply(item_lr=0.01, key_lr=0.01)
    assert counts == {'items': 4, 'keys': 4}      # two visible items in each of two spaces
    for s in SPACES:
        after = kb.read(s, kb._row_ids[s], live=True)
        changed = [not torch.equal(a.values, b) for a, b in zip(after, before_values[s])]
        keys = [not torch.equal(a.key, b) for a, b in zip(after, before_keys[s])]
        assert changed == [True, True, False] and keys == [True, True, False]  # r3 is in the future
        # the frozen bf16 payload is untouched
        stored = kb.read(s, kb._row_ids[s])
        torch.testing.assert_close(stored[0].values.float(), before_values[s][0])


def test_retrieval_loss_scores_missed_targets(tmp_path):
    kb = make_kb(tmp_path, records=tuple((f'r{i}', 1) for i in range(8)))
    r = reader()
    cache = ItemCache(train=True)
    targets = targets_of(kb, ['r0', 'r5', 'r7'])
    read = r.read(torch.randn(HIDDEN), [kb], ['ds'], 3, cache, targets=targets)
    for s, info in read.spaces.items():
        assert info.recall_read == sum(t in set(info.refs) for t in targets[s]) / 3
        assert info.recall >= info.recall_read
        assert len(info.refs) == r.config.keep[s]          # a sparse read
    (read.aux + read.span.sum()).backward()
    for s, info in read.spaces.items():   # scored but not read: no value gradient
        read_ids = {i for _, i in info.refs}
        for (_, space, item_id), v in cache.values.items():
            if space == s and item_id not in read_ids:
                assert v.grad is None or v.grad.abs().sum() == 0
    # every target gets a key gradient from the retrieval loss, retrieved or not
    for s in SPACES:
        for ref in targets[s]:
            assert cache.keys[('ds', s, ref[1])].grad.abs().sum() > 0


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
    changed[11] = 30              # after the second call: must not change either query
    changed[16:] = [40, 41, 42, 43]   # the answer (targets) must not change the reads
    with torch.no_grad():
        _, _, reads, spans = l1.run_episode(ctx, ep, ItemCache(train=False))
        _, _, reads2, spans2 = l1.run_episode(ctx, episode(changed, [3, 10], [6, 13]),
                                              ItemCache(train=False))
        for a, b in zip(spans, spans2):
            torch.testing.assert_close(a, b)
        # a token before the second call changes the second read only
        early = list(IDS)
        early[9] = 31
        _, _, _, spans3 = l1.run_episode(ctx, episode(early, [3, 10], [6, 13]),
                                         ItemCache(train=False))
    torch.testing.assert_close(spans[0], spans3[0])
    assert not torch.equal(spans[1], spans3[1])


def test_later_query_depends_on_earlier_read(tmp_path):
    """The prefix pass of site 2 carries span 1, and its gradient reaches the read that
    produced span 1 (the full graph; nothing is detached)."""
    ctx, _ = context(tmp_path)
    ep = episode(IDS, [3, 10], [6, 13])
    cache = ItemCache(train=True)
    _, _, reads, _ = l1.run_episode(ctx, ep, cache)
    reads[1].aux.backward()           # only the second read's retrieval loss
    # the retrieval loss depends on keys and the query, never directly on item values:
    # a value gradient can only arrive through span 1 -> the prefix pass -> query 2
    first = [('ds', s, i) for s in SPACES for _, i in reads[0].spaces[s].refs]
    assert any(cache.values[r].grad is not None and cache.values[r].grad.abs().sum() > 0
               for r in first)


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
        assert ctx.reader.keys.heads['A'].weight.grad.abs().sum() > 0
        ctx.reader.zero_grad(set_to_none=True)
    torch.testing.assert_close(runs[0][0], runs[1][0], rtol=0, atol=0)
    for a, b in zip(runs[0][1], runs[1][1]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    # the empty-span control reads nothing
    with torch.no_grad():
        empty = [torch.zeros(0, HIDDEN)] * 2
        nll0, n0, reads0, _ = l1.run_episode(ctx, ep, ItemCache(train=False), 'fixed', empty)
    assert reads0 == [] and n0 == n


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


def test_retrieval_loss_is_softmax_over_candidates(tmp_path):
    kb = make_kb(tmp_path, records=tuple((f'r{i}', 1) for i in range(4)))
    r = reader(candidates={'A': 4, 'B': 4})
    cache = ItemCache(train=False)
    state = torch.randn(HIDDEN)
    targets = targets_of(kb, ['r2'])
    read = r.read(state, [kb], ['ds'], 3, cache, targets=targets)
    q = r.keys(state)
    want = []
    for s in SPACES:     # all four items are scored: -log softmax(cos / tau)[target]
        keys = torch.stack([cache.get(kb, s, [i])[0][1] for i in kb._row_ids[s]])
        z = torch.nn.functional.cosine_similarity(q[s][None], keys, dim=-1) / 0.1
        pos = kb._row_ids[s].index(targets[s][0][1])
        want.append(-torch.log_softmax(z, 0)[pos])
        assert len(read.spaces[s].refs) == r.config.keep[s]     # the read stays sparse
    torch.testing.assert_close(read.aux, torch.stack(want).mean())


def test_read_count_is_mass_weighted_length_in_reps_capped(tmp_path):
    kb = make_kb(tmp_path, records=(('r1', 1), ('r2', 1)))   # items 2 and 3 positions long
    targets = targets_of(kb, ['r1', 'r2'])
    read = reader(max_reps=64).read(torch.randn(HIDDEN), [kb], ['ds'], 3,
                                    ItemCache(train=False), targets=targets, gold=True)
    # gates 1: A items stand for 2.5 reps on average (r = 1), B items for 5 (r = 1/2)
    assert read.n == 4          # ceil(mean(2.5, 5.0)), both spaces with mass 2
    capped = reader(max_reps=3).read(torch.randn(HIDDEN), [kb], ['ds'], 3,
                                     ItemCache(train=False), targets=targets, gold=True)
    assert capped.n == 3


def test_retrieval_only_trains_keys_not_values(tmp_path):
    ctx, _ = context(tmp_path)
    ep = episode(IDS, [3, 10], [6, 13])
    cache = ItemCache(train=True)
    nll, _, reads, spans = l1.run_episode(ctx, ep, cache, retrieval_only=True)
    assert nll is None and not any(s.requires_grad for s in spans)
    torch.stack([rd.aux for rd in reads]).mean().backward()
    assert all(v.grad is None for v in cache.values.values())
    assert any(k.grad is not None and k.grad.abs().sum() > 0 for k in cache.keys.values())
    assert ctx.reader.keys.heads['A'].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in ctx.reader.operators.parameters())
