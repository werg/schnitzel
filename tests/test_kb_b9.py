"""B9 round loop (``schnitz.kb.experience``, ``schnitz.kb.stages.b9``); CPU, tiny models,
no downloads."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from schnitz.kb.experience import (GenProtocol, GoldSchedule, KBView, Registry, generate,
                                   read_items, write_prefix, write_source)
from schnitz.kb.read import ItemCache, L1Reader, ReadConfig
from schnitz.kb.stages import b9, l1
from schnitz.kb_store import DEFAULT_SPACES, KnowledgeBase, NewItem, Provenance
from schnitz.span_protocol import ProtocolTokens, untie
from schnitz.span_tokens import SPAN_TOKENS

HIDDEN = 32
SPACES = DEFAULT_SPACES


def reader():
    torch.manual_seed(0)
    return L1Reader(ReadConfig(candidates={'A': 50, 'B': 50, 'C': 50, 'D': 50},
                               keep={'A': 50, 'B': 50, 'C': 50, 'D': 50}, hidden=HIDDEN,
                               span_width=HIDDEN, state=16, op_hidden=12, layers=1,
                               key_hidden=16, checkpointing=False))


def item(space, seed, source, time=1, item_id=None):
    spec = SPACES[space]
    gen = torch.Generator().manual_seed(seed)
    return NewItem(torch.randn(2, spec.width, generator=gen), torch.randn(spec.key_width,
                                                                          generator=gen),
                   Provenance((source,), 'codec'), 1.0, time, item_id)


def corpus_kb(tmp_path, dataset='ds', name='kb'):
    kb = KnowledgeBase.create(tmp_path / name, name=name, dataset=dataset)
    for k, s in enumerate(SPACES):
        kb.append(s, [item(s, 100 * k + i, f'c{i}') for i in range(3)])
    return kb


def write_record(kb, reg, task, record_id, t, seed, split='train', read=()):
    """A round's write as the stage does it: append the first time, then supersede."""
    previous = reg.own.get(task)
    ids = {}
    for k, s in enumerate(SPACES):
        new = item(s, seed * 10 + k, record_id,
                   item_id=reg.records[previous].items[s] if previous else None)
        ids[s] = kb.supersede(s, [new])[0][0] if previous else kb.append(s, [new])[0]
    return reg.add_own(task, record_id, ids, round=t, step=0, split=split, read_items=read)


def write_gold(kb, reg, task, seed):
    ids = {s: kb.append(s, [item(s, seed * 10 + k, f'gold:{task}')])[0]
           for k, s in enumerate(SPACES)}
    return reg.add_gold(task, f'gold:{task}', ids, 0)


def visible(view, space='A'):
    """(id, version) of every item a search of the view can return."""
    hits = view.search(space, torch.randn(1, 256), 100)
    return dict(zip(hits.ids[0], hits.versions[0]))


def test_round_t_reads_only_records_written_before_it(tmp_path):
    kb = corpus_kb(tmp_path)
    reg = Registry('ds')
    write_record(kb, reg, 'other', 'other:r0', 0, 7)       # another task's record
    seen = []

    def view_for(t):
        hidden, sub, _ = reg.visibility('task', 'train', gold_weight=lambda task: 1.0)
        return KBView(kb, hidden, sub)

    def attempt_fn(t, view):
        own = reg.own.get('task')
        seen.append((t, {i: v for i, v in visible(view).items()
                         if reg.record_of(i) is not None and reg.record_of(i).task == 'task'},
                     own))
        return SimpleNamespace(reads=[])

    def write_fn(t, attempt):
        return write_record(kb, reg, 'task', f'task:r{t}', t, t + 1)

    b9.rounds_loop('task', 3, view_for, attempt_fn, None, write_fn)
    assert seen[0][1] == {} and seen[0][2] is None                # round 0: nothing of its own
    for t in (1, 2):                                              # round t: version t (from t-1)
        (item_id, version), = seen[t][1].items()
        assert version == t and seen[t][2] == f'task:r{t - 1}'
    assert reg.supersedes['task'] == 2 and reg.records['task:r0'].current is False
    kb.close()


def test_views_hide_and_substitute_exactly(tmp_path):
    kb = corpus_kb(tmp_path)
    reg = Registry('ds')
    a = write_record(kb, reg, 'a', 'a:r0', 0, 1)
    b = write_record(kb, reg, 'b', 'b:r0', 0, 2)
    view = KBView(kb, hidden={a.items['A']})
    assert a.items['A'] not in visible(view) and len(visible(view)) == 4
    with pytest.raises(PermissionError):
        view.read('A', [a.items['A']])
    hidden, sub, _ = reg.visibility('a', 'train', gold_weight=lambda t: 1.0, mode='swapped',
                                    swap_with='b')
    swapped = KBView(kb, hidden, sub)
    got = swapped.read('A', [a.items['A']])[0]
    theirs = kb.read('A', [b.items['A']])[0]
    mine = kb.read('A', [a.items['A']])[0]
    assert torch.equal(got.values, theirs.values) and torch.equal(got.key, mine.key)
    hidden, _, _ = reg.visibility('a', 'train', gold_weight=lambda t: 1.0, mode='removed')
    assert set(a.items.values()) <= hidden and not set(b.items.values()) & hidden
    kb.close()


def test_gold_weight_recedes_and_scales_the_gate_mass_exactly(tmp_path):
    kb = corpus_kb(tmp_path)
    reg = Registry('ds')
    gold = write_gold(kb, reg, 'task', 5)
    schedule = GoldSchedule(start=1.0, anneal=10, decay=0.5)
    assert schedule.weight(reg, 'task', 0) == 1.0
    write_record(kb, reg, 'task', 'task:r0', 0, 1)
    assert schedule.weight(reg, 'task', 0) == 1.0              # a first write supersedes nothing
    write_record(kb, reg, 'task', 'task:r1', 1, 2)
    assert schedule.weight(reg, 'task', 0) == 0.5              # one supersede
    assert schedule.weight(reg, 'task', 5) == 0.25             # and the global schedule
    assert schedule.weight(reg, 'task', 10) == 0.0
    hidden, _, weights = reg.visibility('task', 'train',
                                        gold_weight=lambda t: schedule.weight(reg, t, 5))
    assert weights == dict.fromkeys(gold.items.values(), 0.25) and not hidden & set(weights)
    hidden, _, weights = reg.visibility('task', 'train',
                                        gold_weight=lambda t: schedule.weight(reg, t, 10))
    assert set(gold.items.values()) <= hidden and not weights  # w = 0: the record is gone
    # gates scale mass: the gold item's share of the read mass is exactly w
    r = reader()
    targets = {s: [('ds', gold.items[s])] for s in SPACES}
    masses = {}
    for w in (1.0, 0.25):
        read = r.read(torch.randn(HIDDEN), [kb], ['ds'], 3, ItemCache('cpu', train=False),
                      targets=targets, gold=True, weights={i: w for i in gold.items.values()})
        masses[w] = {s: info.mass for s, info in read.spaces.items()}
    for s in SPACES:
        assert masses[0.25][s] == pytest.approx(0.25 * masses[1.0][s], rel=1e-6)
    kb.close()


def test_heldout_rounds_read_only_records_without_gold_for_them(tmp_path):
    kb = corpus_kb(tmp_path)
    reg = Registry('ds')
    gold_t = write_gold(kb, reg, 'T', 1)
    gold_e = write_gold(kb, reg, 'E', 2)       # as if the evaluation task had been trained
    hinted_t = write_record(kb, reg, 'U', 'U:r0', 0, 3, read=[gold_t.items['A']])
    hinted_e = write_record(kb, reg, 'V', 'V:r0', 0, 4, read=[gold_e.items['B']])
    model = write_record(kb, reg, 'W', 'W:r0', 0, 5)
    assert hinted_t.kind == 'hinted' and hinted_t.saw_gold == ('T',) and model.kind == 'model'
    # lineage is inherited through a supersede that reads nothing gold
    again = write_record(kb, reg, 'U', 'U:r1', 1, 6)
    assert again.saw_gold == ('T',) and again.kind == 'hinted'
    reg.generation = 1
    own_eval = write_record(kb, reg, 'E', 'E:eval1:r0', 0, 7, split='eval')
    other_eval = write_record(kb, reg, 'F', 'F:eval1:r0', 0, 8, split='eval')
    w = {'gold_weight': lambda t: 0.5}
    hidden, _, weights = reg.visibility('E', 'eval', **w)
    ids = lambda rec: set(rec.items.values())      # noqa: E731
    assert ids(gold_e) <= hidden and ids(hinted_e) <= hidden        # gold for E in the lineage
    assert not (ids(gold_t) | ids(again) | ids(model) | ids(own_eval)) & hidden
    assert ids(other_eval) <= hidden                                # other held-out tasks
    strict, _, _ = reg.visibility('E', 'eval', heldout_lineage='any', **w)
    assert (ids(gold_t) | ids(again)) <= strict and not (ids(model) | ids(own_eval)) & strict
    train, _, _ = reg.visibility('T', 'train', **w)
    assert (ids(own_eval) | ids(other_eval)) <= train               # training never reads eval
    reg.generation = 2                                              # a later evaluation
    later, _, _ = reg.visibility('E', 'eval', **w)
    assert ids(own_eval) <= later
    restored = Registry.from_state(reg.state())
    assert restored.visibility('E', 'eval', **w)[0] == later
    kb.close()


def test_reads_are_authorized_per_dataset_kb(tmp_path):
    kb = corpus_kb(tmp_path, 'ds', 'kb')
    other = corpus_kb(tmp_path, 'secret', 'other')
    r = reader()
    with pytest.raises(PermissionError):            # a view keeps the KB's dataset
        r.read(torch.randn(HIDDEN), [KBView(other)], ['ds'], 3, ItemCache(train=False))
    args = SimpleNamespace(gold_start=1.0, gold_anneal=0, gold_decay=0.5, temperature=0.0,
                           seed=0, heldout_lineage='task')
    exp = b9.Experience(args, None, SimpleNamespace(device='cpu'), r, {'ds': kb}, None, None, 's0')
    with pytest.raises(PermissionError):            # an episode of another dataset
        exp.kb_of({'episode_id': 'e', 'kb': 'secret'})
    with pytest.raises(PermissionError):            # the store refuses another dataset's item
        kb.append('A', [NewItem(torch.randn(1, 256), torch.randn(256),
                                Provenance(('x',), 'codec', 0, 'secret'), 1.0, 1)])
    view, _ = exp.view({'episode_id': 'e', 'kb': 'ds'}, 'train', 0)
    assert view.dataset == 'ds'
    kb.close()
    other.close()


# -- generation ---------------------------------------------------------------------------
@pytest.fixture
def torch_conv(monkeypatch):
    import inspect
    from transformers.models.lfm2 import modeling_lfm2
    for name in ('causal_conv1d_fn', 'causal_conv1d_update'):   # the pure-torch fallbacks
        if hasattr(modeling_lfm2, name):
            monkeypatch.setattr(modeling_lfm2, name, inspect.unwrap(getattr(modeling_lfm2, name)))


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
    return lm


WORDS = {10: '<|tool_call_start|>', 11: '<|tool_call_end|>', 12: '<|im_end|>', 13: '\n',
         20: '[', 21: 'memory_search(', 22: ')', 23: ']', 24: 'SELECT', 25: ' 1', 26: 'lookup(',
         27: 'memory_write(', 40: '<|im_start|>', 41: 'tool', 42: 'assistant'}
MEM, MEM_END = SPAN_TOKENS['mem'][1], SPAN_TOKENS['mem_end'][1]
PROTO = GenProtocol(10, 11, 12, MEM, [12, 13], [40, 41, 13, MEM], [MEM_END, 12, 13], [40, 42, 13],
                    [40, 42, 13, 10, 20, 27, 22, 23, 11],
                    lambda ids: ''.join(WORDS.get(int(i), '?') for i in ids))


def scripted(tokens):
    it = iter(tokens)
    return lambda logits, step: next(it)


def run_generation(script, span_rows=3):
    lm = tiny_lm()
    frozen = l1.Frozen(lm, 2)
    calls = []

    def read(state):
        calls.append(state)
        return SimpleNamespace(span=torch.full((span_rows, HIDDEN), 0.25), spaces={})
    embed = lambda ids: frozen.embed(torch.tensor(ids))        # noqa: E731
    prompt = [1, 2, 3, 4]
    attempt = generate(lm, embed, lambda x: frozen.mid(x[None])[0], read, prompt, PROTO, 50,
                       scripted(script))
    return attempt, calls, frozen, embed


def test_generation_executes_a_read_when_the_call_is_emitted(torch_conv):
    script = [10, 20, 21, 22, 23, 11, 24, 25, 12]
    attempt, calls, frozen, embed = run_generation(script)
    assert len(calls) == 1 and attempt.stop == 'answer' and attempt.answer == 'SELECT 1'
    assert attempt.generated == len(script)
    paren = 4 + 3                        # prompt, then '<|tool_call_start|>' '[' 'memory_search(' ')'
    assert attempt.calls == [paren] and attempt.tokens[paren] == 22
    # the read gets the query-layer states of exactly the prefix, the state at ')' last
    prefix = embed([1, 2, 3, 4, 10, 20, 21, 22])
    torch.testing.assert_close(calls[0], frozen.mid(prefix[None])[0])
    # the span sits between <|mem|> and <|/mem|> in a tool message, then a new assistant turn
    mem = attempt.mems[0]
    assert attempt.tokens[mem] == MEM and attempt.tokens[mem + 1:mem + 4] == [None] * 3
    torch.testing.assert_close(attempt.embeds[mem + 1:mem + 4], torch.full((3, HIDDEN), 0.25))
    assert attempt.tokens[mem + 4] == MEM_END
    assert attempt.tokens[mem + 7:mem + 10] == [40, 42, 13]
    assert attempt.tokens[-3:] == [24, 25, 12]
    # the whole attempt equals one pass over its rows (the cache saw the same sequence)
    assert attempt.embeds.shape[0] == len(attempt.tokens)
    # the write site: the closed turn, then memory_write() up to <|tool_call_end|>
    site = write_prefix(attempt, PROTO, embed)
    torch.testing.assert_close(site[-10:], embed([13] + PROTO.write_call))
    # an external call ends the attempt without a read
    attempt, calls, _, _ = run_generation([10, 20, 26, 22, 23, 11])
    assert calls == [] and attempt.stop == 'call' and 'lookup(' in attempt.answer


def test_read_items_lists_items_with_nonzero_gates():
    info = SimpleNamespace(refs=[('ds', 'a'), ('ds', 'b')], gates=torch.tensor([0.5, 0.0]))
    read = SimpleNamespace(spaces={'A': info})
    assert read_items([read, None]) == {'a'}


# -- gradients into earlier rounds' writes (--backprop-rounds) -------------------------------
def b9_setup(tmp_path, k):
    from test_kb_read import writer_model
    from schnitz.kb.experience import Attempt
    lm = tiny_lm()
    frozen = l1.Frozen(lm, 2)
    model = writer_model(lm)
    r = reader()
    for p in r.stack.codecs.parameters():
        p.requires_grad_(True)
    kb = corpus_kb(tmp_path)
    ctx = l1.Context(frozen, r, {'ds': kb})
    args = SimpleNamespace(gold_start=1.0, gold_anneal=0, gold_decay=0.5, temperature=0.0,
                           seed=0, heldout_lineage='task', backprop_rounds=k, l1b_replay='free',
                           item_lr=0.0, retrieval_weight=0.1)
    exp = b9.Experience(args, model, frozen, r, {'ds': kb}, ctx, PROTO, 's0')
    ids = [1, 9, 10, 11, 20, 21, 22, 23, 11, 12, 13, 40, 41, 13, MEM, MEM_END, 12, 13, 24, 25, 26]
    targets = torch.arange(17, len(ids))
    ep = l1.Episode('task', 'ds', torch.tensor(ids), targets, [6], [14],
                    [{'kb': 'ds', 'record_ids': ['c0']}], 3, {})
    prompt = [1, 2, 3, 4]

    def attempt(read_state=None):
        """An answer after the prompt; with ``read_state`` one read (of the KB as the
        round sees it) spliced into a tool message first, as ``generate`` places it."""
        if read_state is None:
            tokens = prompt + [24, 25, 12]
            return Attempt(frozen.embed(torch.tensor(tokens)), tokens, 'SELECT 1', 3,
                           stop='answer')
        view, weights = exp.view({'episode_id': 'task', 'kb': 'ds'}, 'train', 0)
        cache = ItemCache('cpu', train=False)
        with torch.no_grad():
            read = r.read(read_state, [view], ['ds'], 3, cache, weights=weights)
        head = prompt + PROTO.tool_open
        tail = PROTO.tool_close + [24, 25, 12]
        n = read.span.shape[0]
        embeds = torch.cat([frozen.embed(torch.tensor(head)), read.span.detach().float(),
                            frozen.embed(torch.tensor(tail))])
        return Attempt(embeds, head + [None] * n + tail, 'SELECT 1', 3, mems=[len(head) - 1],
                       reads=[read], stop='answer', cache=cache)
    return exp, model, r, ep, attempt


def run_rounds(exp, ep, attempt, rounds):
    """The round loop of ``train_episode`` with given attempts: view, SFT, write (from
    round 1 on the attempt reads the KB, the task's own record included)."""
    row = {'episode_id': 'task', 'kb': 'ds'}
    exp.begin_episode(row)
    sft = []
    for t in range(rounds):
        view, weights = exp.view(row, 'train', 0)
        att = attempt(torch.randn(HIDDEN, generator=torch.Generator().manual_seed(t))
                      if t else None)
        sft.append(exp.sft(ep, view, weights, 1.0, 'task'))
        exp.write(row, ep, att, t, 0, 'train', True)
    return sft, exp.end_episode()


def test_write_source_rebuilds_the_write_prefix(tmp_path, torch_conv):
    exp, model, _, _, attempt = b9_setup(tmp_path, 1)
    embed = lambda t: exp.frozen.embed(torch.as_tensor(t))      # noqa: E731
    for att in (attempt(), attempt(torch.randn(HIDDEN))):
        ids, mems, spans = write_source(att, PROTO)
        assert torch.equal(exp.writer.inputs(ids, mems, spans), write_prefix(att, PROTO, embed))
    exp.kbs['ds'].close()


@pytest.mark.parametrize('k', [0, 1])
def test_backprop_rounds_reach_the_previous_rounds_write(tmp_path, k, torch_conv):
    exp, model, r, ep, attempt = b9_setup(tmp_path, k)
    _, stats = run_rounds(exp, ep, attempt, 2)
    rep = model.writer.rep.net[1].weight.grad
    codec = [p.grad for p in r.stack.codecs.parameters() if p.grad is not None]
    if k == 0:       # writes detached: nothing reaches the writer or the codecs
        assert stats is None and rep is None and not codec
    else:            # round 1 read round 0's write as its recomputation, exactly
        assert stats['backward_sources'] == 1 and stats['depth_max'] == 1
        assert stats['match_exact'] == 1.0 and stats['drift'] == 0.0
        assert rep.abs().sum() > 0 and any(g.abs().sum() > 0 for g in codec)
    exp.kbs['ds'].close()


@pytest.mark.parametrize('k', [1, 2])
def test_backprop_rounds_chain_through_an_earlier_writes_reads(tmp_path, k, torch_conv):
    """Round 2 reads round 1's write, whose attempt read round 0's write: with k = 2 the
    gradient of round 2's losses reaches round 0's write through that read (a depth-2
    replay); with k = 1 it stops at round 1's write."""
    exp, model, r, ep, attempt = b9_setup(tmp_path, k)
    _, stats = run_rounds(exp, ep, attempt, 3)
    assert stats['depth_max'] == k and stats['match_exact'] == 1.0 and stats['drift'] == 0.0
    if k == 2:
        assert stats['reread_rel'] < 1e-5          # the recomputed read is the logged one
    exp.kbs['ds'].close()
