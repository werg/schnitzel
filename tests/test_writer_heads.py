"""B4b/B4c on tiny fakes (no BGKit, no downloads): head-set parametrization of the
writer path, the output port's state, and the causal prefix of in-context writes."""
from __future__ import annotations

from pathlib import Path
import re
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from schnitz.bgkit_span import PortHeads, SpanWriter
from schnitz.kb.decoder import Model, length_factors
from schnitz.kb.stages.writer import _rollout, load_optimizer, write_example
from schnitz.memory_transcripts import render_ids, site_messages, write_site_prefix
from schnitz.span_protocol import ProtocolTokens, untie
from schnitz.span_tokens import SPAN_TOKENS

WIDTH = 8


class FakeDecoder(nn.Module):
    """Causal stand-in for the LFM2 decoder: running mean of the inputs, then a layer."""

    def __init__(self, vocab: int = 64):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, WIDTH)
        self.mix = nn.Linear(WIDTH, WIDTH)

    def embed(self, ids):
        return self.embed_tokens(ids)

    def _hidden(self, inputs_embeds, attention_mask):
        x = inputs_embeds.float() * attention_mask[..., None]
        steps = torch.arange(1, x.shape[1] + 1, device=x.device)[None, :, None]
        return torch.tanh(self.mix(x.cumsum(1) / steps))


def fake_model() -> Model:
    torch.manual_seed(0)
    model = Model.__new__(Model)
    model.decoder = FakeDecoder()
    model.writer = SpanWriter(WIDTH, 1.0)
    model.device = torch.device('cpu')
    model.prompts = {'memory': (torch.tensor([1, 2]), torch.tensor([3]))}
    model.gate, model.adapter, model.gate_value, model.merged = None, None, 0.0, False
    model.protocol, model.protocol_losses, model.record_protocol = None, [], False
    model.reference, model.port, model.stop_pos_weight = None, None, 1.0
    model.tail = torch.tensor([4])
    model.target_norm = 1.0

    def prefix(examples):  # the exact equivalent of the cached prefix: recompute it
        return {'examples': examples}

    def span_hidden(cache, spans):
        seqs = [torch.cat([model.write_inputs(ex), s.float()]) for ex, s in
                zip(cache['examples'], spans)]
        width = max(x.shape[0] for x in seqs)
        inputs = torch.zeros(len(seqs), width, WIDTH)
        mask = torch.zeros(len(seqs), width, dtype=torch.long)
        for i, x in enumerate(seqs):
            inputs[i, :x.shape[0]], mask[i, :x.shape[0]] = x, 1
        hidden = model.decoder._hidden(inputs, mask)
        return [hidden[i, x.shape[0] - s.shape[0]:x.shape[0]]
                for i, (x, s) in enumerate(zip(seqs, spans))]

    model.prefix, model.span_hidden = prefix, span_hidden
    return model


def examples():
    return [{'ids': torch.tensor([5, 6, 7]), 'prompt': 'memory', 'factor': 4.0,
             'teacher': torch.randn(3, WIDTH)},
            {'ids': torch.tensor([8, 9]), 'prompt': 'memory', 'factor': 8.0,
             'teacher': torch.randn(2, WIDTH)},
            {'prefix_ids': torch.tensor([1, 10, 11, 12, 13]), 'factor': 4.0,
             'teacher': torch.randn(4, WIDTH)}]


def test_default_heads_are_the_writer():
    model, exs = fake_model(), examples()
    feed = [ex['teacher'] for ex in exs]
    for a, b in zip(model.write(exs, feed), model.write(exs, feed, heads=model.writer)):
        assert torch.equal(a, b)
    free_a, stop_a = model.free_run(exs, [3, 2, 4])
    free_b, stop_b = model.free_run(exs, [3, 2, 4], model.writer)
    assert stop_a == stop_b and all(torch.equal(a, b) for a, b in zip(free_a, free_b))
    outs = []
    for heads in (None, model.writer):
        torch.manual_seed(3)
        outs.append(_rollout(model, exs, passes=2, sample=0.5, sequential=2, heads=heads))
    (_, pa, ca, sa), (_, pb, cb, sb) = outs
    assert torch.equal(ca, cb) and torch.equal(sa, sb)
    assert all(torch.equal(a, b) for a, b in zip(pa, pb))


def test_cached_and_full_write_agree():
    model, exs = fake_model(), examples()
    feed = [ex['teacher'] for ex in exs]
    for a, b in zip(model.write(exs, feed), model.write(exs, feed, model.prefix(exs))):
        assert torch.allclose(a, b, atol=1e-6)


def test_port_heads_own_the_port_span():
    model, exs = fake_model(), examples()
    marker = torch.randn(WIDTH)
    port = PortHeads(WIDTH, 1.0, lambda factor: marker + 0 * factor)
    feed = [ex['teacher'] for ex in exs]
    writer_states = model.write(exs, feed)
    port_states = model.write(exs, feed, heads=port)
    assert not torch.allclose(writer_states[0][0], port_states[0][0])  # another marker
    free, _ = model.free_run(exs, [2, 2, 2], port)
    states = model.write(exs, free, heads=port)
    assert torch.allclose(free[0], port.rep(states[0][:-1]), atol=1e-6)
    _, preds, cos, stop = _rollout(model, exs, passes=1, sample=0.5, heads=port)
    (cos + stop + sum(p.sum() for p in preds)).backward()
    assert all(p.grad is None for p in model.writer.parameters())
    assert all(p.grad is not None for p in port.parameters())


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(64, WIDTH)
        self.lm_head = nn.Linear(WIDTH, 64, bias=False)
        self.lm_head.weight = self.embed.weight
        self.config = SimpleNamespace(tie_word_embeddings=True)

    def get_input_embeddings(self):
        return self.embed


def protocol_model() -> Model:
    model = fake_model()
    lm = TinyLM()
    untie(lm)
    model.protocol = ProtocolTokens(lm.embed, lm.lm_head, model.writer.ratio)
    return model


def test_port_install_starts_from_the_writer_heads():
    model = protocol_model()
    with torch.no_grad():
        model.protocol.span_bias.copy_(torch.tensor([0.3, -0.2]))
    model.install_port()
    h = torch.randn(5, WIDTH)
    assert torch.allclose(model.port.stop(h), model.protocol.span_logits(h), atol=1e-6)
    assert torch.allclose(model.port.rep(h), model.writer.rep(h))
    marker = model.port.marker_embedding(torch.tensor(1.0))
    assert torch.allclose(marker, model.protocol.embedding('port')
                          + model.writer.ratio(torch.tensor(1.0)))
    # the ratio code is taken at x1 whatever factor an example carries
    assert torch.allclose(model.port.marker_embedding(torch.tensor(16.0)), marker)
    assert model.port.tokens == ('port', 'port_end')


def test_port_state_round_trip_and_old_states():
    model = protocol_model()
    model.install_port()
    with torch.no_grad():
        for p in model.port.parameters():
            p.add_(torch.randn_like(p))
    state = model.trained_state()
    assert 'port' in state
    fresh = protocol_model()
    fresh.load_trained(state)
    for a, b in zip(model.port.state_dict().values(), fresh.port.state_dict().values()):
        assert torch.equal(a, b)
    old = {k: v for k, v in state.items() if k != 'port'}   # saved before B4b
    older = protocol_model()
    older.load_trained(old)
    assert older.port is None and 'port' not in older.trained_state()


def test_port_group_added_on_resume():
    model = protocol_model()
    args = SimpleNamespace(lr=1e-3, protocol_lr=1e-3, port_lr=1e-3)
    before = torch.optim.AdamW(model.param_groups(args))
    model.writer.rep.net[1].weight.sum().backward()
    before.step()
    saved = before.state_dict()
    model.install_port()
    after = torch.optim.AdamW(model.param_groups(args))
    load_optimizer(after, saved)
    assert len(after.param_groups) == len(saved['param_groups']) + 1
    first = model.writer.rep.net[1].weight
    assert torch.equal(after.state[first]['exp_avg'], before.state[first]['exp_avg'])
    assert all(p not in after.state for p in model.port.parameters())


# -- B4c write-site prefix ----------------------------------------------------------
class FakeTokenizer:
    """Word-level tokenizer with a ChatML-like template (LFM2's shape: tool calls as
    ``<|tool_call_start|>[name()]<|tool_call_end|>`` inside the assistant turn)."""

    PATTERN = re.compile(r'<\|[^|]+\|>|\w+|\s|[^\w\s]')

    def __init__(self):
        self.vocab = {text: ident for text, ident in SPAN_TOKENS.values()}
        self.vocab['<|startoftext|>'] = 1

    def _id(self, piece: str) -> int:
        if piece not in self.vocab:
            self.vocab[piece] = 1000 + len(self.vocab)
        return self.vocab[piece]

    def convert_tokens_to_ids(self, piece: str) -> int:
        return self._id(piece)

    def decode(self, ids) -> str:
        names = {v: k for k, v in self.vocab.items()}
        return ''.join(names[i] for i in ids)

    def apply_chat_template(self, messages, tools=None, tokenize=True, return_dict=False,
                            return_assistant_tokens_mask=False):
        pieces = [('<|startoftext|>', 0)]
        for m in messages:
            loss = int(m['role'] == 'assistant')
            pieces.append((f"<|im_start|>{m['role']}\n", 0))
            body = m.get('content') if isinstance(m.get('content'), str) else ''
            calls = m.get('tool_calls') or []
            if calls:
                names = ', '.join(f"{c['function']['name']}()" for c in calls)
                body += f'<|tool_call_start|>[{names}]<|tool_call_end|>'
            pieces.append((body + '<|im_end|>', loss))
            pieces.append(('\n', 0))
        text = ''.join(p for p, _ in pieces)
        if not tokenize:
            return text
        ids, mask = [], []
        for piece, loss in pieces:
            for token in self.PATTERN.findall(piece):
                ids.append(self._id(token))
                mask.append(loss)
        return {'input_ids': ids, 'assistant_masks': mask}


def write_call():
    return [{'type': 'function', 'function': {'name': 'memory_write', 'arguments': {}}}]


def transcript():
    search = [{'type': 'function', 'function': {'name': 'memory_search', 'arguments': {}}}]
    return {
        'episode_id': 'e1', 'tools': [],
        'messages': [
            {'role': 'system', 'content': 'policy words'},
            {'role': 'user', 'content': 'question alpha'},
            {'role': 'assistant', 'content': '', 'tool_calls': search},
            {'role': 'tool', 'name': 'memory_search', 'content': {'slot': {'record_ids': ['r']}}},
            {'role': 'assistant', 'content': 'answer beta'},
            {'role': 'assistant', 'content': '', 'tool_calls': write_call(),
             'write_span': {'write_site': 0}},
            {'role': 'tool', 'name': 'memory_write', 'content': {'write_result': 'stored'}},
            {'role': 'user', 'content': 'later gamma'},
            {'role': 'assistant', 'content': 'later delta'},
            {'role': 'assistant', 'content': '', 'tool_calls': write_call(),
             'write_span': {'write_site': 1}},
        ],
        'write_sites': [{'message': 5, 'call': 0, 'teacher_text': 'secret omega'},
                        {'message': 9, 'call': 0, 'teacher_text': 'secret kappa'}]}


def test_write_prefix_is_causal():
    tok, row = FakeTokenizer(), transcript()
    prefix = write_site_prefix(tok, row, 0)
    full, _ = render_ids(tok, row['messages'], row['tools'])
    assert prefix == full[:len(prefix)]
    assert full[len(prefix)] == SPAN_TOKENS['bg'][1]          # the model opens the span next
    assert prefix[-1] == tok.convert_tokens_to_ids('<|tool_call_end|>')
    text = tok.decode(prefix)
    for later in ('gamma', 'delta', 'omega', 'kappa', 'stored', SPAN_TOKENS['bg'][0]):
        assert later not in text
    assert SPAN_TOKENS['mem'][0] + SPAN_TOKENS['mem_end'][0] in text    # empty search slot
    # counterfactual: changing anything after the site leaves the prefix unchanged
    other = transcript()
    other['messages'][6]['content'] = {'write_result': 'changed'}
    other['messages'][7]['content'] = 'entirely different words'
    other['messages'] = other['messages'][:8]
    assert write_site_prefix(tok, other, 0) == prefix


def test_second_write_site_sees_the_first_span_empty():
    tok, row = FakeTokenizer(), transcript()
    prefix = write_site_prefix(tok, row, 1)
    full, _ = render_ids(tok, row['messages'], row['tools'])
    assert prefix == full[:len(prefix)] and full[len(prefix)] == SPAN_TOKENS['bg'][1]
    assert prefix.count(SPAN_TOKENS['bg'][1]) == 1 and prefix.count(SPAN_TOKENS['bg_end'][1]) == 1
    assert 'kappa' not in tok.decode(prefix) and 'omega' not in tok.decode(prefix)
    assert len(site_messages(row, 1)) == 10


def test_write_site_must_be_a_write_call():
    row = transcript()
    row['write_sites'][0]['message'] = 4
    with pytest.raises(ValueError):
        site_messages(row, 0)


def test_write_example_truncates_from_the_left():
    tok, row = FakeTokenizer(), transcript()
    model = SimpleNamespace(tok=tok, text_ids=lambda text: torch.arange(40))
    full = write_site_prefix(tok, row, 0)
    ex = write_example(model, row, 0, max_tokens=12, space=0)
    ids = ex['prefix_ids'].tolist()
    assert len(ids) == 12 and ids[0] == full[0] and ids[1:] == full[-11:]
    assert ex['factor'] == length_factors(40)[0] and ex['count'] == 10
    assert write_example(model, row, 0, max_tokens=4096, space=1)['prefix_ids'].tolist() == full


TOKENIZER = Path('/runs/hf-models/LFM2.5-350M')


@pytest.mark.skipif(not TOKENIZER.exists(), reason='local LFM2.5 tokenizer not available')
def test_write_prefix_with_lfm2_template():
    transformers = pytest.importorskip('transformers')
    tok = transformers.AutoTokenizer.from_pretrained(TOKENIZER)
    row = transcript()
    row['tools'] = [{'name': 'memory_search', 'description': 'search',
                     'parameters': {'type': 'object', 'properties': {}}},
                    {'name': 'memory_write', 'description': 'write',
                     'parameters': {'type': 'object', 'properties': {}}}]
    for site in (0, 1):
        prefix = write_site_prefix(tok, row, site)
        full, _ = render_ids(tok, row['messages'], row['tools'])
        assert prefix == full[:len(prefix)] and full[len(prefix)] == SPAN_TOKENS['bg'][1]
        assert tok.decode(prefix).endswith('[memory_write()]<|tool_call_end|>')
        assert 'secret' not in tok.decode(prefix)


def test_write_prefix_slots_are_filled_from_span_caches(tmp_path):
    from types import SimpleNamespace

    from schnitz.kb.bank import SpanCache, kb_dir
    from schnitz.kb.stages.writer import SlotSpans
    from schnitz.memory_transcripts import site_slots, splice_slots
    tok, row = FakeTokenizer(), transcript()
    row['messages'][3]['content']['slot'].update(kb='d:x', record_ids=['r', 'gone', 's'])
    ids = write_site_prefix(tok, row, 0)
    pairs = splice_slots(ids, site_slots(row, 0))
    assert [slot['record_ids'] for _, slot in pairs] == [['r', 'gone', 's']]
    assert ids[pairs[0][0]] == SPAN_TOKENS['mem'][1]
    assert splice_slots(ids[pairs[0][0] + 2:], site_slots(row, 0)) == []   # cut from the left

    class Writer:  # spans: rep j of record t is (sum of its chars) + j
        def text_ids(self, text):
            return torch.tensor([ord(c) for c in text])

        def free_run(self, examples, lengths):
            return [torch.arange(n)[:, None].float() + float(ex['ids'].sum())
                    + torch.zeros(n, 8) for ex, n in zip(examples, lengths)], None
    SpanCache.build(tmp_path / kb_dir('d:x'), Writer(), {'r': 'a' * 40, 's': 'b' * 40}, 's0')
    embed = torch.nn.Embedding(5000, 8)
    model = SimpleNamespace(device='cpu', decoder=SimpleNamespace(embed_tokens=embed))
    ex = {'prefix_ids': torch.tensor(ids), 'slots': pairs}
    slots = SlotSpans(tmp_path, cap=1000)
    filled = slots.fill(model, ex)['inputs']
    r = SpanCache(tmp_path / kb_dir('d:x')).get('r').float()
    s = SpanCache(tmp_path / kb_dir('d:x')).get('s').float()
    at = pairs[0][0] + 1
    assert filled.shape[0] == len(ids) + r.shape[0] + s.shape[0]
    torch.testing.assert_close(filled[at:at + r.shape[0]], r)             # after <|mem|>
    torch.testing.assert_close(filled[at + r.shape[0]:at + r.shape[0] + s.shape[0]], s)
    torch.testing.assert_close(filled[:at], embed(torch.tensor(ids[:at])))
    torch.testing.assert_close(filled[at + r.shape[0] + s.shape[0]:], embed(torch.tensor(ids[at:])))
    assert slots.missing == 1                                             # 'gone' is not cached
    capped = SlotSpans(tmp_path, cap=3).fill(model, ex)['inputs']
    assert capped.shape[0] == len(ids) + 3
