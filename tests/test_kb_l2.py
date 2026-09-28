"""L2 producer path and losses (``schnitz.kb.producer``, ``schnitz.kb.stages.l2``); CPU,
a fake writer and decoder, a tiny stack; no downloads."""
from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from schnitz.kb.producer import (L2Weights, functional_loss, item_losses, key_loss, l2_loss,
                                 produce_items, recombine, rewrite_item, write_spans)
from schnitz.kb.stack import SPACES, KeyHeads, Stack, SuperpositionOperator
from schnitz.kb.stages import l2
from schnitz.kb_store import KnowledgeBase, NewItem, Provenance

W = 16      # decoder width of the fakes
V = 11      # fake vocabulary


class FakeModel:
    """What the producer path uses of ``schnitz.kb.decoder.Model``: the writer is a
    teacher-forcing identity (the hidden state at span position j is the rep fed at j,
    so ``writer.rep(h[:-1])`` returns the fed span), ``free_run`` a deterministic span
    of the text, and ``read`` a differentiable function of the span."""

    device = torch.device('cpu')
    core = None

    def __init__(self):
        torch.manual_seed(0)
        self.writer = SimpleNamespace(rep=nn.Identity())
        self.head = nn.Linear(W, V)

    def text_ids(self, text):
        return torch.tensor([ord(c) % V for c in text])

    def prefix(self, examples):
        return {}

    def write(self, examples, feed, prefix):
        return [torch.cat([f.float(), torch.zeros(1, W)]) for f in feed]

    def free_run(self, examples, lengths):
        out = []
        for ex, n in zip(examples, lengths):
            gen = torch.Generator().manual_seed(int(ex['ids'].sum()))
            out.append(torch.randn(n, W, generator=gen))
        return out, [None] * len(out)

    def read(self, examples, spans):
        logits, targets = [], []
        for ex, span in zip(examples, spans or [None] * len(examples)):
            h = torch.zeros(W) if span is None else span.float().mean(0)
            logits.append(self.head(h).expand(len(ex['target']), V))
            targets.append(ex['target'])
        return torch.cat(logits), torch.cat(targets)


def tiny_reader():
    torch.manual_seed(1)
    stack = Stack(1.0, 16, 12, 1, False, width=W)
    stack.set_statistics([torch.randn(20, W)])
    operators = nn.ModuleDict({s: SuperpositionOperator(s, 16, 12, 1) for s in SPACES})
    return SimpleNamespace(stack=stack, keys=KeyHeads(W, 16), operators=operators)


class Cache:
    def __init__(self, spans):
        self.spans = spans
        self.index = {r: [0, 0, s.shape[0]] for r, s in spans.items()}

    def __contains__(self, r):
        return r in self.spans

    def get(self, r):
        return self.spans[r]


TEXTS = {'r1': 'alpha beta gamma delta', 'r2': 'one two three four five six seven'}


def build(tmp_path, model, reader):
    """A KB whose items are exactly the producer path's output for each record."""
    spans = {r: model.free_run([{'ids': model.text_ids(t)}], [3 + i])[0][0]
             for i, (r, t) in enumerate(TEXTS.items())}
    kb = KnowledgeBase.create(tmp_path / 'kb', name='kb', dataset='ds')
    produced = {}
    for r, span in spans.items():
        items = produce_items(reader.stack, span)
        produced[r] = items
        for s, v in items.items():
            kb.append(s, [NewItem(v.detach(), reader.keys.item_key(s, v).detach(),
                                  Provenance((r,), 'codec'), 1.0, 1)])
    return kb, spans, produced


def test_losses_are_zero_when_the_producer_reproduces_the_item():
    model, reader = FakeModel(), tiny_reader()
    span = model.free_run([{'ids': model.text_ids('abc')}], [4])[0][0]
    items = produce_items(reader.stack, span)
    parts = item_losses(items, {s: v.detach().clone() for s, v in items.items()})
    assert set(parts) == {f'{k}_{s}' for k in ('cos', 'mse') for s in SPACES}
    for s, v in items.items():
        parts['key'] = key_loss(reader.keys, s, v, reader.keys.item_key(s, v).detach())
        assert abs(float(parts['key'].detach())) < 1e-6
    ex = [{'ids': model.text_ids('abc'), 'target': model.text_ids('abc'), 'task': 'reconstruct'}]
    y = recombine(reader.stack, items, 4)
    parts['kl'], info = functional_loss(model, ex, [y], [y.detach().clone()])
    loss = l2_loss(parts, L2Weights())
    assert abs(float(loss)) < 1e-5
    assert info['nll_produced'] == info['nll_target']
    # and not zero for another item
    other = {s: v.detach() + 0.5 * torch.randn_like(v) for s, v in items.items()}
    assert float(l2_loss(item_losses(items, other), L2Weights(key=0, kl=0))) > 0.01


def test_teacher_and_free_feed_reproduce_the_span_with_gradients_into_the_heads():
    model, _ = FakeModel(), None
    model.writer.rep = nn.Linear(W, W)
    with torch.no_grad():
        model.writer.rep.weight.copy_(torch.eye(W))
        model.writer.rep.bias.zero_()
    ex = [{'ids': model.text_ids('hello'), 'prompt': 'memory', 'factor': 4.0}]
    teacher = model.free_run(ex, [5])[0]
    for feed in ('teacher', 'free'):
        out = write_spans(model, ex, [5], feed, teacher if feed == 'teacher' else None)
        torch.testing.assert_close(out[0], teacher[0])
        out[0].sum().backward()
        assert model.writer.rep.weight.grad.abs().sum() > 0
        model.writer.rep.zero_grad()


def test_stage_losses_on_a_kb_of_produced_items(tmp_path):
    model, reader = FakeModel(), tiny_reader()
    kb, spans, _ = build(tmp_path, model, reader)
    records, derived, skipped = l2.collect_targets(kb)
    assert {r.record_id for r in records} == set(TEXTS) and not derived and not skipped
    prod = l2.Producers(model, reader, 's0', 'teacher', {'ds': Cache(spans)}, TEXTS)
    loss, parts = l2.record_losses(prod, records, L2Weights())
    # zero up to the store's bf16 rounding of the item values (keys are stored in fp32)
    assert float(loss) < 1e-4 and parts['key'] < 1e-6
    report = l2.evaluate_records(prod, records, 2)
    assert report['functional_gap'] < 1e-3 and report['cos_A'] < 1e-4
    # a perturbed target is not reproduced
    records[0].target['A'] = records[0].target['A'] + 1.0
    assert float(l2.record_losses(prod, records, L2Weights())[0]) > 0.01
    kb.close()


def test_rewrite_outputs_are_produced_through_their_lineage(tmp_path):
    model, reader = FakeModel(), tiny_reader()
    kb, spans, produced = build(tmp_path, model, reader)
    ids = {r: l2.collect_targets(kb)[0][i].items['A'] for i, r in enumerate(TEXTS)}
    key = torch.nn.functional.normalize(torch.randn(256), dim=0)
    value = rewrite_item(reader.operators['A'], [(produced['r1']['A'], 1.0, None),
                                                 (produced['r2']['A'], 1.0, None)], key, 3)
    kb.rewrite('A', [ids['r1'], ids['r2']],
               [NewItem(value.detach(), key, Provenance((), 'rewrite'), 2.0, 1)])
    export = kb.export_live(tmp_path / 'export')      # lineage inputs survive as history
    records, derived, skipped = l2.collect_targets(export)
    assert len(derived) == 1 and derived[0].space == 'A' and not skipped
    assert sorted(derived[0].inputs) == [('r1', 1.0), ('r2', 1.0)]
    assert all('A' not in r.items for r in records)     # the inputs are no longer current
    prod = l2.Producers(model, reader, 's0', 'teacher', {'ds': Cache(spans)}, TEXTS)
    # the stored key here is the rewrite's condition, not the key heads' key of the values
    loss, parts = l2.derived_losses(prod, derived, L2Weights(key=0))
    assert float(loss) < 1e-4 and parts['cos_A'] < 1e-4
    kb.close()
    export.close()
