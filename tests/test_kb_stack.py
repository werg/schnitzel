import math

import torch

from schnitz.kb.losses import UsageEMA, balance_loss, read_entropy, retrieval_loss
from schnitz.kb.stack import KEY_WIDTH, SPACES, KeyHeads, SuperpositionOperator, read_count
from schnitz.mlp_matrix import MLPMatrix


def test_spaces_come_from_the_store_definition():
    from schnitz.kb_store import DEFAULT_SPACES
    assert SPACES == {n: (s.ratio, s.width) for n, s in DEFAULT_SPACES.items()}
    assert {r * w for r, w in SPACES.values()} == {256}
    assert sum(r * w for r, w in SPACES.values()) == 1024


def test_read_count_rule():
    gates = torch.tensor([1.0, 3.0])
    assert read_count([4, 2], ['A', 'D'], gates, target=7) == 7          # pretraining
    # A: 4 positions = 4 reps; D: 2 positions = 16 reps; mass-weighted mean 13
    assert read_count([4, 2], ['A', 'D'], gates) == 13
    assert read_count([4, 2], ['A', 'D'], gates, budget=8) == 8
    assert read_count([4], ['A'], torch.tensor([0.0])) == 1


def test_extra_features_start_at_zero_and_default_to_none():
    torch.manual_seed(0)
    op = MLPMatrix({'item': 8}, 8, state=16, hidden=8, layers=2, extra=4)
    x = torch.randn(5, 8)
    plain = op([('item', x, 1.0)], 3)[0]
    with_extra = op([('item', x, 1.0, torch.randn(4))], 3)[0]
    assert torch.allclose(plain, with_extra)             # zero-initialized projection
    for layer in op.layers:
        torch.nn.init.normal_(layer.extra.weight)
    assert not torch.allclose(plain, op([('item', x, 1.0, torch.randn(4))], 3)[0])


def test_superposition_operator_k3b_starts_as_k3a_and_gate_zero_removes():
    torch.manual_seed(0)
    op = SuperpositionOperator('D', state=32, hidden=16, layers=2)
    width, key = SPACES['D'][1], KEY_WIDTH['D']
    nbrs = [(torch.randn(3, width), 1.0, torch.randn(key)) for _ in range(3)]
    target = torch.randn(key)
    a, mass = op(nbrs, target, 2)
    b, _ = op(nbrs, target, 2, neighbour_keys=True)
    assert a.shape == (2, width) and float(mass) == 3.0
    assert torch.allclose(a, b)
    dropped = nbrs[:2] + [(torch.randn(3, width), 0.0, torch.randn(key))]
    assert torch.allclose(op(nbrs[:2], target, 2)[0], op(dropped, target, 2)[0], atol=1e-6)


def test_key_heads_are_unit_and_scored():
    heads = KeyHeads(64, hidden=32)
    k = heads.item_key('B', torch.randn(4, 6, SPACES['B'][1]))
    q = heads.query_key('B', torch.randn(2, 64))
    assert k.shape == (4, KEY_WIDTH['B']) and torch.allclose(k.norm(dim=-1), torch.ones(4))
    assert heads.scores('B', q, k).shape == (2, 4)


def test_retrieval_loss_and_recall():
    scores = torch.tensor([[5.0, 0.0, 0.0], [0.0, 0.0, 5.0], [1.0, 1.0, 1.0]],
                          requires_grad=True)
    positive = torch.tensor([[True, False, False], [True, False, False],
                             [False, False, False]])
    loss, stats = retrieval_loss(scores, positive, ks=(1, 3))
    expected = -(torch.log_softmax(scores[0], -1)[0] + torch.log_softmax(scores[1], -1)[0]) / 2
    assert torch.allclose(loss, expected)
    assert stats == {'recall@1': 0.5, 'recall@3': 1.0}
    loss.backward()
    assert scores.grad[2].abs().sum() == 0                 # no positives: skipped


def test_balance_loss_is_one_for_uniform_use_and_grows_with_concentration():
    usage = UsageEMA(4, decay=0.0)
    usage.update(torch.tensor([0, 1, 2, 3]), torch.ones(4))
    ids = torch.tensor([0, 1, 2, 3])
    assert math.isclose(float(balance_loss(ids, torch.ones(4), usage)), 1.0, rel_tol=1e-6)
    usage.update(torch.tensor([0]), torch.ones(1))         # item 0 now takes all use
    assert float(balance_loss(torch.tensor([0]), torch.ones(1), usage)) > 1.0
    assert float(balance_loss(torch.tensor([1]), torch.ones(1), usage)) < 1.0
    usage.grow(6)
    assert usage.share.shape[0] == 6 and math.isclose(float(usage.share.sum()), 1.0,
                                                      rel_tol=1e-6)


def test_read_entropy():
    assert math.isclose(float(read_entropy(torch.tensor([1.0, 1.0, 0.0]))), 2.0, rel_tol=1e-5)


def test_stack_standardizes_spans_with_corpus_statistics():
    from schnitz.kb.stack import Stack
    torch.manual_seed(0)
    stack = Stack(0.8, state=32, hidden=16, layers=1, checkpointing=False, width=24)
    common = torch.randn(24)
    spans = [common + 0.1 * torch.randn(9, 24) for _ in range(20)]
    stack.set_statistics(spans)
    z = stack.standardize(torch.cat(spans))
    assert torch.allclose(z.mean(0), torch.zeros(24), atol=1e-5)
    assert torch.allclose(z.std(0), torch.ones(24), atol=1e-4)
    items = stack.encode(spans[0])
    assert {k: v.shape[0] for k, v in items.items()} == {'A': 9, 'B': 5, 'C': 3, 'D': 2}
    out = stack.decode(items, {s: 1.0 for s in SPACES}, 9)
    assert out.shape == (9, 24)
    assert torch.allclose(stack.standardize(out) * stack.std + stack.mean, out, atol=1e-5)
