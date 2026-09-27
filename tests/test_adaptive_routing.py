import torch

from schnitz.routing import AdaptiveDistanceGate, union_support_loss


def test_distance_gate_is_density_relative_and_trains_both_keys():
    gate = AdaptiveDistanceGate(3, density_k=2, min_temperature=.01,
                                max_temperature=.4, initial_temperature=.1).double()
    query = torch.tensor([[1., 0., 0.]], dtype=torch.double, requires_grad=True)
    keys = torch.tensor([[[.95, .05, 0.], [.8, .2, 0.], [-1., 0., 0.]]],
                        dtype=torch.double, requires_grad=True)
    scores = torch.einsum('bd,bnd->bn', torch.nn.functional.normalize(query, dim=-1),
                          torch.nn.functional.normalize(keys, dim=-1))
    weights, diagnostics = gate(query, scores, scores, torch.ones_like(scores, dtype=torch.bool),
                                torch.ones_like(scores, dtype=torch.bool))
    assert weights[0, 0] > weights[0, 1] > weights[0, 2]
    assert 0 < diagnostics['effective_records'].item() <= 3
    (weights * torch.tensor([[1., -2., 3.]], dtype=torch.double)).sum().backward()
    assert query.grad.abs().sum() > 0
    assert keys.grad.abs().sum() > 0
    assert gate.adjust.weight.grad.abs().sum() > 0


def test_support_floor_only_raises_labelled_materialized_records():
    gate = AdaptiveDistanceGate(2, density_k=1)
    query = torch.tensor([[1., 0.]])
    scores = torch.tensor([[-1., -1.]])
    candidates = torch.tensor([[0., -1.]])
    support = torch.tensor([[True, False]])
    weights, _ = gate(query, scores, candidates, torch.ones_like(support),
                      torch.ones_like(support), support, .3)
    assert weights[0, 0] >= .3
    assert weights[0, 1] < .3


def test_support_anchor_rewards_union_coverage_without_forcing_every_space():
    weak = torch.tensor([0., 0.], requires_grad=True)
    strong = torch.tensor([5., 0.], requires_grad=True)
    covered = union_support_loss([weak, strong], [(0,), (0,)])
    repeated_weak = union_support_loss([weak, weak], [(0,), (0,)])
    assert covered < repeated_weak
    covered.backward()
    assert strong.grad is not None


def test_scaled_routing_logits_sharpen_a_positive_without_changing_rank():
    scores = torch.tensor([.3] + [0.] * 63, requires_grad=True)
    weak = union_support_loss([scores], [(0,)])
    sharp = union_support_loss([scores * 10], [(0,)])
    assert sharp < weak
    assert (scores * 10).argmax() == scores.argmax()
    sharp.backward()
    assert scores.grad[0] < 0
    assert scores.grad[1] > 0
