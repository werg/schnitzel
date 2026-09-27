"""The diagnostic's batched objective must match the actual routing objective."""
import importlib.util
from pathlib import Path

import torch
from schnitz.routing import group_plan_loss


def test_feature_probe_pair_objective_and_gradient():
    path = Path(__file__).parents[1] / 'scripts/probe_routing_features.py'
    spec = importlib.util.spec_from_file_location('feature_probe', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scores = torch.tensor([[3., -2., 1., .5], [-1., 4., 0., 2.], [90., -90., 0., 1.]],
                          dtype=torch.float64, requires_grad=True)
    required = torch.tensor([[0, 2], [3, 3], [1, 2]])
    lengths = torch.tensor([2, 1, 2])
    actual = module.pair_loss(scores, required, lengths)
    expected = torch.stack([group_plan_loss(row, [tuple(indices[:count].tolist())])
                            for row, indices, count in zip(scores, required, lengths, strict=True)]).mean()
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    a, = torch.autograd.grad(actual, scores, retain_graph=True)
    b, = torch.autograd.grad(expected, scores)
    torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)


def test_feature_probe_shared_initial_forward_and_frozen_head():
    from types import SimpleNamespace
    from torch import nn
    path = Path(__file__).parents[1] / 'scripts/probe_routing_features.py'
    spec = importlib.util.spec_from_file_location('feature_probe', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    agent = SimpleNamespace(key_head=nn.Linear(8, 4, bias=False),
                            address_maps=[nn.Linear(4, 4, bias=False)],
                            query_maps=[nn.Linear(4, 4, bias=False)],
                            query_head=nn.Linear(8, 4, bias=False))
    a, b = module.AddressProbe(agent, False), module.AddressProbe(agent, True)
    features = {'key': torch.randn(3, 4, 8), 'raw_query': torch.randn(3, 8)}
    assert torch.equal(a(features), b(features))
    a(features).sum().backward()
    b(features).sum().backward()
    assert a.query_head.weight.grad is None
    assert b.query_head.weight.grad is not None
    assert b.query_head.weight.grad.abs().sum() > 0
