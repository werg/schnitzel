import pytest
import torch

from schnitz.compaction import (SyntheticCompactor, mean_and_mass, contribution_loss,
    adaptive_field_responsibilities, field_responsibilities, FullClusterCode,
    random_partition, storage_noise)
from schnitz.readers import SetReader, merge_statistics


@pytest.mark.parametrize("kind", ["mlp", "attention"])
def test_duplicate_records_compact_exactly_to_mean_and_mass(kind):
    reader = SetReader(6, 4, 9, width=12, slots=3, rounds=3, kind=kind).double()
    x = torch.randn(2, 1, 6).double().expand(-1, 8, -1)
    query, weights = torch.randn(2, 4).double(), torch.rand(2, 8).double()
    mean, mass = mean_and_mass(x, weights)
    loss = contribution_loss(reader, x, weights, mean, mass, query)
    assert loss.item() < 1e-20


def test_synthetic_compactor_permutation_and_mass():
    compact = SyntheticCompactor(8, width=16, records=2)
    x, weights = torch.randn(3, 7, 8), torch.rand(3, 7)
    codes, mass = compact(x, weights)
    p = torch.randperm(7)
    alternative, alternative_mass = compact(x[:, p], weights[:, p])
    torch.testing.assert_close(codes, alternative)
    torch.testing.assert_close(mass, alternative_mass)
    torch.testing.assert_close(mass.sum(1), weights.sum(1))
    codes.square().mean().backward()
    assert compact.encode[0].weight.grad is not None


@pytest.mark.parametrize("kind", ["mlp", "attention"])
def test_overlapping_fields_conserve_contribution_and_mass(kind):
    reader = SetReader(6, 4, 9, width=12, slots=3, rounds=2, kind=kind).double()
    x, q = torch.randn(1, 9, 6).double(), torch.randn(1, 4).double()
    alpha = field_responsibilities(x[0], x[0, :3], memberships=2).double()
    torch.testing.assert_close(alpha.sum(1), torch.ones(9).double())
    state = reader.initial_state(q)
    full = reader.aggregate(0, x, q, state, torch.ones(1, 9).double())
    pieces = [reader.aggregate(0, x, q, state, alpha[:, c][None]) for c in range(3)]
    merged = merge_statistics(pieces)
    torch.testing.assert_close(full.mean(), merged.mean(), atol=1e-7, rtol=1e-7)
    torch.testing.assert_close(full.log_mass(), merged.log_mass(), atol=1e-7, rtol=1e-7)


def test_adaptive_fields_use_local_density_and_conserve_each_record():
    points = torch.tensor([[0.], [.01], [.02], [5.], [7.], [9.]])
    centers = torch.tensor([[0.], [7.]])
    alpha = adaptive_field_responsibilities(points, centers, memberships=2, density_k=3)
    torch.testing.assert_close(alpha.sum(1), torch.ones(len(points)))
    # The dense first chart receives a much smaller raw radius than the sparse chart.
    assert alpha[2, 0] > alpha[2, 1]
    assert alpha[4, 1] > alpha[4, 0]


def test_full_cluster_scope_is_enforced():
    code = FullClusterCode(("a", "b"), "private", "v1")
    code.require_selection(("b", "a"), "private")
    with pytest.raises(ValueError):
        code.require_selection(("a",), "private")
    with pytest.raises(PermissionError):
        code.require_selection(("a", "b"), "other")


def test_regrouping_is_a_partition():
    groups = random_partition(11, 3, torch.Generator().manual_seed(1))
    assert sorted(i for group in groups for i in group) == list(range(11))
    assert all(1 <= len(group) <= 3 for group in groups)


def test_noise_is_opt_in_and_has_gradients():
    x = torch.randn(3, 8, requires_grad=True)
    assert storage_noise(x) is x
    y = storage_noise(x, noise_std=0.01, quantization_step=0.01)
    assert not torch.equal(x, y)
    y.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))


def test_relu_counterexample_is_not_silently_assumed_compactable():
    points = torch.tensor([-1.0, 1.0])
    query = torch.tensor([[-2.0], [0.0], [2.0]])
    full = torch.relu(query - points).sum(-1)
    compact = 2 * torch.relu(query[:, 0] - points.mean())
    assert not torch.equal(full, compact)
