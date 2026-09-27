import pytest
import torch

from schnitz.key_geometry import (RetrievalLoad, effective_rank, geometry_summary, koleo_loss,
                               mean_direction_cosine, occurrence_statistics, uniformity,
                               variance_covariance_loss)


def _spread(n=512, d=32, seed=0):
    return torch.randn(n, d, generator=torch.Generator().manual_seed(seed))


def _collapsed(n=512, d=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    direction = torch.randn(d, generator=g)
    return direction + 0.01 * torch.randn(n, d, generator=g)


def test_collapse_metrics_separate_spread_from_collinear_keys():
    spread, collapsed = _spread(), _collapsed()
    assert mean_direction_cosine(collapsed) > 0.99 > 0.3 > mean_direction_cosine(spread)
    assert effective_rank(spread) > 20
    assert uniformity(spread) < uniformity(collapsed)
    low_rank = _spread()[:, :2] @ torch.randn(2, 32)
    assert effective_rank(low_rank) < 2.5


def test_regularizers_are_lower_for_spread_keys_and_differentiable():
    spread, collapsed = _spread(), _collapsed()
    variance_s, covariance_s = variance_covariance_loss(spread)
    variance_c, _ = variance_covariance_loss(collapsed)
    assert variance_s < variance_c
    assert covariance_s < 0.1
    assert koleo_loss(spread) < koleo_loss(collapsed)
    keys = collapsed.clone().requires_grad_(True)
    variance, covariance = variance_covariance_loss(keys)
    (variance + covariance + koleo_loss(keys)).backward()
    assert torch.isfinite(keys.grad).all() and keys.grad.abs().sum() > 0


def test_correlated_dimensions_raise_covariance():
    x = _spread()
    x[:, 1] = x[:, 0] + 0.01 * x[:, 1]
    assert variance_covariance_loss(x)[1] > variance_covariance_loss(_spread())[1]


def test_occurrence_statistics_detect_hubs():
    balanced = torch.arange(100).reshape(25, 4)
    hubbed = torch.cat([torch.zeros(25, 1, dtype=torch.long),
                        torch.arange(75).reshape(25, 3) % 10 + 1], 1)
    even, hub = occurrence_statistics(balanced, 100), occurrence_statistics(hubbed, 100)
    assert even['gini'] == pytest.approx(0.0, abs=1e-6)
    assert even['never_retrieved'] == 0
    assert hub['gini'] > 0.8 and hub['never_retrieved'] == pytest.approx(0.89)
    assert hub['max_share'] == pytest.approx(0.25) and hub['skewness'] > 3


def test_geometry_summary_reports_every_metric():
    summary = geometry_summary(_spread(256), _spread(64, seed=1), k=8)
    assert {'key_effective_rank', 'query_effective_rank', 'key_uniformity',
            'hubness_at_8'} <= summary.keys()


def test_retrieval_load_penalizes_only_hubs_and_explores_cold_records():
    load = RetrievalLoad(10, decay=0.9, threshold=3.0)
    for _ in range(20):
        load.update(torch.tensor([0, 0, 0, 1]))
    overload = load.overload(torch.arange(10))
    assert overload[0] > 0 and (overload[1:] == 0).all()
    queries = torch.randn(4, 8)
    keys = torch.randn(10, 8, requires_grad=True)
    load.penalty(queries, keys, torch.arange(10)).backward()
    assert keys.grad[0].abs().sum() > 0 and keys.grad[1:].abs().sum() == 0
    g = torch.Generator().manual_seed(0)
    picks = torch.cat([load.explore(torch.arange(10), 3, generator=g) for _ in range(200)])
    counts = torch.bincount(picks, minlength=10)
    assert counts[0] < counts[5] and counts[1] < counts[5]
    restored = RetrievalLoad(10, decay=0.9, threshold=3.0)
    restored.load_state_dict(load.state_dict())
    assert torch.equal(restored.counts, load.counts)
    with pytest.raises(ValueError):
        RetrievalLoad(11, decay=0.9, threshold=3.0).load_state_dict(load.state_dict())


def test_bank_load_uses_step_snapshots_and_restores_by_bank_identity():
    from schnitz.key_geometry import BankLoad
    load = BankLoad(['a', 'b', 'c', 'd'], spaces=2, decay=0.5, threshold=1.5)
    load.begin_step()
    load.record(0, ['a', 'a', 'b'])
    assert (load.overload(0, ['a']) == 0).all()  # snapshot is still empty
    load.commit()
    load.begin_step()
    assert load.overload(0, ['a'])[0] > 0 and load.overload(0, ['c'])[0] == 0
    assert (load.overload(1, ['a']) == 0).all()
    picks = [record for seed in range(200) for record in load.explore(0, 1, seed)]
    assert picks.count('a') < picks.count('c')
    assert 'c' not in load.explore(0, 3, 0, exclude={'c'})
    assert load.explore(0, 2, 7) == load.explore(0, 2, 7)
    stats = load.statistics()
    assert stats[0]['gini'] > 0 and stats[1]['cold_fraction'] == 1.0
    restored = BankLoad(['a', 'b', 'c', 'd'], spaces=2, decay=0.5, threshold=1.5)
    restored.load_state_dict(load.state_dict())
    assert torch.equal(restored.loads[0].counts, load.loads[0].counts)
    with pytest.raises(ValueError):
        BankLoad(['a', 'b', 'c', 'e'], spaces=2, decay=0.5,
                 threshold=1.5).load_state_dict(load.state_dict())


def test_read_selection_forces_gold_into_the_tail_and_keeps_size():
    from schnitz.spatial_training import _read_selection, _site_uniform
    found = ('x1', 'g1', 'x2', 'x3', 'x4')
    chosen, missing = _read_selection(found, ('g1', 'g2'), (), 4, True)
    assert missing == ('g2',) and chosen == ('g1', 'g2', 'x1', 'x2')
    chosen, missing = _read_selection(found, ('g1', 'g2'), (), 4, False)
    assert missing == () and chosen == ('g1', 'x1', 'x2', 'x3')
    chosen, _ = _read_selection(found, ('g1',), ('e1',), 4, False)
    assert chosen == ('g1', 'x1', 'x2', 'e1')
    # Always forcing reproduces the supplied-support reference ordering.
    reference = (('g1', 'g2') + tuple(r for r in found if r not in ('g1', 'g2')))[:4]
    assert _read_selection(found, ('g1', 'g2'), (), 4, True)[0] == reference
    item = {'episode_id': 'e', 'call_id': 'c', 'query_position': 3, 'training_step': 5}
    assert _site_uniform(item, 'gold') == _site_uniform(dict(item), 'gold')
    assert _site_uniform(item, 'gold') != _site_uniform(item | {'training_step': 6}, 'gold')
    draws = [_site_uniform(item | {'training_step': i}, 'gold') for i in range(2000)]
    assert 0.45 < sum(d < 0.5 for d in draws) / 2000 < 0.55


def test_gate_additive_floor_keeps_gradient_and_density_scales_with_reads():
    from schnitz.routing import AdaptiveDistanceGate
    torch.manual_seed(0)
    query = torch.randn(1, 8)
    candidates = torch.linspace(0.9, -0.5, 64)[None]
    selected = candidates.clone().requires_grad_(True)
    ones = torch.ones_like(candidates, dtype=torch.bool)
    support = torch.zeros_like(ones)
    support[0, -1] = True  # a distant forced gold record
    blocked = AdaptiveDistanceGate(8, min_temperature=.05, floor_mode='max')
    weights, stats = blocked(query, selected, candidates, ones, ones, support, 0.2)
    weights[0, -1].backward()
    assert float(weights[0, -1]) == pytest.approx(0.2) and selected.grad[0, -1] == 0
    assert stats['slope'][0, -1] == 0
    selected.grad = None
    passing = AdaptiveDistanceGate(8, min_temperature=.05, floor_mode='additive')
    weights, stats = passing(query, selected, candidates, ones, ones, support, 0.2)
    weights[0, -1].backward()
    assert weights[0, -1] >= 0.2 and selected.grad[0, -1] > 0
    assert stats['slope'][0, -1] > 0
    local = AdaptiveDistanceGate(8, density_k=8)
    scaled = AdaptiveDistanceGate(8, density_k=8, density_fraction=0.25)
    local_radius = local(query, candidates, candidates, ones, ones)[1]['radius']
    scaled_radius = scaled(query, candidates, candidates, ones, ones)[1]['radius']
    # 0.25 × 64 reads puts the boundary at the 16th record, below the 8th.
    assert scaled_radius < local_radius
    assert (scaled(query, candidates, candidates, ones, ones)[0].sum()
            > local(query, candidates, candidates, ones, ones)[0].sum())
    with pytest.raises(ValueError):
        AdaptiveDistanceGate(8, floor_mode='sometimes')
