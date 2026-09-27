import pytest
import torch

from schnitz.agent import SchnitzelAgent
from schnitz.bank_curriculum import BankCurriculum

from test_record_gradients import _bank, _direct_agent


def _rows(count):
    return [[f'g{row}a', f'g{row}b'] for row in range(count)]


def test_curriculum_stages_are_nested_and_rows_have_active_golds():
    records = sorted({r for row in _rows(50) for r in row} | {f'x{i:03d}' for i in range(100)})
    curriculum = BankCurriculum((20, 60, 0), _rows(50), records, seed=3,
                                min_steps=2, max_steps=10, window=2)
    small, large, full = (stage for stage, _ in curriculum.stages)
    assert len(small) == 20 and len(large) == 60 and full is None and small <= large
    for active, rows in curriculum.stages[:2]:
        assert rows and all(set(_rows(50)[row]) <= active for row in rows)
        assert sum(r.startswith('g') for r in active) <= len(active)
    assert len(curriculum.stages[2][1]) == 50
    drawn = {curriculum.row(step, offset, 2) for step in range(10) for offset in range(2)}
    assert drawn <= set(curriculum.rows)
    with pytest.raises(ValueError):
        BankCurriculum((60, 20, 0), _rows(50), records, seed=3)


def test_curriculum_advances_on_stability_or_budget_and_round_trips():
    records = sorted({r for row in _rows(50) for r in row})
    curriculum = BankCurriculum((20, 0), _rows(50), records, seed=1, min_steps=3,
                                max_steps=50, window=2, decay=0.5)
    # Unstable predictions hold the stage past the minimum.
    for step in range(1, 6):
        assert not curriculum.observe(step, prediction_cosine=0.5, value_cosine=1.0,
                                      recall=0.5)
    state = curriculum.state_dict()
    advanced = [curriculum.observe(step, prediction_cosine=1.0, value_cosine=1.0, recall=0.5)
                for step in range(6, 20)]
    assert any(advanced) and curriculum.stage == 1 and curriculum.active is None
    restored = BankCurriculum((20, 0), _rows(50), records, seed=1, min_steps=3,
                              max_steps=50, window=2, decay=0.5)
    restored.load_state_dict(state)
    assert restored.stage == 0 and restored.ema == state['ema']
    budget = BankCurriculum((20, 0), _rows(50), records, seed=1, min_steps=1, max_steps=2)
    assert not budget.observe(1, prediction_cosine=None, value_cosine=None, recall=None)
    assert budget.observe(2, prediction_cosine=None, value_cosine=None, recall=None)


def test_index_restriction_limits_both_search_paths(tiny_config, tmp_path):
    agent = SchnitzelAgent(_direct_agent(tiny_config))
    _, _, _, index, cache = _bank(agent, tmp_path, count=4)
    queries = torch.randn(2, index.spaces['s0'].keys.shape[1])
    kwargs = dict(top_k=20, namespace='corpus', space='s0', generation='g1',
                  domains=('research', 'research'), query_times=(10, 10))
    active = frozenset(cache.ids[::2])
    index.restrict(active)
    reference = index.search_batch(queries, **kwargs)
    index.use_device('cpu')
    device = index.search_batch(queries, **kwargs)
    for a, b in zip(reference, device, strict=True):
        ids = [s.record_id for s in a.selections]
        assert ids and set(ids) <= active and ids == [s.record_id for s in b.selections]
    assert set(index.eligible_ids('s0', cache.ids, domain='research', query_time=10)) <= active
    index.restrict(None)
    assert len(index.search_batch(queries, **kwargs)[0].selections) > len(ids)
    assert cache.stalest(len(cache.ids), active=active) == sorted(active)


def test_bank_load_respects_active_subset_and_query_eligibility():
    from schnitz.key_geometry import BankLoad
    ids = [f'r{i:03d}' for i in range(100)]
    load = BankLoad(ids, 1, threshold=2.0)
    active = set(ids[:10])
    load.restrict(active)
    load.begin_step()
    load.record(0, ids[:10])
    load.commit()
    load.begin_step()
    # Each active record carries its fair share, so none is overloaded; averaging
    # over the whole bank would have flagged all ten.
    assert not bool(load.overload(0, ids[:10]).any())
    eligible = torch.zeros(100, dtype=torch.bool)
    eligible[:5] = True
    drawn = load.explore(0, 20, seed=1, eligible=eligible.numpy())
    assert drawn and set(drawn) <= set(ids[:5])
    assert set(load.explore(0, 50, seed=2)) <= active
    assert load.statistics()[0]['cold_fraction'] == 0.0
    load.restrict(None)
    assert load.statistics()[0]['cold_fraction'] == pytest.approx(0.9)


def test_index_eligible_mask_matches_eligible_ids(tiny_config, tmp_path):
    agent = SchnitzelAgent(_direct_agent(tiny_config))
    _, _, _, index, cache = _bank(agent, tmp_path, count=4)
    index.restrict(cache.ids[::2])
    mask = index.eligible_mask('s0', domain='research', query_time=10)
    array = index.spaces['s0']
    assert set(array.ids[mask].tolist()) == set(
        index.eligible_ids('s0', cache.ids, domain='research', query_time=10))
    index.restrict(None)
    assert index.eligible_mask('s0', domain='research', query_time=10).sum() > mask.sum()


def test_index_eligible_mask_cache_is_bounded_and_correct(tiny_config, tmp_path, monkeypatch):
    import schnitz.key_index as key_index
    monkeypatch.setattr(key_index, 'ELIGIBLE_CACHE_ENTRIES', 2)
    agent = SchnitzelAgent(_direct_agent(tiny_config))
    _, _, _, index, _ = _bank(agent, tmp_path, count=4)
    first = index.eligible_mask('s0', domain='research', query_time=10).copy()
    for query_time in (0, 5, 10, 20, 1):
        index.eligible_mask('s0', domain='research', query_time=query_time)
    assert len(index._eligible) == 2
    assert (index.eligible_mask('s0', domain='research', query_time=10) == first).all()
    assert not index.eligible_mask('s0', domain='research', query_time=0).any()
