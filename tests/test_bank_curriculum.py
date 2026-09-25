import pytest
import torch

from sdkb.agent import SDKBAgent
from sdkb.bank_curriculum import BankCurriculum

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
    agent = SDKBAgent(_direct_agent(tiny_config))
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
