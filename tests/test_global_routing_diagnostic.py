import importlib.util
from pathlib import Path

import pytest
import torch

from schnitz.store import DiskStore, StoredRecord

spec = importlib.util.spec_from_file_location('global_diagnostic',
    Path(__file__).resolve().parents[1] / 'scripts/evaluate_global_routing.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_nested_candidate_pools_are_order_independent_and_full_bank_has_no_exclusions(tmp_path):
    worlds = ['a', 'b', 'c', 'd']
    pools = [module.pool_worlds(worlds, 'b', n) for n in (1, 2, 4)]
    assert pools[0] == {'b'} and pools[0] < pools[1] < pools[2] == set(worlds)
    assert module.pool_worlds(list(reversed(worlds)), 'b', 2) == pools[1]
    # Expanding the diagnostic pool must retain the store's real time/domain gates.
    store = DiskStore(tmp_path / 'bank.sqlite')
    for world in worlds:
        store.put(StoredRecord(world, torch.ones(2), torch.ones(2), created_at=1))
    store.put(StoredRecord('future', torch.ones(2), torch.ones(2), created_at=5))
    store.put(StoredRecord('private', torch.ones(2), torch.ones(2), domain='private'))
    for pool in pools:
        plan = store.search(torch.ones(2), top_k=20, query_time=5, exclude_ids=frozenset(worlds) - pool)
        assert {s.record_id for s in plan.selections} == pool


@pytest.mark.parametrize('worlds,target,size', [(['a','a'],'a',1), (['a'],'b',1), (['a'],'a',0), (['a'],'a',2)])
def test_candidate_pool_rejects_invalid_scope(worlds, target, size):
    with pytest.raises(ValueError):
        module.pool_worlds(worlds, target, size)


def test_query_subset_does_not_shrink_candidate_universe():
    from schnitz.data import make_multiuse_world
    episodes = [e for seed in range(4) for e in make_multiuse_world(seed, bindings=2)]
    selected, names = module.query_subset(episodes, 2)
    assert len(selected) == 20 and len(names) == 2
    assert len(episodes) == 40  # caller retains the full candidate corpus
    worlds = list(dict.fromkeys(e.environment for e in episodes))
    assert module.pool_worlds(worlds, selected[0].environment, 4) == set(worlds)
    assert module.query_subset(episodes, None)[0] == episodes
    for invalid in (0, 5, True):
        with pytest.raises(ValueError):
            module.query_subset(episodes, invalid)
