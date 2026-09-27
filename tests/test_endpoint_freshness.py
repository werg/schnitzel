"""Breadth controls must preserve the sampled task/source-count/depth schedule."""
import importlib.util
from pathlib import Path


def load_module():
    path = Path(__file__).parents[1]/'scripts/make_endpoint_freshness_control.py'
    spec = importlib.util.spec_from_file_location('endpoint_freshness', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_slot_major_corpora_match_draws_and_source_prefix(tmp_path):
    module = load_module()
    small = module.build(4, 'freshness-test')
    large = module.build(16, 'freshness-test')
    assert len(small) == 22*4 and len(large) == 22*16
    for slot in range(22):
        assert small[slot*4:(slot+1)*4] == large[slot*16:slot*16+4]
    a = module.sampling_plan(small, seed=127, steps=100, accumulation=4)
    b = module.sampling_plan(large, seed=127, steps=100, accumulation=4)
    assert a['schedule_sha256'] == b['schedule_sha256']
    assert a['final_rng_state'] == b['final_rng_state']
    assert a['families'] == b['families']
    assert set(a['families']) == {'multiuse/identifier', 'multiuse/action', 'multiuse/permission', 'multiuse/restoration'}
    assert a['families']['multiuse/identifier'] > a['families']['multiuse/action']
    from schnitz.data import save_episodes
    from schnitz.episode_index import EpisodeIndex
    path = tmp_path/'episodes.jsonl'
    save_episodes(path, small)
    assert len(EpisodeIndex(path)) == len(small)


def test_freshness_requires_power_of_two_world_counts():
    import pytest
    module = load_module()
    for count in (0, 3, True):
        with pytest.raises(ValueError, match='power of two'):
            module.build(count, 'invalid')
