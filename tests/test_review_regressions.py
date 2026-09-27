import copy
from dataclasses import replace

import pytest
import torch

from schnitz.agent import SchnitzelAgent
from schnitz.data import make_episode, make_boolean_world, save_episodes
from schnitz.evaluation import build_shared_bank, evaluate_transfer_run
from schnitz.sessions import read_session
from schnitz.store import DiskStore
from schnitz.training import train
from schnitz.trajectory_eval import evaluate_teacher_run


@pytest.mark.parametrize('timing', ['prefix', 'loop_boundary'])
def test_fixed_plans_do_not_search_and_revalidate_visibility(tmp_path, tiny_config, monkeypatch, timing):
    c = copy.deepcopy(tiny_config)
    c.memory.read_steps = 2
    c.train.retrieval = 'learned'
    if timing == 'loop_boundary':
        c.model.recurrence_mode = 'middle_block'
        c.model.recurrent_start, c.model.recurrent_end = 0, 1
        c.model.loops, c.model.writer_loops = 3, 1
        c.memory.read_timing = timing
    agent = SchnitzelAgent(c).eval()
    e = make_episode(0, distractors=2)
    store = DiskStore(tmp_path / 'bank.sqlite')
    build_shared_bank(agent, store, [e])
    prompt = agent.prompt_ids(e.query)
    args = dict(namespace='global', generation='frozen-v1', query_time=e.query_time)
    original = read_session(agent, store, prompt, **args)
    monkeypatch.setattr(store, 'search', lambda *a, **kw: pytest.fail('Ablation rerouted'))
    altered = read_session(agent, store, prompt, **args, fixed_plans=original.plans, ablate_values=True)
    assert altered.plans == original.plans
    assert altered.selected_ids == original.selected_ids
    with pytest.raises(ValueError, match='boundary'):
        read_session(agent, store, prompt, **(args | {'domain': 'other'}), fixed_plans=original.plans)
    store.delete('global', original.selected_ids[0][0])
    with pytest.raises(KeyError):
        read_session(agent, store, prompt, **args, fixed_plans=original.plans)


def test_compaction_teacher_evaluation_uses_stored_codes(tmp_path, tiny_config, monkeypatch):
    c = copy.deepcopy(tiny_config)
    c.memory.compaction, c.memory.compact_records = 'synthetic', 1
    c.memory.compaction_warmup, c.memory.compaction_probability = 0, 1.
    run = tmp_path / 'run'
    train(c, run)
    episodes = tmp_path / 'test.jsonl'
    save_episodes(episodes, [make_episode(42, distractors=0)])
    result = evaluate_teacher_run(run, episodes)
    assert 'compact' in result['summary']
    assert result['persistent_codes']['codes'] == 1


def test_learned_counterfactuals_keep_original_plans(tmp_path, tiny_config):
    c = copy.deepcopy(tiny_config)
    c.train.retrieval = 'learned'
    c.memory.read_steps = 2
    run = tmp_path / 'run'
    train(c, run)
    path = tmp_path / 'boolean.jsonl'
    save_episodes(path, make_boolean_world(0))
    result = evaluate_transfer_run(run, path, boolean_counterfactuals=True)
    rows = {r['condition']: r for r in result['rows']}
    assert rows['all']['selected_ids'] == rows['zero_values']['selected_ids']
    assert rows['all']['selected_ids'] == rows['cf_a']['selected_ids'] == rows['cf_b']['selected_ids']


def test_provided_context_cannot_be_used_as_verified_routing_labels(tmp_path, tiny_config):
    c = copy.deepcopy(tiny_config)
    c.train.retrieval = 'learned'
    e = replace(make_episode(0), support_annotation='provided_context', sufficient_groups=())
    path = tmp_path / 'data.jsonl'
    save_episodes(path, [e])
    c.train.episodes_file = str(path)
    with pytest.raises(ValueError, match='verified'):
        train(c, tmp_path / 'run')
    assert not (tmp_path / 'run/CURRENT').exists()


def test_learned_routing_rejects_dataset_without_competing_candidates(tmp_path, tiny_config):
    c = copy.deepcopy(tiny_config)
    c.train.retrieval = 'learned'
    path = tmp_path / 'data.jsonl'
    save_episodes(path, make_boolean_world(0, operations=('xor',)))
    c.train.episodes_file = str(path)
    with pytest.raises(ValueError, match='competing'):
        train(c, tmp_path / 'run')


def test_all_live_training_skips_unused_cache_and_resumes_exactly(tmp_path, tiny_config, monkeypatch):
    import schnitz.training as training
    c = copy.deepcopy(tiny_config)
    c.train.live_fraction = 1.
    monkeypatch.setattr(training, 'read_cached', lambda *a, **kw: pytest.fail('Unused cache read'))
    monkeypatch.setattr(training, 'persist_outputs', lambda *a, **kw: pytest.fail('Unused cache write'))
    full, resumed = tmp_path / 'full', tmp_path / 'resumed'
    assert train(c, full)['store']['records'] == 0
    train(c, resumed, stop_after=1)
    train(c, resumed, resume=True)
    from safetensors.torch import load_file
    a, b = [load_file(str(p / 'model.safetensors')) for p in (full, resumed)]
    for name in a:
        torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)


def test_legacy_evaluation_also_reuses_captured_plans(tmp_path, tiny_config, monkeypatch):
    import schnitz.sessions as sessions
    from schnitz.training import build_evaluation_store, stored_evaluation
    tiny_config.train.retrieval = 'learned'
    tiny_config.memory.read_steps = 2
    agent = SchnitzelAgent(tiny_config).eval()
    es = [make_episode(0, distractors=2)]
    store = DiskStore(tmp_path / 'bank.sqlite')
    build_evaluation_store(agent, store, es, 'frozen')
    original, captures = sessions.read_session, []
    def read(*args, **kwargs):
        if kwargs.get('ablate_values') or kwargs['namespace'].startswith('cf_'):
            assert kwargs.get('fixed_plans') is not None
            captures.append(kwargs['fixed_plans'])
        return original(*args, **kwargs)
    monkeypatch.setattr(sessions, 'read_session', read)
    stored_evaluation(agent, store, es, 'frozen')
    assert len(captures) == 3
