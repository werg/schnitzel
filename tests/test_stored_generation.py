"""Candidate-free generation keeps routing eligibility and payload controls intact."""
import importlib.util
from pathlib import Path

import pytest
from safetensors.torch import load_model

from schnitz.agent import SchnitzelAgent
from schnitz.data import make_multiuse_world, save_episodes
from schnitz.evaluation import build_shared_bank
from schnitz.store import DiskStore
from schnitz.training import train


@pytest.mark.parametrize('learned', [False, True])
def test_generation_uses_stored_records_and_fixed_plans(tmp_path, tiny_config, monkeypatch, learned):
    tiny_config.train.steps = 1
    tiny_config.train.max_prompt_tokens = 1500
    run = tmp_path / 'run'
    train(tiny_config, run)
    agent = SchnitzelAgent(tiny_config).eval()
    load_model(agent, str(run / 'model.safetensors'))
    episodes = make_multiuse_world(2, bindings=2) + make_multiuse_world(3, bindings=2)
    path = tmp_path / 'episodes.jsonl'
    save_episodes(path, episodes)
    bank = tmp_path / 'bank.sqlite'
    store = DiskStore(bank)
    build_shared_bank(agent, store, episodes)
    calls, search = [], DiskStore.search

    def observe(self, query, **kwargs):
        calls.append(kwargs)
        return search(self, query, **kwargs)

    monkeypatch.setattr(DiskStore, 'search', observe)
    script = Path(__file__).parents[1] / 'scripts/evaluate_stored_generation.py'
    spec = importlib.util.spec_from_file_location('stored_generation', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.evaluate(run, bank, path, 2, 2, learned_world=learned, read_budget=2)
    rows = {(r['episode'], r['condition']): r for r in result['rows']}
    if learned:
        assert calls and all(c['top_k'] == 2 and len(c['exclude_ids']) == 4 for c in calls)
    else:
        assert not calls
    for episode in episodes:
        normal = rows[episode.episode_id, 'all']['selected_ids']
        if learned:
            assert len(normal) == 2
        else:
            assert set(normal) == set(episode.required_ids)
        assert set(normal) <= {s.record_id for s in episode.supports}
        assert rows[episode.episode_id, 'zero_values']['selected_ids'] == normal
        assert rows[episode.episode_id, 'none']['selected_ids'] == []
