"""Oracle-only producer work may shrink without changing the actual selected reads."""
from copy import deepcopy

import pytest
import torch
from safetensors.torch import load_file

from schnitz.agent import SchnitzelAgent
from schnitz.checkpoints import resolve_checkpoint
from schnitz.data import make_multiuse_world, save_episodes
from schnitz.training import train


@pytest.mark.parametrize('family', ['identifier', 'action'])
@pytest.mark.parametrize('replay', [False, True])
def test_selected_producers_preserve_updates_rng_and_full_selected_evidence(tmp_path, tiny_config, monkeypatch, replay, family):
    episodes = [e for e in make_multiuse_world(12, bindings=2) if e.task_family == 'multiuse/'+family]
    selected_count = len(episodes[0].required_ids)
    if family == 'action':
        tiny_config.model.tiny_layers = 3
        tiny_config.model.recurrence_mode = 'middle_block'
        tiny_config.model.recurrent_start, tiny_config.model.recurrent_end = 1, 2
        tiny_config.model.loops, tiny_config.model.writer_loops = 3, 1
        tiny_config.memory.read_timing = 'loop_boundary'
        tiny_config.memory.read_steps, tiny_config.memory.read_top_k = 2, 1
    data = tmp_path/'episodes.jsonl'
    save_episodes(data, episodes)
    tiny_config.train.episodes_file = str(data)
    tiny_config.train.live_fraction = 1.
    tiny_config.train.replay = replay
    tiny_config.train.steps = 2
    tiny_config.train.gradient_accumulation = 2
    tiny_config.memory.noise_std = .02
    original = SchnitzelAgent.produce
    calls = []
    def produce(self, source_ids):
        calls.append(source_ids.clone())
        return original(self, source_ids)
    monkeypatch.setattr(SchnitzelAgent, 'produce', produce)
    reference = tmp_path/'reference'
    train(tiny_config, reference)
    reference_calls = len(calls)
    calls.clear()
    selected = deepcopy(tiny_config)
    selected.train.selected_producers_only = True
    output = tmp_path/'selected'
    train(selected, output)
    assert len(calls) < reference_calls
    expected_calls = 4 * selected_count * (2 if replay else 1)
    assert len(calls) == expected_calls
    left, right = [resolve_checkpoint(p) for p in (reference, output)]
    a, b = [load_file(str(p/'model.safetensors')) for p in (left, right)]
    assert a.keys() == b.keys()
    assert all(torch.equal(a[k], b[k]) for k in a)
    a, b = [torch.load(p/'training_state.pt', weights_only=False) for p in (left, right)]
    def equal(x, y):
        if isinstance(x, torch.Tensor):
            return torch.equal(x, y)
        if isinstance(x, dict):
            return x.keys() == y.keys() and all(equal(x[k], y[k]) for k in x)
        if isinstance(x, (list, tuple)):
            return len(x) == len(y) and all(equal(u, v) for u, v in zip(x, y))
        return x == y
    assert equal(a, b)
    if family == 'action':
        import json
        rows = [json.loads(line) for line in (output/'metrics.jsonl').read_text().splitlines()]
        assert all(row['read_count'] == 2 for row in rows)


@pytest.mark.parametrize('field,value', [('live_fraction', .5), ('retrieval', 'learned'), ('arm', 'oracle_text')])
def test_selected_producers_reject_changed_cache_or_selection_semantics(tiny_config, field, value):
    tiny_config.train.selected_producers_only = True
    tiny_config.train.live_fraction = 1.
    setattr(tiny_config.train, field, value)
    with pytest.raises(ValueError, match='fully live oracle memory'):
        tiny_config.validate()


def test_selected_producer_policy_cannot_change_on_exact_resume(tmp_path, tiny_config):
    tiny_config.train.live_fraction = 1.
    path = tmp_path/'run'
    train(tiny_config, path, stop_after=1)
    tiny_config.train.selected_producers_only = True
    with pytest.raises(ValueError, match='hyperparameters differ'):
        train(tiny_config, path, resume=True)
