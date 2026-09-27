import copy

import pytest
import torch
from safetensors.torch import load_file

from schnitz.checkpoints import resolve_checkpoint
from schnitz.training import EpisodeSampler, train


def test_shuffled_passes_cover_every_episode_before_repeating():
    sampler = EpisodeSampler(11, seed=233)
    first = [sampler.index(i) for i in range(11)]
    second = [sampler.index(i) for i in range(11, 22)]
    assert set(first) == set(second) == set(range(11))
    assert first != second
    assert first == [EpisodeSampler(11, seed=233).index(i) for i in range(11)]
    with pytest.raises(ValueError):
        EpisodeSampler(0, seed=233)


def test_shuffled_passes_resume_exactly_after_partial_update(tmp_path, tiny_config, monkeypatch):
    from schnitz.operations import request_stop
    from schnitz.replay import ReplayTape

    config = copy.deepcopy(tiny_config)
    config.train.steps = 3
    config.train.gradient_accumulation = 3
    config.train.train_worlds = 4
    config.train.sampling_policy = 'shuffled_passes'
    config.train.checkpoint_every = 1000
    full, interrupted = tmp_path / 'full', tmp_path / 'interrupted'
    train(config, full)
    original = ReplayTape.backward
    calls = 0

    def stop_after_microbatch(self, *args, **kwargs):
        nonlocal calls
        result = original(self, *args, **kwargs)
        calls += 1
        if calls == 4:
            request_stop(interrupted)
        return result

    monkeypatch.setattr(ReplayTape, 'backward', stop_after_microbatch)
    partial = train(config, interrupted)
    assert partial['steps'] == 1
    monkeypatch.setattr(ReplayTape, 'backward', original)
    train(config, interrupted, resume=True)
    a, b = [load_file(str(resolve_checkpoint(path) / 'model.safetensors'))
            for path in (full, interrupted)]
    for name in a:
        torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)
