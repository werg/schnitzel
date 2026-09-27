"""Explicit reader-capacity forks retain all other weights and exact resume."""
from copy import deepcopy
import json

import pytest
import torch
from safetensors.torch import load_file

from schnitz.checkpoints import resolve_checkpoint
from schnitz.training import train


@pytest.mark.parametrize('width_factor', [1, 2])
def test_reader_reset_retains_every_other_initial_weight(tmp_path, tiny_config, width_factor):
    source = tmp_path/'source'
    train(tiny_config, source)
    before = load_file(str(resolve_checkpoint(source)/'model.safetensors'))
    config = deepcopy(tiny_config)
    config.memory.reader_width *= width_factor
    config.train.reinitialize_reader = True
    config.train.seed = 29
    target = tmp_path/'target'
    train(config, target, init_from=source)
    initial = next((target/'checkpoints').glob('step-000000000-*'))
    after = load_file(str(initial/'model.safetensors'))
    for name in before:
        if not name.startswith('reader.'):
            assert torch.equal(before[name], after[name]), name
    assert not torch.equal(before['reader.initial_query.weight'], after['reader.initial_query.weight'])
    provenance = json.loads((target/'initialization.json').read_text())
    assert provenance['reader_reinitialized']
    assert set(provenance['reinitialized_parameters']) == {k for k in after if k.startswith('reader.')}
    # Changing the reader does not authorize changing the storage/interface shape.
    invalid = deepcopy(config)
    invalid.memory.payload_dims = [48]
    with pytest.raises(ValueError, match='stored interface'):
        train(invalid, tmp_path/'invalid', init_from=source)


def test_reader_reset_is_explicit_and_resumes_without_reset(tmp_path, tiny_config):
    source = tmp_path/'source'
    train(tiny_config, source)
    config = deepcopy(tiny_config)
    config.memory.reader_width *= 2
    with pytest.raises(ValueError, match='reader architecture'):
        train(config, tmp_path/'implicit', init_from=source)
    config.train.reinitialize_reader = True
    config.train.optimizer = 'muon'
    config.train.steps = 3
    config.train.seed = 31
    full, resumed = tmp_path/'full', tmp_path/'resumed'
    train(config, full, init_from=source)
    train(config, resumed, init_from=source, stop_after=1)
    train(config, resumed, resume=True)
    a, b = [resolve_checkpoint(path) for path in (full, resumed)]
    left, right = [load_file(str(path/'model.safetensors')) for path in (a, b)]
    assert left.keys() == right.keys()
    assert all(torch.equal(left[k], right[k]) for k in left)
    def equal(x, y):
        if isinstance(x, torch.Tensor):
            return torch.equal(x, y)
        if isinstance(x, dict):
            return x.keys() == y.keys() and all(equal(x[k], y[k]) for k in x)
        if isinstance(x, (tuple, list)):
            return len(x) == len(y) and all(equal(u, v) for u, v in zip(x, y))
        return x == y
    left, right = [torch.load(path/'training_state.pt', weights_only=False) for path in (a, b)]
    assert equal(left, right)


def test_reader_reset_requires_warm_start(tmp_path, tiny_config):
    tiny_config.train.reinitialize_reader = True
    with pytest.raises(ValueError, match='requires a warm-start'):
        train(tiny_config, tmp_path/'fresh')
