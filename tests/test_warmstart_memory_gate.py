"""A declared warm-start gate fork changes one parameter and resumes exactly."""
from copy import deepcopy
import json
import math

import pytest
import torch
from safetensors.torch import load_file

from schnitz.checkpoints import resolve_checkpoint
from schnitz.training import train


def native(config):
    config.model.tiny_layers = 3
    config.model.recurrence_mode = 'middle_block'
    config.model.recurrent_start, config.model.recurrent_end = 1, 2
    config.model.loops, config.model.writer_loops = 2, 1
    config.model.freeze_backbone = True
    config.memory.read_timing = 'loop_boundary'
    config.memory.read_steps = 1
    config.train.live_fraction = 1.
    config.train.optimizer = 'muon'
    return config


def equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            equal(x, y)
    else:
        assert a == b


def test_gate_override_changes_only_declared_warmstart_weight_and_resumes(tmp_path, tiny_config):
    source_config = native(tiny_config)
    source_config.train.steps = 1
    source = tmp_path/'source'
    train(source_config, source)
    before = load_file(str(resolve_checkpoint(source)/'model.safetensors'))
    config = deepcopy(source_config)
    config.train.steps = 3
    config.train.seed = 233
    config.train.warmstart_memory_gate = .30
    full, partial = tmp_path/'full', tmp_path/'partial'
    train(config, full, init_from=source)
    initial = next((full/'checkpoints').glob('step-000000000-*'))
    after = load_file(str(initial/'model.safetensors'))
    for name in before:
        if name == 'backbone.bridge.memory_logit':
            assert float(after[name].sigmoid()) == pytest.approx(.30)
        else:
            torch.testing.assert_close(before[name], after[name], rtol=0, atol=0, msg=name)
    provenance = json.loads((full/'initialization.json').read_text())
    assert provenance['warmstart_memory_gate'] == .30
    train(config, partial, init_from=source, stop_after=1)
    train(config, partial, resume=True)
    for filename in ('model.safetensors', 'training_state.pt'):
        a, b = [resolve_checkpoint(path)/filename for path in (full, partial)]
        if filename.endswith('.safetensors'):
            equal(load_file(str(a)), load_file(str(b)))
        else:
            equal(torch.load(a, weights_only=True), torch.load(b, weights_only=True))
    changed = deepcopy(config)
    changed.train.warmstart_memory_gate = .4
    with pytest.raises(ValueError, match='hyperparameters differ'):
        train(changed, partial, resume=True)


@pytest.mark.parametrize('value', [0., 1., -0.1, math.nan])
def test_gate_override_validates_probability(tiny_config, value):
    config = native(tiny_config)
    config.train.warmstart_memory_gate = value
    with pytest.raises(ValueError, match='memory gate'):
        config.validate()


def test_gate_override_requires_explicit_warmstart(tmp_path, tiny_config):
    config = native(tiny_config)
    config.train.warmstart_memory_gate = .3
    with pytest.raises(ValueError, match='warm-start'):
        train(config, tmp_path/'fresh')


def test_gate_override_is_exercised_by_model_preflight(tiny_config):
    from schnitz.probes import model_probe
    config = native(tiny_config)
    config.train.warmstart_memory_gate = .3
    report = model_probe(config)
    assert report['recurrence']['memory_mix'] == pytest.approx(.3)
    assert report['gradient_norms']['bridge_memory_projection'] > 0
