"""Optional telemetry must not discard completed optimizer work or mask failures."""
import copy
import json
import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file

from schnitz.checkpoints import resolve_checkpoint
from schnitz.tracking import Tracking
from schnitz.training import train


def equal_state(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal_state(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b, strict=True):
            equal_state(left, right)
    else:
        assert a == b


def test_wandb_runtime_failures_preserve_training_and_complete_state(tmp_path, tiny_config, monkeypatch):
    calls = []
    def fail_log(row):
        calls.append(('log', row['optimizer_step']))
        raise RuntimeError('backend failure with private diagnostic text')
    def fail_finish(**kwargs):
        calls.append(('finish', kwargs['exit_code']))
        raise RuntimeError('finish failed')
    fake = SimpleNamespace(define_metric=lambda *a, **kw: None, log=fail_log, finish=fail_finish)
    monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(init=lambda **kwargs: fake))
    config = copy.deepcopy(tiny_config)
    config.train.steps = 3
    config.train.checkpoint_every = 10000
    reference, tracked = tmp_path/'reference', tmp_path/'tracked'
    train(config, reference)
    config.train.wandb_mode = 'offline'
    assert train(config, tracked)['steps'] == 3
    left, right = resolve_checkpoint(reference, verify=True), resolve_checkpoint(tracked, verify=True)
    equal_state(load_file(str(left/'model.safetensors')), load_file(str(right/'model.safetensors')))
    equal_state(torch.load(left/'training_state.pt', weights_only=True),
                torch.load(right/'training_state.pt', weights_only=True))
    assert calls == [('log', 1), ('finish', 1)]
    files = list((tracked/'tracking').glob('failure-*.json'))
    assert len(files) == 1
    record = json.loads(files[0].read_text())
    assert [r['operation'] for r in record['errors']] == ['log', 'finish']
    assert record['errors'][0]['optimizer_step'] == 1
    assert 'private diagnostic text' not in files[0].read_text()


def test_finish_failure_does_not_replace_original_training_exception(tmp_path, tiny_config, monkeypatch):
    def fail_finish(**kwargs):
        raise RuntimeError('telemetry finish failed')
    fake = SimpleNamespace(define_metric=lambda *a, **kw: None, finish=fail_finish)
    monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(init=lambda **kwargs: fake))
    config = copy.deepcopy(tiny_config)
    config.train.wandb_mode = 'offline'
    with pytest.raises(ValueError, match='original training error'):
        with Tracking(config, tmp_path):
            raise ValueError('original training error')


def test_telemetry_error_record_failure_is_nonfatal(tmp_path, tiny_config, monkeypatch):
    from schnitz import tracking
    def fail_log(row):
        raise RuntimeError('telemetry failed')
    def fail_record(*args, **kwargs):
        raise OSError('diagnostic filesystem unavailable')
    fake = SimpleNamespace(define_metric=lambda *a, **kw: None, log=fail_log, finish=lambda **kw: None)
    monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(init=lambda **kwargs: fake))
    config = copy.deepcopy(tiny_config)
    config.train.wandb_mode = 'offline'
    with Tracking(config, tmp_path) as tracker:
        monkeypatch.setattr(tracking, 'atomic_json', fail_record)
        tracker.log({'step': 1, 'loss': 2.})
        tracker.log({'step': 2, 'loss': 1.})


def test_new_attempt_retries_tracking_with_the_same_run_identity(tmp_path, tiny_config, monkeypatch):
    starts, logged = [], []
    def log(row):
        logged.append(row)
        if len(logged) == 1:
            raise RuntimeError('first attempt backend failure')
    fake = SimpleNamespace(define_metric=lambda *a, **kw: None, log=log, finish=lambda **kw: None)
    monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(init=lambda **kw: starts.append(kw) or fake))
    config = copy.deepcopy(tiny_config)
    config.train.wandb_mode = 'offline'
    for step in (1, 2):
        with Tracking(config, tmp_path) as tracker:
            tracker.log({'step': step, 'loss': 1.})
    assert starts[0]['id'] == starts[1]['id']
    assert [row['optimizer_step'] for row in logged] == [1, 2]
    assert logged[0]['resume_attempt'] != logged[1]['resume_attempt']


def test_requested_tracking_setup_failure_still_prevents_training(tmp_path, tiny_config, monkeypatch):
    def fail_init(**kwargs):
        raise RuntimeError('requested tracking setup failed')
    monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(init=fail_init))
    config = copy.deepcopy(tiny_config)
    config.train.wandb_mode = 'offline'
    output = tmp_path/'run'
    with pytest.raises(RuntimeError, match='requested tracking setup failed'):
        train(config, output)
    assert not (output/'CURRENT').exists()
