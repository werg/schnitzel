"""The study driver must not advance past interrupted training."""
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from schnitz.config import load_config
from schnitz.operations import control_dir
from schnitz.trajectories import file_sha256


@pytest.fixture
def study(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        'continuation_runner', repository / 'experiments/binding-continuation-20260919/run.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'manifest.json').write_text('{}')
    episodes = tmp_path / 'episodes.jsonl'
    episodes.write_text('identity fixture')
    root = tmp_path / 'study'
    root.mkdir()
    config = load_config(repository / 'configs/tiny_cpu.yaml')
    config.train.episodes_file = str(episodes)
    records = {}
    for arm in ('recurrent_core', 'all'):
        path = root / f'{arm}.yaml'
        path.write_text(yaml.safe_dump(asdict(config)))
        records[arm] = {'config': str(path), 'sha256': file_sha256(path),
                        'run': str(root / arm)}
    (root / 'inputs.json').write_text(json.dumps({
        'source_checkpoint': str(source),
        'source_manifest_sha256': file_sha256(source / 'manifest.json'),
        'episodes_sha256': file_sha256(episodes), 'configs': records}))
    return module, root


def test_stopped_first_arm_does_not_start_second(study, monkeypatch):
    module, root = study
    calls = []

    def interrupted(config, output, **kwargs):
        calls.append(output.name)
        assert kwargs['stop_output'] == root
        return {'stopped_early': True, 'stop_requested': True}

    monkeypatch.setattr(module, 'train', interrupted)
    module.run(root, False)
    assert calls == ['recurrent_core']
    state = json.loads((control_dir(root) / 'process.json').read_text())
    assert state['status'] == 'stopped'
    assert state['completed_arms'] == []


def test_failure_is_recorded_without_advancing(study, monkeypatch):
    module, root = study

    def failed(*args, **kwargs):
        raise RuntimeError('training failure')

    monkeypatch.setattr(module, 'train', failed)
    with pytest.raises(RuntimeError, match='training failure'):
        module.run(root, False)
    state = json.loads((control_dir(root) / 'process.json').read_text())
    assert state['status'] == 'failed'
    assert state['active_arm'] == 'recurrent_core'


def test_changed_config_fails_before_training(study, monkeypatch):
    module, root = study
    (root / 'all.yaml').write_text('changed')
    monkeypatch.setattr(module, 'train', lambda *args, **kwargs: pytest.fail('Training started'))
    with pytest.raises(ValueError, match='Prepared config changed'):
        module.run(root, False)


def test_resume_restores_existing_arm_and_warm_starts_only_new_arm(study, monkeypatch):
    module, root = study
    existing = root / 'recurrent_core'
    existing.mkdir()
    (existing / 'manifest.json').write_text(json.dumps({'step': 1}))
    monkeypatch.setattr(module, 'resolve_checkpoint', lambda path, **kwargs: path)
    calls = []

    def completed(config, output, **kwargs):
        calls.append((output.name, kwargs['resume'], kwargs['init_from']))
        return {'stopped_early': False, 'stop_requested': False}

    monkeypatch.setattr(module, 'train', completed)
    module.run(root, True)
    assert calls[0] == ('recurrent_core', True, None)
    assert calls[1][0:2] == ('all', False)
    assert calls[1][2].name == 'source'
    state = json.loads((control_dir(root) / 'process.json').read_text())
    assert state['status'] == 'complete'
    assert state['completed_arms'] == ['recurrent_core', 'all']


def test_invalid_arm_order_fails_before_training(study, monkeypatch):
    module, root = study
    path = root / 'inputs.json'
    inputs = json.loads(path.read_text())
    inputs['order'] = ['recurrent_core', 'recurrent_core']
    path.write_text(json.dumps(inputs))
    monkeypatch.setattr(module, 'train', lambda *args, **kwargs: pytest.fail('Training started'))
    with pytest.raises(ValueError, match='exactly once'):
        module.run(root, False)
