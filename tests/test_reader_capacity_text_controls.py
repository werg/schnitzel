"""Supplemental controls must not clear an active child's stop request."""
import importlib.util
import json
from pathlib import Path

import pytest

from schnitz.operations import control_dir, run_lock
from schnitz.trajectories import file_sha256


def setup(tmp_path, monkeypatch):
    repository = Path(__file__).resolve().parents[1]
    script = repository/'experiments/binding-reader-capacity-20260920/run_text_controls.py'
    spec = importlib.util.spec_from_file_location('capacity_text_controls', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv('GIT_WORK_TREE', str(repository))
    root = tmp_path/'study'
    root.mkdir()
    episodes = root/'episodes.jsonl'
    episodes.write_text('fixture')
    stage = root/'width'
    stage.mkdir()
    (stage/'manifest.json').write_text(json.dumps({'step': 2}))
    inputs = {'configs': {'width': {}}, 'steps': 2, 'heldout_sha256': file_sha256(episodes)}
    (root/'inputs.json').write_text(json.dumps(inputs))
    relative = 'experiments/binding-fixed-query-detail-20260920/text_control.py'
    (root/'text-controls.json').write_text(json.dumps({'script': relative,
        'script_sha256': file_sha256(repository/relative), 'arms': ['width'],
        'episodes': str(episodes), 'episodes_sha256': file_sha256(episodes)}))
    primary = root/'confirmation/width'
    primary.mkdir(parents=True)
    (primary/'results.json').write_text(json.dumps({'inputs': {
        'checkpoint_manifest_sha256': file_sha256(stage/'manifest.json'),
        'episodes_sha256': file_sha256(episodes),
        'script_sha256': file_sha256(repository/'scripts/evaluate_oracle_transfer.py')}}))
    monkeypatch.setattr(module, 'resolve_checkpoint', lambda path, **_kwargs: path)
    calls = []
    queue = {'stop': lambda *_: None, 'check_stop': lambda *_: None,
             'run_jobs': lambda *_args, **_kwargs: calls.append('launched')}
    monkeypatch.setattr(module.runpy, 'run_path', lambda *_: queue)
    monkeypatch.setattr(module.signal, 'signal', lambda *_: None)
    return module, repository, root, calls


def test_resume_clears_only_inactive_text_child_control(tmp_path, monkeypatch):
    module, frozen, root, calls = setup(tmp_path, monkeypatch)
    child = root/'confirmation/width-text'
    stop = control_dir(child)/'STOP'
    stop.touch()
    with run_lock(child, clear_stop=False):
        with pytest.raises(RuntimeError, match='already running'):
            module.run(root, frozen, resume=True)
        assert stop.exists()
        assert not calls
    module.run(root, frozen, resume=True)
    assert not stop.exists()
    assert calls == ['launched']


def test_incomplete_primary_does_not_launch_text_controls(tmp_path, monkeypatch):
    module, frozen, root, calls = setup(tmp_path, monkeypatch)
    (root/'width/manifest.json').write_text(json.dumps({'step': 1}))
    with pytest.raises(ValueError, match='incomplete'):
        module.run(root, frozen)
    assert not calls
