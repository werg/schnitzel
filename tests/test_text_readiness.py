from dataclasses import asdict
import json
from pathlib import Path
import runpy

import pytest
import yaml

from schnitz.agent import SchnitzelAgent
from schnitz.data import make_multiuse_world, save_episodes
from schnitz.operations import request_stop


def test_checkpoint_text_evaluation_resumes_and_reads_immutable_checkpoint(tmp_path, tiny_config, monkeypatch):
    from safetensors.torch import save_model
    from schnitz.trajectories import file_sha256
    helper = runpy.run_path(str(Path(__file__).parents[1]/'experiments/trajectory-readiness-20260920/evaluate_text.py'))
    tiny_config.train.arm = 'oracle_text'
    tiny_config.train.evidence_scope = 'available'
    tiny_config.model.loops = 2
    config = tmp_path/'config.yaml'
    config.write_text(yaml.safe_dump(asdict(tiny_config)))
    episodes = tmp_path/'episodes.jsonl'
    save_episodes(episodes, make_multiuse_world(4, bindings=1)[:1])
    checkpoint = tmp_path/'immutable-checkpoint'
    checkpoint.mkdir()
    save_model(SchnitzelAgent(tiny_config), str(checkpoint/'model.safetensors'))
    digest = file_sha256(checkpoint/'model.safetensors')
    (checkpoint/'manifest.json').write_text(json.dumps({'format': 1, 'sha256': {'model.safetensors': digest}}))
    before = {p.name: file_sha256(p) for p in checkpoint.iterdir()}
    output = tmp_path/'score'
    def forbidden(*args, **kwargs):
        raise AssertionError('Checkpoint text scoring cannot call the writer')
    monkeypatch.setattr(SchnitzelAgent, 'produce', forbidden)
    helper['run'](config, episodes, output, checkpoint_run=checkpoint, depths=(1, 2))
    result = json.loads((output/'results.json').read_text())
    assert set(result['summary']) >= {'r1_selected_text', 'r1_none', 'r2_selected_text', 'r2_none'}
    assert len(result['rows']) == 4
    assert before == {p.name: file_sha256(p) for p in checkpoint.iterdir()}
    monkeypatch.setattr(SchnitzelAgent, '__init__', forbidden)
    helper['run'](config, episodes, output, checkpoint_run=checkpoint, depths=(1, 2))
    with pytest.raises(ValueError, match='inputs changed'):
        helper['run'](config, episodes, output, checkpoint_run=checkpoint, depths=(2,))


def test_text_readiness_resume_reuses_rows_and_never_writes_sources(tmp_path, tiny_config, monkeypatch):
    helper = runpy.run_path(str(Path(__file__).parents[1]/'experiments/trajectory-readiness-20260920/evaluate_text.py'))
    tiny_config.train.arm = 'oracle_text'
    tiny_config.train.evidence_scope = 'available'
    config = tmp_path/'config.yaml'
    config.write_text(yaml.safe_dump(asdict(tiny_config)))
    episodes = tmp_path/'episodes.jsonl'
    save_episodes(episodes, make_multiuse_world(3, bindings=1)[:2])
    def forbidden(*args, **kwargs):
        raise AssertionError('Text readiness must never invoke the writer')
    monkeypatch.setattr(SchnitzelAgent, 'produce', forbidden)
    full, partial = tmp_path/'full', tmp_path/'partial'
    helper['run'](config, episodes, full)
    original = SchnitzelAgent.conditioned_nll
    calls = []
    def stop(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        calls.append(True)
        request_stop(partial)
        return result
    monkeypatch.setattr(SchnitzelAgent, 'conditioned_nll', stop)
    helper['run'](config, episodes, partial)
    assert len(calls) == 1 and not (partial/'results.json').exists()
    assert len(json.loads((partial/'progress.json').read_text())['rows']) == 1
    def count(self, *args, **kwargs):
        calls.append(True)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(SchnitzelAgent, 'conditioned_nll', count)
    helper['run'](config, episodes, partial)
    assert len(calls) == 4
    a, b = [json.loads((p/'results.json').read_text()) for p in (full, partial)]
    assert a['rows'] == b['rows'] and a['summary'] == b['summary']
    monkeypatch.setattr(SchnitzelAgent, '__init__', forbidden)
    helper['run'](config, episodes, partial)
    episodes.write_text(episodes.read_text()+'\n')
    with pytest.raises(ValueError, match='inputs changed'):
        helper['run'](config, episodes, partial)
