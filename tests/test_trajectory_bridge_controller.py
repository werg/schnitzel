from dataclasses import asdict
import json
from pathlib import Path
import runpy

import yaml

from schnitz.data import make_multiuse_world, save_episodes
from schnitz.operations import request_stop
from schnitz.trajectories import file_sha256


def test_bridge_resume_after_preflight_stop_and_completed_stage(tmp_path, tiny_config, monkeypatch):
    helper = runpy.run_path(str(Path(__file__).parents[1]/'experiments/trajectory-prefix-muon-20260920/run_bridge.py'))
    run = helper['run']
    root = tmp_path/'study'
    root.mkdir()
    data = root/'episodes.jsonl'
    save_episodes(data, make_multiuse_world(7, bindings=1))
    tiny_config.train.episodes_file = str(data)
    config = root/'recurrence_bridge.yaml'
    config.write_text(yaml.safe_dump(asdict(tiny_config)))
    (root/'inputs.json').write_text(json.dumps(dict(config_sha256=file_sha256(config),
        train=str(data), train_sha256=file_sha256(data), validation=str(data),
        validation_sha256=file_sha256(data), revision=tiny_config.model.revision,
        steps=tiny_config.train.steps)))
    def probe(config):
        request_stop(root)
        return {'preflight': 'fixture'}
    monkeypatch.setitem(run.__globals__, 'model_probe', probe)
    run(root)
    assert not (root/'recurrence_bridge').exists()
    # Resume must start a fresh stage because stopping happened before training.
    run(root, resume=True)
    assert json.loads((root/'training-result.json').read_text())['steps'] == tiny_config.train.steps
    before = {str(p.relative_to(root)): file_sha256(p) for p in root.rglob('*') if p.is_file()}
    def forbidden(*args, **kwargs):
        raise AssertionError('A verified complete stage must not be retrained or re-saved')
    monkeypatch.setitem(run.__globals__, 'train', forbidden)
    run(root, resume=True)
    after = {str(p.relative_to(root)): file_sha256(p) for p in root.rglob('*') if p.is_file()}
    assert before == after
