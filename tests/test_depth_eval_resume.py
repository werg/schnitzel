import json

import pytest

from schnitz.agent import SchnitzelAgent
from schnitz.data import make_multiuse_world, save_episodes
from schnitz.operations import request_stop
from schnitz.training import train
from schnitz.depth_eval import evaluate_depths


def test_teacher_depth_sweep_resumes_same_bank_plans_and_rows(tmp_path, tiny_config, monkeypatch):
    tiny_config.model.tiny_layers = 4
    tiny_config.model.recurrence_mode = 'middle_block'
    tiny_config.model.recurrent_start, tiny_config.model.recurrent_end = 1, 3
    tiny_config.model.loops, tiny_config.model.writer_loops = 3, 1
    tiny_config.memory.read_timing = 'loop_boundary'
    tiny_config.memory.read_steps = 1
    tiny_config.train.live_fraction = 1.
    tiny_config.train.steps = 1
    run = tmp_path/'trained'
    train(tiny_config, run)
    episodes = tmp_path/'episodes.jsonl'
    save_episodes(episodes, make_multiuse_world(13, bindings=1)[:1])
    full, stopped = tmp_path/'full', tmp_path/'stopped'
    reference = evaluate_depths(run, episodes, full, depths=(1, 2),
                                protocol='teacher', max_episodes=1)
    assert reference['status'] == 'complete'
    original = SchnitzelAgent.conditioned_nll
    calls = []
    def interrupt(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        calls.append(True)
        if len(calls) == 5:
            request_stop(stopped)
        return result
    monkeypatch.setattr(SchnitzelAgent, 'conditioned_nll', interrupt)
    assert evaluate_depths(run, episodes, stopped, depths=(1, 2),
                           protocol='teacher', max_episodes=1)['status'] == 'checkpointed'
    assert len(calls) == 5 and not (stopped/'summary.json').exists()
    assert len(json.loads((stopped/'depth-2-progress.json').read_text())['rows']) == 1
    def forbid_writer(*args, **kwargs):
        raise AssertionError('Completed frozen depth bank re-encoded sources')
    monkeypatch.setattr(SchnitzelAgent, 'produce', forbid_writer)
    def counted(self, *args, **kwargs):
        calls.append(True)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(SchnitzelAgent, 'conditioned_nll', counted)
    resumed = evaluate_depths(run, episodes, stopped, depths=(1, 2),
                              protocol='teacher', max_episodes=1)
    assert resumed['status'] == 'complete' and len(calls) == 8
    for depth in (1, 2):
        assert json.loads((full/f'depth-{depth}.json').read_text())['rows'] == json.loads(
            (stopped/f'depth-{depth}.json').read_text())['rows']
    assert evaluate_depths(run, episodes, stopped, depths=(1, 2),
                           protocol='teacher', max_episodes=1) == resumed
    with pytest.raises(ValueError, match='identity changed'):
        evaluate_depths(run, episodes, stopped, depths=(1, 3),
                        protocol='teacher', max_episodes=1)
