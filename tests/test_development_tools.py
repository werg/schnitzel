import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from schnitz.data import make_boolean_world, counterfactual_boolean
from schnitz.probes import model_probe


@pytest.mark.parametrize('spaces', [1, 2])
def test_probe_reports_actual_backend_and_gradients(tiny_config, spaces):
    if spaces == 2:
        tiny_config.memory.payload_dims = [24, 48]
        tiny_config.memory.neighbors = [2, 1]
    result = model_probe(tiny_config)
    assert result['backend'] == 'tiny' and result['resolved_revision'] == 'local-tiny'
    assert result['one_loop_identity_max_error'] == 0
    assert result['zero_gate_two_loop_max_error'] == 0
    assert result['causal_prefix_max_error'] == 0
    assert all(value > 0 for value in result['gradient_norms'].values())
    if spaces == 2:
        for name in ('codec_0', 'codec_1', 'reader_local_0', 'reader_local_1',
                     'reader_fusion_0', 'address_map_0', 'address_map_1',
                     'query_map_0', 'query_map_1'):
            assert result['gradient_norms'][name] > 0


def test_boolean_counterfactual_preserves_nonpayload_channels():
    episodes = [e for i in range(4) for e in make_boolean_world(i, operations=('a', 'b', 'xor'))]
    for episode in episodes:
        for bit in ('a', 'b'):
            changed = counterfactual_boolean(episode, bit)
            assert (changed.episode_id, changed.query, changed.query_time, changed.required_ids) == (
                episode.episode_id, episode.query, episode.query_time, episode.required_ids)
            assert [s.record_id for s in changed.supports] == [s.record_id for s in episode.supports]
            assert [s.created_at for s in changed.supports] == [s.created_at for s in episode.supports]
            assert sum(a.text != b.text for a, b in zip(changed.supports, episode.supports, strict=True)) == 1
            should_flip = episode.task_family in {f'boolean/{bit}', 'boolean/xor'}
            assert (changed.answer != episode.answer) == should_flip


def test_matrix_prepares_pinned_information_matched_arms(tmp_path, tiny_config):
    script = Path(__file__).parents[1] / 'scripts' / 'prepare_matrix.py'
    spec = importlib.util.spec_from_file_location('prepare_matrix', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump(asdict(tiny_config)))
    result = module.prepare(config, tmp_path / 'matrix', steps=2, worlds=2, eval_worlds=2, bindings=1)
    assert len(result['runs']) == 7
    configs = [yaml.safe_load(Path(run['config']).read_text()) for run in result['runs']]
    assert len({c['train']['episodes_file'] for c in configs}) == 1
    assert len({run['test'] for run in result['runs']}) == 1
    assert (tmp_path / 'matrix' / 'run_all.sh').stat().st_mode & 0o100
    tiny_config.model.backend = 'hf'
    config.write_text(yaml.safe_dump(asdict(tiny_config)))
    with pytest.raises(ValueError, match='Pin'):
        module.prepare(config, tmp_path / 'bad')
    assert json.loads((tmp_path / 'matrix' / 'matrix.json').read_text())['runs'] == result['runs']
