"""Evaluation and preflight must check shared-memory reserve before model allocation."""
import importlib

import pytest


@pytest.mark.parametrize('module_name,function_name', [
    ('evaluation_adapter', 'load_frozen_agent'),
    ('training', 'evaluate_run'),
    ('training', 'evaluate_episode_file'),
    ('evaluation', 'evaluate_transfer_run'),
    ('trajectory_eval', 'evaluate_teacher_run'),
    ('depth_eval', 'evaluate_depths'),
    ('probes', 'model_probe'),
])
def test_evaluation_rejects_memory_pressure_before_model(
        monkeypatch, tiny_config, tmp_path, module_name, function_name):
    module = importlib.import_module('schnitz.'+module_name)
    tiny_config.train.min_system_available_bytes = 1024
    monkeypatch.setattr('schnitz.runtime.available_host_memory', lambda: 512)
    if hasattr(module, 'config_from_run'):
        monkeypatch.setattr(module, 'config_from_run', lambda _path: tiny_config)
    if hasattr(module, 'load_episodes'):
        monkeypatch.setattr(module, 'load_episodes', lambda _path: [])
    def forbidden(*_args, **_kwargs):
        pytest.fail('Allocated model before checking host-memory reserve')
    monkeypatch.setattr(module, 'SchnitzelAgent', forbidden)
    if function_name == 'load_frozen_agent':
        args = (tiny_config, tmp_path)
    elif function_name == 'model_probe':
        args = (tiny_config,)
    elif function_name == 'evaluate_depths':
        tiny_config.model.recurrence_mode = 'middle_block'
        tiny_config.model.writer_loops = 1
        args = (tmp_path, tmp_path/'episodes.jsonl', tmp_path/'output')
    elif function_name == 'evaluate_run':
        args = (tmp_path,)
    else:
        args = (tmp_path, tmp_path/'episodes.jsonl')
    with pytest.raises(MemoryError, match='reserve'):
        getattr(module, function_name)(*args)
