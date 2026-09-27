import copy
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from schnitz.checkpoints import resolve_checkpoint
from schnitz.store import DiskStore, StoredRecord
from schnitz.training import train


def test_interrupted_resume_matches_uninterrupted_with_cache_and_noise(tmp_path, tiny_config):
    config = copy.deepcopy(tiny_config)
    config.train.steps = 5
    config.train.train_worlds = 5
    config.train.live_fraction = 0.5
    config.train.checkpoint_every = 1
    config.memory.noise_std = 0.03
    config.memory.compaction = 'synthetic'
    config.memory.compact_records = 1
    config.memory.compaction_warmup = 0
    config.memory.compaction_probability = 0.5
    full, resumed = tmp_path / 'full', tmp_path / 'resumed'
    train(config, full)
    partial = train(config, resumed, stop_after=2)
    assert partial['stopped_early'] and partial['steps'] == 2
    # These writes and log rows simulate work performed after the last commit point.
    DiskStore(resumed / 'training_cache.sqlite').put(StoredRecord(
        'uncommitted', torch.ones(12), torch.ones(24), namespace='train'))
    with (resumed / 'metrics.jsonl').open('a') as handle:
        handle.write('{"step": 999, "loss": 0}\n{"step":')
    result = train(config, resumed, resume=True)
    assert result['steps'] == 5
    a = load_file(str(resolve_checkpoint(full) / 'model.safetensors'))
    b = load_file(str(resolve_checkpoint(resumed) / 'model.safetensors'))
    assert a.keys() == b.keys()
    for name in a:
        torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)
    def rows(path):
        # Wall time and allocator/host pressure are attempt telemetry, not
        # restored scientific state. Every training metric must still match.
        return [{k: v for k, v in json.loads(s).items() if k not in {'elapsed_seconds', 'memory'}}
                for s in (path / 'metrics.jsonl').read_text().splitlines()]
    assert rows(full) == rows(resumed)
    with DiskStore(resumed / 'training_cache.sqlite').connect() as db:
        assert db.execute("SELECT count(*) FROM records WHERE record_id='uncommitted'").fetchone()[0] == 0
    assert len(list((resumed / 'checkpoints').glob('step-*'))) == 2


def test_partial_publish_keeps_old_checkpoint(tmp_path, tiny_config, monkeypatch):
    import schnitz.checkpoints as checkpoints
    run = tmp_path / 'run'
    original = checkpoints._atomic_text
    writes = 0
    def fail(path, text):
        nonlocal writes
        if Path(path).name == 'CURRENT':
            writes += 1
            if writes == 2:
                raise OSError('simulated interrupted pointer update')
        return original(path, text)
    monkeypatch.setattr(checkpoints, '_atomic_text', fail)
    with pytest.raises(OSError, match='simulated'):
        train(tiny_config, run)
    cp = resolve_checkpoint(run, verify=True)
    assert json.loads((cp / 'manifest.json').read_text())['step'] == 0
    monkeypatch.setattr(checkpoints, '_atomic_text', original)
    assert train(tiny_config, run, resume=True)['steps'] == tiny_config.train.steps


def test_detects_corrupt_optimizer_before_loading(tmp_path, tiny_config):
    run = tmp_path / 'run'
    train(tiny_config, run)
    path = resolve_checkpoint(run) / 'training_state.pt'
    with path.open('ab') as handle:
        handle.write(b'bad')
    with pytest.raises(ValueError, match='checksum'):
        resolve_checkpoint(run, verify=True)


def test_direct_checkpoint_path_still_verifies_its_manifest(tmp_path, tiny_config):
    run = tmp_path / 'run'
    train(tiny_config, run)
    checkpoint = resolve_checkpoint(run)
    assert resolve_checkpoint(checkpoint, verify=True) == checkpoint
    with (checkpoint / 'training_state.pt').open('ab') as handle:
        handle.write(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        resolve_checkpoint(checkpoint, verify=True)


def test_raw_to_paired_compaction_warm_start(tmp_path, tiny_config):
    raw = tmp_path / 'raw'
    train(tiny_config, raw)
    compact = copy.deepcopy(tiny_config)
    compact.memory.compaction = 'synthetic'
    compact.memory.compact_records = 1
    compact.memory.compaction_objective = 'paired'
    compact.memory.compaction_warmup = 0
    compact.memory.compaction_probability = 1.
    compact.train.steps = 1
    run = tmp_path / 'compact'
    result = train(compact, run, init_from=raw)
    assert result['steps'] == 1
    assert result['last']['compact_nll'] > 0
    provenance = json.loads((run / 'initialization.json').read_text())
    assert provenance['optimizer_reset']
    assert all(name.startswith('compactor.') for name in provenance['missing_initialized'])
    with pytest.raises(ValueError, match='resume or warm-start'):
        train(compact, run, resume=True, init_from=raw)


def test_compactor_only_does_not_move_raw_system(tmp_path, tiny_config):
    raw = tmp_path / 'raw'
    train(tiny_config, raw)
    original = load_file(str(resolve_checkpoint(raw) / 'model.safetensors'))
    compact = copy.deepcopy(tiny_config)
    compact.memory.compaction = 'synthetic'
    compact.memory.compact_records = 1
    compact.memory.compaction_objective = 'paired'
    compact.memory.compaction_warmup = 0
    compact.memory.compaction_probability = 1.
    compact.train.optimization_scope = 'compactor'
    compact.train.live_fraction = 0.
    run = tmp_path / 'compact'
    train(compact, run, init_from=raw)
    after = load_file(str(resolve_checkpoint(run) / 'model.safetensors'))
    for name, tensor in original.items():
        torch.testing.assert_close(after[name], tensor, rtol=0, atol=0)
    initial = load_file(str(sorted((run / 'checkpoints').glob('step-000000000-*'))[0] / 'model.safetensors'))
    assert any(not torch.equal(initial[k], after[k]) for k in after if k.startswith('compactor.'))


@pytest.mark.parametrize('native_compaction', [False, 'interleaved', 'paired'])
def test_mid_accumulation_emergency_resume_is_exact(tmp_path, tiny_config, monkeypatch, native_compaction):
    from schnitz.operations import request_stop
    from schnitz.replay import ReplayTape
    config = copy.deepcopy(tiny_config)
    config.train.steps = 3
    config.train.gradient_accumulation = 3
    config.train.checkpoint_every = 1000
    config.train.live_fraction = .5
    config.memory.noise_std = .03
    if native_compaction:
        config.model.tiny_layers = 4
        config.model.recurrence_mode = 'middle_block'
        config.model.recurrent_start, config.model.recurrent_end = 1, 3
        config.model.loops, config.model.writer_loops = 3, 1
        config.train.loop_counts = [2, 3]
        config.memory.read_timing, config.memory.read_steps = 'loop_boundary', 1
        config.memory.compaction, config.memory.compact_records = 'synthetic', 1
        config.memory.compaction_objective = native_compaction
        config.memory.behavior_kl_weight = .2
        config.memory.compaction_warmup = 0
        config.memory.compaction_probability = .5
        config.memory.compaction_grouping = 'random'
    full, interrupted = tmp_path / 'full', tmp_path / 'interrupted'
    train(config, full)
    original = ReplayTape.backward
    calls = 0
    def stop_after_microbatch(self, *args, **kwargs):
        nonlocal calls
        result = original(self, *args, **kwargs)
        calls += 1
        if calls == 4:
            request_stop(interrupted)
        return result
    monkeypatch.setattr(ReplayTape, 'backward', stop_after_microbatch)
    partial = train(config, interrupted)
    assert partial['steps'] == 1
    saved = torch.load(resolve_checkpoint(interrupted) / 'training_state.pt', weights_only=True)
    assert saved['accumulation']['microbatches'] == 1
    assert saved['gradients']
    assert saved['optimizer']['param_groups'][0]['lr'] == config.train.backbone_learning_rate
    monkeypatch.setattr(ReplayTape, 'backward', original)
    train(config, interrupted, resume=True)
    a = load_file(str(resolve_checkpoint(full) / 'model.safetensors'))
    b = load_file(str(resolve_checkpoint(interrupted) / 'model.safetensors'))
    for name in a:
        torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)
    a_state = torch.load(resolve_checkpoint(full) / 'training_state.pt', weights_only=True)
    b_state = torch.load(resolve_checkpoint(interrupted) / 'training_state.pt', weights_only=True)
    assert a_state['optimizer']['param_groups'] == b_state['optimizer']['param_groups']
    for index, state in a_state['optimizer']['state'].items():
        for name, value in state.items():
            torch.testing.assert_close(value, b_state['optimizer']['state'][index][name], rtol=0, atol=0)
    assert a_state['python_rng'] == b_state['python_rng']
    torch.testing.assert_close(a_state['torch_rng'], b_state['torch_rng'], rtol=0, atol=0)


def test_immutable_checkpoint_is_not_a_training_output(tmp_path, tiny_config):
    import hashlib
    run = tmp_path/'run'
    train(tiny_config, run)
    checkpoint = resolve_checkpoint(run)
    before = {p.relative_to(checkpoint): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in checkpoint.rglob('*') if p.is_file()}
    with pytest.raises(ValueError, match='Immutable checkpoint.*training output'):
        train(tiny_config, checkpoint, resume=True)
    after = {p.relative_to(checkpoint): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in checkpoint.rglob('*') if p.is_file()}
    assert after == before


def test_direct_restore_checks_dataset_identity(tmp_path, tiny_config):
    import hashlib
    import random
    from schnitz.agent import SchnitzelAgent
    from schnitz.checkpoints import restore_checkpoint
    from schnitz.optimizers import make_optimizer
    run = tmp_path/'run'
    train(tiny_config, run)
    agent = SchnitzelAgent(tiny_config)
    with pytest.raises(ValueError, match='Episode contents changed'):
        restore_checkpoint(agent, make_optimizer(agent), resolve_checkpoint(run), random.Random(1), 'changed')
    checkpoint = resolve_checkpoint(run)
    # Even an additional immutable diagnostic log must remain byte-identical.
    (checkpoint/'metrics.jsonl').write_text('{"step": 999, "loss": 0}\n')
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoint.iterdir() if p.is_file()}
    manifest = json.loads((checkpoint/'manifest.json').read_text())
    assert restore_checkpoint(agent, make_optimizer(agent), checkpoint, random.Random(1),
                              manifest['dataset_sha256']) == tiny_config.train.steps
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in checkpoint.iterdir() if p.is_file()}


@pytest.mark.parametrize('missing', ['global_python_rng', 'optimizer_parameter_names', 'optimizer_type'])
def test_incomplete_legacy_state_requires_explicit_warm_start(tmp_path, tiny_config, missing):
    import hashlib
    run = tmp_path/'run'
    train(tiny_config, run)
    checkpoint = resolve_checkpoint(run)
    state_path = checkpoint/'training_state.pt'
    state = torch.load(state_path, weights_only=True)
    del state[missing]
    torch.save(state, state_path)
    # Simulate a valid older checkpoint, not accidental byte corruption.
    manifest_path = checkpoint/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['sha256']['training_state.pt'] = hashlib.sha256(state_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='complete resume state'):
        train(tiny_config, run, resume=True)


@pytest.mark.parametrize('stop_at_saved_step', [0, 1])
def test_stop_during_save_reuses_that_committed_state(tmp_path, tiny_config, monkeypatch, stop_at_saved_step):
    from schnitz import training
    from schnitz.operations import request_stop
    config = copy.deepcopy(tiny_config)
    config.train.steps = 3
    config.train.checkpoint_every = 1 if stop_at_saved_step else 10000
    full, stopped = tmp_path/'full', tmp_path/'stopped'
    train(config, full)
    original = training.save_checkpoint
    saved_steps = []
    def save(*args, **kwargs):
        checkpoint = original(*args, **kwargs)
        saved_steps.append(args[3])
        if args[3] == stop_at_saved_step:
            request_stop(stopped)
        return checkpoint
    monkeypatch.setattr(training, 'save_checkpoint', save)
    result = train(config, stopped)
    assert result['steps'] == stop_at_saved_step and result['stopped_early']
    assert saved_steps == ([0] if stop_at_saved_step == 0 else [0, 1])
    monkeypatch.setattr(training, 'save_checkpoint', original)
    assert train(config, stopped, resume=True)['steps'] == 3
    a, b = resolve_checkpoint(full, verify=True), resolve_checkpoint(stopped, verify=True)
    a_weights, b_weights = load_file(str(a/'model.safetensors')), load_file(str(b/'model.safetensors'))
    for name, tensor in a_weights.items():
        torch.testing.assert_close(tensor, b_weights[name], rtol=0, atol=0)
    a_state = torch.load(a/'training_state.pt', weights_only=True)
    b_state = torch.load(b/'training_state.pt', weights_only=True)
    assert a_state['python_rng'] == b_state['python_rng']
    assert a_state['optimizer_parameter_names'] == b_state['optimizer_parameter_names']
    assert a_state['optimizer']['param_groups'] == b_state['optimizer']['param_groups']
    for index, state in a_state['optimizer']['state'].items():
        for key, value in state.items():
            torch.testing.assert_close(value, b_state['optimizer']['state'][index][key], rtol=0, atol=0)
    torch.testing.assert_close(a_state['torch_rng'], b_state['torch_rng'], rtol=0, atol=0)


def test_stopped_resume_commits_an_extended_step_budget(tmp_path, tiny_config, monkeypatch):
    from schnitz import training
    from schnitz.operations import request_stop
    config = copy.deepcopy(tiny_config)
    run = tmp_path/'run'
    train(config, run, stop_after=1)
    config.train.steps = 4
    original = training.restore_checkpoint
    def restore(*args, **kwargs):
        step = original(*args, **kwargs)
        request_stop(run)
        return step
    monkeypatch.setattr(training, 'restore_checkpoint', restore)
    result = train(config, run, resume=True)
    assert result['steps'] == 1 and result['stopped_early']
    assert training.config_from_run(run).train.steps == 4
