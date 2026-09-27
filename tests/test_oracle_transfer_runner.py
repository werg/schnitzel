import json
from pathlib import Path
import sys

import pytest

from schnitz.data import make_multiuse_world, save_episodes
from schnitz.training import train
from schnitz.trajectories import file_sha256

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
try:
    import evaluate_oracle_transfer as evaluator
    from evaluate_oracle_transfer import run
finally:
    sys.path.pop(0)


def test_oracle_runner_offline_banks_free_generation_and_completed_restart(tmp_path, tiny_config, monkeypatch):
    path = tmp_path/'episodes.jsonl'
    episodes = make_multiuse_world(9, bindings=1)
    save_episodes(path, episodes)
    tiny_config.train.steps = 1
    tiny_config.train.episodes_file = str(path)
    tiny_config.train.evidence_scope = 'required'
    source = tmp_path/'source'
    train(tiny_config, source)
    output = tmp_path/'evaluation'
    run(source, path, output)
    report = json.loads((output/'results.json').read_text())
    assert len(report['generation_rows']) == len(episodes)*6
    assert set(report['writes']) == {'global', 'permission', 'restoration', 'identifier'}
    assert report['inputs']['routing'] == 'oracle required sources'
    hashes = {name: file_sha256(output/name) for name in ('bank.sqlite', 'results.json')}
    def forbidden(*args, **kwargs):
        raise AssertionError('Completed restart loaded a model')
    monkeypatch.setattr('evaluate_oracle_transfer.load_frozen_agent', forbidden)
    run(source, path, output)
    assert hashes == {name: file_sha256(output/name) for name in hashes}
    with (output/'bank.sqlite').open('ab') as handle:
        handle.write(b'changed')
    with pytest.raises(ValueError, match='bank changed'):
        run(source, path, output)


def test_endpoint_counterfactual_changes_only_endpoint_information():
    original = make_multiuse_world(8, bindings=2)
    changed = [evaluator.identifier_counterfactual(e) for e in original]
    shared_sources = {}
    for a, b in zip(original, changed):
        assert (a.episode_id, a.query, a.query_time, a.required_ids) == (b.episode_id, b.query, b.query_time, b.required_ids)
        assert (a.restore, a.allowed_capability, a.capability) == (b.restore, b.allowed_capability, b.capability)
        assert (a.answer != b.answer) == (a.task_family == 'multiuse/identifier')
        if a.task_family == 'multiuse/identifier':
            assert b.answer in b.choices
            assert b.answer in next(s.text for s in b.supports if s.record_id in b.required_ids)
        for old, new in zip(a.supports, b.supports):
            assert (old.record_id, old.created_at, old.kind) == (new.record_id, new.created_at, new.kind)
            if old.kind == 'restoration':
                assert old == new
            else:
                assert old.text.split('Its endpoint')[0] == new.text.split('Its endpoint')[0]
                assert old.text != new.text
            assert shared_sources.setdefault(new.record_id, new) == new


@pytest.mark.parametrize('stop_mode', ['control', 'signal', 'memory'])
def test_oracle_runner_resumes_scoring_and_generation_without_repeating_work(tmp_path, tiny_config, monkeypatch, stop_mode):
    import signal
    from schnitz.agent import SchnitzelAgent
    from schnitz.operations import control_dir, request_stop
    path = tmp_path/'episodes.jsonl'
    episodes = make_multiuse_world(11, bindings=1)
    save_episodes(path, episodes)
    tiny_config.train.steps = 1
    tiny_config.train.episodes_file = str(path)
    tiny_config.train.evidence_scope = 'required'
    if stop_mode == 'memory':
        tiny_config.train.min_system_available_bytes = 1
    source, full, partial = tmp_path/'source', tmp_path/'full', tmp_path/'partial'
    train(tiny_config, source)
    run(source, path, full)
    generate = evaluator.FrozenScorer.generate
    pressure = {'low': False}
    if stop_mode == 'memory':
        monkeypatch.setattr(evaluator, 'available_host_memory', lambda: 0 if pressure['low'] else 2**40, raising=False)
    calls = 0
    def interrupted(self, *args, **kwargs):
        nonlocal calls
        prediction = generate(self, *args, **kwargs)
        calls += 1
        if calls == 5:
            if stop_mode == 'control':
                request_stop(partial)
            elif stop_mode == 'signal':
                signal.raise_signal(signal.SIGTERM)
            else:
                pressure['low'] = True
        return prediction
    monkeypatch.setattr(evaluator.FrozenScorer, 'generate', interrupted)
    with pytest.raises(RuntimeError, match='Stopped'):
        run(source, path, partial)
    assert calls == 5
    progress = json.loads((partial/'generation-progress.json').read_text())
    assert len(progress['rows']) == 5
    (control_dir(partial)/'STOP').unlink(missing_ok=True)
    pressure['low'] = False
    calls = 0
    def count(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        return generate(self, *args, **kwargs)
    def forbidden(*args, **kwargs):
        raise AssertionError('Resume repeated committed writer/scoring work')
    monkeypatch.setattr(evaluator.FrozenScorer, 'generate', count)
    monkeypatch.setattr(evaluator, 'stored_transfer_evaluation', forbidden)
    monkeypatch.setattr(SchnitzelAgent, 'produce', forbidden)
    run(source, path, partial)
    assert calls == len(episodes)*6 - 5
    expected = json.loads((full/'results.json').read_text())
    actual = json.loads((partial/'results.json').read_text())
    assert actual['generation_rows'] == expected['generation_rows']
    assert {k: v for k, v in actual['scores'].items() if k != 'resources'} == {
        k: v for k, v in expected['scores'].items() if k != 'resources'}
    # A changed committed bank must invalidate partial progress too.
    (partial/'results.json').unlink()
    progress_path = partial/'generation-progress.json'
    committed = progress_path.read_text()
    changed = json.loads(committed)
    changed['rows'][0]['episode'] = 'wrong-query'
    progress_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='prefix changed'):
        run(source, path, partial)
    progress_path.write_text(committed)
    with (partial/'bank.sqlite').open('ab') as handle:
        handle.write(b'changed')
    with pytest.raises(ValueError, match='bank changed'):
        run(source, path, partial)


@pytest.mark.parametrize('method', ['mean', 'trained'])
def test_oracle_runner_reads_native_compact_codes_and_raw_subsets(tmp_path, tiny_config, method, monkeypatch):
    path = tmp_path/'episodes.jsonl'
    save_episodes(path, make_multiuse_world(9, bindings=1))
    c = tiny_config
    c.model.tiny_layers = 4
    c.model.recurrence_mode = 'middle_block'
    c.model.recurrent_start, c.model.recurrent_end = 1, 3
    c.model.loops, c.model.writer_loops = 3, 1
    c.memory.read_timing, c.memory.read_steps = 'loop_boundary', 1
    c.memory.compaction, c.memory.compact_records = 'synthetic', 1
    c.memory.compaction_warmup, c.memory.compaction_probability = 0, 1.
    c.train.steps, c.train.episodes_file, c.train.evidence_scope = 1, str(path), 'required'
    source, output = tmp_path/'source', tmp_path/'evaluation'
    train(c, source)
    run(source, path, output, compact_method=method)
    report = json.loads((output/'results.json').read_text())
    assert report['inputs']['compact_method'] == method
    assert all(size['codes'] == 1 for size in report['code_storage'].values())
    actions = [r for r in report['scores']['rows'] if r['task_family'] == 'multiuse/action']
    for row in actions:
        accounting = row['payload_accounting']
        if row['condition'] in {'all', 'cf_permission', 'cf_restoration', 'cf_identifier'}:
            assert any(a['clusters'] for a in accounting)
            assert not any(a['raw_fallback_ids'] for a in accounting)
        elif row['condition'].startswith('drop_'):
            assert any(a['raw_fallback_ids'] for a in accounting)
            assert not any(a['clusters'] for a in accounting)
    for row in report['generation_rows']:
        if row['task_family'] == 'multiuse/action' and row['condition'] == 'all':
            assert any(a['clusters'] for a in row['payload_accounting'])
    bank_hash = file_sha256(output/'bank.sqlite')
    (output/'results.json').unlink()
    from schnitz.agent import SchnitzelAgent
    def forbidden(*args, **kwargs):
        raise AssertionError('Restart re-encoded a source')
    monkeypatch.setattr(SchnitzelAgent, 'produce', forbidden)
    run(source, path, output, compact_method=method)
    resumed = json.loads((output/'results.json').read_text())
    assert resumed['generation_rows'] == report['generation_rows']
    assert file_sha256(output/'bank.sqlite') == bank_hash
    with pytest.raises(ValueError, match='identity changed'):
        run(source, path, output, compact_method='raw')


def test_oracle_offline_pressure_rolls_back_partial_bank(tmp_path, tiny_config, monkeypatch):
    from schnitz.agent import SchnitzelAgent
    from schnitz.store import DiskStore
    path = tmp_path/'episodes.jsonl'
    save_episodes(path, make_multiuse_world(17, bindings=1))
    tiny_config.train.steps = 1
    tiny_config.train.episodes_file = str(path)
    tiny_config.train.evidence_scope = 'required'
    tiny_config.train.min_system_available_bytes = 1
    source, output = tmp_path/'source', tmp_path/'output'
    train(tiny_config, source)
    pressure = {'low': False}
    original = SchnitzelAgent.produce
    def produce(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        pressure['low'] = True
        return result
    monkeypatch.setattr(SchnitzelAgent, 'produce', produce)
    monkeypatch.setattr(evaluator, 'available_host_memory', lambda: 0 if pressure['low'] else 2**40, raising=False)
    with pytest.raises(RuntimeError, match='Stopped'):
        run(source, path, output)
    with DiskStore(output/'bank.sqlite').connect() as db:
        assert db.execute('SELECT count(*) FROM records').fetchone()[0] == 0
    pressure['low'] = False
    monkeypatch.setattr(SchnitzelAgent, 'produce', original)
    run(source, path, output)
    report = json.loads((output/'results.json').read_text())
    assert all(writes['writer_calls'] == 2 for writes in report['writes'].values())
