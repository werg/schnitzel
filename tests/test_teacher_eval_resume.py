import json

import pytest

from schnitz.agent import SchnitzelAgent
from schnitz.data import make_episode, make_multiuse_world
from schnitz.store import DiskStore
from schnitz.trajectory_eval import build_teacher_bank, stored_teacher_evaluation


def test_teacher_bank_manifest_reuses_frozen_records_without_writer(tmp_path, tiny_config, monkeypatch):
    agent = SchnitzelAgent(tiny_config).eval()
    episodes = make_multiuse_world(7, bindings=1)
    store = DiskStore(tmp_path / 'bank.sqlite')
    identity = 'checkpoint-model-sha256'
    first = build_teacher_bank(agent, store, episodes, writer_identity=identity)
    assert first['writer_calls'] > 0
    with store.connect() as db:
        before = db.execute('SELECT * FROM records ORDER BY namespace,record_id,space').fetchall()
    monkeypatch.setattr(agent, 'produce', lambda *a, **k: pytest.fail('Completed bank re-encoded sources'))
    again = build_teacher_bank(agent, store, episodes, writer_identity=identity)
    assert again['writer_calls'] == 0
    with store.connect() as db:
        assert db.execute('SELECT * FROM records ORDER BY namespace,record_id,space').fetchall() == before
    with pytest.raises(ValueError, match='identity changed'):
        build_teacher_bank(agent, store, episodes, writer_identity='different-checkpoint')


def test_teacher_score_progress_preserves_original_plans_and_skips_completed_rows(tmp_path, tiny_config, monkeypatch):
    agent = SchnitzelAgent(tiny_config).eval()
    episodes = make_multiuse_world(8, bindings=1)[:1]
    store = DiskStore(tmp_path / 'bank.sqlite')
    build_teacher_bank(agent, store, episodes)
    full = stored_teacher_evaluation(agent, store, episodes)
    saved = {}
    def progress(rows, plans):
        saved['rows'] = json.loads(json.dumps(rows))
        saved['plans'] = json.loads(json.dumps(plans))
    def stop():
        return len(saved.get('rows', [])) == 1
    assert stored_teacher_evaluation(agent, store, episodes, progress=progress,
                                     want_stop=stop) is None
    assert len(saved['rows']) == 1 and episodes[0].episode_id in saved['plans']
    original = SchnitzelAgent.conditioned_nll
    calls = []
    def counted(self, *args, **kwargs):
        calls.append(True)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(SchnitzelAgent, 'conditioned_nll', counted)
    resumed = stored_teacher_evaluation(agent, store, episodes, rows=saved['rows'],
                                        saved_plans=saved['plans'], progress=progress)
    assert len(calls) == 3
    assert resumed['rows'] == full['rows']
    assert resumed['summary'] == full['summary']


def test_teacher_bank_namespace_failure_leaves_only_committed_scopes(tmp_path, tiny_config, monkeypatch):
    import schnitz.offline_bank as offline_bank
    agent = SchnitzelAgent(tiny_config).eval()
    episodes = make_multiuse_world(1, bindings=1) + make_multiuse_world(2, bindings=1)
    store = DiskStore(tmp_path / 'bank.sqlite')
    original = offline_bank.ensure_offline_records
    calls = []
    def interrupt(*args, **kwargs):
        calls.append(kwargs['namespace'])
        if len(calls) == 2:
            raise RuntimeError('interrupted namespace')
        return original(*args, **kwargs)
    monkeypatch.setattr(offline_bank, 'ensure_offline_records', interrupt)
    with pytest.raises(RuntimeError, match='interrupted namespace'):
        build_teacher_bank(agent, store, episodes, writer_identity='frozen')
    with store.connect() as db:
        scopes = db.execute('SELECT DISTINCT namespace FROM records').fetchall()
        manifests = db.execute('SELECT namespace FROM offline_banks').fetchall()
    assert scopes == manifests and len(scopes) == 1
    monkeypatch.setattr(offline_bank, 'ensure_offline_records', original)
    build_teacher_bank(agent, store, episodes, writer_identity='frozen')


def test_teacher_bank_cooperative_stop_resumes_committed_namespace(tmp_path, tiny_config, monkeypatch):
    import schnitz.offline_bank as offline_bank
    agent = SchnitzelAgent(tiny_config).eval()
    episodes = make_multiuse_world(11, bindings=1) + make_multiuse_world(12, bindings=1)
    store = DiskStore(tmp_path / 'bank.sqlite')
    original = offline_bank.ensure_offline_records
    calls = []
    def counted(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append((kwargs['namespace'], result))
        return result
    monkeypatch.setattr(offline_bank, 'ensure_offline_records', counted)
    assert not build_teacher_bank(agent, store, episodes, writer_identity='frozen',
                                  want_stop=lambda: len(calls) == 1)['complete']
    assert len(calls) == 1
    assert build_teacher_bank(agent, store, episodes, writer_identity='frozen')['complete']
    assert calls[1] == (calls[0][0], False)


def test_teacher_run_resumes_without_rewriting_checkpoint_or_scored_rows(tmp_path, tiny_config, monkeypatch):
    from schnitz.checkpoints import resolve_checkpoint
    from schnitz.data import save_episodes
    from schnitz.operations import request_stop
    from schnitz.training import train
    from schnitz.trajectory_eval import evaluate_teacher_run
    from schnitz.trajectories import file_sha256
    tiny_config.train.steps = 1
    source_run = tmp_path / 'trained'
    train(tiny_config, source_run)
    checkpoint = resolve_checkpoint(source_run, verify=True)
    before = {p.name: file_sha256(p) for p in checkpoint.iterdir() if p.is_file()}
    episodes = tmp_path / 'episodes.jsonl'
    save_episodes(episodes, make_multiuse_world(9, bindings=1)[:1])
    output = tmp_path / 'teacher-score'
    original = SchnitzelAgent.conditioned_nll
    calls = []
    def interrupted(self, *args, **kwargs):
        calls.append(True)
        value = original(self, *args, **kwargs)
        request_stop(output)
        return value
    monkeypatch.setattr(SchnitzelAgent, 'conditioned_nll', interrupted)
    assert evaluate_teacher_run(checkpoint, episodes, output=output)['status'] == 'checkpointed'
    assert len(calls) == 1 and not (output/'results.json').exists()
    assert len(json.loads((output/'progress.json').read_text())['rows']) == 1
    def counted(self, *args, **kwargs):
        calls.append(True)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(SchnitzelAgent, 'conditioned_nll', counted)
    complete = evaluate_teacher_run(checkpoint, episodes, output=output)
    assert complete['status'] == 'complete' and len(calls) == 4
    assert before == {p.name: file_sha256(p) for p in checkpoint.iterdir() if p.is_file()}
    monkeypatch.setattr(SchnitzelAgent, '__init__', lambda *a, **k: pytest.fail('Completed evaluation rebuilt model'))
    assert evaluate_teacher_run(checkpoint, episodes, output=output) == complete
    episodes.write_text(episodes.read_text() + '\n')
    with pytest.raises(ValueError, match='identity changed'):
        evaluate_teacher_run(checkpoint, episodes, output=output)


def test_teacher_writer_and_scoring_guard_compute_but_not_progress_io(tmp_path, tiny_config, monkeypatch):
    from contextlib import contextmanager
    import schnitz.runtime as runtime
    agent = SchnitzelAgent(tiny_config).eval()
    episodes = make_multiuse_world(14, bindings=1)[:1]
    store = DiskStore(tmp_path/'bank.sqlite')
    armed = []
    @contextmanager
    def guard(*args, **kwargs):
        armed.append(True)
        try:
            yield
        finally:
            armed.pop()
    monkeypatch.setattr(runtime, 'compute_watchdog', guard)
    original_produce = agent.produce
    def produce(*args, **kwargs):
        assert armed
        return original_produce(*args, **kwargs)
    monkeypatch.setattr(agent, 'produce', produce)
    build_teacher_bank(agent, store, episodes, writer_identity='frozen')
    original_nll = agent.conditioned_nll
    def nll(*args, **kwargs):
        assert armed
        return original_nll(*args, **kwargs)
    monkeypatch.setattr(agent, 'conditioned_nll', nll)
    stored_teacher_evaluation(agent, store, episodes,
                              progress=lambda rows, plans: assert_disarmed(armed))


def assert_disarmed(armed):
    assert not armed


def test_teacher_offline_writer_bounds_cached_encoded_sources(tmp_path, tiny_config):
    agent = SchnitzelAgent(tiny_config).eval()
    episodes = [make_episode(i, distractors=0) for i in range(70)]
    store = DiskStore(tmp_path/'bank.sqlite')
    report = build_teacher_bank(agent, store, episodes, writer_identity='frozen')
    assert report['unique_sources'] == 140
    assert report['writer_calls'] >= 140
    assert 0 < report['writer_peak_cached_sources'] <= 64
    assert store.sizes()['records'] == 280


def test_teacher_wrong_value_scopes_reuse_serialized_peer_values(tmp_path, tiny_config, monkeypatch):
    agent = SchnitzelAgent(tiny_config).eval()
    episodes = make_multiuse_world(15, bindings=1) + make_multiuse_world(16, bindings=1)
    reference, managed = DiskStore(tmp_path/'reference.sqlite'), DiskStore(tmp_path/'managed.sqlite')
    build_teacher_bank(agent, reference, episodes)
    original = agent.produce
    calls = []
    def counted(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(agent, 'produce', counted)
    result = build_teacher_bank(agent, managed, episodes, writer_identity='frozen')
    assert len(calls) == result['unique_sources'] == result['writer_calls']
    with reference.connect() as a, managed.connect() as b:
        query = 'SELECT * FROM records ORDER BY namespace,record_id,space'
        assert a.execute(query).fetchall() == b.execute(query).fetchall()
