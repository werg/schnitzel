import json
from pathlib import Path

import pytest

from schnitz.cli import main
from schnitz.config import load_config
from schnitz.diagnostics import compact_probe, io_benchmark


def test_doctor_cli(capsys):
    main(['doctor'])
    assert 'torch' in json.loads(capsys.readouterr().out)


def test_teacher_evaluation_cli_accepts_checkpoint_specific_output(monkeypatch, capsys):
    import schnitz.trajectory_eval as trajectory_eval
    received = {}
    def fake_evaluate(run, episodes, *, max_episodes, generate_tokens, output):
        received.update(run=run, episodes=episodes, count=max_episodes,
                        tokens=generate_tokens, output=output)
        return {'status': 'complete'}
    monkeypatch.setattr(trajectory_eval, 'evaluate_teacher_run', fake_evaluate)
    main(['evaluate-teachers', '--run', 'run-a', '--episodes', 'heldout.jsonl',
          '--max-episodes', '16', '--generate-tokens', '96', '--output', 'run-a-step-100'])
    assert received == {'run': 'run-a', 'episodes': 'heldout.jsonl', 'count': 16,
                        'tokens': 96, 'output': 'run-a-step-100'}
    assert json.loads(capsys.readouterr().out)['status'] == 'complete'


def test_make_data_cli(tmp_path, capsys):
    target = tmp_path / 'episodes.jsonl'
    main(['make-data', '--output', str(target), '--count', '3'])
    assert len(target.read_text().splitlines()) == 3
    assert json.loads(capsys.readouterr().out)['count'] == 3
    with pytest.raises(FileExistsError):
        main(['make-data', '--output', str(target)])


def test_all_shipped_configs():
    root = Path(__file__).resolve().parents[1]
    for path in (root / 'configs').glob('*.yaml'):
        load_config(path)


def test_io_probe(tmp_path):
    result = io_benchmark(tmp_path / 'io.sqlite', records=8, reads=2, payload_dim=12,
                          key_dim=4, neighbors=2)
    assert result['ideal_value_bytes_per_read'] == 48
    assert result['fetch']['p50_ms'] >= 0


@pytest.mark.parametrize('kind', ['mlp', 'attention'])
def test_compactor_probe(kind):
    result = compact_probe(steps=2, reader_kind=kind)
    assert result['last_training_loss'] >= 0
    assert set(result['after']) == {'redundant', 'independent'}
