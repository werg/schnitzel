import json


def test_probe_metrics_reconcile_to_saved_step_and_drop_torn_tail(tmp_path):
    from schnitz.checkpoints import reconcile_metrics
    path = tmp_path / 'metrics.jsonl'
    path.write_text('{"step": 20, "loss": 1}\n{"step": 40, "loss": 0.5}\n{"step":')
    reconcile_metrics(tmp_path, 25)
    assert [json.loads(line) for line in path.read_text().splitlines()] == [{'step': 20, 'loss': 1}]
    reconcile_metrics(tmp_path, 0)
    assert path.read_text() == ''
    reconcile_metrics(tmp_path / 'not-created', 0)
