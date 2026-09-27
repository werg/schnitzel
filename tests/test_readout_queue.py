import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest

from schnitz.operations import request_stop, run_lock, stop_requested
from schnitz.trajectories import file_sha256


@pytest.mark.parametrize('active_child', [False, True])
def test_readout_resume_clears_only_released_child_controls(tmp_path, monkeypatch, active_child):
    repo = Path(__file__).parents[1]
    run = runpy.run_path(str(repo/'experiments/binding-freshness-readout-20260920/run.py'))['run']
    queue = runpy.run_path(str(repo/'experiments/binding-compact-aware-20260920/run_confirmation.py'))
    calls = []
    queue['run_jobs'] = lambda root, jobs, **kwargs: calls.append(jobs)
    monkeypatch.setitem(run.__globals__, 'runpy', SimpleNamespace(run_path=lambda _: queue))
    monkeypatch.setitem(run.__globals__, 'signal', SimpleNamespace(SIGTERM=15, SIGINT=2, signal=lambda *a: None))
    root = tmp_path/'study'
    root.mkdir()
    (root/'episodes.jsonl').write_text('fixture')
    checkpoint = tmp_path/'checkpoint'
    checkpoint.mkdir()
    (checkpoint/'manifest.json').write_text('{}')
    monkeypatch.setitem(run.__globals__, 'resolve_checkpoint', lambda *a, **kw: checkpoint)
    (root/'inputs.json').write_text(json.dumps({'episodes_sha256': file_sha256(root/'episodes.jsonl'),
        'models': {'source': {'checkpoint': str(checkpoint),
                             'manifest_sha256': file_sha256(checkpoint/'manifest.json')}},
        'representations': ['payload', 'reader'], 'steps': 1600, 'heldout_worlds': 32}))
    child = root/'source/payload'
    request_stop(root)
    request_stop(child)
    with pytest.raises(RuntimeError, match='stop requested'):
        run(root)
    assert not calls and stop_requested(child)
    if active_child:
        with run_lock(child, clear_stop=False):
            with pytest.raises(RuntimeError, match='already running'):
                run(root, resume=True)
            assert stop_requested(child)
        assert not calls
    else:
        run(root, resume=True)
        assert not stop_requested(root) and not stop_requested(child)
        assert len(calls) == 3
