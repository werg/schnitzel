"""Calibration must use prior source/query content without answer annotations."""
from dataclasses import replace
import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def centered():
    path = Path(__file__).parents[1]/'experiments/binding-centered-lexical-20260920/run.py'
    spec = importlib.util.spec_from_file_location('centered_lexical_control', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_calibration_ignores_targets_and_rejects_future_sources(centered):
    from schnitz.data import make_multiuse_world
    episodes = make_multiuse_world(1, bindings=2)
    original = centered.calibrate(episodes, width=64, seed=11)
    changed = [replace(e, answer='unrelated target', choices=('other',), required_ids=()) for e in episodes]
    assert np.array_equal(original, centered.calibrate(changed, width=64, seed=11))
    future = replace(episodes[0].supports[0], created_at=episodes[0].query_time)
    bad = replace(episodes[0], supports=(future, *episodes[0].supports[1:]))
    with pytest.raises(ValueError, match='Future'):
        centered.calibrate([bad], width=64, seed=11)


def test_centered_index_reopens_without_calibration_text(centered, tmp_path):
    records = [dict(record_id='opaque-a', namespace='global', space='s0', generation='frozen-v1',
                    domain='research', created_at=1),
               dict(record_id='opaque-b', namespace='global', space='s0', generation='frozen-v1',
                    domain='private', created_at=1)]
    keys = np.stack([centered.DENSE['encode']({'alpha': 1}, width=64, seed=11),
                     centered.DENSE['encode']({'beta': 1}, width=64, seed=11)])
    centers = np.stack([keys.mean(0), keys.mean(0)])
    output = tmp_path/'index'
    centered.publish(output, records, keys, centers, {'width': 64, 'seed': 11})
    metadata, stored, persisted_centers = centered.reopen(output, {'width': 64, 'seed': 11})
    assert np.array_equal(persisted_centers, centers)
    assert all('terms' not in r for r in metadata)
    result = centered.rank(metadata, stored, centers[1], 'alpha', width=64, seed=11,
                           query_time=2, top_k=2)
    assert [r['record_id'] for r in result] == ['opaque-a']
    assert centered.rank(metadata, stored, centers[1], 'alpha', width=64, seed=11,
                         query_time=1, top_k=2) == []
    with (output/'vectors.npz').open('ab') as f:
        f.write(b'corrupt')
    with pytest.raises(ValueError, match='digest'):
        centered.reopen(output, {'width': 64, 'seed': 11})
