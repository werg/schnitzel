from dataclasses import replace
from pathlib import Path
import sys

import pytest

from schnitz.data import make_multiuse_world, save_episodes
from schnitz.episode_index import EpisodeIndex

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
try:
    from make_fixed_query_identifiers import fixed_query_episode
finally:
    sys.path.pop(0)


def test_fixed_query_preserves_sources_targets_and_causal_metadata(tmp_path):
    original = [e for i in range(2) for e in make_multiuse_world(i, bindings=2)]
    changed = [fixed_query_episode(e) for e in original]
    identifiers = [e for e in changed if e.task_family == 'multiuse/identifier']
    assert len({e.query for e in identifiers}) == 1
    assert len({e.answer for e in identifiers}) == 4
    for old, new in zip(original, changed, strict=True):
        if old.task_family != 'multiuse/identifier':
            assert new == old
            continue
        assert new.episode_id != old.episode_id
        assert new.provenance['fixed_query_parent_episode_id'] == old.episode_id
        for field in ('supports', 'answer', 'required_ids', 'sufficient_groups', 'query_time',
                      'environment', 'choices', 'restore', 'allowed_capability', 'capability'):
            assert getattr(old, field) == getattr(new, field)
    assert changed == [fixed_query_episode(e) for e in original]
    path = tmp_path/'episodes.jsonl'
    save_episodes(path, changed)
    index = EpisodeIndex(path)
    assert len(index) == len(changed)
    assert index[-1] == changed[-1]


def test_fixed_query_rejects_future_or_wrong_source_answer():
    episode = next(e for e in make_multiuse_world(5, bindings=2) if e.task_family == 'multiuse/identifier')
    for malformed in (replace(episode, query_time=0), replace(episode, answer='api_abcdef'),
                      replace(episode, required_ids=()), replace(episode, required_ids=('missing',))):
        with pytest.raises(ValueError):
            fixed_query_episode(malformed)
