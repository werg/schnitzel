from dataclasses import replace
from pathlib import Path
import sys

import pytest

from schnitz.data import make_multiuse_world, save_episodes
from schnitz.episode_index import EpisodeIndex

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
try:
    from make_endpoint_views import make_views, endpoint_view
finally:
    sys.path.pop(0)


def test_endpoint_views_have_immutable_sources_and_ambiguous_query_only_targets(tmp_path):
    base = [e for i in range(2) for e in make_multiuse_world(i, bindings=2)]
    episodes, provenance = make_views(base, 8)
    assert len(episodes) == 160 and len(provenance) == 64
    for e in base:
        views = [endpoint_view(e, v) for v in range(8)]
        assert views[0] == e
        assert len({v.query for v in views}) == 1
        assert len({v.answer for v in views}) == (8 if e.task_family == 'multiuse/identifier' else 1)
        for view in views:
            sources = {s.record_id: s for s in view.supports}
            assert all(s.created_at < view.query_time for s in sources.values())
            assert set(view.required_ids) <= sources.keys()
            assert all(set(g) <= sources.keys() for g in view.sufficient_groups)
            assert (view.restore, view.allowed_capability, view.capability) == (e.restore, e.allowed_capability, e.capability)
            if e.task_family == 'multiuse/identifier':
                assert f'Its endpoint is {view.answer}.' in sources[view.required_ids[0]].text
                assert view.answer in view.choices
    path = tmp_path/'episodes.jsonl'
    save_episodes(path, episodes)
    indexed = EpisodeIndex(path)
    assert len(indexed) == len(episodes)
    assert indexed[-1] == episodes[-1]


def test_endpoint_views_reject_future_or_misattributed_targets():
    identifiers = [e for e in make_multiuse_world(3, bindings=2) if e.task_family == 'multiuse/identifier']
    with pytest.raises(ValueError):
        endpoint_view(replace(identifiers[0], query_time=0), 1)
    with pytest.raises(ValueError):
        endpoint_view(replace(identifiers[0], answer=identifiers[1].answer), 1)
