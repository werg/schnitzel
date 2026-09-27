"""Position-level copy supervision must stay causal and source-matched."""
from dataclasses import replace
import importlib.util
from pathlib import Path

import pytest

from schnitz.data import make_multiuse_world, save_episodes
from schnitz.episode_index import EpisodeIndex


@pytest.fixture
def characters():
    path = Path(__file__).parents[1]/'scripts/make_identifier_character_control.py'
    spec = importlib.util.spec_from_file_location('identifier_characters', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_paired_character_corpora_preserve_sources_and_nonidentifier_tasks(characters, tmp_path):
    original = make_multiuse_world(3, bindings=2)
    full, positions = characters.paired_episodes(original)
    assert len(full) == len(positions) == 22
    assert [e for e in full if e.task_family != 'multiuse/identifier'] == [
        e for e in original if e.task_family != 'multiuse/identifier']
    assert sum(e.task_family == 'identifier_character' for e in positions) == 12
    for a, b in zip(full, positions):
        assert (a.supports, a.required_ids, a.query_time, a.environment) == (
            b.supports, b.required_ids, b.query_time, b.environment)
        if b.task_family == 'identifier_character':
            index = b.provenance['endpoint_character_position']
            assert b.answer == a.answer[4+index-1]
            assert a.answer not in b.query
            assert b.choices == tuple('0123456789abcdef')
    for name, episodes in [('full', full), ('positions', positions)]:
        path = tmp_path/(name+'.jsonl')
        save_episodes(path, episodes)
        assert len(EpisodeIndex(path)) == 22
    assert characters.paired_episodes(original) == (full, positions)


@pytest.mark.parametrize('position', [0, 7, True])
def test_character_position_requires_valid_index(characters, position):
    e = next(e for e in make_multiuse_world(1) if e.task_family == 'multiuse/identifier')
    with pytest.raises(ValueError, match='position'):
        characters.character_episode(e, position)


def test_character_labels_require_the_actual_prior_selected_source(characters):
    e = next(e for e in make_multiuse_world(1) if e.task_family == 'multiuse/identifier')
    future = replace(e, supports=tuple(replace(s, created_at=e.query_time) for s in e.supports))
    with pytest.raises(ValueError, match='prior'):
        characters.character_episode(future, 1)
    wrong = replace(e, answer='api_000000' if e.answer != 'api_000000' else 'api_ffffff')
    with pytest.raises(ValueError, match='selected source'):
        characters.character_episode(wrong, 1)
