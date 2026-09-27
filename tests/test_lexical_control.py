"""A source-text retrieval comparator must keep causal and visibility boundaries."""
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def lexical():
    path = Path(__file__).parents[1]/'scripts/evaluate_lexical_routing.py'
    spec = importlib.util.spec_from_file_location('lexical_control', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def record(record_id, terms, **overrides):
    return dict(record_id=record_id, terms=terms, namespace='global', space='s0',
                generation='frozen-v1', domain='research', created_at=1) | overrides


def test_rank_uses_only_visible_features_and_handles_empty(lexical):
    records = [record('a', {'service': 1, 'alpha': 1}), record('b', {'service': 1, 'beta': 1})]
    def rank(docs):
        return lexical.rank(docs, 'service alpha', query_time=4, top_k=2)
    original = rank(records)
    assert [r['record_id'] for r in original] == ['a', 'b']
    for field, value in [('created_at', 4), ('created_at', 99), ('domain', 'private'),
                         ('namespace', 'other'), ('space', 'other'), ('generation', 'old')]:
        hidden = record('hidden', {'service': 1000, 'alpha': 1000})
        hidden[field] = value
        assert rank(records+[hidden]) == original
    assert lexical.rank(records, 'alpha', query_time=1, top_k=2) == []
    assert lexical.rank(records, 'outofvocabulary', query_time=4, top_k=2) == []
    with pytest.raises(ValueError, match='top_k'):
        lexical.rank(records, 'alpha', query_time=4, top_k=0)


def test_query_target_annotations_cannot_enter_ranker(lexical):
    # Textual source/query identifiers are content; operational IDs are opaque.
    rows = [record('opaque-x', lexical.terms('svc_blue permits retry.')),
            record('opaque-y', lexical.terms('svc_red forbids retry.'))]
    before = lexical.rank(rows, 'May svc_blue retry?', query_time=4, top_k=1)
    changed = [dict(row, record_id='renamed-'+row['record_id']) for row in rows]
    after = lexical.rank(changed, 'May svc_blue retry?', query_time=4, top_k=1)
    assert before[0]['record_id'] == 'opaque-x'
    assert after[0]['record_id'] == 'renamed-opaque-x'
    assert before[0]['score'] == after[0]['score']
    assert lexical.terms('RUN rm -rf / ; API_ab12ef') == {'run': 1, 'rm': 1, 'rf': 1, 'api_ab12ef': 1}


def test_sufficient_group_is_not_the_full_annotated_pair(lexical):
    from types import SimpleNamespace
    episode = SimpleNamespace(required_ids=('permission', 'restoration'),
                              sufficient_groups=(('permission',),))
    assert lexical.support_metrics(episode, ['permission', 'distractor']) == {
        'complete_support': True, 'all_required': False}
    assert lexical.support_metrics(episode, ['permission', 'restoration']) == {
        'complete_support': True, 'all_required': True}


def test_index_preserves_source_versions_and_bank_visibility(lexical, tmp_path):
    import json
    import sqlite3
    from schnitz.data import make_multiuse_world, save_episodes
    episodes = make_multiuse_world(1, bindings=2)
    corpus = tmp_path/'episodes.jsonl'
    save_episodes(corpus, episodes)
    bank = tmp_path/'bank.sqlite'
    sources = {s.record_id: s for e in episodes for s in e.supports}
    with sqlite3.connect(bank) as db:
        db.execute('CREATE TABLE records (record_id TEXT, source_id TEXT, namespace TEXT, space TEXT, generation TEXT, domain TEXT, created_at INTEGER, deleted INTEGER)')
        db.executemany('INSERT INTO records VALUES (?,?,?,?,?,?,?,?)',
                       [(s.record_id, s.record_id, 'global', 's0', 'frozen-v1', 'research', s.created_at, 0)
                        for s in sources.values()])
    index = lexical.build_index(bank, corpus)
    assert len(index['records']) == 4
    assert {r['record_id'] for r in index['records']} == sources.keys()
    encoded = json.dumps(index)
    assert episodes[0].query not in encoded
    with sqlite3.connect(bank) as db:
        db.execute('UPDATE records SET created_at=999 WHERE record_id=?', (next(iter(sources)),))
    with pytest.raises(ValueError, match='source time'):
        lexical.build_index(bank, corpus)
