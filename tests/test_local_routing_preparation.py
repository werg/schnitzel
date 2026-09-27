from dataclasses import asdict
import json
import pytest

from schnitz.data import Episode, Source, load_episodes


@pytest.mark.parametrize(('family', 'locator'), [
    ('passage_qa', 'source_article_title'),
    ('located_passage_span', 'article-and-passage-prefix-v1'),
])
def test_local_routing_prep_retains_verified_causal_source_and_deterministic_distractors(
        tmp_path, family, locator):
    from scripts.prepare_local_routing import prepare
    sources = [Source(f's{i}', f'Title: Article_{i}\nPassage: fact {i}', 1, 'passage')
               for i in range(4)]
    source_path = tmp_path / 'sources.jsonl'
    source_path.write_text(''.join(json.dumps(asdict(source) | {
        'provenance': {'article_title': f'Article_{i}'}}) + '\n'
        for i, source in enumerate(sources)))
    episode = Episode('q', 'squad-train', (sources[0],),
                      'Article: Article_0\nQuestion: Which fact?', 'fact 0',
                      ('s0',), False, 0, 0, 2, family, (), (('s0',),),
                      'verified', {'article_title': 'Article_0',
                                   'query_locator': locator})
    episodes_path = tmp_path / 'input.jsonl'
    episodes_path.write_text(json.dumps(asdict(episode)) + '\n')
    first = prepare(episodes_path, source_path, tmp_path / 'out1',
                    distractors=2, require_external=False, candidate_source_limit=3)
    second = prepare(episodes_path, source_path, tmp_path / 'out2',
                     distractors=2, require_external=False, candidate_source_limit=3)
    assert first['episodes_sha256'] == second['episodes_sha256']
    prepared = load_episodes(tmp_path / 'out1' / 'episodes.jsonl')[0]
    assert prepared.query == episode.query and prepared.answer == episode.answer
    assert prepared.required_ids == ('s0',) and prepared.sufficient_groups == (('s0',),)
    assert prepared.supports[0] == sources[0]
    assert len({source.record_id for source in prepared.supports}) == 3
    assert 's3' not in {source.record_id for source in prepared.supports}
    assert first['candidate_source_limit'] == 3
    assert all(source.created_at < prepared.query_time for source in prepared.supports)
    assert prepared.provenance['distractor_ids'] == [source.record_id
                                                    for source in prepared.supports[1:]]
