import json

from schnitz.routing_curriculum import RoutingCandidateIndex, routing_mix


def test_candidate_index_respects_time_domain_and_lexical_signal_is_soft(tmp_path):
    sources = tmp_path / 'sources.jsonl'
    sources.write_text(''.join(json.dumps(row) + '\n' for row in (
        {'record_id': 'a', 'text': 'Title: Copper River\nPassage: salmon',
         'domain': 'research', 'created_at': 1},
        {'record_id': 'b', 'text': 'Title: Copper River\nPassage: later',
         'domain': 'research', 'created_at': 5},
        {'record_id': 'c', 'text': 'Title: Copper River\nPassage: private',
         'domain': 'private', 'created_at': 1},
        {'record_id': 'd', 'text': 'Title: Granite Hill\nPassage: stone',
         'domain': 'research', 'created_at': 1},
    )))
    index = RoutingCandidateIndex(sources)
    assert set(index.sample('episode', 3, domain='research', query_time=2,
                            limit=4)) == {'a', 'd'}
    scores = index.lexical_similarities('Copper River salmon', ('a', 'd'))
    assert scores[0] > scores[1] >= 0
    assert index.lexical_similarities('Copper River', ('b',),
                                      domain='research', query_time=2) == ()


def test_routing_mix_moves_from_easy_to_global():
    assert routing_mix(0) == (1.0, 0.0)
    easy, global_weight = routing_mix(1000)
    assert 0 < easy < 1 and 0 < global_weight < 1
    assert routing_mix(3000) == (0.25, 0.75)
    assert routing_mix(1000, ramp_steps=1000) == (0.25, 0.75)
