import hashlib
import json

from schnitz.backbones import ByteTokenizer
from schnitz.data import load_episodes
from schnitz.corpus_data import prepare_squad


def test_short_reconstruction_keeps_source_before_query_and_target_out_of_prompt():
    from schnitz.corpus_data import short_reconstruction
    from schnitz.data import Source
    source = Source('opaque-source', 'Title: Example\nPassage: Alpha beta gamma delta epsilon.', 1, 'passage')
    episode = short_reconstruction(source, max_words=3)
    assert episode.supports == (source,)
    assert episode.answer == 'Alpha beta gamma'
    assert episode.answer not in episode.query
    assert episode.query_time > source.created_at
    assert episode.required_ids == (source.record_id,)


def test_short_reconstruction_respects_complete_target_token_budget():
    from schnitz.corpus_data import short_reconstruction
    from schnitz.data import Source
    tokenizer = ByteTokenizer()
    source = Source('opaque', 'Title: Example\nPassage: alpha beta gamma delta', 1, 'passage')
    episode = short_reconstruction(source, max_words=4, tokenizer=tokenizer, max_target_tokens=8)
    assert episode.answer == 'alpha'
    assert 'first 1 word' in episode.query
    assert len(tokenizer.encode(episode.answer, add_special_tokens=False)) + 1 <= 8


def test_short_reconstruction_preserves_source_whitespace():
    from schnitz.corpus_data import short_reconstruction
    from schnitz.data import Source
    source = Source('opaque', 'Title: Example\nPassage: Alpha  beta\ngamma.', 1, 'passage')
    assert short_reconstruction(source, max_words=2).answer == 'Alpha  beta'


def test_eligible_distractor_preserves_verified_target_and_causal_identity():
    from schnitz.corpus_data import short_reconstruction, with_distractor
    from schnitz.data import Source
    primary = Source('a', 'Title: A\nPassage: Alpha beta.', 1, 'passage')
    decoy = Source('b', 'Title: B\nPassage: Copper silver.', 1, 'passage')
    original = short_reconstruction(primary)
    mixed = with_distractor(original, decoy)
    assert mixed.required_ids == original.required_ids
    assert mixed.answer == original.answer and mixed.query == original.query
    assert mixed.supports == (primary, decoy)
    assert mixed.sufficient_groups == original.sufficient_groups
    assert all(s.created_at < mixed.query_time for s in mixed.supports)


def test_title_located_question_retains_target_and_prior_source():
    from schnitz.corpus_data import title_located_question
    from schnitz.data import Episode, Source
    source = Source('source', 'Title: Helios\nPassage: Helios uses the code aqua.', 1, 'passage')
    original = Episode('q', 'squad-train', (source,), 'What code does it use?', 'aqua',
                       ('source',), False, 0, 0, 2, 'passage_qa', (),
                       (('source',),), 'verified', {'article_title': 'Helios'})
    located = title_located_question(original)
    assert 'Helios' in located.query and located.answer not in located.query
    assert located.supports == original.supports
    assert located.required_ids == original.required_ids
    assert located.query_time == original.query_time
    assert located.answer == original.answer


def test_title_locator_rejects_answer_hidden_by_underscores():
    import pytest
    from schnitz.corpus_data import title_located_question
    from schnitz.data import Episode, Source
    source = Source('source', 'Title: Royal_Institute_of_British_Architects', 1, 'passage')
    original = Episode('q', 'squad-train', (source,), 'What does RIBA stand for?',
                       'Royal Institute of British Architects', ('source',), False,
                       0, 0, 2, 'passage_qa', (), (('source',),), 'verified',
                       {'article_title': 'Royal_Institute_of_British_Architects'})
    with pytest.raises(ValueError, match='disclose'):
        title_located_question(original)


def _title(want_validation):
    for i in range(100):
        title = f'Article {i}'
        if (int(hashlib.sha256(title.encode()).hexdigest()[:8], 16) % 10 == 0) == want_validation:
            return title
    raise AssertionError('No title for test split')


def test_squad_source_manifest_preserves_passage_identity_and_causal_answers(tmp_path):
    train, heldout = _title(False), _title(True)
    context = 'Aster stores the access phrase violet-73 in the field manual.'
    paragraph = {'context': context, 'qas': [
        {'id': 'q1', 'question': 'What is the access phrase?', 'is_impossible': False,
         'answers': [{'text': 'violet-73', 'answer_start': context.index('violet-73')}]},
        {'id': 'q2', 'question': 'Where is the phrase stored?', 'is_impossible': False,
         'answers': [{'text': 'field manual', 'answer_start': context.index('field manual')}]},
        {'id': 'q3', 'question': 'What does not exist?', 'is_impossible': True, 'answers': []},
    ]}
    heldout_context = 'Beryl stores the access phrase silver-28 in another field manual.'
    heldout_paragraph = {'context': heldout_context, 'qas': [
        {'id': 'heldout-q1', 'question': 'What is the access phrase?', 'is_impossible': False,
         'answers': [{'text': 'silver-28', 'answer_start': heldout_context.index('silver-28')}]},
        {'id': 'heldout-q2', 'question': 'Where is the phrase stored?', 'is_impossible': False,
         'answers': [{'text': 'field manual', 'answer_start': heldout_context.index('field manual')}]},
    ]}
    raw = tmp_path / 'raw.json'
    raw.write_text(json.dumps({'version': 'v2.0', 'data': [
        {'title': train, 'paragraphs': [paragraph]},
        {'title': heldout, 'paragraphs': [heldout_paragraph]},
    ]}))
    output = tmp_path / 'prepared'
    manifest = prepare_squad(raw, output, ByteTokenizer(), max_train_sources=1,
                             max_validation_sources=1, max_source_tokens=512)
    assert manifest['sources']['train'] == manifest['sources']['validation'] == 1
    training = load_episodes(output / 'train.jsonl')
    validation = load_episodes(output / 'validation.jsonl')
    assert len(training) == len(validation) == 2
    assert {e.provenance['article_title'] for e in training} == {train}
    assert {e.provenance['article_title'] for e in validation} == {heldout}
    assert {e.supports[0].record_id for e in training}.isdisjoint(
        {e.supports[0].record_id for e in validation})
    assert {e.supports[0].record_id for e in training} == {
        json.loads(line)['record_id'] for line in (output / 'sources-train.jsonl').read_text().splitlines()}
    assert all(e.supports[0].created_at < e.query_time and e.required_ids == (e.supports[0].record_id,)
               and e.answer not in e.query and e.supports[0].text.endswith(
                   context if e.provenance['article_title'] == train else heldout_context)
               for e in training + validation)
    assert manifest['sha256']['sources-train.jsonl']


def test_squad_prep_fails_on_unfilled_source_budget(tmp_path):
    raw = tmp_path / 'raw.json'
    raw.write_text(json.dumps({'version': 'v2.0', 'data': []}))
    try:
        prepare_squad(raw, tmp_path / 'data', ByteTokenizer(), max_train_sources=1,
                      max_validation_sources=1)
    except ValueError as exc:
        assert 'source budget' in str(exc)
    else:
        raise AssertionError('Empty corpus must not publish a complete manifest')
