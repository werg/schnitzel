"""Held-out evaluation preparation on tiny project-authored fixtures (no downloads)."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
try:
    import prepare_heldout_memory_eval as heldout
finally:
    sys.path.pop(0)


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def by_id(path: Path) -> dict[str, dict]:
    return {row['record_id']: row for row in read(path / 'sources.jsonl')}


def session(*turns):
    return [{'role': role, 'content': text, **({'has_answer': True} if gold else {})}
            for role, text, gold in turns]


def lme_fixture(path: Path) -> None:
    filler = session(('user', 'Can you suggest a pasta recipe for dinner tonight?', False),
                     ('assistant', 'Try a lemon garlic spaghetti with parsley.', False))
    questions = [
        {'question_id': 'ku1', 'question_type': 'knowledge-update',
         'question': 'Which city does my sister live in now?', 'answer': 'Lisbon',
         'question_date': '2023/05/30 (Tue) 10:00',
         'answer_session_ids': ['a_old', 'a_new'],
         'haystack_session_ids': ['a_old', 'noise', 'a_new'],
         'haystack_dates': ['2023/05/01 (Mon) 09:00', '2023/05/02 (Tue) 09:00',
                            '2023/05/30 (Tue) 08:15'],
         'haystack_sessions': [
             session(('user', 'My sister just moved to Porto for work.', True),
                     ('assistant', 'Porto is lovely in spring.', False)),
             filler,
             session(('user', 'Update: my sister relocated again and lives in Lisbon.', True),
                     ('assistant', 'Lisbon has great trams.', False))]},
        {'question_id': 'late1', 'question_type': 'single-session-user',
         'question': 'What is my dog called?', 'answer': 'Rex',
         'question_date': '2023/05/10 (Wed) 12:00', 'answer_session_ids': ['a_dog'],
         'haystack_session_ids': ['a_dog', 'noise'],
         'haystack_dates': ['2023/05/11 (Thu) 12:00', '2023/05/02 (Tue) 09:00'],
         'haystack_sessions': [session(('user', 'My dog Rex loves the beach.', True)), filler]},
        {'question_id': 'pet_abs', 'question_type': 'single-session-user',
         'question': 'What is my hamster called?', 'answer': 'You did not mention a hamster.',
         'question_date': '2023/06/01 (Thu) 12:00', 'answer_session_ids': ['a_cat'],
         'haystack_session_ids': ['a_cat'], 'haystack_dates': ['2023/05/20 (Sat) 12:00'],
         'haystack_sessions': [session(('user', 'My cat Mila sleeps all day.', True))]},
    ]
    path.write_text(json.dumps(questions))


def test_json_array_is_streamed_in_small_blocks(tmp_path):
    path = tmp_path / 'rows.json'
    rows = [{'id': index, 'text': 'x' * (index * 7)} for index in range(40)]
    path.write_text(json.dumps(rows, indent=1))
    assert list(heldout.iter_json_array(path, block=16)) == rows


def test_dialogue_passages_keep_speakers_and_turn_ownership():
    long_turn = ' '.join(f'Sentence number {index} is about gardening.' for index in range(40))
    passages = heldout.dialogue_passages(
        [('user: ', 'Hi there.'), ('assistant: ', long_turn), ('user: ', 'Thanks!')],
        'Chat session on 2023/05/01. ')
    assert all(len(text) <= heldout.MAX_CHARS for text, _ in passages)
    assert all(text.startswith('Chat session on 2023/05/01. ') for text, _ in passages)
    assert passages[0][1][0] == 0 and passages[-1][1][-1] == 2
    assert sum(1 in owners for _, owners in passages) > 1
    assert all('assistant: ' in text for text, owners in passages if 1 in owners)


def test_longmemeval_time_gold_update_and_abstention(tmp_path):
    raw = tmp_path / 'lme.json'
    lme_fixture(raw)
    manifest = heldout.build_longmemeval(raw, tmp_path / 'out', 'longmemeval_s', tokenizer=None,
                                         max_sources=1000, max_questions=10, revision='fixture')
    out = tmp_path / 'out'
    sources = by_id(out)
    assert manifest['role'] == 'heldout_evaluation_only'
    assert manifest['episodes'] == {'test': 1, 'test-abstention': 1}
    assert manifest['filtered']['test'] == {'evidence_not_before_query': 1}
    (update,) = read(out / 'episodes-test.jsonl')
    assert update['query'].endswith('Current date: 2023/05/30 (Tue) 10:00. '
                                    'Which city does my sister live in now?')
    assert update['query_time'] == heldout.minute_number(
        heldout.lme_time('2023/05/30 (Tue) 10:00'))
    (gold,) = update['required_ids']
    assert 'Lisbon' in sources[gold]['text'] and 'Porto' not in sources[gold]['text']
    # Same-day evidence counts because time is in minutes, and it precedes the query.
    assert sources[gold]['created_at'] < update['query_time']
    (old,) = update['provenance']['superseded_ids']
    assert 'Porto' in sources[old]['text']
    assert old in {row['record_id'] for row in update['supports']}
    assert update['provenance']['question_type'] == 'knowledge-update'
    (abstention,) = read(out / 'episodes-test-abstention.jsonl')
    assert abstention['answer'] == 'You did not mention a hamster.'
    # Full haystacks of the selected questions are bank sources, with dated text.
    texts = [row['text'] for row in sources.values()]
    assert any('pasta recipe' in text for text in texts)
    assert all(text.startswith('Chat session on ') for text in texts)
    for row in read(out / 'episodes-test.jsonl') + read(out / 'episodes-test-abstention.jsonl'):
        assert all(item['created_at'] < row['query_time'] for item in row['supports'])
        assert 'Lisbon' not in row['query']


def test_longmemeval_source_cap_keeps_whole_haystacks(tmp_path):
    raw = tmp_path / 'lme.json'
    lme_fixture(raw)
    manifest = heldout.build_longmemeval(raw, tmp_path / 'out', 'longmemeval_m', tokenizer=None,
                                         max_sources=5, max_questions=10, revision='fixture')
    assert manifest['sources'] <= 5
    assert manifest['stats']['selected_questions'] == sum(manifest['episodes'].values())


def locomo_fixture(path: Path) -> None:
    conversation = {
        'speaker_a': 'Ana', 'speaker_b': 'Ben',
        'session_1_date_time': '1:56 pm on 8 May, 2023',
        'session_1': [{'speaker': 'Ana', 'dia_id': 'D1:1', 'text': 'I adopted a parrot named Kiwi.'},
                      {'speaker': 'Ben', 'dia_id': 'D1:2', 'text': 'Cute!',
                       'blip_caption': 'a green parrot on a perch'}],
        'session_2_date_time': '10:00 am on 20 May, 2023',
        'session_2': [{'speaker': 'Ben', 'dia_id': 'D2:1', 'text': 'I started pottery classes.'}],
        'session_3_date_time': '9:00 am on 1 June, 2023',
    }
    qa = [{'question': 'What is the name of the parrot Ana adopted?', 'answer': 'Kiwi',
           'evidence': ['D1:1'], 'category': 4},
          {'question': 'When did Ben start pottery?', 'answer': 'May 2023',
           'evidence': ['D2:1; D1:2'], 'category': 2},
          {'question': 'What is the name of the parrot Ben adopted?', 'evidence': ['D1:1'],
           'category': 5, 'adversarial_answer': 'Kiwi'}]
    path.write_text(json.dumps([{'sample_id': 'conv-1', 'conversation': conversation, 'qa': qa}]))


def test_locomo_times_gold_and_adversarial(tmp_path):
    raw = tmp_path / 'locomo.json'
    locomo_fixture(raw)
    manifest = heldout.build_locomo(raw, tmp_path / 'out', tokenizer=None, revision='fixture')
    out = tmp_path / 'out'
    sources = by_id(out)
    assert manifest['episodes'] == {'test': 2, 'test-adversarial': 1}
    assert manifest['stats']['dated_sessions_without_turns'] == 1
    rows = read(out / 'episodes-test.jsonl')
    last = heldout.day_number(heldout.locomo_time('10:00 am on 20 May, 2023').date())
    assert {row['query_time'] for row in rows} == {last + 1}
    assert 'Kiwi' in sources[rows[0]['required_ids'][0]]['text']
    assert len(rows[1]['required_ids']) == 2
    assert rows[1]['provenance']['category_name'] == 'temporal'
    assert any('[shares a photo: a green parrot' in row['text'] for row in sources.values())
    (adversarial,) = read(out / 'episodes-test-adversarial.jsonl')
    assert adversarial['answer'] == heldout.LOCOMO_REFUSAL
    assert adversarial['provenance']['adversarial_answer'] == 'Kiwi'


def test_multihop_rag_fact_gold_null_split_and_corpus_query_time(tmp_path):
    raw = tmp_path / 'mh'
    raw.mkdir()
    body_a = ('The river festival opened on Friday. Organisers counted 4,000 visitors. '
              'The mayor praised the volunteers.')
    body_b = 'A new bridge will open in spring. It cost 12 million euros to build.'
    corpus = [
        {'title': 'Festival', 'author': 'x', 'source': 'Daily A', 'category': 'news',
         'published_at': '2023-10-01T08:00:00+00:00', 'url': 'u/a', 'body': body_a},
        {'title': 'Bridge', 'author': 'y', 'source': 'Daily B', 'category': 'news',
         'published_at': '2023-10-05T08:00:00+00:00', 'url': 'u/b', 'body': body_b},
        {'title': 'Later', 'author': 'z', 'source': 'Daily C', 'category': 'news',
         'published_at': '2023-12-24T08:00:00+00:00', 'url': 'u/c', 'body': 'Snow fell.'},
    ]
    queries = [
        {'query': 'How many visitors did the festival reported by Daily A have, and what did '
                  'the Daily B bridge cost?', 'answer': '4,000 and 12 million euros',
         'question_type': 'inference_query',
         'evidence_list': [{'url': 'u/a', 'fact': 'Organisers counted 4,000 visitors.'},
                           {'url': 'u/b', 'fact': 'It cost 12 million euros to build.'}]},
        {'query': 'Did Daily A report the festival before Daily B reported the bridge?',
         'answer': 'Yes', 'question_type': 'temporal_query',
         'evidence_list': [{'url': 'u/a', 'fact': 'The river festival opened on Friday.'},
                           {'url': 'u/b', 'fact': 'A new bridge will open in spring.'}]},
        {'query': 'Who won the chess final?', 'answer': 'Insufficient information.',
         'question_type': 'null_query', 'evidence_list': []},
    ]
    (raw / 'corpus.json').write_text(json.dumps(corpus))
    (raw / 'MultiHopRAG.json').write_text(json.dumps(queries))
    manifest = heldout.build_multihop_rag(raw, tmp_path / 'out', tokenizer=None,
                                          revision='fixture')
    out = tmp_path / 'out'
    sources = by_id(out)
    newest = heldout.day_number(heldout.dt.date(2023, 12, 24))
    assert manifest['query_time'] == newest + 1
    assert manifest['episodes'] == {'test': 2, 'test-null': 1}
    first, second = read(out / 'episodes-test.jsonl')
    assert first['sufficient_groups'] == [first['required_ids']]
    texts = [sources[key]['text'] for key in first['required_ids']]
    assert any('4,000' in text for text in texts) and any('12 million' in text for text in texts)
    assert second['answer'] == 'Yes'
    (null,) = read(out / 'episodes-test-null.jsonl')
    assert null['required_ids'] == [] and null['sufficient_groups'] == []
    assert all(row['created_at'] < null['query_time'] for row in null['supports'])


HTML = '''<div class="mw-parser-output"><div class="shortdescription">Short</div>
<table class="infobox"><tr><th>Born</th><td>May 9, 1830</td></tr></table>
<p>Harriet Lane was the niece of James Buchanan.<sup class="reference">[1]</sup> She acted as
first lady.</p><div class="mw-heading"><h2 id="Life">Life<span class="mw-editsection">edit</span></h2></div>
<p>She married Henry Johnston in 1866.</p>
<div class="mw-heading"><h2 id="References">References</h2></div><ol class="references"><li>Cite</li></ol>
<p>Trailing reference-section text.</p></div>'''


def test_wikipedia_html_to_units():
    units = heldout.wiki_units(HTML)
    assert 'Born | May 9, 1830' in units
    assert 'Harriet Lane was the niece of James Buchanan.' in units
    assert 'She married Henry Johnston in 1866.' in units
    assert not any('[1]' in unit or 'Cite' in unit or 'Trailing' in unit or 'Short' in unit
                   or 'edit' in unit for unit in units)


def test_frames_heuristic_gold_and_source_cap(tmp_path):
    raw = tmp_path / 'frames'
    raw.mkdir()
    with (raw / 'test.tsv').open('w', newline='') as handle:
        writer = csv.writer(handle, delimiter='\t')
        writer.writerow(['', 'Prompt', 'Answer', 'reasoning_types', 'wiki_links'])
        writer.writerow(['0', 'Whom did the niece of the fifteenth president marry?',
                         'Henry Johnston', 'Multiple constraints',
                         "['https://en.wikipedia.org/wiki/A', 'https://en.wikipedia.org/wiki/B']"])
        writer.writerow(['1', 'What is on page C?', 'nothing', 'Numerical',
                         "['https://en.wikipedia.org/wiki/C']"])

    def article(title, pageid, units):
        return {'title': title, 'pageid': pageid, 'revid': 10 + pageid, 'host': 'en.wikipedia.org',
                'timestamp': '2024-09-01T00:00:00Z', 'units': units}

    filler = [f'Filler sentence {index} about unrelated weather records and statistics.' * 3
              for index in range(60)]
    articles = {
        'https://en.wikipedia.org/wiki/A': article('James Buchanan', 1, filler + [
            'Buchanan was the fifteenth president; his niece Harriet Lane served as hostess.']),
        'https://en.wikipedia.org/wiki/B': article('Harriet Lane', 2, filler + [
            'Harriet Lane married Henry Johnston in 1866.']),
    }
    manifest = heldout.build_frames(raw, tmp_path / 'out', tokenizer=None, revision='fixture',
                                    max_sources=12, articles=articles)
    out = tmp_path / 'out'
    sources = by_id(out)
    assert manifest['sources'] <= 12
    assert manifest['stats']['article_chunks_dropped_by_cap'] > 0
    assert manifest['filtered']['test'] == {'unfetched_article': 1}
    (row,) = read(out / 'episodes-test.jsonl')
    assert row['support_annotation'] == 'answer_match' and len(row['required_ids']) == 2
    texts = [sources[key]['text'] for key in row['required_ids']]
    assert any('married Henry Johnston' in text for text in texts)
    assert any('niece Harriet Lane' in text for text in texts)
    assert all(sources[key]['provenance']['revid'] in {11, 12} for key in row['required_ids'])


def test_memoryagentbench_conflict_gold_is_latest_fact(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    context = ('Here is a list of facts:\n'
               '0. Ada Lovelace was born in the city of London.\n'
               '1. The capital of Ruritania is Strelsau.\n'
               '2. Ada Lovelace was born in the city of Paris.\n'
               '3. The capital of Ruritania is Zenda.\n'
               '4. Bob plays the position of goalkeeper.\n')
    row = {'context': context,
           'questions': ['Where was Ada Lovelace born?', 'What is the capital of Ruritania?'],
           'answers': [['Paris'], ['Strelsau']],
           'metadata': {'source': 'factconsolidation_sh_32k', 'qa_pair_ids': ['a', 'b']}}
    path = tmp_path / 'cr.parquet'
    pq.write_table(pa.Table.from_pylist([row]), path)
    manifest = heldout.build_memoryagentbench_cr(path, tmp_path / 'out', tokenizer=None,
                                                 revision='fixture')
    out = tmp_path / 'out'
    sources = by_id(out)
    assert manifest['sources'] == 5
    # The Ruritania label contradicts the newer fact: dropped and counted, not guessed.
    assert manifest['filtered']['test'] == {'gold_not_latest': 1}
    (item,) = read(out / 'episodes-test.jsonl')
    (gold,) = item['required_ids']
    assert sources[gold]['text'] == '2. Ada Lovelace was born in the city of Paris.'
    assert sources[gold]['created_at'] == 3 and item['query_time'] == 6
    (old,) = item['provenance']['superseded_ids']
    assert 'London' in sources[old]['text']
