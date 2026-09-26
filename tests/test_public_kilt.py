"""KILT / TopiOCQA preparation on tiny project-authored fixtures (no downloads)."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

pa = pytest.importorskip('pyarrow')
pq = pytest.importorskip('pyarrow.parquet')

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

import prepare_public_kilt as kilt  # noqa: E402
from public_corpus_common import PROMPT, split_sentences  # noqa: E402


def _prov(wid, paragraph, start, end, title='Page'):
    return {'bleu_score': 1.0, 'start_character': start, 'start_paragraph_id': paragraph,
            'end_character': end, 'end_paragraph_id': paragraph,
            'meta': {'fever_page_id': '', 'fever_sentence_id': -1, 'annotation_id': '-1',
                     'yes_no_answer': '', 'evidence_span': []},
            'section': 'Section::::Abstract.', 'title': title, 'wikipedia_id': wid}


def _row(identifier, text, outputs):
    return {'id': identifier, 'input': text, 'output': [
        {'answer': answer, 'meta': {'score': -1}, 'provenance': provenance}
        for answer, provenance in outputs]}


FILLER = ('This filler sentence talks about the weather in a distant valley at length. ' * 3).strip()
PAGES = {
    '10': ['Glass Harbor\n',
           'Glass Harbor is a fictional port. It was founded by Mara Vell in 1702. ' + FILLER + '\n',
           'Section::::History.\n',
           'BULLET::::- The harbor lighthouse is painted green and white along its tower. '
           + FILLER + '\n',
           'The harbor exports woven baskets to three neighbouring islands every year. '
           + FILLER + '\n'],
    '20': ['Copper Finch\n',
           'The copper finch is a songbird. It nests in hollow reeds near slow rivers. '
           + FILLER + '\n',
           'Its song has four notes and repeats at dawn during the whole breeding season. '
           + FILLER + '\n'],
    '30': ['Unrelated\n', 'Nothing here is needed. ' + FILLER + '\n'],
}


def _span(wid, paragraph, needle):
    start = PAGES[wid][paragraph].index(needle)
    return start, start + len(needle)


def _fixtures(tmp_path: Path) -> Path:
    raw = tmp_path / 'raw'
    founded = _span('10', 1, 'Mara Vell')
    reeds = _span('20', 1, 'hollow reeds')
    green = _span('10', 3, 'green and white')
    nq = [_row('n1', 'who founded glass harbor', [
              ('Mara Vell', [_prov('10', 1, *founded, 'Glass Harbor')]),
              ('Somebody Else', [])]),
          _row('n2', 'where does the copper finch nest', [
              ('in hollow reeds', [_prov('20', 1, reeds[0] - 3, reeds[1], 'Copper Finch')])]),
          _row('n3', 'what colour is the lighthouse', [
              ('green and white', [_prov('10', 3, *green, 'Glass Harbor')])])]
    fever = [_row('f1', 'Glass Harbor was founded by Mara Vell.', [
                 ('SUPPORTS', [_prov('10', 1, 0, len(PAGES['10'][1]) - 1, 'Glass Harbor')])]),
             _row('f2', 'The copper finch sings seven notes.', [
                 ('REFUTES', [_prov('20', 2, 0, 70, 'Copper Finch')])]),
             _row('f3', 'The copper finch nests in trees.', [
                 ('REFUTES', [_prov('20', 1, *reeds, 'Copper Finch')])])]
    zsre = [_row('z1', 'Glass Harbor [SEP] founded by', [
                ('Mara Vell', [_prov('10', 1, *founded, 'Glass Harbor')])]),
            _row('z2', 'Copper Finch [SEP] nesting site', [
                ('hollow reeds', [_prov('20', 1, *reeds, 'Copper Finch')])])]
    for task, rows in (('nq', nq), ('fever', fever), ('structured_zeroshot', zsre)):
        for split in ('train', 'validation'):
            path = raw / 'kilt_tasks' / task / f'{split}-00000-of-00001.parquet'
            path.parent.mkdir(parents=True, exist_ok=True)
            pick = rows if split == 'train' else rows[:1]
            if split == 'validation':
                pick = [{**row, 'id': row['id'] + 'v'} for row in pick]
            pq.write_table(pa.Table.from_pylist(pick), path)
    source = raw / kilt.KNOWLEDGE_SOURCE
    source.parent.mkdir(parents=True)
    with source.open('w') as handle:
        for wid, paragraphs in PAGES.items():
            handle.write(json.dumps({'_id': wid, 'wikipedia_id': wid,
                                     'wikipedia_title': paragraphs[0].strip(),
                                     'text': paragraphs, 'anchors': []}) + '\n')
    passage = {'id': 'wiki:1', 'title': 'Glass Harbor [SEP] History',
               'text': 'Glass Harbor was founded by Mara Vell in 1702. Its lighthouse is green.'}
    other = {'id': 'wiki:2', 'title': 'Glass Harbor [SEP] Trade',
             'text': 'The harbor exports woven baskets to three islands.'}
    turns = [
        {'Conversation_no': 1, 'Turn_no': 1, 'Question': 'who founded glass harbor',
         'Answer': 'Mara Vell', 'Rationale': 'founded by Mara Vell', 'Context': [],
         'Gold_passage': passage},
        {'Conversation_no': 1, 'Turn_no': 2, 'Question': 'in what year did she do it?',
         'Answer': '1702', 'Rationale': 'Mara Vell in 1702',
         'Context': ['who founded glass harbor', 'Mara Vell'], 'Gold_passage': passage},
        {'Conversation_no': 1, 'Turn_no': 3, 'Question': 'what does it export?',
         'Answer': 'woven baskets', 'Rationale': 'exports woven baskets',
         'Context': ['who founded glass harbor', 'Mara Vell', 'in what year did she do it?',
                     '1702'], 'Gold_passage': other},
        {'Conversation_no': 1, 'Turn_no': 4, 'Question': 'anything else?',
         'Answer': 'UNANSWERABLE', 'Rationale': '', 'Context': [], 'Gold_passage': other},
    ]
    for name in ('topiocqa_train.jsonl', 'topiocqa_valid.jsonl'):
        path = raw / 'topiocqa' / 'data' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps(turn) + '\n' for turn in turns))
    return raw


def _episodes(path: Path, split: str) -> list[dict]:
    return [json.loads(line) for line in (path / f'episodes-{split}.jsonl').open()]


def test_sentence_spans_match_shared_splitter():
    text = 'One sentence here. Another one!  A third?\nAnd a fourth'
    assert [text[a:b] for a, b in kilt.sentence_spans(text)] == split_sentences(text)


def test_paragraph_chunks_mark_span_and_strip_markup():
    paragraph = PAGES['10'][3]
    start, end = _span('10', 3, 'green and white')
    pieces = kilt.paragraph_chunks(paragraph, [(start, end)])
    assert not pieces[0][0].startswith('BULLET')
    gold = [text for text, is_gold in pieces if is_gold]
    assert gold and all('green and white' in text for text in gold)
    assert [t for t, _ in pieces] == [t for t, _ in kilt.paragraph_chunks(paragraph)]
    assert all(len(text) <= 700 for text, _ in pieces)


def test_scan_streams_only_needed_pages(tmp_path):
    raw = _fixtures(tmp_path)
    pages = kilt.scan_knowledge_source(raw / kilt.KNOWLEDGE_SOURCE, {'10': {1}}, log_every=0)
    assert set(pages) == {'10'}
    assert 1 in pages['10']['paragraphs']
    assert 2 not in pages['10']['distractors']  # section headers are not distractors
    assert set(pages['10']['distractors']) <= {3, 4}


def test_prepare_all_domains(tmp_path):
    raw = _fixtures(tmp_path)
    out = tmp_path / 'out'
    summaries = kilt.prepare(raw, out, list(kilt.DEFAULTS), tokenizer=None)
    assert set(summaries) == set(kilt.DEFAULTS)
    for domain in kilt.DEFAULTS:
        sources = {json.loads(line)['record_id']: json.loads(line)
                   for line in (out / domain / 'sources.jsonl').open()}
        for split in ('train', 'validation'):
            for row in _episodes(out / domain, split):
                assert row['query'].startswith(PROMPT)
                if domain == 'kilt_fever':
                    claim = row['query'].split('Claim: ', 1)[1]
                    assert row['answer'] not in claim.lower()
                else:
                    assert row['answer'].lower() not in row['query'].lower()
                assert row['query_time'] == 2
                assert len(row['supports']) <= len(row['required_ids']) + 2
                for record in row['supports']:
                    assert sources[record['record_id']]['created_at'] == 1
                    assert sources[record['record_id']]['domain'] == domain
                    assert len(sources[record['record_id']]['text']) <= 700 + 200
                if domain != 'kilt_fever':
                    for key in row['required_ids']:
                        if row['answer'] in {'Mara Vell', '1702', 'woven baskets',
                                             'in hollow reeds', 'hollow reeds',
                                             'green and white'}:
                            assert row['answer'].lower() in sources[key]['text'].lower()
                    distractors = {r['record_id'] for r in row['supports']} - set(row['required_ids'])
                    assert all(row['answer'].lower() not in sources[key]['text'].lower()
                               for key in distractors)

    nq = {row['episode_id']: row for row in _episodes(out / 'kilt_nq', 'train')}
    assert nq['kilt_nq-n1']['answer'] == 'Mara Vell'
    assert nq['kilt_nq-n1']['sufficient_groups'] == [nq['kilt_nq-n1']['required_ids']]
    assert len(nq) == 3

    fever = {row['episode_id']: row for row in _episodes(out / 'kilt_fever', 'train')}
    assert fever['kilt_fever-f1']['answer'] == 'supported'
    assert fever['kilt_fever-f2']['answer'] == 'refuted'
    assert fever['kilt_fever-f1']['query'].endswith('Claim: Glass Harbor was founded by Mara Vell.')

    zsre = {row['episode_id']: row for row in _episodes(out / 'kilt_zsre', 'train')}
    assert zsre['kilt_zsre-z1']['query'].endswith('What is the founded by of Glass Harbor?')
    assert zsre['kilt_zsre-z1']['provenance']['relation'] == 'founded by'

    topics = {row['episode_id']: row for row in _episodes(out / 'topiocqa', 'train')}
    assert 'topiocqa-train-1-4' not in topics  # unanswerable dropped
    second = topics['topiocqa-train-1-2']
    assert 'Q: who founded glass harbor A: Mara Vell' in second['query']
    assert second['query'].endswith('Current question: in what year did she do it?')
    third = topics['topiocqa-train-1-3']
    gold = {r['record_id']: r['text'] for r in third['supports']}
    assert any('woven baskets' in gold[key] for key in third['required_ids'])
    manifest = json.loads((out / 'topiocqa' / 'manifest.json').read_text())
    assert 'NC-SA' in manifest['license']


def test_alternative_provenances_are_any_one_groups(tmp_path):
    raw = _fixtures(tmp_path)
    pages = kilt.scan_knowledge_source(raw / kilt.KNOWLEDGE_SOURCE, {'10': {1, 3}}, log_every=0)
    founded = _span('10', 1, 'Mara Vell')
    row = _row('a', 'who founded glass harbor', [
        ('Mara Vell', [_prov('10', 1, *founded)]),
        ('mara vell', [_prov('10', 3, 0, 20)])])  # second provenance lacks the answer
    plan = kilt.plan_nq(row)
    assert len(plan['groups']) == 2
    writer = kilt.Writer(tmp_path / 'w', 'kilt_nq')
    groups, used = kilt.resolve_kilt('kilt_nq', plan, pages, writer.filters_for('train'))
    assert len(groups) == 1 and ('10', 3) in used
    assert writer.filters_for('train').counts['provenance_without_answer'] == 1


def test_query_history_truncates_oldest_first():
    class Words:
        def encode(self, text, add_special_tokens=False):
            return text.split()

    row = {'Question': 'and then?', 'Context': [f'q{i} ' + 'w ' * 30 if i % 2 == 0 else f'a{i}'
                                                for i in range(8)]}
    text = kilt.conversational_question(row, Words())
    assert text is not None and text.endswith('Current question: and then?')
    assert 'q6' in text and 'q0' not in text
    assert kilt.conversational_question({'Question': 'x', 'Context': ['w ' * 200, 'a']},
                                        Words()) is None


def test_zsre_relation_cap():
    rows = [_row(f'r{i}', f'S{i} [SEP] {"common" if i < 80 else f"rare{i}"}',
                 [('A', [_prov('1', 1, 0, 1)])]) for i in range(100)]
    chosen = kilt.select_rows('kilt_zsre', rows, 'train', 100, 1)
    assert kilt.relation_cap('train', 100, 21) == 3
    assert sum(plan['meta']['relation'] == 'common' for _, plan in chosen) <= 10
