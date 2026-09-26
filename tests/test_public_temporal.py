"""Temporal public-corpus preparation (StreamingQA, TimeQA) on project-authored fixtures."""
from __future__ import annotations

import base64
import datetime as dt
import gzip
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
try:
    import prepare_public_temporal as temporal
finally:
    sys.path.pop(0)


def _ts(date: str, hour: int = 7) -> int:
    return int(dt.datetime.fromisoformat(date).replace(hour=hour, tzinfo=dt.timezone.utc).timestamp())


def _doc_line(date: str, sentences: list[str]) -> tuple[bytes, str]:
    split = base64.b64encode('\n'.join(sentences).encode())
    unsplit = base64.b64encode(' '.join(sentences).encode())
    key = temporal.wmt_key(date.replace('-', ''), unsplit)
    return date.replace('-', '').encode() + b'\t' + split + b'\t' + unsplit + b'\n', key


def _question(qa_id: str, question: str, answer: str, asked: str, key: str, evidence: str) -> dict:
    return {'qa_id': qa_id, 'question': question, 'answers': [answer], 'answers_additional': '',
            'question_ts': _ts(asked), 'evidence_ts': _ts(evidence, 8), 'evidence_id': key,
            'recent_or_past': 'recent', 'written_or_generated': 'generated',
            'toxicity_insult': 0.01, 'toxicity_threat': 0.0}


def _write_gz(path: Path, lines: list[bytes]) -> None:
    with gzip.open(path, 'wb') as handle:
        handle.writelines(lines)


def _streamingqa_fixture(tmp_path: Path) -> tuple[Path, list[Path], Path]:
    raw = tmp_path / 'raw'
    raw.mkdir()
    evidence_a, key_a = _doc_line('2010-03-04', [
        'The harbour council met on Tuesday.', 'It chose Marlow Quay as the new ferry terminal.',
        'Construction starts next spring.'])
    evidence_b, key_b = _doc_line('2016-05-02', [
        'The orchard festival returned this weekend.', 'Its prize went to a pear grower from Tilby.'])
    same_day, key_c = _doc_line('2010-03-04', [
        'A ferry strike was called off by the harbour union.', 'Talks resume next week.'])
    earlier, key_d = _doc_line('2010-03-02', ['The harbour council published its budget.'])
    later, key_e = _doc_line('2010-03-06', ['The harbour council chose a terminal, reports say.'])
    leak, key_f = _doc_line('2010-03-04', ['Marlow Quay residents celebrated the harbour news.'])
    orchard, key_g = _doc_line('2016-05-01', ['Orchard growers prepared for the festival.'])
    duplicate, _ = _doc_line('2010-03-03', ['An undeduplicated harbour copy.'])
    wmt = [tmp_path / 'news-docs.2010.en.filtered.gz', tmp_path / 'news-docs.2016.en.filtered.gz']
    _write_gz(wmt[0], [evidence_a, same_day, earlier, later, leak, duplicate, b'broken line\n'])
    _write_gz(wmt[1], [evidence_b, orchard])
    dedup = tmp_path / 'keys.txt.gz'
    _write_gz(dedup, [(key + '\n').encode() for key in (key_a, key_b, key_c, key_d, key_e, key_f, key_g)])
    train = [
        _question('train-0', 'Which site did the harbour council choose for the ferry terminal?',
                  'Marlow Quay', '2010-03-10', key_a, '2010-03-04'),
        _question('train-1', 'What did the harbour council choose on its meeting day?',
                  'Marlow Quay', '2010-03-04', key_a, '2010-03-04'),
        _question('train-2', 'Where does the harbour council plan the ferry terminal?',
                  'Marlow Quay', '2016-06-01', key_a, '2010-03-04'),
        # The date prefix alone is not evidence: the body never states the date.
        _question('train-3', 'On what day did the harbour council choose the terminal?',
                  'March 4, 2010', '2010-03-12', key_a, '2010-03-04'),
        _question('train-4', 'March 3, 2010', 'Marlow Quay', '2010-03-11', key_a, '2010-03-04'),
    ]
    valid = [_question('valid-0', 'Where was the pear grower who won the orchard prize from?',
                       'Tilby', '2016-05-09', key_b, '2016-05-02'),
             _question('valid-1', 'Which quay was chosen for the ferry terminal?',
                       'Marlow Quay', '2016-02-01', key_a, '2010-03-04')]
    for name, rows in (('streaminqa_train.jsonl.gz', train), ('streaminqa_valid.jsonl.gz', valid)):
        _write_gz(raw / name, [(json.dumps(row) + '\n').encode() for row in rows])
    return raw, wmt, dedup


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_wmt_key_matches_upstream_sorting_key():
    unsplit = base64.b64encode(b'A sentence.')
    key = temporal.wmt_key('20170305', unsplit)
    assert key == ('20170305000000000000\x00\x01' + hashlib.sha256(unsplit).hexdigest() + '\x00\x01')
    assert temporal.key_day(key) == (dt.date(2017, 3, 5) - dt.date(2000, 1, 1)).days
    assert temporal.day_of_timestamp(_ts('2017-03-05', 8)) == temporal.key_day(key)


def test_streamingqa_times_are_causal_days_and_splits_are_ordered(tmp_path):
    raw, wmt, dedup = _streamingqa_fixture(tmp_path)
    manifest = temporal.build_streamingqa(
        raw, tmp_path / 'out', wmt_files=wmt, dedup_keys=dedup, train=5, validation=5,
        max_sources=100, cutoff=dt.date(2016, 4, 1), end=dt.date(2017, 1, 1), workers=1)
    train = _read(tmp_path / 'out' / 'episodes-train.jsonl')
    validation = _read(tmp_path / 'out' / 'episodes-validation.jsonl')
    sources = {row['record_id']: row for row in _read(tmp_path / 'out' / 'sources.jsonl')}
    # Same-day question dropped (never shifted); question after cutoff not in train;
    # valid question before cutoff not in validation.
    assert [row['provenance']['qa_id'] for row in train] == ['train-0']
    assert [row['provenance']['qa_id'] for row in validation] == ['valid-0']
    assert manifest['filtered']['train'] == {'evidence_not_before_query': 1,
                                             'answer_not_in_article_chunk': 1,
                                             'question_too_short': 1}
    assert max(r['query_time'] for r in train) < min(r['query_time'] for r in validation)
    item = train[0]
    day = lambda text: (dt.date.fromisoformat(text) - dt.date(2000, 1, 1)).days  # noqa: E731
    assert item['query_time'] == day('2010-03-10')
    assert 'Date: Wednesday, March 10, 2010' in item['query']
    assert 'Marlow Quay' not in item['query']
    gold = [sources[key] for key in item['required_ids']]
    assert len(gold) == 1 and 'Marlow Quay' in gold[0]['text']
    assert gold[0]['created_at'] == day('2010-03-04')
    assert gold[0]['text'].startswith('Thursday, March 4, 2010. ')
    others = [s for s in item['supports'] if s['record_id'] not in item['required_ids']]
    texts = ' '.join(s['text'] for s in others)
    # Distractors: deduplicated, causally prior (never the later article), no answer leak.
    assert len(others) == 2 and 'Marlow Quay' not in texts
    assert 'undeduplicated' not in texts and 'reports say' not in texts
    assert all(s['created_at'] <= day('2010-03-04') < item['query_time'] for s in item['supports'])
    assert item['sufficient_groups'] == [[key] for key in item['required_ids']]
    assert manifest['distinct_query_times_total'] == 2
    assert manifest['scan']['malformed'] == 1


def test_streamingqa_source_budget_is_enforced(tmp_path):
    raw, wmt, dedup = _streamingqa_fixture(tmp_path)
    manifest = temporal.build_streamingqa(
        raw, tmp_path / 'out', wmt_files=wmt, dedup_keys=dedup, train=5, validation=5,
        max_sources=3, cutoff=dt.date(2016, 4, 1), end=dt.date(2017, 1, 1), workers=1)
    assert manifest['sources'] <= 3
    assert manifest['filtered']['validation'] == {'source_budget': 1}


def _timeqa_rows(split: str) -> list[dict]:
    title = 'Ada Brook'
    body = (' Ada Brook ( 1901 – 1980 ) was a rower . She coached Harbor Club from 1930 to 1935 . '
            + ''.join(f'She joined county board {i} in {1900 + i} . ' for i in range(40))
            + 'She coached River Club from 1940 to 1948 . She retired in 1950 .')
    context = title + body
    rows = []
    for mode, question in (('easy', 'Which team did Ada Brook coach from 1940 to 1948?'),
                           ('hard', 'Which team did Ada Brook coach in 1944?')):
        start = context.index('River Club')
        # Upstream paragraph titles can be section names rather than the page title.
        row = {'idx': f'/wiki/Ada_Brook#P286#{mode}', 'question': question, 'context': context,
               'targets': ['River Club'], 'from': [start], 'end': [start + len('River Club')],
               'paragraphs': [{'title': 'Career', 'text': body}]}
        if split == 'dev':
            row = {key: value for key, value in row.items() if key not in {'from', 'end'}}
            row['targets'], row['paragraphs'] = str(row['targets']), str(row['paragraphs'])
            row['idx'] = row['idx'].replace('Ada_Brook', 'Ada_Brook_%28rower%29')
        rows.append(row)
    rows.append({'idx': '/wiki/Ada_Brook#P286#9', 'question': 'Who employed Ada Brook in 1999?',
                 'context': context, 'targets': [''], 'from': [], 'end': [],
                 'paragraphs': [{'title': title, 'text': body}]})
    return rows


def test_timeqa_gold_and_same_page_distractors(tmp_path):
    raw = tmp_path / 'timeqa'
    raw.mkdir()
    train = _timeqa_rows('train')
    for mode, index in (('easy', 0), ('hard', 1)):
        _write_gz(raw / f'train.{mode}.json.gzip', [(json.dumps(train[index]) + '\n').encode(),
                                                   (json.dumps(train[2]) + '\n').encode()])
        dev = _timeqa_rows('dev')
        (raw / f'dev.{mode}.json').write_text(json.dumps(dev[index]) + '\n')
    manifest = temporal.build_timeqa(raw, tmp_path / 'out', train=4, validation=4, max_sources=50)
    assert manifest['episodes'] == {'train': 2, 'validation': 2}
    assert manifest['filtered']['train'] == {'unanswerable': 2}
    rows = _read(tmp_path / 'out' / 'episodes-train.jsonl') + _read(
        tmp_path / 'out' / 'episodes-validation.jsonl')
    sources = {row['record_id']: row for row in _read(tmp_path / 'out' / 'sources.jsonl')}
    assert {row['provenance']['mode'] for row in rows} == {'easy', 'hard'}
    for row in rows:
        assert row['query_time'] == 2 and 'River Club' not in row['query']
        assert all('River Club' in sources[key]['text'] for key in row['required_ids'])
        others = [s for s in row['supports'] if s['record_id'] not in row['required_ids']]
        assert len(others) == 2 and all('River Club' not in s['text'] for s in others)
        assert all(sources[s['record_id']]['created_at'] == 1 for s in row['supports'])
        title = 'Ada Brook' if row['provenance']['split'] == 'train' else 'Ada Brook (rower)'
        texts = [sources[s['record_id']]['text'] for s in row['supports']]
        assert all(text.startswith(f'Title: {title}\nPassage: ') for text in texts)
        assert not any('Ada Brook Ada Brook' in text for text in texts)  # title prefix stripped
    annotations = {row['provenance']['split']: row['support_annotation'] for row in rows}
    assert annotations == {'train': 'verified_span', 'validation': 'answer_match'}


def test_timeqa_rejects_contexts_that_disagree(tmp_path):
    raw = tmp_path / 'timeqa'
    raw.mkdir()
    first, second = _timeqa_rows('train')[:2]
    second['context'] = second['context'] + ' Extra.'
    second['idx'] = first['idx'].replace('easy', 'other')
    path = raw / 'train.easy.json.gzip'
    _write_gz(path, [(json.dumps(row) + '\n').encode() for row in (first, second)])
    with pytest.raises(ValueError, match='two different contexts'):
        temporal.load_timeqa(path, 'easy', temporal.Writer(tmp_path / 'x', 'timeqa').filters_for('t'))
